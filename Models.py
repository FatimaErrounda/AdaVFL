import torch
import torch.nn as nn
from torch.nn.modules.linear import Linear
import torch.optim as optim
import math
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.nn.utils import clip_grad_norm_
from prettytable import PrettyTable

###############################
# Debugging
###############################
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

###############################
# Structure for managing the privacy budgets when parallel training
###############################

class Budget:
  Validation_accuracy = []
  Loop_accuracy = []
  def __init__(self,args, initial_sigma):
    self.cumulated_budget = 0.
    if args.PrivacyMode !=  "Concentrated": 
      self.Validation_accuracy =  np.full(args.validation_period, 0.) 
      self.Loop_accuracy =  np.full(args.epoch_period, 0.) 
      self.privacy_budgets = initial_sigma
      self.local_loss = 0.
      self.minimum_training_accuracy = args.gamma_min
    else:
      total_epsilon = args.epsilon_0
      total_delta = args.delta_0
      self.eps_nmax =  (total_epsilon * 0.5) / args.epochs
      self.rho_nmax = initial_sigma
      self.rho_ng = self.rho_nmax
      self.rho = 4*args.total_rho
      self.chosen_step_sizes = []
      self.max_step_size =0.001
      self.n_candidate = 5
      self.exp_dec = 1.0
    
Shuffling = True
###############################
# GRU model
###############################
class GRU(nn.Module):
  def __init__(self, args, lr=0.0001):
    super().__init__()
    self.args = args
    if args.dataset == "bikeNYC":
      self.lr = 0.0001
    else:
      if (args.dataset == "Yelp"):
        self.lr = 0.0001 

    self.number_features = 1
    self.hidden_layer = 2*self.number_features
    self.output_size= args.output_length
    self.batch_size = 1
    self.sequence_length = args.input_length

    self.gru = nn.GRU(self.number_features, self.hidden_layer, self.output_size, batch_first=False)
    self.linear = nn.Linear(self.hidden_layer, self.output_size)
    self.relu = nn.ReLU()

    torch.nn.init.xavier_uniform_(self.gru.weight_ih_l0)
    torch.nn.init.xavier_uniform_(self.gru.weight_hh_l0)
    torch.nn.init.xavier_uniform_(self.linear.weight)
    self.gru.bias_ih_l0.data.fill_(0.01)
    self.gru.bias_hh_l0.data.fill_(0.01)
    self.linear.bias.data.fill_(0.01)
    
    if(args.local_model_loss == 'MSE'):
      self.loss_function = nn.MSELoss().to(self.args.device_name)
    else:
      if(args.local_model_loss == 'MAE'):
        self.loss_function = nn.L1Loss().to(self.args.device_name)
      else:
        raise SystemError('Need to specify a loss function')

    for param in self.parameters():
      param.grad = None
    
    for param in self.parameters():
      param.accumulated_grads = []
    
    if self.args.PrivacyMode == "None":
      self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
  
  def debug(self):
    for p in self.parameters():
      print(p.grad.shape)
      
  def InitializeBudget(self,initial_sigma):
    self.Budget = Budget(self.args,initial_sigma) 
  
  def init_hidden(self):
    weight = next(self.parameters()).data
    hidden = weight.new(1*self.output_size, self.batch_size, self.hidden_layer).zero_().to(self.args.device_name)
    return hidden

  def fwd_pass(self, input_Sequence, train=False, noise = 0., o_clipping = False):
    acc = {}
    losses = []
    n = len(input_Sequence)
    if train:
      if self.args.PrivacyMode == "None":
        self.optimizer.zero_grad()
        for param in self.parameters():    
          param.grad = None

#     print(input_Sequence[0][0],input_Sequence[0][1] )    
    hidden_cell= self.init_hidden()
#     if Shuffling == True:
    Indexes = [i for i in range(n-1)]
    np.random.shuffle(Indexes)
#     print(Indexes)
#     for seq, labels in input_Sequence:
    for i in Indexes:
      seq = input_Sequence[i][0]
      labels = input_Sequence[i][1]
      seq = seq.to(self.args.device_name)
      labels = labels.to(self.args.device_name)
      labels = labels.squeeze()
      if train:
        hidden_cell = hidden_cell.data

      y_pred, hidden_cell = self.forward(seq,hidden_cell)
      y_pred = y_pred.squeeze()
      if "MAE" in acc:
        acc["MAE"] += abs(labels - y_pred)
      else: 
        acc["MAE"] = abs(labels - y_pred)
      if "MSE" in acc:
        acc["MSE"] += (labels - y_pred) * (labels - y_pred) 
      else:
        acc["MSE"] = (labels - y_pred) * (labels - y_pred) 
      if "AE" in acc:
        acc["AE"] += labels - y_pred
      else:
        acc["AE"] = labels - y_pred
      if "WMAPE" in acc:
        acc["WMAPE"] += abs(labels)
      else:
        acc["WMAPE"] = abs(labels)

#       print("GRU:seq.shape",seq.shape)
#       print("GRU:hidden_cell.shape",hidden_cell.shape)
#       print("GRU:y_pred.shape",y_pred.shape)
#       print("GRU:labels.shape",labels.shape)
        
      single_loss = self.loss_function(y_pred, labels)
      losses.append(single_loss.item())
      
      if train:
        # compute the gradient for this sample
        single_loss.backward()
        if self.args.PrivacyMode != "None":
          for param in self.parameters():
            per_sample_grad = param.grad.detach().clone()
            clip_grad_norm_(per_sample_grad, max_norm=self.args.clipping_threshold)  # in-place
#             per_sample_grad.add_(1.,torch.normal(mean=0, std=noise * self.args.clipping_threshold,size=per_sample_grad.shape))
            per_sample_grad.add_(torch.normal(mean=0, std=noise * self.args.clipping_threshold,size=per_sample_grad.shape),alpha=1.)
            param.accumulated_grads.append(per_sample_grad) 

    if train:
      if self.args.PrivacyMode == "None":
        self.optimizer.step() 
      else:
        for param in self.parameters():
          param.grad = torch.stack(param.accumulated_grads, dim=0).sum(dim=0)/len(param.accumulated_grads)
#           param.data.add_(-self.lr,param.grad)
          param.data.add_(param.grad, alpha=-self.lr)  
          param.grad = None  # Reset for next iteration
      
    acc_WMAPE = float((acc["MAE"]/acc["WMAPE"]).nanmean())
    
    acc["MAE"] /= n
    acc["MSE"] /= n
    acc["AE"] /= n

    acc_MAE = float(acc["MAE"].mean())
    acc_MSE = float(acc["MSE"].mean())
    acc_AE = float(acc["AE"].mean())
    acc_RMSE = math.sqrt(acc_MSE)

    acc_floats = {}
    acc_floats["MAE"] = acc_MAE
    acc_floats["MSE"] = acc_MSE
    acc_floats["RMSE"] = acc_RMSE
    acc_floats["AE"] = acc_AE
    acc_floats["WMAPE"] = acc_WMAPE
    loss = sum(losses)/len(losses)
    return acc_floats, loss

  def assess_gradients(self, input_Sequence):
    acc = {}
    losses = []
    n = len(input_Sequence)

    hidden_cell= self.init_hidden()
    
    Indexes = [i for i in range(n-1)]
    np.random.shuffle(Indexes)
#     print(Indexes)
#     for seq, labels in input_Sequence:
    for i in Indexes:
#     for seq, labels in input_Sequence:
      seq = input_Sequence[i][0]
      labels = input_Sequence[i][1]  
      seq = seq.to(self.args.device_name)
      labels = labels.to(self.args.device_name)
      labels = labels.squeeze()
      hidden_cell = hidden_cell.data

      y_pred, hidden_cell = self.forward(seq,hidden_cell)

      y_pred = y_pred.squeeze()
      if "MAE" in acc:
        acc["MAE"] += abs(labels - y_pred)
      else: 
        acc["MAE"] = abs(labels - y_pred)
      if "MSE" in acc:
        acc["MSE"] += (labels - y_pred) * (labels - y_pred) 
      else:
        acc["MSE"] = (labels - y_pred) * (labels - y_pred) 
      if "AE" in acc:
        acc["AE"] += labels - y_pred
      else:
        acc["AE"] = labels - y_pred
      if "WMAPE" in acc:
        acc["WMAPE"] += abs(labels)
      else:
        acc["WMAPE"] = abs(labels)

      single_loss = self.loss_function(y_pred, labels)
      losses.append(single_loss.item())
      
      single_loss.backward()
      for param in self.parameters():
        per_sample_grad = param.grad.detach().clone()
        clip_grad_norm_(per_sample_grad, max_norm=self.args.clipping_threshold)  # in-place
        param.accumulated_grads.append(per_sample_grad) 

    for param in self.parameters():
      param.grad = torch.stack(param.accumulated_grads, dim=0).sum(dim=0)/len(param.accumulated_grads)
    
    acc_WMAPE = float((acc["MAE"]/acc["WMAPE"]).nanmean())
    
    acc["MAE"] /= n
    acc["MSE"] /= n
    acc["AE"] /= n

    acc_MAE = float(acc["MAE"].mean())
    acc_MSE = float(acc["MSE"].mean())
    acc_AE = float(acc["AE"].mean())
    acc_RMSE = math.sqrt(acc_MSE)

    acc_floats = {}
    acc_floats["MAE"] = acc_MAE
    acc_floats["MSE"] = acc_MSE
    acc_floats["RMSE"] = acc_RMSE
    acc_floats["AE"] = acc_AE
    acc_floats["WMAPE"] = acc_WMAPE
    loss = sum(losses)/len(losses)
    return acc_floats, loss

  def forward(self, input_seq, hidden_cell):
    input_seq = input_seq.unsqueeze(0)
    gru_out, hidden_cell = self.gru(input_seq.view(self.sequence_length ,self.batch_size, -1), hidden_cell)
    predictions = self.linear(self.relu(gru_out))
    predictions = predictions.squeeze(0)
    return predictions[-1],hidden_cell 

  def test(self,LocalPredictionSamples):
    acc = {}
    input_Sequence = LocalPredictionSamples.sequence
    predictions = []
    
    n = len(input_Sequence)
    for seq, labels in input_Sequence:
      h = self.init_hidden()
      seq = seq.to(self.args.device_name)
      y_pred,h = self.forward(seq,h)
      predictions.append(y_pred)
      
      if "MAE" in acc:
        acc["MAE"] += abs(labels - y_pred)
      else: 
        acc["MAE"] = abs(labels - y_pred)
      if "MSE" in acc:
        acc["MSE"] += (labels - y_pred) * (labels - y_pred) 
      else:
        acc["MSE"] = (labels - y_pred) * (labels - y_pred) 
      if "AE" in acc:
        acc["AE"] += labels - y_pred
      else:
        acc["AE"] = labels - y_pred
      if "WMAPE" in acc:
        acc["WMAPE"] += abs(labels)
      else:
        acc["WMAPE"] = abs(labels)

      self.args.groundTruth.append(labels.item())
      self.args.prediction.append(y_pred.item())
    
    acc_WMAPE = float((acc["MAE"]/acc["WMAPE"]).nanmean())
    
    acc["MAE"] /= n
    acc["MSE"] /= n
    acc["AE"] /= n

    acc_MAE = float(acc["MAE"].mean())
    acc_MSE = float(acc["MSE"].mean())
    acc_AE = float(acc["AE"].mean())
    acc_RMSE = math.sqrt(acc_MSE)

    acc_floats = {}
    acc_floats["MAE"] = acc_MAE
    acc_floats["MSE"] = acc_MSE
    acc_floats["RMSE"] = acc_RMSE
    acc_floats["AE"] = acc_AE
    acc_floats["WMAPE"] = acc_WMAPE

    return acc_floats

  def predict(self,LocalPredictionSamples,args):
    input_Sequence = LocalPredictionSamples.sequence
    predictions = []
    acc = {}
    n = len(input_Sequence)
    for seq, labels in input_Sequence:
      seq = seq.to(self.args.device_name)
      labels = labels.to(self.args.device_name)
      h = self.init_hidden()
      seq = seq.to(self.args.device_name)
      y_pred,h = self.forward(seq,h)
      predictions.append(y_pred)
      if "MAE" in acc:
        acc["MAE"] += abs(labels - y_pred)
      else: 
        acc["MAE"] = abs(labels - y_pred)
      if "MSE" in acc:
        acc["MSE"] += (labels - y_pred) * (labels - y_pred) 
      else:
        acc["MSE"] = (labels - y_pred) * (labels - y_pred) 
      if "AE" in acc:
        acc["AE"] += labels - y_pred
      else:
        acc["AE"] = labels - y_pred
      if "WMAPE" in acc:
        acc["WMAPE"] += abs(labels)
      else:
        acc["WMAPE"] = abs(labels)

      args.groundTruth.append(labels.item())
      args.prediction.append(y_pred.item())
    acc_WMAPE = float((acc["MAE"]/acc["WMAPE"]).nanmean())
    
    acc["MAE"] /= n
    acc["MSE"] /= n
    acc["AE"] /= n

    acc_MAE = float(acc["MAE"].mean())
    acc_MSE = float(acc["MSE"].mean())
    acc_AE = float(acc["AE"].mean())
    acc_RMSE = math.sqrt(acc_MSE)

    acc_floats = {}
    acc_floats["MAE"] = acc_MAE
    acc_floats["MSE"] = acc_MSE
    acc_floats["RMSE"] = acc_RMSE
    acc_floats["AE"] = acc_AE
    acc_floats["WMAPE"] = acc_WMAPE
    
    return predictions

  def displayWeights(self):
    for name, parameter in self.named_parameters():
      if not parameter.requires_grad: continue
      params = parameter.numel()
      print("[name:",name," params:", params, "]\n")

  def special_loop(self,seq,labels):
    gradients = []
    hidden_cell= self.init_hidden()
    seq = seq.to(self.args.device_name)
    labels = labels.to(self.args.device_name)
    labels = labels.squeeze()
    y_pred, hidden_cell = self.forward(seq,hidden_cell)
    y_pred = y_pred.squeeze()
#     print("special_loop:seq.shape",seq.shape)
#     print("special_loop:hidden_cell.shape",hidden_cell.shape)
#     print("special_loop:y_pred.shape",y_pred.shape)
#     print("special_loop:labels.shape",labels.shape)
    single_loss = self.loss_function(y_pred, labels)
    single_loss.backward()
    loss = single_loss.item()
    prediction = y_pred.detach().clone()

    for param in self.parameters():
      per_sample_grad = param.grad.detach().clone()
      gradients.append(per_sample_grad)
    
    gradient1 = self.gru.weight_ih_l0.grad.detach().clone()
    gradient1= gradient1.squeeze()
    gradient1 = torch.flatten(gradient1)
    
    gradient2 = self.gru.weight_hh_l0.grad.detach().clone()
    gradient2 = torch.flatten(gradient2)
    
    gradient3 = self.gru.bias_ih_l0.grad.detach().clone()
    
    gradient4 = self.gru.bias_hh_l0.grad.detach().clone()
    
    gradient5 = self.linear.weight.grad.detach().clone()
    gradient5 = torch.flatten(gradient5)
    gradient6 = self.linear.bias.grad.detach().clone()
    gradient6 = torch.cat((gradient5,gradient6,gradient5,gradient6))
    
    gradients = torch.cat((gradient1,gradient2,gradient3,gradient4,gradient6),axis = 0)
    gradients = torch.reshape(gradients,(6,6))

    return loss,prediction,gradients

###############################
# GNN model
###############################
class MyGraphOutputLayer(nn.Module):
  def __init__(self,args, in_features):
    super(MyGraphOutputLayer, self).__init__()
    self.args = args
    self.W = nn.Parameter(torch.randn(in_features,args.x_axis, args.y_axis).to(args.device_name),requires_grad=True)
    torch.nn.init.xavier_uniform_(self.W)
    self.sigmoid = nn.Sigmoid()
  
  def forward(self, h, att):
    adj_dimension = self.args.x_axis*self.args.y_axis
    e = torch.mul(self.W,h)
    e = torch.sum(e, dim=0)
    e = torch.reshape(e,(1,adj_dimension))
    e = e.expand(adj_dimension,adj_dimension)
    e = torch.mul(e,att)
    e = torch.sum(e, dim=0)
    return e.reshape(self.args.x_axis,self.args.y_axis)

class MyGraphAttentionLayer(nn.Module):
  def __init__(self, args, in_features,out_features, x_index, y_index, i_1,j_1,i_2,j_2, neighbour, dropout, alpha, concat=True):
    super(MyGraphAttentionLayer, self).__init__()
    self.dropout = dropout
    self.in_features = in_features
    self.x_index = x_index
    self.y_index = y_index
    self.i_1 = i_1
    self.j_1 = j_1
    self.i_2 = i_2
    self.j_2 = j_2
    self.neighbour = neighbour
    self.args = args
    if(i_1>=args.x_axis or i_2>=args.x_axis or j_1>=args.y_axis or j_2>=args.y_axis):
      print("a problem in creating the attention layer in ",i_1,j_1)
      print("a problem in creating the attention layer in ",i_2,j_2)
    self.alpha = alpha

    if neighbour == 1:
      self.W = nn.Parameter(torch.randn(in_features).to(args.device_name),requires_grad=True)
      self.W.data.uniform_(0.0, 1.0)
    else:
      self.W = nn.Parameter(torch.randn(in_features).to(args.device_name),requires_grad=False)
      self.W.data.uniform_(0.0, 1.0)
    if neighbour == 1:
      self.a = nn.Parameter(torch.randn(2*out_features).to(args.device_name),requires_grad=True)
      self.a.data.uniform_(0.0, 1.0)
    else:
      self.a = nn.Parameter(torch.randn(2*out_features).to(args.device_name),requires_grad=False)
      self.a.data.uniform_(0.0, 1.0)
    
    self.leakyrelu = nn.LeakyReLU(self.alpha)
    for param in self.parameters():
      param.grad = None

  def forward(self, h, adj):
    if self.neighbour:
      x1 = h[:,self.i_1,self.j_1]
      x2 = h[:,self.i_2,self.j_2] 
      Wh1 = torch.matmul(x1, self.W) 
      Wh2 = torch.matmul(x2, self.W) 
      WH = torch.stack((Wh1,Wh2), dim=0)
      WH = torch.matmul(WH, self.a)
      return self.leakyrelu(WH)
    else:
      return torch.tensor(0.).to(self.args.device_name)

class MyGAT(nn.Module):
  def __init__(self,args, adj, dropout=0.6, alpha=0.2):
    super(MyGAT, self).__init__()
    self.dropout = dropout
    self.args = args
    
#     self.LayerNormalization = True
    self.LayerNormalization = False
    self.epsilon = 1e-10
    if self.LayerNormalization == True:
      self.reset_parameters()

    if(args.global_model_loss == 'MSE'):
      self.loss_function = nn.MSELoss().to(self.args.device_name)
    else:
      if(args.global_model_loss == 'MAE'):
        self.loss_function = nn.L1Loss().to(self.args.device_name)
      else:
        raise SystemError('Need to specify a loss function')

    self.attentions = []
    self.out_layers = MyGraphOutputLayer(args, args.input_length)
    self.add_module('output_layer',self.out_layers)
    self.totalattentions = args.x_axis * args.y_axis
    for x in range(self.totalattentions):
      for y in range(self.totalattentions):
        j_1 = x // self.args.x_axis
        i_1 = x % self.args.x_axis
        j_2 = y // self.args.x_axis
        i_2 = y % self.args.x_axis  
        self.attentions.append(MyGraphAttentionLayer(args, args.input_length,args.output_length, x,y, i_1,j_1, i_2, j_2, adj[x][y]==1, dropout=dropout, alpha=alpha))
    
    for i, attention in enumerate(self.attentions):
        self.add_module('attention_{}'.format(i), attention)
    
    self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0015)


  def fwd_pass(self, input_Sequence, adj,train=False):
    acc = {}
    acc["MAE"] = torch.zeros(self.args.output_length,self.args.x_axis,self.args.y_axis)
    acc["MSE"] = torch.zeros(self.args.output_length,self.args.x_axis,self.args.y_axis)
    acc["RMSE"] = torch.zeros(self.args.output_length,self.args.x_axis,self.args.y_axis)
    acc["AE"] = torch.zeros(self.args.output_length,self.args.x_axis,self.args.y_axis)
    acc["WMAPE"] = torch.zeros(self.args.output_length,self.args.x_axis,self.args.y_axis)
    
    losses = []
    output = []
    i=0
    n = len(input_Sequence)
    if train:
      self.optimizer.zero_grad()
    Indexes = [i for i in range(n-1)]
    np.random.shuffle(Indexes)
#     print(Indexes)
#     for seq, labels in input_Sequence:
    for i in Indexes:
#     for seq, labels in input_Sequence:
      seq = input_Sequence[i][0]
      labels = input_Sequence[i][1] 
      seq = seq.unsqueeze(0)
      #changes the shape to 4 dimensions: 1,self.args.input_length,self.args.bike_x,self.args.bike_y)
      seq = seq.to(self.args.device_name)
      #changes the shape to self.args.bike_x,self.args.bike_y)
      labels = labels.squeeze()
      labels = labels.to(self.args.device_name)
      y_pred = self.forward(seq,adj)
      #changes the shape to self.args.axis_x,self.args.axis_y)
      y_pred = y_pred.squeeze() 
      single_loss = self.loss_function(y_pred, labels)
      if train:
        single_loss.backward()
      else:
        y_pred2 = y_pred.detach().clone()
        if(i==0):
          for x in seq.squeeze():
            output.append(x[:][:])
        else:
          output.append(y_pred2)
        i+=1
      
      losses.append(single_loss.item())
      y_pred2 = y_pred.detach().clone()
      acc["MAE"] += torch.abs(labels - y_pred2)
      acc["MSE"] += (labels - y_pred2) **2  
      acc["AE"] += labels - y_pred2 
      acc["WMAPE"] += torch.abs(labels)  

    if train:
      self.optimizer.step()  
    acc_WMAPE = float((acc["MAE"]/acc["WMAPE"]).nanmean())
    
    acc["MAE"] /= n
    acc["MSE"] /= n
    acc["AE"] /= n

    acc_MAE = float(acc["MAE"].mean())
    acc_MSE = float(acc["MSE"].mean())
    acc_AE = float(acc["AE"].mean())
    acc_RMSE = math.sqrt(acc_MSE)

    acc_floats = {}
    acc_floats["MAE"] = acc_MAE
    acc_floats["MSE"] = acc_MSE
    acc_floats["RMSE"] = acc_RMSE
    acc_floats["AE"] = acc_AE
    acc_floats["WMAPE"] = acc_WMAPE
    loss = sum(losses)/len(losses)
    return output, acc_floats, loss
  
  def forward(self, x, adj):
    x = F.dropout(x, self.dropout, training=self.training)
    x=x.squeeze()
    #calculate the attention coefficient of each pair of nodes
    attx = torch.stack([att(x, adj) for att in self.attentions], dim=0)
    attx = torch.reshape(attx,(self.totalattentions,self.totalattentions))
    zero_vec = 0.0*torch.ones_like(attx)
    #keep only the neighbours
    attx = torch.where(adj > 0, attx, zero_vec)
    #readjust the coefficient per neighbour
    if self.layer_normalization == True:
      attx=self.layer_normalization(attx,self.gamma_1,self.beta_1)
    attx = F.softmax(attx, dim=0)
    #calculate the next timestamp
    return self.out_layers(x,attx)

  def predict(self,x,adj):
    x = x.to(self.args.device_name)
    return self.forward(x,adj)

  def reset_parameters(self):
    self.gamma_1 = nn.Parameter(torch.ones(1))
    self.beta_1 = nn.Parameter(torch.zeros(1))

  def layer_normalization(self, input_layer, gamma, beta):
    mean = input_layer.mean(dim=-1, keepdim=True)
    var = ((input_layer - mean) ** 2).mean(dim=-1, keepdim=True)
    std = (var + self.epsilon).sqrt()
    y = (input_layer - mean) / std
    if gamma is not None:
        y *= gamma
    if beta is not None:
        y += beta
    return y

###############################
# Feature inference attack model
###############################
class MLP(nn.Module):
  def __init__(self,args, dim_output):
    super(MLP, self).__init__()
    self.dim_output = dim_output
    self.args = args
    
    self.LayerNormalization = True
    self.epsilon = 1e-10
    self.reset_parameters()

    input_dim = args.x_axis*args.y_axis*args.input_length
    input_layer_size = 1

    if self.args.dataset == "bikeNYC":
      hidden1_size = 6
      hidden2_size = 2
      hidden3_size = 2
    else:
#       hidden1_size = 60
#       hidden2_size = 20
#       hidden3_size = 10
      hidden1_size = 2*(args.featureRatio//10)
      hidden2_size = 2*(args.featureRatio//10)
      hidden3_size = args.featureRatio//10

    self.input_fc = nn.Linear(input_dim, input_layer_size*input_dim)
    torch.nn.init.xavier_uniform_(self.input_fc.weight)
    self.input_fc.bias.data.fill_(0.01)

    ### non-linear solution 
    self.hiddenLayer1 = nn.Parameter(torch.normal(0., std=1.0, size=(input_layer_size*input_dim, hidden1_size*input_dim)),requires_grad=True)
    self.hiddenLayer2 = nn.Parameter(torch.normal(0., std=1.0, size=(hidden1_size*input_dim, hidden2_size*input_dim)),requires_grad=True)
    self.hiddenLayer3 = nn.Parameter(torch.normal(0., std=1.0, size=(hidden2_size*input_dim, hidden3_size*input_dim)))

    self.hiddenBias1 = nn.Parameter(torch.zeros(hidden1_size*input_dim),requires_grad=True)
    self.hiddenBias2 = nn.Parameter(torch.zeros(hidden2_size*input_dim),requires_grad=True)
    self.hiddenBias3 = nn.Parameter(torch.zeros(hidden3_size*input_dim),requires_grad=True)

    self.outputLayer = nn.Parameter(torch.normal(0., std=1.0, size=(hidden3_size*input_dim, self.dim_output)),requires_grad=True)
    self.outputBias = nn.Parameter(torch.zeros(self.dim_output),requires_grad=True)


    self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
     
    for param in self.parameters():
      param.grad = None
  
  def my_loss(self,original, output, target):
    # copy the gradients from the calculated model's output
    loss = original.sum()
    loss.data = torch.mean(torch.abs(output - target))
    return loss

  def reset_parameters(self):
    self.gamma_1 = nn.Parameter(torch.ones(1))
    self.gamma_2 = nn.Parameter(torch.ones(1))
    self.gamma_3 = nn.Parameter(torch.ones(1))
    self.gamma_fc = nn.Parameter(torch.ones(1))
    self.beta_1 = nn.Parameter(torch.zeros(1))
    self.beta_2 = nn.Parameter(torch.zeros(1))
    self.beta_3 = nn.Parameter(torch.zeros(1))
    self.beta_fc = nn.Parameter(torch.zeros(1))
  
  def layer_normalization(self, input_layer, gamma, beta):
    mean = input_layer.mean(dim=-1, keepdim=True)
    var = ((input_layer - mean) ** 2).mean(dim=-1, keepdim=True)
    std = (var + self.epsilon).sqrt()
    y = (input_layer - mean) / std
    if gamma is not None:
        y *= gamma
    if beta is not None:
        y += beta
    return y
  
  def forward_single(self,seq, training = False):
    seq = seq.squeeze()
    seq = seq.to(self.args.device_name)
    seq = torch.flatten(seq)
    I_1 = self.input_fc(seq)

    ### non-linear solution with different activation functions
    h_1 = torch.matmul(I_1,self.hiddenLayer1) + self.hiddenBias1
    if self.LayerNormalization == True:
      l_h_1 = self.layer_normalization(h_1,self.gamma_1,self.beta_1)
      h_1_O = torch.tanh(l_h_1)
    else:
      h_1_O = torch.tanh(h_1)

    h_2 = torch.matmul(h_1_O,self.hiddenLayer2) + self.hiddenBias2
    h_2_O = torch.tanh(h_2)
    h_3 = torch.matmul(h_2_O,self.hiddenLayer3) + self.hiddenBias3
    if self.LayerNormalization == True:
      l_h_3 = self.layer_normalization(h_3,self.gamma_3,self.beta_3)
      h_3_O = torch.tanh(l_h_3)
    else:
      h_3_O = torch.tanh(h_3)

    B = torch.matmul(h_3_O,self.outputLayer) + self.outputBias
    Y = torch.tanh(B)
    return Y

  def backward_single(self,original, guessed_input,true_input):
    true_input = true_input.to(self.args.device_name)
    guessed_input = guessed_input.to(self.args.device_name)
    single_loss = self.my_loss(original, guessed_input,true_input)
    single_loss.backward()
    single_loss.grad = None
    return single_loss.item()

  def guess(self, seq, flat_organization_contribution_distribution, organization_id):
    predictions = []
    seq = seq.to(self.args.device_name)
    guessed = self.forward_single(seq,False)
    label = seq.squeeze(0)
    label = torch.flatten(label)
    indeces = list(*np.where(flat_organization_contribution_distribution == organization_id))
    ## calculate the accuracy
    acc = {}
    acc["MSE"] = 0.
    acc["MAE"] = 0.
    j=0
    for i in indeces:
      diff = label[i]-guessed[j]  
      acc["MAE"] += diff
      acc["MSE"] += diff**2
      self.args.groundTruth.append(label[i].item())
      self.args.prediction.append(guessed[j].item())
      # self.args.prediction.append(label[i].item())
      predictions.append(guessed[j].item())
      j+=1
#     return MSE/len(indeces), predictions
    acc["MSE"] /= len(indeces)
    acc["MAE"] /= len(indeces)
#     print(acc["MSE"])
#     print(acc["MAE"])
    return acc, predictions

###############################
# Membership inference attack model
###############################
class MembershipInferenceAttacker(nn.Module):
  def __init__(self,args):
    super().__init__()
    self.args = args
    self.output_component = nn.Sequential(
            nn.Linear(len(args.observedEpochs), 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Dropout(p=0.2),
            nn.ReLU()
        )
    self.label_component = nn.Sequential(
            nn.Linear(len(args.observedEpochs), 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Dropout(p=0.2),
            nn.ReLU()
        )
    self.loss_component = nn.Sequential(
            nn.Linear(len(args.observedEpochs), 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Dropout(p=0.2),
            nn.ReLU()
        )
    self.gradient_component = nn.Conv2d(len(args.observedEpochs), 6, 6, stride=3)  
    self.encoder = Encoder(198) #3*64+6
    self.decoder = Decoder(198) #3*64+6
#     self.decoder = Decoder(3*len(args.observedEpochs)+6*6*len(args.observedEpochs)) #3*4+36*4
#     count_parameters(self)
#     for param in self.parameters():
#         print(param.grad)
    self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
    self.loss_function = nn.MSELoss()

  def fwd_pass(self,input_Sequence, observed_epochs):
    losses = [] 
    i=0
    n = len(input_Sequence)
    self.optimizer.zero_grad()
    Indexes = [i for i in range(n-1)]
    np.random.shuffle(Indexes)
    for i in Indexes:
      seq = input_Sequence[i][0]
      labels = input_Sequence[i][1] 
      seq = seq.to(self.args.device_name)
      labels = labels.to(self.args.device_name)
      labels = labels.squeeze()
      #for each observed epochs do the training loop
      j=0 
#       for param in self.parameters():
#         print(param.grad)
      for model in observed_epochs:
        loss,output,gradient = model.special_loop(seq,labels)
#         for param in self.parameters():
#           print(param.grad)
        if j==0:
            stacked_output = output.unsqueeze(0)
            stacked_labels = labels.unsqueeze(0)
            stacked_loss = torch.tensor([loss])
            stacked_gradients = gradient
        else:
            stacked_output = torch.cat((stacked_output,output.unsqueeze(0)))
            stacked_labels = torch.cat((stacked_labels,labels.unsqueeze(0)))
            stacked_loss = torch.cat((stacked_loss,torch.tensor([loss])))
            stacked_gradients = torch.cat((stacked_gradients,gradient)) 
        j+=1      
      stacked_output = stacked_output.squeeze()  
      stacked_gradients = torch.reshape(stacked_gradients,(len(self.args.observedEpochs), 6, 6))
      
#       print("stacked_output.shape",stacked_output.shape)
#       print("stacked_labels.shape",stacked_labels.shape)
#       print("stacked_loss.shape",stacked_loss.shape)
#       print("gradient.shape",gradient.shape)
        
      output = self.output_component(stacked_output)
      label = self.label_component(stacked_labels)
      loss = self.loss_component(stacked_loss)
      gradient = self.gradient_component(stacked_gradients)
      gradient = torch.flatten(gradient)
      input_encoder = torch.cat((output,label,loss,gradient))  
        
      reconstructed = self.forward_stacked(input_encoder)
#       print("reconstructed.shape",reconstructed.shape)
#       print("input_encoder.shape",input_encoder.shape)
      single_loss = self.loss_function(reconstructed, input_encoder)
      single_loss.backward()
#       print("fwd_pass:single_loss",single_loss.item()) 
      losses.append(single_loss.item())
    #backward  
    self.optimizer.step()  
    return sum(losses)/len(losses)

  def fwd(self,input_Sequence, target_model):
    scores = []
    i=0
    n = len(input_Sequence)
    # print("fwd:n is ",n)
    Indexes = [i for i in range(n-1)]
    np.random.shuffle(Indexes)
    for i in Indexes:
      seq = input_Sequence[i][0]
      labels = input_Sequence[i][1] 
      seq = seq.to(self.args.device_name)
      labels = labels.to(self.args.device_name)
      labels = labels.squeeze()
      #for each observed epochs do the training loop
      j=0 
      for j in range(len(self.args.observedEpochs)):
        loss,output,gradient = target_model.special_loop(seq,labels)
        
        if j==0:
            stacked_output = output.unsqueeze(0)
            stacked_labels = labels.unsqueeze(0)
            stacked_loss = torch.tensor([loss])
            stacked_gradients = gradient
        else:
            stacked_output = torch.cat((stacked_output,output.unsqueeze(0)))
            stacked_labels = torch.cat((stacked_labels,labels.unsqueeze(0)))
            stacked_loss = torch.cat((stacked_loss,torch.tensor([loss])))
            stacked_gradients = torch.cat((stacked_gradients,gradient)) 
        j+=1      
      stacked_output = stacked_output.squeeze()  
      stacked_gradients = torch.reshape(stacked_gradients,(len(self.args.observedEpochs), 6, 6))
        
#       print("stacked_output.shape",stacked_output.shape)
#       print("stacked_labels.shape",stacked_labels.shape)
#       print("stacked_loss.shape",stacked_loss.shape)
#       print("gradient.shape",gradient.shape)
        
      output = self.output_component(stacked_output)
      label = self.label_component(stacked_labels)
      loss = self.loss_component(stacked_loss)
      gradient = self.gradient_component(stacked_gradients)
      gradient = torch.flatten(gradient)
      input_encoder = torch.cat((output,label,loss,gradient))  
      score = self.forward_stacked(input_encoder,False)
      # print("score",score)
      # print("score.shape",score.shape)
      scores.append(score)
    # print("length",len(scores))
    scores = torch.stack(scores, 0)    
    return scores

  def forward_stacked(self,input_encoder, train = True):
#     print("input_encoder.shape",input_encoder.shape)
    score = self.encoder(input_encoder) 
#     print("score.shape",score.shape) 
    if train == True:
      reconstructed = self.decoder(score)
#       print("reconstructed.shape",reconstructed.shape)  
      return reconstructed
    else:
      value = score.item()
      score = score.unsqueeze(0)
      score = torch.tensor([value,1.-value])
      return score #Must be [a,1-a]

class Encoder(nn.Module):
  def __init__(self,inputshape):
    super().__init__()
    self.hidden_layer = nn.Sequential(
        nn.Linear(in_features=inputshape, out_features=256), 
        nn.ReLU(),
        nn.Linear(in_features=256, out_features=128),
        nn.ReLU(),
        nn.Linear(in_features=128, out_features=64),
        nn.Dropout(0.2),
        nn.ReLU(),
        nn.Linear(in_features=64, out_features=1),
        nn.Sigmoid())

  def forward(self, features):
    code = self.hidden_layer(features)
    return code

class Decoder(nn.Module):
  def __init__(self,outputshape):
    super().__init__()
    self.hidden_layer = nn.Sequential(
        nn.Linear(in_features=1, out_features=4),
        nn.ReLU(),
        nn.Linear(in_features=4, out_features=64),  
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(in_features=64, out_features=outputshape))

  def forward(self, code):
    reconstructed = self.hidden_layer(code)
    return reconstructed