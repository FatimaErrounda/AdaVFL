import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import math
import copy
from math import log, sqrt, exp
from scipy.optimize import fsolve
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.cluster import KMeans

class Pair(object):
  def __init__(self,x_axis, y_axis,model):
    self.x_axis = x_axis
    self.y_axis = y_axis
    self.model = model

###############################
# Common
###############################
def sigma_to_epsilon(sigma, delta):
  return math.sqrt(2.0 * np.log(1.25/delta))/sigma

def epsilon_to_sigma(eps,delta):
  return math.sqrt(2.0 * np.log(1.25/delta))/eps

def sigma_to_rho(sigma):
  return 1/(2.0*sigma*sigma)

def rho_to_sigma(rho):
  return math.sqrt(1./(2.*rho))

def compute_epsilon(rho):
  return math.sqrt(2.0*rho)

def rho_to_dp(rho, delta):
  return rho + (2 * math.sqrt(rho*log(1./delta)))

def compute_advcomp_budget(eps, delta, T):
  denom = math.sqrt(2.0 * T * log(2.0 / delta))
  ep = eps / (2.0 * denom)
  dp = delta / (2.0 * T)
  return ep, dp 

def compute_advcomp_sigma(eps, delta, T):
  return math.sqrt(T*log(1/delta)*log(T/delta))/eps

def compute_cumulated_budget(iter_budget, total_budget):
  return total_budget+iter_budget

def dp_to_zcdp(eps, delta):
  def eq_epsilon(rho):
    if rho <= 0.0:
        rhs = rho
    else:
        rhs = rho + 2.0 * math.sqrt(rho * math.log(1.0/delta))
    return eps - rhs
  rho = fsolve(eq_epsilon, 0.0)
  return rho[0]

def pertub_weights(timestamp, dataset, model, budget, args):   
  found_timestamp = False
  index_timestamp = -1
  n = len(dataset.sequence_timestamps)
  first = dataset.sequence_timestamps[0]
  last = dataset.sequence_timestamps[n-1]
  
  index_timestamp = (timestamp-first[0][0])/(last[0][0]-first[0][0])*n
  index_timestamp = int(index_timestamp.item())

  endtimestamp = timestamp+args.trainingInterval
  index_end = (endtimestamp-first[0][0])/(last[0][0]-first[0][0])*n
  index_end = int(index_end.item())
  
  args.batch_size = index_end-index_timestamp
  args.overall_size = n

  if index_timestamp == -1:
    print("a problem finding the timestamp in the training data for dataset index ",dataset.x_axis,",", dataset.y_axis)
  model.to(args.device_name)

  acc, loss = model.fwd_pass(dataset.sequence[index_timestamp:index_end], True, budget)
  return acc, loss
  
def update_global_weights(model, GlobalSequences, GlobalLabels,args,adj):
  #build the sequence with labels
  input_Sequence = []
  i = 0
  for item in GlobalSequences.sequence:
    seq, label1 = item
    input_Sequence.append((seq,GlobalLabels[i]))
    i+=1
  model.to(args.device_name)
  _, acc, loss = model.fwd_pass(input_Sequence,adj,True) 
  return acc, loss 

def train_global_weights(timestamp, model, global_data,args,adj):
  found_timestamp = False
  index_timestamp = -1
  n = len(global_data.sequence_timestamps)
  first = global_data.sequence_timestamps[0]
  last = global_data.sequence_timestamps[n-1]

  index_timestamp = (timestamp-first)/(last-first)*n
  index_timestamp = int(index_timestamp)

  endtimestamp = timestamp+args.trainingInterval
  index_end = (endtimestamp-first)/(last-first)*n
  index_end = int(index_end)

  _, acc, loss = model.fwd_pass(global_data.sequence[index_timestamp:index_end],adj,True) 
  return acc, loss 

def update_weights(timestamp, dataset, model, args):  
  found_timestamp = False
  index_timestamp = -1
  n = len(dataset.sequence_timestamps)
  first = dataset.sequence_timestamps[0]
  last = dataset.sequence_timestamps[n-1]
  
  index_timestamp = (timestamp-first[0][0])/(last[0][0]-first[0][0])*n
  index_timestamp = int(index_timestamp.item())

  endtimestamp = timestamp+args.trainingInterval
  index_end = (endtimestamp-first[0][0])/(last[0][0]-first[0][0])*n
  index_end = int(index_end.item())

  if index_timestamp == -1:
    print("a problem finding the timestamp in the training data for dataset index ",dataset.x_axis,",", dataset.y_axis)
  model.to(args.device_name)
  acc, loss = model.fwd_pass(dataset.sequence[index_timestamp:index_end],True)
  return acc, loss


def predict_model(model, seq, adj, args):
  return model.predict(seq,adj)

def test_model(model, adj, dataset,args):
  return model.fwd_pass(dataset.sequence,adj, False)
###############################
# Debugging
###############################
def output_results(args,exec_average_local_RMSE,exec_average_local_WMAPE,exec_average_local_AE,
      device_model, LocalTestData, local_training_loss, assigned_sigma):
  
  RMSE_Epoch = []
  WMAPE_Epoch = []
  AE_Epoch = []    

  print("Training RMSEs",sum(exec_average_local_RMSE) / len(exec_average_local_RMSE) )
  for i in exec_average_local_RMSE:
    print(i)

  print("Training WMAPEs",sum(exec_average_local_WMAPE) / len(exec_average_local_WMAPE) )
  for i in exec_average_local_WMAPE:
    print(i)

  print("Training AEs",sum(exec_average_local_AE) / len(exec_average_local_AE) )
  for i in exec_average_local_AE:
    print(i)

  device_model.eval()
  for epoch in tqdm(range(10)):
    with torch.no_grad():
      acc = device_model.test(LocalTestData[epoch])
      RMSE_Epoch.append(acc["RMSE"])
      WMAPE_Epoch.append(acc["WMAPE"])
      AE_Epoch.append(acc["AE"])

  print("Testing RMSEs",sum(RMSE_Epoch) / len(RMSE_Epoch) )
  for i in RMSE_Epoch:
    print(i)

  print("Testing WMAPEs",sum(WMAPE_Epoch) / len(WMAPE_Epoch) )
  for i in WMAPE_Epoch:
    print(i)

  print("Testing AEs",sum(AE_Epoch) / len(AE_Epoch) )
  for i in AE_Epoch:
    print(i)

  x = range(0,len(local_training_loss))

  plt.subplot(3, 2, 4)
  plt.plot(local_training_loss, color='g', label = 'local loss')
  plt.legend(loc="upper left")
  plt.title('local loss')
  plt.xlabel('Epoch')

  if args.PrivacyMode != "None":
    plt.subplot(3, 2, 5)
    plt.plot(assigned_sigma, color='g', label = 'budget')
    plt.legend(loc="upper left")
    plt.title('budget')
    plt.xlabel('Epoch')
    plt.subplot(3, 2, 6)
    plt.plot(args.tracked_error, color='r', label = 'error')

###############################
# Membership inference attacker
###############################
def train_attacker(timestamp, AttackerTrainData, attacker_model, observed_epochs, args):
  found_timestamp = False
  index_timestamp = -1
  n = len(AttackerTrainData.sequence_timestamps)
  first = AttackerTrainData.sequence_timestamps[0]
  last = AttackerTrainData.sequence_timestamps[n-1]

  index_timestamp = (timestamp-first[0][0])/(last[0][0]-first[0][0])*n
  index_timestamp = int(index_timestamp)

  endtimestamp = timestamp+args.trainingInterval
  index_end = (endtimestamp-first[0][0])/(last[0][0]-first[0][0])*n
  index_end = int(index_end)

  loss = attacker_model.fwd_pass(AttackerTrainData.sequence[index_timestamp:index_end],observed_epochs) 
  return loss 

def test_attacker(AttackerTestData,attacker_model,target_model,args):
  # AttackerTestData[0]=member_data
  # AttackerTestData[1]=non_member_data

  #get the scores for the testing data
  NonMembersScores = []
  MembersScores = []
  scores = []
  beginningTime = args.beginTestNonMemberTimestamp
  endingTime = args.endTestNonMemberTimestamp
#   print("beginningTime",beginningTime)   
#   print("endingTime",endingTime)   
  number_of_iterations = ((endingTime - beginningTime)//args.testingInterval)+1
#   print("number_of_iterations",number_of_iterations) 
  # for iteration in tqdm(range(number_of_iterations)):
  for timestamp in tqdm(range(beginningTime, endingTime, args.testingInterval)):
    found_timestamp = False
    index_timestamp = -1
    n = len(AttackerTestData[1].sequence_timestamps)
    first = AttackerTestData[1].sequence_timestamps[0]
    last = AttackerTestData[1].sequence_timestamps[n-1]
    index_timestamp = (timestamp-first[0][0])/(last[0][0]-first[0][0])*n
    index_timestamp = int(index_timestamp.item())
    endtimestamp = timestamp+args.testingInterval
    index_end = (endtimestamp-first[0][0])/(last[0][0]-first[0][0])*n
    index_end = int(index_end)
#     print("first timestamp ",first[0][0])
#     print("last timestamp ",last[0][0])
#     print("timestamp",timestamp, "index_timestamp",index_timestamp,"index_end",index_end)
#     print("input array",len(AttackerTestData[1].sequence[index_timestamp:index_end]))
    NonMembersScores[index_timestamp:index_end] = attacker_model.fwd(AttackerTestData[1].sequence[index_timestamp:index_end],target_model) 
  
  beginningTime = args.beginTestMemberTimestamp
  endingTime = args.endTestMemberTimestamp
  number_of_iterations = ((endingTime - beginningTime)//args.testingInterval)+1
#   print("number_of_iterations",number_of_iterations)  
  # for iteration in tqdm(range(number_of_iterations)):
  for timestamp in tqdm(range(beginningTime, endingTime, args.testingInterval)):  
    found_timestamp = False
    index_timestamp = -1
    n = len(AttackerTestData[0].sequence_timestamps)
    first = AttackerTestData[0].sequence_timestamps[0]
    last = AttackerTestData[0].sequence_timestamps[n-1]
    index_timestamp = (timestamp-first[0][0])/(last[0][0]-first[0][0])*n
    index_timestamp = int(index_timestamp.item())
    endtimestamp = timestamp+args.testingInterval
    index_end = (endtimestamp-first[0][0])/(last[0][0]-first[0][0])*n
    index_end = int(index_end)
#     print("timestamp",timestamp, "index_timestamp",index_timestamp,"index_end",index_end)
    MembersScores[index_timestamp:index_end] = attacker_model.fwd(AttackerTestData[0].sequence[index_timestamp:index_end],target_model) 
#   print("len(MembersScores) ",len(MembersScores))
#   print("len(NonMembersScores) ",len(NonMembersScores))
  tensorMembers = torch.stack(MembersScores,0)  
#   print("tensorMembers.shape",tensorMembers.shape)
  tensorNonMembers = torch.stack(NonMembersScores, 0)
#   print("tensorNonMembers.shape",tensorNonMembers.shape)
  scores = torch.cat((tensorMembers,tensorNonMembers ))
#   print("scores.shape",scores.shape)
  # cluster the output
  acc = cluster_and_check(scores,len(MembersScores))  
  return acc 

def cluster_and_check(scores, cutPoint):
  kmeans = KMeans(n_clusters=2)
  kmeans.fit(scores)
  membership = kmeans.labels_
  Size = len(membership)
  TruePositive = (membership[:cutPoint] == membership[0]).sum()
  TrueNegative = (membership[cutPoint:] == membership[Size-1]).sum()  
#   unique, TruePositive = np.unique(membership[:cutPoint], return_counts=True)
#   unique, TrueNegative = np.unique(membership[cutPoint:], return_counts=True)
  #accuracy = (TP+TN)/(P+N) ==> (True positive+true negative)/size of AttackerTestData
  return (TruePositive+TrueNegative)/len(membership)

###############################
# DP-AGD
###############################
def noisyMax(candidate, lmbda, bmin=False):
  scores = np.array(candidate)
  noise = np.random.exponential(lmbda, size=len(scores))
  # choose the minimum?
  if bmin:
      scores *= -1.0
      noise *= -1.0
  # add noise
  scores += noise
  idx = np.argmax(scores)
  return idx, candidate[idx]

def override_model(model, weights):
  for w, param in zip(weights,model.parameters()):
    param.grad = None
    param.data.copy_(w)

def grad_avg(rho_old, rho_H, model, noisy_grad, args):
    sigma = args.clipping_threshold / math.sqrt(2.0 * (rho_H - rho_old))
    # new estimate
    g_2 = perturb_gradients(model, sigma, args.clipping_threshold,args.batch_size)
    beta = rho_old / rho_H
    # weighted average
    s_tilde = []
    for p_param, n_param in zip(g_2,noisy_grad):
      s_tilde.append(beta * n_param + (1.0 - beta) * p_param)
    return s_tilde

def perturb_gradients(model, sigma,clipping_threshold,batch_size):
  weights = []
  for p in model.parameters():
    copy = p.grad.detach().clone()
    copy.add_(torch.normal(mean=0, std=(sigma * clipping_threshold)/batch_size,size=copy.shape),alpha=1.)
    weights.append(copy)
  return weights
  
def build_candidates(model, noisy_grad, step_sizes):
  candidate = []
  for step in step_sizes:
    w=[]
    for param, n_param in zip(model.parameters(), noisy_grad):
      copy = param.data.detach().clone()
      copy.add_(n_param, alpha=-step)
      w.append(copy) 
    candidate.append(w)
  return candidate
  
def loss_score(model, weights, dataset, obj_clip, args):
  for new_param, param in zip(weights,model.parameters()):
    param.data.copy_(new_param)
  with torch.no_grad():
    _,loss = model.fwd_pass(dataset.sequence, False)
    return loss

def grad_func(timestamp, dataset, model, args):
  found_timestamp = False
  index_timestamp = -1
  n = len(dataset.sequence_timestamps)
  first = dataset.sequence_timestamps[0]
  last = dataset.sequence_timestamps[n-1]
  
  index_timestamp = (timestamp-first[0][0])/(last[0][0]-first[0][0])*n
  index_timestamp = int(index_timestamp.item())

  endtimestamp = timestamp+args.trainingInterval
  index_end = (endtimestamp-first[0][0])/(last[0][0]-first[0][0])*n
  index_end = int(index_end.item())

  if index_timestamp == -1:
    print("a problem finding the timestamp in the training data for dataset index ",dataset.x_axis,",", dataset.y_axis)
  model.to(args.device_name)
  acc, loss = model.assess_gradients(dataset.sequence[index_timestamp:index_end])  
  return acc, loss
  
def pertub_weights_process_conc(timestamp, trainingdataset, validation_dataset, model, delta_t, number_of_training_rounds, args): 
  local_training_acc,local_loss = grad_func(timestamp, trainingdataset, model, args)
  sigma = rho_to_sigma(model.Budget.rho_ng)
  model.Budget.rho -= model.Budget.rho_ng
  if model.Budget.rho < 0:
    args.Halt = True
  idx = 0
  noisy_grad = perturb_gradients(model,sigma,args.clipping_threshold, args.batch_size)
  model.Budget.privacy_budgets = model.Budget.rho_ng  
  model_copy=copy.deepcopy(model)
  while idx == 0:
    step_sizes = np.linspace(0, model.Budget.max_step_size, model.Budget.n_candidate+1)
    candidate = build_candidates(model_copy, noisy_grad, step_sizes)
    scores = [loss_score(model_copy, theta, validation_dataset, args.obj_clip,args)
                for theta in candidate]
    scores[0] *= model.Budget.exp_dec
    lmbda = args.obj_clip/math.sqrt(2.0 * model.Budget.rho_nmax)
    idx, _= noisyMax(scores, lmbda, bmin=True)
    model.Budget.rho -= model.Budget.rho_nmax
    if idx > 0:
      # don't do the update when the remain budget is insufficient
      if model.Budget.rho >= 0:
        override_model(model, candidate[idx-1])
      model.Budget.rho -= model.Budget.rho_ng
      privacy_budgets = rho_to_dp(model.Budget.rho_ng,delta_t)   
    else:
      rho_old = model.Budget.rho_ng
      model.Budget.rho_ng *= (1.0 + args.gamma)
      noisy_grad = grad_avg(rho_old, model.Budget.rho_ng, model, noisy_grad, args)
      sum = 0.
      for n_grad in noisy_grad:
        sum += torch.linalg.norm(n_grad)
      norm = math.sqrt(sum)
      for n_grad in noisy_grad:
        n_grad /= norm
      model.Budget.rho -= (model.Budget.rho_ng - rho_old)
  
  model.Budget.chosen_step_sizes.append(step_sizes[idx])
  if (number_of_training_rounds % 10) == 0:
    max_step_size = min(1.1*max(model.Budget.chosen_step_sizes), 2.0)
    del model.Budget.chosen_step_sizes[:] 
  model.Budget.cumulated_budget += model.Budget.privacy_budgets
  return local_training_acc,local_loss

###############################
# Validation-based
###############################
def calculate_validation_accuracy(model, validation_data, args):
  acc_floats,loss = model.fwd_pass(validation_data.sequence, False)
  return acc_floats[args.validation_accuracy_metric]

def update_budget_accuracy(remainder, validation_table, sigma, validation_accuracy, args):
  ## calculate the averaged validation accuracy
  average_validation = validation_table.sum()/args.validation_period

  if(validation_accuracy - average_validation >= args.validation_threshold):
#     print("validation updating the sigma because it reached the threshold")
    new_sigma = sigma * args.validation_coefficient
    return new_sigma
  else:
    return sigma

def pertub_weights_process_val(timestamp, trainingdataset, validation_dataset, model, quotient, remainder, args):   
  validation_accuracy = calculate_validation_accuracy(model,validation_dataset, args)
  if quotient == 0:
    model.Budget.Validation_accuracy[remainder] = validation_accuracy
    privacy_budgets = model.Budget.privacy_budgets
    privacy_rho = sigma_to_rho(privacy_budgets)
  else:
    if remainder == 0:
      budget = update_budget_accuracy(remainder,model.Budget.Validation_accuracy, model.Budget.privacy_budgets, validation_accuracy, args)
      model.Budget.privacy_budgets = budget
      privacy_budgets = budget
      privacy_rho = sigma_to_rho(privacy_budgets)
      # print("recalculate the budget to ",budget)
    else:
      privacy_budgets= model.Budget.privacy_budgets   
      model.Budget.Validation_accuracy[remainder] = validation_accuracy
      privacy_rho = sigma_to_rho(privacy_budgets)
  model.Budget.cumulated_budget = compute_cumulated_budget(privacy_rho,model.Budget.cumulated_budget)
  acc, loss = pertub_weights(timestamp, trainingdataset, model, privacy_budgets, args)  
#   print("model budget =",model.Budget.privacy_budgets)  
  if model.Budget.cumulated_budget >= args.total_rho:
    print("Halt=True")
    args.Halt = True
  return acc, loss
  
###############################
# Increase-based
###############################
def update_budget_increase(epoch, args):
  ## args.Increase_curb =  "Log" "Exp", "Intrvl"
  if args.Increase_curb == "Log":
    return np.minimum(args.Increase_e_min+np.log((epoch*(np.exp(args.Increase_e_max-args.Increase_e_min)-1.)/args.Increase_gamma) + 1.),args.Increase_e_max)
  else:
    if args.Increase_curb == "Exp":
      return np.minimum(args.Increase_e_min+((np.exp(epoch)-1.)*(args.Increase_e_max-args.Increase_e_min))/(np.exp(args.Increase_gamma) - 1.),args.Increase_e_max)
    else:
      if args.Increase_curb == "Intrvl":
        return np.minimum(args.Increase_e_min+epoch*(args.Increase_e_max-args.Increase_e_min)/args.Increase_gamma,args.Increase_e_max)
      else:
        exit('Error: unrecognized update_budget_increase mode')

def pertub_weights_process_inc(timestamp, trainingdataset, model, delta_t, quotient, remainder, args):   
  if quotient == 0:
    privacy_budgets = model.Budget.privacy_budgets
    privacy_rho = dp_to_zcdp(privacy_budgets,delta_t)
  else:
    if remainder == 0:
      budget = update_budget_increase(quotient, args)
      model.Budget.privacy_budgets=budget
      privacy_budgets = budget 
      privacy_rho = dp_to_zcdp(privacy_budgets,delta_t)
    else:
      privacy_budgets = model.Budget.privacy_budgets  
      privacy_rho = dp_to_zcdp(model.Budget.privacy_budgets,delta_t)   
  model.Budget.cumulated_budget = compute_cumulated_budget(privacy_rho,model.Budget.cumulated_budget)
  acc, loss = pertub_weights(timestamp, trainingdataset, model, privacy_budgets, args)
  if model.Budget.cumulated_budget >= args.total_rho:
    args.Halt = True
  return acc, loss

###############################
# Adaptive
###############################
def update_budget_training(rho,local_training_acc, Loop_accuracy, minimum_training_accuracy, args):
  average_training = Loop_accuracy.sum()/len(Loop_accuracy)
  error = (local_training_acc - average_training)/np.max([local_training_acc,minimum_training_accuracy])
  return error, rho*(1.+args.alpha*error)

def pertub_weights_process_ada(timestamp, trainingdataset, model, quotient, remainder, number_of_training_rounds, args):   
  if args.AdaptiveError == "Training":
    if quotient == 0:
      if(number_of_training_rounds == 0):
        model.Budget.Loop_accuracy[remainder] = 0.
        privacy_budgets = model.Budget.privacy_budgets
      else:
        # Loop_accuracy[remainder] = local_training_acc[args.validation_accuracy_metric]
        model.Budget.Loop_accuracy[remainder] = model.Budget.local_loss
        privacy_budgets = model.Budget.privacy_budgets
        # if minimum_training_accuracy > local_training_acc[args.trining_accuracy_metric]:
        #   minimum_training_accuracy = local_training_acc[args.trining_accuracy_metric]  
      args.tracked_error.append(0.)
    else:
      # if minimum_training_accuracy > local_training_acc[args.trining_accuracy_metric]:
      #     minimum_training_accuracy = local_training_acc[args.trining_accuracy_metric]
      if remainder == 0:
        if args.Halt: 
          privacy_budgets = model.Budget.privacy_budgets
        else:
          error, budget = update_budget_training(model.Budget.privacy_budgets, model.Budget.local_loss, model.Budget.Loop_accuracy, model.Budget.minimum_training_accuracy, args)
          args.tracked_error.append(error)
          privacy_budgets = budget
          model.Budget.privacy_budgets = budget
      else:
        # args.tracked_error.append(local_training_acc[args.trining_accuracy_metric])
        # Loop_accuracy[remainder] = local_training_acc[args.trining_accuracy_metric]
        # args.tracked_error.append(model.Budget.local_loss)
        model.Budget.Loop_accuracy[remainder] = model.Budget.local_loss
        privacy_budgets = model.Budget.privacy_budgets
  sigma_t = rho_to_sigma(privacy_budgets)    
  local_training_acc,model.Budget.local_loss = pertub_weights(timestamp, trainingdataset, model,sigma_t,args)  
  model.Budget.cumulated_budget = compute_cumulated_budget(privacy_budgets,model.Budget.cumulated_budget)
  if model.Budget.cumulated_budget >= args.total_rho:
    args.Halt = True
  return local_training_acc,model.Budget.local_loss
