#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import math 
import random
import sys
from Models import GRU
from collections import defaultdict
import datetime
from random import seed
from random import randint

class LocalSequentialDataset:
  def __init__(self, timestamps, x_axis, y_axis, begin, end, args):
    self.timestamps = timestamps
    self.x_axis = x_axis
    self.y_axis = y_axis
    self.begin = begin
    self.end = end
    self.args = args
    self.sequence = []
    self.sequence_timestamps = []
    self.overall_timestamps = []

  def make_data(self):  
    total_sample_length = self.args.input_length + self.args.output_length
    endpartitionIndex = int(self.end*(len(self.timestamps)-1))
    endpartitionIndex = int(endpartitionIndex/total_sample_length)
    endpartitionIndex *= total_sample_length
    beginpartitionIndex = int(self.begin*(len(self.timestamps)-1))
    beginpartitionIndex = int(beginpartitionIndex/total_sample_length)
    beginpartitionIndex *= total_sample_length
    
    data = self.timestamps["stats"][beginpartitionIndex:endpartitionIndex]
    self.args.start_training_time = self.timestamps["timestamp"][beginpartitionIndex]
    self.args.start_ending_time = self.timestamps["timestamp"][endpartitionIndex]
    self.overall_timestamps = self.timestamps["timestamp"][beginpartitionIndex:endpartitionIndex].values
    data_timestamps = torch.LongTensor(self.timestamps["timestamp"][beginpartitionIndex:endpartitionIndex].values)
    data_timestamps =  [x for x in self.timestamps["timestamp"][beginpartitionIndex:endpartitionIndex].values]
    data_stats = torch.Tensor(data.values)
    if self.args.train_normalization == True:
      scaler = MinMaxScaler(feature_range=(-1, 1))
      self.scaler = scaler
      data_normalized = scaler.fit_transform(data_stats.reshape(-1, 1))
      data_normalized = torch.FloatTensor(data_normalized).view(-1)
    else:
      data_normalized = torch.FloatTensor(data_stats).view(-1)
    self.min_value = torch.min(data_normalized)
    self.max_value = torch.max(data_normalized)
    L = len(data_normalized)
    for i in range(L-total_sample_length):
        seq = data_normalized[i:i+self.args.input_length]
        label = data_normalized[i+self.args.input_length:i+total_sample_length]
        seq_timestamp = data_timestamps[i:i+self.args.input_length]
        label_timestamp = data_timestamps[i+self.args.input_length:i+total_sample_length]
        self.sequence.append((seq,label))
        self.sequence_timestamps.append((seq_timestamp,label_timestamp))                         
        
class LocalSampledDataset:
  def __init__(self, timestamps,x_axis, y_axis, args):
    self.timestamps = timestamps
    self.args = args
    self.x_axis = x_axis
    self.y_axis = y_axis
    self.sequence = []
    self.sequence_timestamps = []

  def make_data(self, sampled_list):
    total_sample_length = self.args.input_length + self.args.output_length
    datas = self.timestamps["stats"]
    datas = torch.Tensor(datas.values)
    if self.args.train_normalization == True:
      scaler = MinMaxScaler(feature_range=(-1, 1))
      self.scaler = scaler
      datas = scaler.fit_transform(datas.reshape(-1, 1))
      datas = torch.FloatTensor(datas).view(-1)

    self.min_value = torch.min(datas)
    self.max_value = torch.max(datas)

    data_timestamps = torch.LongTensor(self.timestamps["timestamp"].values)
    data_timestamps =  [x for x in self.timestamps["timestamp"].values]
    for i in sampled_list:
      seq = datas[i:i+self.args.input_length]
      label = datas[i+self.args.input_length:i+total_sample_length]
      seq_timestamp = data_timestamps[i:i+self.args.input_length]
      label_timestamp = data_timestamps[i+self.args.input_length:i+total_sample_length]
      self.sequence.append((seq,label))
      self.sequence_timestamps.append((seq_timestamp,label_timestamp))   
    
class GlobalSequentialDataset:
  def __init__(self, args, ratioBegin, ratioEnd):
    self.args = args
    self.ratioBegin = ratioBegin
    self.ratioEnd = ratioEnd
    self.sequence = []
    self.sequence_timestamps = []
    self.seqDict = {}
    self.labelDict = {}
    self.raw_stats_dict = defaultdict(dict)
    self.raw_timestamps_dict= defaultdict(dict)
    
  def add_data(self, grid_id, timestamps): 
    grid = grid_id.split(".")
    axis = grid[0].split("X")
    x_axis = int(axis[0])
    y_axis = int(axis[1])
    total_sample_length = self.args.input_length + self.args.output_length

    endpartitionIndex = int(self.ratioEnd *(len(timestamps)-1))
    endpartitionIndex = int(endpartitionIndex/total_sample_length)
    endpartitionIndex *= total_sample_length
    beginpartitionIndex = int(self.ratioBegin*(len(timestamps)-1))
    beginpartitionIndex = int(beginpartitionIndex/total_sample_length)
    beginpartitionIndex *= total_sample_length
    
    data = timestamps["stats"][beginpartitionIndex:endpartitionIndex]
    data_timestamps = torch.LongTensor(timestamps["timestamp"][beginpartitionIndex:endpartitionIndex].values)
    data_timestamps =  [x for x in timestamps["timestamp"][beginpartitionIndex:endpartitionIndex].values]
    self.start_timestamp = timestamps["timestamp"][beginpartitionIndex]
#     print("beginpartitionIndex:",beginpartitionIndex,"self.start_timestamp",self.start_timestamp)
    self.end_timestamp = timestamps["timestamp"][endpartitionIndex]
    starting_timestamp = timestamps["timestamp"][beginpartitionIndex]
    self.raw_stats_dict[x_axis][y_axis] = []
    self.raw_stats_dict[x_axis][y_axis] = np.full(endpartitionIndex-beginpartitionIndex, 0.) 
    self.raw_stats_dict[x_axis][y_axis] = data
    
    self.raw_timestamps_dict[x_axis][y_axis] = []
    self.raw_timestamps_dict[x_axis][y_axis] = np.full(endpartitionIndex-beginpartitionIndex, 0.) 
    self.raw_timestamps_dict[x_axis][y_axis] = timestamps["timestamp"][beginpartitionIndex:endpartitionIndex]
    self.raw_timestamps_dict[x_axis][y_axis][:] = [x - starting_timestamp for x in self.raw_timestamps_dict[x_axis][y_axis]]
    data_stats = torch.Tensor(data.values)

    if self.args.test_normalization == True:
      scaler = MinMaxScaler(feature_range=(-1, 1))
      data_normalized = scaler.fit_transform(data_stats.reshape(-1, 1))
      data_normalized = torch.FloatTensor(data_normalized).view(-1)
    else:
      data_normalized = torch.FloatTensor(data_stats).view(-1)
    L = len(data_normalized)
    self.min_value = torch.min(data_normalized)
    self.max_value = torch.max(data_normalized)

    for i in range(L-total_sample_length):
      seq = data_normalized[i:i+self.args.input_length]
      label = data_normalized[i+self.args.input_length:i+total_sample_length]
      seq_timestamp = data_timestamps[i:i+self.args.input_length]
      if seq_timestamp[0].item() in self.seqDict:
        seq_maps = self.seqDict.get(seq_timestamp[0].item())
        k=0
        for s in seq:
          seq_maps[k][x_axis][y_axis]= s
          k += 1 
        item = {seq_timestamp[0].item(): seq_maps}
        self.seqDict.update(item)
        label_maps = self.labelDict.get(seq_timestamp[0].item())
        j=0 
        for l in label:
          label_maps[j][x_axis][y_axis]= l
        item = {seq_timestamp[0].item(): label_maps}
        self.labelDict.update(item)
      else:
        seq_maps = [[[]]]
        seq_maps = np.full((self.args.input_length,self.args.x_axis,self.args.y_axis), 0.) 
        i=0
        for s in seq:
          seq_maps[i][x_axis][y_axis]= s
          i += 1
        self.seqDict[seq_timestamp[0].item()] = seq_maps
        seq_timestamps = [[[]]]
        seq_timestamps = np.full((self.args.output_length,self.args.x_axis,self.args.y_axis), 0.) 
        j=0
        for l in label:
          seq_timestamps[j][x_axis][y_axis]= l
        self.labelDict[seq_timestamp[0].item()] = seq_timestamps   
    
  def make_data(self):
    for t,seq in self.seqDict.items():
      seq = torch.FloatTensor(seq)
      seq = seq.unsqueeze(0) 
      label = self.labelDict.get(t)
      label = torch.FloatTensor(label)
      self.sequence_timestamps.append(t)
      self.sequence.append((seq,label)) 
    
  def check_data(self):
    print("GlobalSequentialDataset:size of seqDict", len(self.seqDict))
    print("GlobalSequentialDataset:size of labelDict", len(self.labelDict))
    if(len(self.seqDict) != len(self.labelDict)):
      print("error in building the data")
    print("GlobalSequentialDataset min value",self.min_value)
    print("GlobalSequentialDataset max value",self.min_value)
    print("start timestamp of global sequential",self.start_timestamp)
    print("end timestamp of global sequential",self.end_timestamp)
    
class GlobalSampledDataset:
  def __init__(self, args):
    self.args = args
    self.sequence = []
    self.seqDict = {}
    self.labelDict = {}
    self.total_sample_length = self.args.input_length + self.args.output_length

  def add_data(self, grid_id, grid_dataset):
    grid = grid_id.split(".")
    axis = grid[0].split("X")
    x_axis = int(axis[0])
    y_axis = int(axis[1])
    for data, timestamps in zip(grid_dataset.sequence, grid_dataset.sequence_timestamps):
      seq,label = data
      seq_timestamp, label_timestamp = timestamps  
      if seq_timestamp[0].item() in self.seqDict:
        seq_maps = self.seqDict.get(seq_timestamp[0].item())
        i=0
        for s in seq:
          seq_maps[i][x_axis][y_axis]= s
          i += 1
        item = {seq_timestamp[0].item(): seq_maps}
        self.seqDict.update(item)
        label_maps = self.labelDict.get(seq_timestamp[0].item())
        j=0
        for l in label:
          label_maps[j][x_axis][y_axis]= l
        item = {seq_timestamp[0].item(): label_maps}
        self.labelDict.update(item)
      else:
        seq_maps = [[[]]]
        seq_maps = np.full((self.args.input_length,self.args.x_axis,self.args.y_axis), 0.) 
        i=0
        for s in seq:
          seq_maps[i][x_axis][y_axis]= s
          i += 1
        self.seqDict[seq_timestamp[0].item()] = seq_maps
        seq_timestamps = [[[]]]
        seq_timestamps = np.full((self.args.output_length,self.args.x_axis,self.args.y_axis), 0.) 
        j=0
        for l in label:
          seq_timestamps[j][x_axis][y_axis]= l.item()
        self.labelDict[seq_timestamp[0].item()] = seq_timestamps                    
    
  def make_data(self):
    self.min_value = sys.float_info.max
    self.max_value = sys.float_info.min
    for t,seq in self.seqDict.items():
      seq = torch.FloatTensor(seq)
      min_v = torch.min(seq).item()
      max_v = torch.max(seq).item()
      if min_v < self.min_value:
        self.min_value = min_v
      if max_v > self.max_value:
        self.max_value = max_v
      seq = seq.unsqueeze(0)
      label = self.labelDict.get(t)
      seq = torch.FloatTensor(seq)
      label = torch.FloatTensor(label)
      self.sequence.append((seq,label))

  def check_data(self):
    print("GlobalSampledDataset")
    print("GlobalSampledDataset:size of seqDict", len(self.sequence))
    print("GlobalSampledDataset min value",self.min_value)
    print("GlobalSampledDataset max value",self.max_value)
    self.args.max_value = self.max_value
    self.args.min_value = self.min_value
    if(len(self.seqDict) != len(self.labelDict)):
      print("error in building the data")
    
def GenerateRandomSamples(timestamps, args):
  total_sample_length = args.input_length + args.output_length
  samplenumbers = int(args.predictionSampleRatio*len(timestamps["stats"]))
  maxlength = len(timestamps["stats"])-total_sample_length
  sampled_list = random.sample(range(maxlength), samplenumbers)
  sampled_list= np.sort(sampled_list)
  return sampled_list

def MakeTrainingTimes(LocalTrainData, args):
  seq_timestamp, label_timestamps = LocalTrainData[0].sequence_timestamps[0]
  args.beginTrainingTimestamp = seq_timestamp[0].item()
  seq_timestamp, label_timestamps = LocalTrainData[0].sequence_timestamps[len(LocalTrainData[0].sequence_timestamps)-1]
  args.endTrainingTimestamp = seq_timestamp[0].item()

def MakeAttackTimes(AttackTrainingDataset, args):
  args.beginAttackTimestamp = AttackTrainingDataset.start_timestamp
  args.endAttackTimestamp = AttackTrainingDataset.end_timestamp

def MakeMembershipAttackTimes(AttackerTrainData,AttackerMemberTestData,AttackerNonMemberTestData, args):
  seq_timestamp, label_timestamps = AttackerMemberTestData[0].sequence_timestamps[0]
  args.beginTestMemberTimestamp = seq_timestamp[0].item()
  seq_timestamp, label_timestamps = AttackerMemberTestData[0].sequence_timestamps[len(AttackerMemberTestData[0].sequence_timestamps)-1]
  args.endTestMemberTimestamp = seq_timestamp[0].item()

  seq_timestamp, label_timestamps = AttackerNonMemberTestData[0].sequence_timestamps[0]
  args.beginTestNonMemberTimestamp = seq_timestamp[0].item()
  seq_timestamp, label_timestamps = AttackerNonMemberTestData[0].sequence_timestamps[len(AttackerNonMemberTestData[0].sequence_timestamps)-1]
  args.endTestNonMemberTimestamp = seq_timestamp[0].item()

  seq_timestamp, label_timestamps = AttackerTrainData[0].sequence_timestamps[0]
  args.beginTrainMembershipTimestamp = seq_timestamp[0].item()
  seq_timestamp, label_timestamps = AttackerTrainData[0].sequence_timestamps[len(AttackerTrainData[0].sequence_timestamps)-1]
  args.endTrainMembershipTimestamp = seq_timestamp[0].item()

def CheckLocalMembershipData(args):
  if args.training_attacker_begin > args.trainRatioEnd:
    print("The training data for the membership attack must contain some members")
  
  if args.training_attacker_end < args.trainRatioEnd:
    print("The training data for the membership attack must also contain some non members")

  if args.test_member_begin > args.trainRatioEnd:
    print("The test data for the membership attack must contain some members")  

  if args.test_non_member_begin < args.trainRatioEnd:
    print("The data points that are considered non members must no pertain to targeted model's training data")

  if args.test_non_member_begin > args.trainRatioEnd:
    print("The data points that are considered non members must no pertain to targeted model's training data")

def build_member_test_data(member_data, non_member_data):
  AttackerTestData = []
  AttackerTestData.append(member_data)
  AttackerTestData.append(non_member_data)
  return AttackerTestData
  
# def CheckLocalTestData(LocalTestData, args):
#   print("CheckLocalTestData")
#   beginningTime = args.beginTrainingTimestamp
#   endingTime = args.endTrainingTimestamp
#   print("beginningTime=",beginningTime)
#   print("endingTime=",endingTime)
#   print("args.trainingInterval=",args.trainingInterval)
#   for dataset in LocalTrainData:
#     for timestamp in range(beginningTime, endingTime, args.trainingInterval):
#       found_timestamp = False
#       index_timestamp = -1
#       i = 0
#       for seq_timestamp, label_timestamps in dataset.sequence_timestamps:
#         if timestamp >= seq_timestamp[0].item() and timestamp <= seq_timestamp[len(seq_timestamp)-1].item():
#           found_timestamp = True
#           index_timestamp = i
#         i += 1

#       if index_timestamp == -1:
#         print("a problem finding the timestamp in the training data for dataset index ",dataset.x_axis,",", dataset.y_axis)

def CheckLocalTrainData(LocalTrainData, args):
  print("CheckLocalTrainData")
  beginningTime = args.beginTrainingTimestamp
  endingTime = args.endTrainingTimestamp
  print("beginningTime=",beginningTime)
  print("endingTime=",endingTime)
  print("args.trainingInterval=",args.trainingInterval)
  for dataset in LocalTrainData:
    for timestamp in range(beginningTime, endingTime, args.trainingInterval):
      found_timestamp = False
      index_timestamp = -1
      i = 0
      for seq_timestamp, label_timestamps in dataset.sequence_timestamps:
        if timestamp >= seq_timestamp[0].item() and timestamp <= seq_timestamp[len(seq_timestamp)-1].item():
          found_timestamp = True
          index_timestamp = i
        i += 1

      if index_timestamp == -1:
        print("a problem finding the timestamp in the training data for dataset index ",dataset.x_axis,",", dataset.y_axis)

def CheckLocalPredictionData(datasets,samples):
  print("CheckLocalPredictionData")
  
  if "0X0" not in datasets:
    print("a problem in the dictionary of prediction datasets")
    exit("CheckLocalPredictionData: datasets[0X0] is not in the datasets")
  
  print("size of seqDict", len(datasets["0X0"].sequence))

  reference_timestamps = []
  for t in datasets["0X0"].sequence_timestamps:
    seq_t, label_t = t
    reference_timestamps.append(seq_t[0])
  totalsamples = len(datasets["0X0"].sequence_timestamps)
  for data in datasets.values():
    if totalsamples != len(data.sequence_timestamps):
      print("a problem in the timestamps length of the local prediction data (", data.x_axis,",",data.y_axis)
    for t in data.sequence_timestamps: 
      seq_t,label_t = t
      if not seq_t[0] in reference_timestamps:
        print("a problem in the timestamps of the local prediction data(", data.x_axis,",",data.y_axis)
        return False
  return True

def generate_local_prediction(weight_pair, LocalPredictionSample, args):
  if args.scratch_prediction == True:
    if args.local_model == 'GRU':
      weight_pair.model.to(args.device_name)
      predictionModel = GRU(args)
      predictionModel.load_state_dict(weight_pair.model.state_dict())
      predictionModel.to(args.device_name)
      localpredictionresult = predictionModel.predict(LocalPredictionSample,args)
      return localpredictionresult
    else:
      raise SystemError('Unrecognized local model')
  else:
    weight_pair.model.to(args.device_name)
    localpredictionresult = weight_pair.model.predict(LocalPredictionSample,args)
  return localpredictionresult

def build_adjMatrix(args):
  dimension = args.x_axis*args.y_axis
  adj = torch.zeros(dimension,dimension)

  for i,j in zip(range(args.x_axis),range(args.y_axis)):
    #build matrix
    ij_index = j * args.y_axis + i
    i = i+1
    j = j+1
    matrix_adj = torch.zeros(args.x_axis+2,args.y_axis+2)
    matrix_adj[i-1][j-1] = 1
    matrix_adj[i-1][j] = 1
    matrix_adj[i-1][j+1] = 1
    matrix_adj[i][j-1] = 1
    matrix_adj[i][j+1] = 1
    matrix_adj[i+1][j-1] = 1
    matrix_adj[i+1][j] = 1
    matrix_adj[i+1][j+1] = 1
    matrix_adj = matrix_adj[1:,:]
    matrix_adj = matrix_adj[:-1,:]
    matrix_adj = matrix_adj[:,1:]
    matrix_adj = matrix_adj[:,:-1]
    indeces = (matrix_adj == 1.0).nonzero()
    for index in indeces:
      x,y = index
      index = y * args.x_axis + x
      adj[ij_index][index] = 1
  return adj

def build_map(localpredictionresults, args):
  totalNumberOfSequences = len(localpredictionresults["0X0"])
  seq_predictions = [[[[]]]]
  seq_predictions = np.full((totalNumberOfSequences,args.output_length,args.x_axis,args.y_axis), 0.) 
  for index, localpredictions in localpredictionresults.items():
    axis = index.split("X")
    x_axis = int(axis[0])
    y_axis = int(axis[1])
    i = 0;
    for prediction in localpredictions:
      seq_predictions[i][0][x_axis][y_axis]= prediction
      i+=1
  seq_predictions = torch.Tensor(seq_predictions)
  return seq_predictions

def RandomFeature(sequence, i,j,args):
  out_seq = sequence.detach().clone()
  out_seq = out_seq.squeeze(0)
  if args.dataset == "bikeNYC":
    out_seq[:,i,j] = torch.normal(mean=0., std=1.,size=(args.input_length,))
  else:
    out_seq[:,i,j] =  torch.normal(mean=0., std=1.,size=(args.input_length,))
  out_seq = out_seq.unsqueeze(0)
  return out_seq

def marginal(vec1,vec2):
  mae = nn.L1Loss()
  return mae(vec1,vec2)

def build_attacker_input(args, seq, guessed_features, organization_contribution_distribution):
  out_seq = seq.squeeze(0)
  flat_out_seq = torch.flatten(out_seq.detach().clone())
  flat_guessed_seq = torch.flatten(guessed_features.detach().clone())
  
  indeces = list(*np.where(organization_contribution_distribution == args.attacker_organization_id))
  j = 0
  for i in indeces:
    flat_out_seq[i] = flat_guessed_seq[j]
    j+=1

  out_seq = flat_out_seq.reshape(args.input_length, args.x_axis, args.y_axis)
  out_seq = out_seq.unsqueeze(0)
  return out_seq

def extract_true_input(seq,organization_contribution_distribution,organization_id ):
  flat_seq = seq.squeeze(0)
  flat_seq = torch.flatten(flat_seq)
  indeces = list(*np.where(organization_contribution_distribution == organization_id))
  flat_out_seq = torch.zeros(len(indeces))
  j = 0
  for i in indeces:
    flat_out_seq[j] = flat_seq[i]
  return flat_out_seq

def compute_feature_contribution(organization_contribution_distribution, feature_contribution_metric, args):
  total_features = feature_contribution_metric.sum()
  feature_contribution_metrics = [[]]
  feature_contribution_metrics = np.full((args.x_axis,args.y_axis), 0.)

#   print("total_features:",total_features)
  attacker_organization_contribution = feature_contribution_metric[np.where(organization_contribution_distribution == args.attacker_organization_id)].sum()
#   print("attacker contribution",attacker_organization_contribution, " for organization_id", args.attacker_organization_id)
  
  victim_organization_contribution = feature_contribution_metric[np.where(organization_contribution_distribution == args.victim_organization_id)].sum()
  
  organization_contribution_metrics = []
  organization_contribution_metrics = np.full(3, 0.)
  organization_contribution_metrics[args.attacker_organization_id] = attacker_organization_contribution/total_features
  organization_contribution_metrics[args.victim_organization_id] = victim_organization_contribution/total_features

  for i in range(0,args.x_axis):
    for j in range(0,args.y_axis):
      feature_contribution_metrics[i][j] = organization_contribution_metrics[organization_contribution_distribution[i][j]]
  return feature_contribution_metrics

def modify_input_attacker(seq, organization_contribution_distribution, args):
  out_seq = seq.detach().clone()
  out_seq = out_seq.squeeze(0)
  for pair in list(zip(*np.where(organization_contribution_distribution == args.attacker_organization_id))):
    i, j, k = pair
    if args.dataset == "bikeNYC":
      out_seq[i,j,k] = np.random.normal(0., 1.)
    else:
      out_seq[i,j,k] = np.random.normal(0., 1.)
  out_seq = out_seq.unsqueeze(0)
  return out_seq

def initialize_ratio_outputsize(args):
  ratio = args.featureRatio
  output = -1 
  if args.dataset == "bikeNYC":
    ## Bike 
    # 10% == 12 cells
    if ratio == 10:
      output= 12
    if ratio == 20:
      # 20% == 25 cells
      output= 25
    ## 30% == 38 cells
    if ratio == 30:
      output= 38
    ## 40% == 51 cells
    if ratio == 40:
      output= 51
    ## 50% == 64 cells
    if ratio == 50:
      output= 64
    # 60% == 76 cells
    if ratio == 60:
      output= 60
    # 70% == 90 cells
    if ratio == 70:
      output= 90
    # 80% == 102 cells
    if ratio == 80:
      output= 102
    # 90% == 115 cells
    if ratio == 90:
      output= 115

  ##############################################################
  ##############################################################
  ## Yelp 
  if args.dataset == "Yelp":
    # 10% == 6 cells
    if ratio == 10:
      output= 6
    ## 20% == 12 cells
    if ratio == 20:
      output= 12
    ## 30% == 19 cells
    if ratio == 30:
      output= 19
    ## 40% == 26 cells
    if ratio == 40:
      output= 26
    ## 50% == 32 cells
    if ratio == 50:
      output= 32
    # 60% == 38 cells
    if ratio == 60:
      output= 38
    # 70% == 45 cells
    if ratio == 70:
      output= 45
    # 80% == 51 cells
    if ratio == 80:
      output= 51
    # 90% == 58 cells
    if ratio == 90:
      output= 58
  if output == -1:
    print("a problem assigning the attacker output size")
  return output*args.input_length

  
def initialize_ratio_contribution(args,feature_contribution_metric):
  ratio = args.featureRatio
  ratio_contribution_metrics = [[]]
  ratio_contribution_metrics = np.full((args.x_axis,args.y_axis), 1.)

  organization_contribution_distribution = [[]]
  organization_contribution_distribution = np.full((args.x_axis,args.y_axis), args.victim_organization_id)

  number_Features = initialize_ratio_outputsize(args)//args.input_length  
  number_features_counter = 0 
  if args.feature_sampling == "Random":
    total_features = args.x_axis*args.y_axis
    sampledIndeces = random.sample(range(total_features), number_Features)
    for sample in sampledIndeces:
      x = sample//args.y_axis
      y = sample-x*args.y_axis
      organization_contribution_distribution[x][y] = args.attacker_organization_id  
  else:
    feature_dict = defaultdict(dict)
    for i in range(args.x_axis):
      for j in range(args.y_axis):
        index = i*args.y_axis+j
        feature_dict[index] = feature_contribution_metric[i][j]
    if args.feature_sampling == "Increasing":
      for k in sorted(feature_dict, key=feature_dict.get, reverse=False):
        if number_features_counter < number_Features:
          x = k//args.y_axis
          y = k-x*args.y_axis  
          organization_contribution_distribution[x][y] = args.attacker_organization_id 
          number_features_counter+=1
    else:
      if args.feature_sampling == "Decreasing":
        for k in sorted(feature_dict, key=feature_dict.get, reverse=True):
          if number_features_counter < number_Features:  
            x = k//args.y_axis
            y = k-x*args.y_axis  
            organization_contribution_distribution[x][y] = args.attacker_organization_id  
            number_features_counter+=1

  return organization_contribution_distribution

def train_attacker(attacker,timestamp,AttackerTrainData,globalModel,adj,organization_contribution_distribution, args):
  flat_organization_contribution_distribution = np.ndarray.flatten(organization_contribution_distribution)
  loss = 0.
  acc = {}
  acc["MAE"] = torch.zeros(1,args.input_length,args.x_axis,args.y_axis)
  acc["MSE"] = torch.zeros(1,args.input_length,args.x_axis,args.y_axis)
  acc["RMSE"] = torch.zeros(1,args.input_length,args.x_axis,args.y_axis)
  acc["AE"] = torch.zeros(1,args.input_length,args.x_axis,args.y_axis)
  acc["WMAPE"] = torch.zeros(1,args.input_length,args.x_axis,args.y_axis)

  found_timestamp = False
  index_timestamp = -1
  n = len(AttackerTrainData.sequence_timestamps)
  first = AttackerTrainData.sequence_timestamps[0]
  last = AttackerTrainData.sequence_timestamps[n-1]

  index_timestamp = (timestamp-first)/(last-first)*n
  index_timestamp = int(index_timestamp)

  endtimestamp = timestamp+args.trainingInterval
  index_end = (endtimestamp-first)/(last-first)*n
  index_end = int(index_end)

  if index_timestamp == -1:
    print("a problem finding the timestamp in the training data for dataset index ",AttackerTrainData.x_axis,",", AttackerTrainData.y_axis)
  losses = []
  
  input_Sequence = AttackerTrainData.sequence[index_timestamp:index_end]
  batch_size = len(input_Sequence) 
  attacker.optimizer.zero_grad()
  Indexes = [i for i in range(batch_size-1)]
  np.random.shuffle(Indexes)
  for i in Indexes:      
    seq = input_Sequence[i][0]
    labels = input_Sequence[i][1] 
    seq = seq.to(args.device_name)
    labels = labels.to(args.device_name)

    #initialize randomly the guessed targeted feature
    rand_seq = modify_input_attacker(seq,organization_contribution_distribution, args)
    
    #make the attacker model guess the input
    rand_seq = rand_seq.to(args.device_name)
    guessed_features = attacker.forward_single(rand_seq, True)
    
    #build the targeted model's input using the adversary best guess
    modified_input = build_attacker_input(args, seq,guessed_features,flat_organization_contribution_distribution)
    
    #calculate the training accuracy
    acc["MAE"] += torch.abs(seq - modified_input)
    acc["MSE"] += (seq - modified_input) **2  
    acc["AE"] += seq - modified_input 
    acc["WMAPE"] += torch.abs(seq)

    #compare the output of the targeted model's predictions
    modified_input = modified_input.to(args.device_name)
    with torch.no_grad():
      attackerprediction = globalModel.predict(modified_input,adj).unsqueeze(0)
      attackerprediction = attackerprediction.to(args.device_name)

    #backprop the attacker's model
    loss = attacker.backward_single(guessed_features,attackerprediction, labels)
    losses.append(loss)    
  attacker.optimizer.step()

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

  return acc_floats, sum(losses)/len(losses)

def initialize_weight(args):
  if args.featureMode == "None":
    return 0
