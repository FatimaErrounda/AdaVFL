from datetime import datetime
class arguments():
  def __init__(self):
    #######
    #data
    self.dataset = "Yelp" #"Yelp" or "bikeNYC"
    # bike link
    self.dataset_link = "https://drive.google.com/drive/folders/1diJwebRNa5AQ16Jy6eHNGtYGmIGeqcrt"
    # yelp link
    self.dataset_link = "https://drive.google.com/drive/folders/1K2Y_txKAda0TOEEDYvoa7sPPMYLCXI-U"
    self.input_length = 6
    self.output_length = 1
    self.trainRatioBegin = 0
    self.trainRatioEnd = 0.6
    self.testRatioBegin = 0.8
    self.testRatioEnd = 1.0
    self.predictionSampleRatio = 0.5
    self.trainAttackerBegin = 0.5
    self.trainAttackerEnd = 0.7
    self.trainMemberRatio = 0.5
    self.testMemberRatio = 0.7
    self.data_capture_interval = 1800000
    self.feature_sampling="Increasing" #"Random","Increasing", "Decreasing"

    #Model
    self.local_model_loss = "MAE"
    self.global_model_loss = "MAE"
    self.local_model = 'GRU'
    self.global_model = 'GNN'
    self.epochs = 16
    self.pretrainepochs = 6
    self.attacker_epochs = 6
    self.test_normalization = True
    self.train_normalization = True

    #membership privacy attack
    self.observedEpochs = [6,8,10,12]
    self.cluster_threshold = 0.5
    self.training_attacker_begin = 0.20
    self.training_attacker_end = 0.45
    self.test_member_begin = 0.2
    self.test_member_end = 0.25
    self.test_non_member_begin = 0.3
    self.test_non_member_end = 0.4

    #feature privacy
    self.featureMode = "Uniform" # "InvContribution" "LinContribution" "Uniform" "None"
    self.victim_organization_id = 2
    self.attacker_organization_id = 1
    self.featureRatio = 90

    #learning privacy
    self.epsilon_0 = 1.6
    self.epsilon_1 = 3.2
    self.delta_0 = 0.00001
    self.clipping_threshold = 4
    self.sigma_0 = 4.
    self.PrivacyMode = "None" #"None", "Concentrated", "Uniform", "Adaptive", "Validation", "Increase"

    #adaptive
    self.AdaptiveError = "Training" #"Accuracy" "Training"
    self.alpha = 1
    self.epoch_period = 5
    self.gamma_min = 0.06

    #DP-AGD
    self.obj_clip = 1.0
    self.gamma=0.1

    # validation-based
    self.validation_period = 5
    self.validation_sigma = 10.
    self.validation_threshold = 0.1
    self.validation_coefficient = 0.7
    self.validation_accuracy_metric = "MAE" # "RMSE", "WMAPE", "AE"

    # Increase-based
    self.Increase_curb = "Intrvl" ## "Log", "Exp", "Intrvl"
    self.Increase_e_max = 1.
    self.Increase_e_min = 0.1
    self.Increase_gamma = 10 
    
    #capturing output
    self.groundTruth =[]
    self.prediction = []
    self.scratch_prediction = True

    