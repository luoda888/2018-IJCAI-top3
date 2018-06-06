# 2018-IJCAI-top3
This is 2018 IJCAI alimama Top3 Code. 3/5204
<br>We open source parts of the code and explain all the feature engineering.

## Introduction
<br>CTR estimation problem is a classic and valuable problem in the field of advertising algorithms.At present, the industry has a more mature solution to the problem of CTR estimation in steady flow.
<br>The problem of this competition is to find a stable and reliable way to estimate the CTR problem during the promotion period in abnormal traffic.
<br>We make exploratory analysis of the data on the change of abnormal flow, and construct the characteristics of sales volume, price and display times.And based on analysis of the distribution of data, we have constructed four different training sets.Each training set utilizes an integrated learning and neural network model.
<br>The constructed offline validation strategy is the last hour and last two hours of the morning for abnormal traffic, and the evaluation indicators are `Auc` and `Logloss`.
  
## Feature Engineering 
	User-Item/Shop/Brand/City
	User/Shop/Item portrait
	Click Time Feature
	High order interaction characteristics
	Sequence Statistical Feature
	Trick Feature
  
## Model

	Lightgbm
	Xgboost
	Catboost
	GBDT+LR
	NN (DeepFFM,DeepFM,FNN)
  
