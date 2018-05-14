---
categories: kaggle
title: 캐글 타이타닉 생존자 예측
author: inoray
tags: kaggle titanic
---


## 1. 데이터 분석

아래 강좌를 참조해서 분석

- [허민석님 유투브 강좌](https://www.youtube.com/watch?v=aqp_9HV58Ls)

[구현 소스](https://github.com/inoray/ML_DL_Tensorflow_study/tree/master/season_02_Kaggle/01_Titanic_Machine_Learning_from_Disaster/김성헌/김성헌_타이타닉_데이터분석.ipynb) 참조.



## 2. Modeling

- tensorflow DNN ensemble model 구성
- corss validation 으로 최적 hyper parameter 탐색
- 최종 hyper parameter
  - learning_rate: 0.001
  - training_epoches: 200
  - hidden layers: 3
    - dense, dropout, relu
  - units:100
  - ensemble models: 10

[구현 소스](https://github.com/inoray/ML_DL_Tensorflow_study/tree/master/season_02_Kaggle/01_Titanic_Machine_Learning_from_Disaster/김성헌/김성헌_타이타닉_DNN_Ensemble.ipynb) 참조.



## 3. 결과

![결과](/assets/images/2018-05-05-kaggle-taitanic/김성헌_타이타닉_DNN_Ensemble.png)
