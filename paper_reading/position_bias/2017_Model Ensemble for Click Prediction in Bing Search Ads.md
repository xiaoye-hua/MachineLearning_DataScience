## TODO & Questions

1. [x] How to solve position bias 
    1. [x] Logistic regression model 
2. [x] Inversed position bias has better performance than normal one. 
    1. [x] Note that we use the inverse position bias, i.e., given pb = σ(x),
    2. We need the inversed form because the final sigmoid will convert it back to position CTR which is empirical CTR.
we use x instead of pb.???
3. [x] COEC -> Click Over Expected Clicks -> check ref paper
4. 特征处理
    1. 所有的统计特征包括position特征都需要做归一化处理【目前常规的做法是统计值分桶，然后学习不同桶的低纬嵌入，落地方面，效果稳定，还能建模包括统计特征在内的特征交叉】
5. Position features    
    1. position bias whose value is roughly the expected CTR of all samples
collected from an bucket traffic with **randomized ad order**

## Problem Definition

1. [x] key metrics
    1. Offline: AUC

## Previous Methods & New Methods


1. 8 ensemble methods are evaluted
2. Boosting neural networks with gradient boosting decision trees turns out to be the best.
    1. [x] how does it work??

## Impacts

1. With larger training data, it shows near 0.9% AUC improvement in offline testing and significant click yield gains in online traffic.


## Ref

1. [Model Ensemble for Click Prediction in Bing Search Ads 论文讲解](https://zhuanlan.zhihu.com/p/542734071)
2. Related papers:
    1. the concept of COEC: [Personalized click prediction in
sponsored search, 2010, Yahoo](2010_%20Personalized%20click%20prediction%20in%20sponsored%20search.md)
