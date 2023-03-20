# Machine Learning Material
Materials or code that I have used to learn machine/deep learning.

## Project Structure
1. src/: source code
2. Unittest/: unit test for source code
4. notebooks/: jupyter notebooks
5. scripts/: python scripts 
3. data/: training data

Refer to [cookiecutter-data-science-structure](https://drivendata.github.io/cookiecutter-data-science/#example) for machine learning project structure


## Machine Learning Algorithms Code Implementation

- Traditional Machine Learning
    - Generalized Linear Model
        - [ ] Linear regression (Lasso, Ridge)
        - [ ] logistic regression
        - [ ] Linear discriminant analysis
    - Decision tree
        - [x] [ID3 decision tree](src/Model/DecisionTree/ID3DecisionTree.py)
        - [x] [C4.5 decision tree](src/Model/DecisionTree/C45DecisionTree.py)
        - [x] [CART](src/Model/DecisionTree/CART.py)
        - [ ] Random Forest
        - [ ] Gradient Boosted Tree 
    - SVM
        - [x] [Linear SVM (Binary classification)](src/Model/SVM/LinearSVM.py)
        - [x] [Kernel SVM (Binary classification)](src/Model/SVM/KernelSVM.py)
    - [x] [KNN](src/Model/KNN.py)
    - [ ] Naive Bayes
    - Clustering
        - [ ] K-means
        - [ ] Hierarchical Clustering
- Deep Learning 
    - [x] [Neural Network](src/Model/NN/NNModel.py)
    - Sequence Model
        - [x] [RNN](src/Model/SequenceModel/RNN.py)
        - [x] [LSTM](src/Model/SequenceModel/LSTM.py)
        - [x] [GRU](src/Model/SequenceModel/GRU.py)
        - [x] [Attention](src/Model/SequenceModel/Attention.py)
        - [x] [seq2seq](src/Model/SequenceModel/Seq2Seq.py)
        - [x] [transformer](src/Model/SequenceModel/Transformer.py)
    - CNN: 
        - [x] [AlexNet](src/Model/CNN/AlexNet.py)
        - [x] [VGG11](src/Model/CNN/VGG11.py)
        - [x] [NiN](src/Model/CNN/NiN.py)
        - [x] [GoogLeNet](src/Model/CNN/GoogLeNet.py)
        - [x] [ResNet18](src/Model/CNN/ResNet18.py)
        - [x] [DenseNet](src/Model/CNN/DenseNet.py)
- (Deep) Reinforcement Learning
    - Multi-bandit
        - [ ] Contextual multi-bandit
    - [x] [Q learning](src/ReinforcementLearning/Q_learning_Sarsa/run_q_learing.py)
    - [x] [SARSA](src/ReinforcementLearning/Q_learning_Sarsa/run_Sarsa.py)
    - [x] [Deep Q-learning](src/ReinforcementLearning/Deep_Q_Learning)
    - [x] [Policy gradient](src/ReinforcementLearning/policy_gradient)
    - [x] [Actor-Critic](src/ReinforcementLearning/Actor_Critic)
    - ....
-  NLP 
    - [ ] Word2Vector
- Application 
    - Recommendation
        - [ ] Deep Semantic Similarity Model(DSSM)
        - [ ] YoutubeDNN
        - [ ] MIND
        - [x] [Matrix Factorization (MF)](src/Model/Recommender/MF.py)
        - [x] [Factorization Machine (FM)](src/Model/Recommender/FM.py)
        - [x] [Wide & deep](src/Model/Recommender/WideDeep.py)
        - [x] [DeepFM](src/Model/Recommender/DeepFM.py)



## Course 

1. [Multi-agent System](https://github.com/xiaoye-hua/Multi_Agent_System)
2. [Udacity - Machine Learning DevOps Nanodegree Program](report/Udaciy_ML_DevOps_nanodegree.md)

## Competition codes:

1. [Personalize Expedia Hotel Searches - ICDM 2013](https://github.com/xiaoye-hua/expedia_hotel_recommendation)

## More learning codes


1. Kaggle 
    1. [categorical feature with LR, XGBoost and CatBoost](https://www.kaggle.com/code/huaguo/categorical-feature-with-lr-xgboost-and-catboost)
 
## How to run the code

### Step 1: environment setup
```shell script
conda create --name ds_env --file requirements.txt python=3.6
conda activate ds_env
export PYTHONPATH=./:PYTHONPATH
```
