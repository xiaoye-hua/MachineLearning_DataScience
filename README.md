# Machine Learning Material
Materials or code that I have used to learn machine/deep learning.

## Project Structure
1. src/ source code
2. Unittest/ unit test for source code
4. notebooks/ jupyter notebooks
5. scripts/ python scripts 
3. data/ training data

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
        - [x] [Linear SVM](src/Model/SVM/LinearSVM.py)
        - [ ] Kernel SVM
    - [ ] Naive Bayes
- Deep Learning 
    - [x] [Neural Network](src/Model/NN/NNModel.py)
    - Sequence Model
        - RNN
        - LSTM
        - GRU
        - [x] [Attention](src/Model/SequenceModel/Attention.py)
        - [x] [seq2seq](src/Model/SequenceModel/Seq2Seq.py)
        - [x] [transformer](src/Model/SequenceModel/Transformer.py)
    - CNN: 
        - [x] [AlexNet](src/Model/CNN/AlexNet.py)
- (Deep) Reinforcement Learning

Refer to [cookiecutter-data-science-structure](https://drivendata.github.io/cookiecutter-data-science/#example) for machine learning project structure
