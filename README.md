
![Logo](https://raw.githubusercontent.com/saranpan/dipple/main/logo.jpg)

# dipple: deep but simple to build..

[![open in colab](https://camo.githubusercontent.com/52feade06f2fecbf006889a904d221e6a730c194/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667)](https://colab.research.google.com/drive/1qLPAn6oXnh96rKPn_LrpxPCBxW4rzgJT?usp=sharing) [![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/wallik2/dipple/blob/main/LICENSE) [![Git](https://img.shields.io/github/forks/wallik2/dipple)](https://github.com/saranpan/dipple) [![Discord](https://img.shields.io/discord/911220061287616594)](https://discord.gg/XS8Znh7HPs)

## what is it?
<b>Dipple</b> is a Python package that simplifies the process of creating neural network architectures, particularly for beginners in data science. It offers a simple and easy-to-use interface for building Linear, Logistic, Shallow, and Deep L-layer Neural networks using only a few lines of code. 

It currently supports Multi-layer Perceptron, with various regularization options such as L2 and Dropout, as well as optimizers and weight initialization techniques to improve training and avoid saddle points. The name "Dipple" is a combination of the words "deep" and "simple", and it reflects the package's goal of making building deep neural networks easy and accessible to beginners.

The project, Dipple, was initiated in 2022 by Saran Pannasuriyaporn as a means of self-study in the field of deep learning. The author chose to learn by writing code from scratch, as a way of gaining a deeper understanding of the concepts. This package is not intended to replace existing libraries such as Tensorflow or Pytorch, but rather to provide an opportunity for aspiring learners to not only learn about deep learning, but also advanced concepts such as object-oriented programming by examining the code samples of Dipple Repository available on [GitHub](https://github.com/saranpan/dipple).

## Requirement
Python 3.7 +

## Installation
```sh
pip install dipple
```

## Quick Start


#### 1. Import the dataset

Get started quickly by exploring the Crescent dataset, a popular toy dataset for binary-class classification tasks and a benchmark for machine learning models. With Dipple, loading and using the Crescent dataset is easy and straightforward

```sh
from dipple.import_dataset import Dataset_Library

dlib_borrow = Dataset_Library('crescent') 
df_train, df_test = dlib_borrow.get_2df()
```
![output_code](https://i.ibb.co/KWnvCqp/dee.png)

```sh
# Preview the first 3 rows of train set
display(df_train.head(3))
```

|    | x1 | x2 | y |
| ------ | ------ | ------ | ------ |
| 0 | -0.216870 | 1.015449 | 0 |
| 1 | 0.805050 | -0.557973 | 1 |
| 2 | 0.711275 | -0.410060 | 1 |

```sh
#splitting predictor and class
X_train = df_train[['x1','x2']]
Y_train = df_train[['y']]
```

#### 2. Define the Multilayer Perceptron Architecture

If you wish to build a multi-layer perceptron with 2 hidden layers, containing 5 and 2 units respectively, you can define the details in the ```hyperparam_setting``` dictionary. In addition, you can also specify the activation function for both hidden and output layer as relu and sigmoid respectively

```sh
hyperparam_setting = {
              "layer_dims" : [2,5,2,1],
              "hidden_activation_function" : 'relu',
              "output_activation_function" : 'sigmoid',}   
```

The Dipple's MLP implementation expects the ```hyperparam_setting``` to have specific keys named "layer_dims", "hidden_activation_function", and "output_activation_function" respectively, in order to define the architecture of the multi-layer perceptron.

The available activation functions for both of hidden and output layers are as the following :
- ```linear```
- ```sigmoid```
- ```tanh```
- ```relu```
- ```leakyrelu```
- ```softmax```

Once choose, you can define the model object by input ```hyperparam_setting``` 
```sh
from dipple.mlp import *

model = MultilayerPerceptron.initiate_by_hyperparam_dict(hyperparam_setting)
```

#### 3. Configuring Gradient Descent
This step is used to specify the method for updating the parameters via gradient descent.

If you want to set the gradient descent with loss function binary_cross_entropy_loss, learning rate 0.0001, weight initialization as he, regularization as L2 with lambda value of 0.001, and optimizer adam with beta1 = 0.9, beta2 = 0.99, eps = 10e-8, you can use the following code:


```sh
model.compiles(loss='binary_cross_entropy_loss',lr=1e-3,initialization='he',regularization="L2",lambd= 1e-2,optimizer='adam',beta1=0.9,beta2=0.99,eps=10e-8)
```

The details of argument setting for method compile are shown as the following:
- ```loss``` : ['binary_cross_entropy_loss','cross_entropy_loss','MSE']
- ```learning rate```
- ```weight initialization techniques``` : ['zero','random','he','xavier']
- ```regularization``` : ['dropout','L2']
- ```optimizer``` : : ['momentum','adagrad','rmsprop','adam']

Note that if you wish to use dropout instead of L2, the keyword argument ```lambd``` must be replaced by tuple ```keep_prob_sequence```, which indicate the keep probability of a sequence of layer respectively 

#### 4. Fit Data to our model
Once the model is configured, we can use it to fit our data using mini-batch gradient descent with a batch size of 32 for 27000 epochs. To track the progress of the model and report the cost function every 1000 epochs. we can use the following code 

```sh
model.fit(X_train,Y_train,Epochs=27000,batch_size=32,
report_cost=True, evry_report_epoch = 1000)
```
```sh
Output:
========
> Epoch 1000/27000 : ===Cost=== : 0.4433844520553876
> Epoch 2000/27000 : ===Cost=== : 0.3674708272179111
> Epoch 3000/27000 : ===Cost=== : 0.34272523427485757
                            .
                            .
                            .
Epoch 26000/27000 : ===Cost=== : 0.1516102412308588
Epoch 27000/27000 : ===Cost=== : 0.15146985031429971
```

![output_code2](https://i.ibb.co/52s9rYh/dee2.png)



Once the model is trained, we can access the updated parameters (weights and biases) for each layer by using the following code:



```sh
model.param
```

```sh
Output
=======
{'W1': array([[ 1.66269748,  0.18280045],
            [ 0.98504132,  1.58239975],
            [ 1.23171595,  0.07314983],
            [ 1.56213207,  0.05702136],
            [-0.39345288,  0.88787371]]),
        .
        .
 'W3': array([[-0.51395741,  4.60415329]]), 
 'b3': array([[-4.84589836]])}
```


#### 5. Predict 
The trained model can be used to make predictions using the predict method. There are two options for the output: probability and cut-off value.

If you want the probability, you can directly use the following code:

```sh
model.predict(X_train)
```

```sh
Output
=======
array([[0.12525224, 0.96623857, 0.96625601, 0.99820462, 0.00779925, ....]])
```
However, if you prefer the cut-off value with a threshold of 0.5, you can use the following code:

```sh
model.predict(X_train, predict_proba=False, threshold=0.5)
```

```sh
Output
=======
array([[0, 1, 1, 1, 0, ....]])
```

It's worth noting that when predict_proba is set to False, the threshold parameter is not required for multi-class classification (with softmax as the output activation function) as the class with the highest probability will be selected automatically.


#### 6. Evaluate and Interpret the result
If there are 2 predictors, it is worth-try to plot 2D decision boundary. In this crescent dataset

```sh
from dipple.interpret import plot_decision_boundary_2D
from dipple.metrics import accuracy_score

threshold = 0.5

plot_decision_boundary_2D(model=model,X=X,y=Y,threshold=threshold)
Y_pred = model.predict(X,threshold = threshold,predict_proba=False)
print(f'Accuracy on train set : {accuracy_score(Y_pred,Y)}')
```
![output_code3](https://i.ibb.co/WpCQXTJ/dee3.png)
Accuracy on train set : 0.94

## Dependencies
our package dipple implements by these packages with the following versions

| Dependency | Version |
| ------ | ------ |
| numpy | 1.21.6 |
| pandas | 1.3.5 |
| matplotlib | 3.2.2 |



## License

- [MIT]




[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

   [MIT]: <https://github.com/wallik2/dipple/blob/main/LICENSE>

