# %% External module
import numpy as np
import matplotlib.pyplot as plt

def dictionary_to_vector(parameters):
    """
    Roll all our parameters dictionary into a single vector satisfying our specific required shape.
    """
    keys = []
    count = 0
    params = []
    num_params = len(parameters)//2
    for l in range(1,num_params+1):
        params = params + ["W"+str(l)]
        params = params + ["b"+str(l)]
        
    for key in params:
        
        # flatten parameter
        new_vector = np.reshape(parameters[key], (-1,1))
        keys = keys + [key]*new_vector.shape[0]
        
        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count = count + 1

    return theta, keys

def vector_to_dictionary(theta,parameters):
    """
    Unroll all our parameters dictionary from a single vector satisfying our specific required shape.
    """
    params = {}
    num_params = len(parameters)//2
    num_data_n = 0
    for l in range(1,num_params+1):
        num_data = num_data_n
        num_data_n = num_data + parameters["W"+str(l)].shape[0]*parameters["W"+str(l)].shape[1]
        
        params["W"+str(l)] =theta[num_data:num_data_n].reshape((parameters["W"+str(l)].shape[0],
                                                                parameters["W"+str(l)].shape[1]))
        
        num_data = num_data_n
        num_data_n = num_data + parameters["b"+str(l)].shape[0]*parameters["b"+str(l)].shape[1]
        params["b"+str(l)] = theta[num_data:num_data_n].reshape((parameters["b"+str(l)].shape[0],
                                                                 parameters["b"+str(l)].shape[1]))

    return params

def gradients_to_vector(parameters,gradients):
    """
    Roll all our gradients dictionary into a single vector satisfying our specific required shape.
    """
    
    count = 0
    params = []
    num_params = len(parameters)//2
    for l in range(1,num_params+1):
        params = params + ["dW"+str(l)]
        params = params + ["db"+str(l)]
        
    for key in params:
        # flatten parameter
        new_vector = np.reshape(gradients[key], (-1,1))
        
        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count = count + 1

    return theta

def pd_to_np(any_func):
    """
    A Higher-order function
    Work as Decorator of any function which like to turn every pandas object into numpy
    
    """

    import pandas as pd
    
    def wrapper_function(*args,**kwargs):

        for key, value in kwargs.items():
            if isinstance(value,pd.DataFrame):
                value_ = value.to_numpy()
                kwargs[key] = value_       #updated from value pandas to numpy
        
        return any_func(*args, **kwargs)

    return wrapper_function