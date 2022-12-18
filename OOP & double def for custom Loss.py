"""
Object Oriented Programming for creating our custom Loss, Huber besides double def function

"""

import tensorflow as tf
import numpy as np
from tensorflow import keras

# inputs
xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)

# labels
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

## ►►►  y= 2x-1

############################################################
""" double def: outer def and inner def for custom Loss, Huber """

def my_huber_loss_with_threshold(threshold): #Outer def
    def my_huber_loss(y_true, y_pred): #Inner def
        error = y_true - y_pred
        is_small_error = tf.abs(error) <= threshold  #Boolean variable
        small_error_loss = tf.square(error) / 2
        big_error_loss = threshold * (tf.abs(error) - (0.5 * threshold))
        
        return tf.where(is_small_error, small_error_loss, big_error_loss) 

    # return the inner function tuned by the hyperparameter
    return my_huber_loss


model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss=my_huber_loss_with_threshold(threshold=1.2))
model.fit(xs, ys, epochs=500,verbose=0)
print(model.predict([10.0]))




#_______________________________________________________________

""" OOP for custom Loss, Huber (Init and Call) """

from tensorflow.keras.losses import Loss

class MyHuberLoss(Loss):  #Inherit from the aforementioned Keras
    def __init__(self, threshold=1): #setting the default value to 1
        super().__init__()
        self.threshold = threshold  #we can use it further as hyperparameter (default=1)

    # compute loss
    def call(self, y_true, y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) <= self.threshold
        small_error_loss = tf.square(error) / 2
        big_error_loss = self.threshold * (tf.abs(error) - (0.5 * self.threshold))
        return tf.where(is_small_error, small_error_loss, big_error_loss)



#input shape: 1? → 1D matrix of float numbers connected to one neuron
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])]) #it is a rgression → 1 neuron
model.compile(optimizer='sgd', loss=MyHuberLoss(threshold=0.7)) #Stochastic Gradient descent
model.fit(xs, ys, epochs=10000,verbose=0)  #verbose=0 → dont show epoch line
print(model.predict([10.0]))   # based on  y= 2x-1






