# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Conda (system)
#     language: python
#     name: system
# ---

# %%
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers


# %%
class DiscretizedMSE(tf.losses.Loss):
    
    def call(self, y_true, y_pred):  # Inputs: (None, 1), (None, num_bins, 2) s.t. last_dim of y_pred is pred, prob_mass
        bin_loss = (tf.expand_dims(tf.cast(y_true, y_pred.dtype), -1)-y_pred[:,:,0])**2  # Calculate the error for each bin
        bin_loss_weighted = bin_loss * y_pred[:,:,1]
        # tf.print("bin_loss", bin_loss[:2])
        return tf.reduce_mean(bin_loss_weighted)
    
class DiscretizedMAE(tf.losses.Loss):
    
    def call(self, y_true, y_pred):  # Inputs: (None, 1), (None, num_bins, 2) s.t. last_dim of y_pred is pred, prob_mass
        bin_loss = tf.math.abs(tf.expand_dims(tf.cast(y_true, y_pred.dtype), -1)-y_pred[:,:,0])  # Calculate the error for each bin
        bin_loss_weighted = bin_loss * y_pred[:,:,1]
        # tf.print("bin_loss", bin_loss[:2])
        return tf.reduce_mean(bin_loss_weighted)
        
class AdaptedCrossentropy(tf.losses.Loss):
    
    def __init__(self, **kwargs):
        super().__init__()
        self.xent_loss = tf.keras.losses.SparseCategoricalCrossentropy(reduction="none", **kwargs)
    
    def call(self, y_true, y_pred):  # Inputs: (None, 1), (None, num_bins, 2) s.t. last_dim of y_pred is pred, prob_mass
        xent = self.xent_loss(tf.cast(y_true - 1, y_pred.dtype), y_pred[:,:,1])  # Reducing y_true by 1 to match sparsexent's assumption of class_id >= 0
        return tf.reduce_mean(xent)
    
class KWNegativeLoglikelihood(tf.losses.Loss):
    
    def __init__(self, **kwargs):
        super().__init__()
    
    def call(self, y_true, y_pred):  # Inputs: (None, 1), (None, 2) s.t. the last dim of the second input are alpha and beta values
        alpha, beta = y_pred[:,0], y_pred[:,1]
        y_true = tf.cast(y_true/5, y_pred.dtype)  # scale y_true between 0 and 1
        lower, upper = y_true - 0.2, y_true
        mask_lower = lower > 0
        mask_upper = upper < 1
        
        term1 = beta*tf.math.log(1 - lower**alpha)
        term2 = tf.math.log(1 - ((1-upper**alpha)/(1-lower**alpha))**beta)
        loglik = tf.reduce_mean(term1 + term2)
        return -loglik
        
        

class SampleMSE(tf.losses.Loss):
    
    def call(self, y_true, y_pred):  # Inputs: (None, 1), (None, num_samples) 
        sample_loss = (tf.cast(y_true, y_pred.dtype)-y_pred)**2 / tf.cast(tf.shape(y_pred)[-1], y_pred.dtype)  # Calculate the error for each sample
        return tf.reduce_mean(sample_loss)