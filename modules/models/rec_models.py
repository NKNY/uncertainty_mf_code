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
import os
import sys

from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow import keras
from tensorflow.keras import layers
from tensorboard.plugins import projector

sys.path.append("/Users/nknyazev/Documents/Radboud/Research/Projects/uncertainty_mf/code")

from modules.models.rec_layers import *
from modules.models.rec_losses import *


# %%
def init_compile_mf(params, compilation_params):
    
    regularizer = params["regularizer"]
    initializer = params["initializer"]
    num_users, num_items = params["num_users"], params["num_items"]
    num_hidden = params["num_hidden"]
    
    optimizer = compilation_params["optimizer"]
    loss = compilation_params["loss"]
    metrics = compilation_params["metrics"]
    
    # Get each input to the model
    uid_input = keras.Input(shape=(), name="uid")
    iid_input = keras.Input(shape=(), name="iid")

    # Embed each uid and iid
    uid_features = layers.Embedding(
        num_users, num_hidden, name="uid_features", activity_regularizer=regularizer, embeddings_initializer=initializer
    )(uid_input)
    iid_features = layers.Embedding(
        num_items, num_hidden, name="iid_features", activity_regularizer=regularizer, embeddings_initializer=initializer
    )(iid_input)

    uid_bias = layers.Embedding(num_users, 1, name="uid_bias", activity_regularizer=regularizer)(uid_input)
    iid_bias = layers.Embedding(num_items, 1, name="iid_bias", activity_regularizer=regularizer)(iid_input)

    # Predictions
    pred = layers.Add(name="pred")([layers.Dot(axes=(1,1))([uid_features, iid_features]), uid_bias, iid_bias])

    # Create MF
    mf_model = keras.Model(
        inputs=[uid_input, iid_input],
        outputs=[pred]
    )

    # Compile MF
    mf_model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    
    return mf_model


# %%
def init_compile_ff(params, compilation_params):
    
    # Outputs range from 0 to 4, not 1 to 5!
    
    # Parameters
    regularizer = params["regularizer"]
    initializer = params["initializer"]
    num_users, num_items = params["num_users"], params["num_items"]
    num_hidden = params["num_hidden"]
    
    optimizer = compilation_params["optimizer"]
    loss = compilation_params["loss"]
    metrics = compilation_params["metrics"] 
    
    # Get each input to the model
    uid_input = keras.Input(shape=(), name="uid")
    iid_input = keras.Input(shape=(), name="iid")

    # Embed each uid and iid
    uid_features = layers.Embedding(
        num_users, num_hidden[0], name="uid_features", embeddings_regularizer=regularizer, 
        embeddings_initializer=initializer
    )(uid_input)
    iid_features = layers.Embedding(
        num_items, num_hidden[0], name="iid_features", embeddings_regularizer=regularizer,
        embeddings_initializer=initializer
    )(iid_input)

    features = layers.Concatenate()([uid_features[:,:10], iid_features[:,:10]])
    dot = layers.Multiply()([uid_features[:,10:], iid_features[:,10:]])
    features = layers.Concatenate()([features, dot])
    ff1 = layers.Dense(num_hidden[1], activation="relu", kernel_regularizer=regularizer)(features)
    ff2 = layers.Dense(num_hidden[2], activation="relu", kernel_regularizer=regularizer)(ff1)
    ff3 = layers.Dense(num_hidden[3], activation="relu", kernel_regularizer=regularizer)(ff2)

    # Predictions
    regression = layers.Dense(1, name="regression", kernel_regularizer=regularizer)(ff3)
    softmax_inputs = layers.Dense(5, name="softmax_inputs")(ff3)
    softmax = layers.Softmax(axis=-1, name="softmax")(softmax_inputs)
    softmax_argmax = layers.Lambda(lambda x: tf.argmax(x, axis=-1), name="softmax_argmax")(softmax_inputs)
    softmax_expectation = layers.Dot(axes=(-1,-1), name="softmax_expectation")([tf.expand_dims(tf.range(5.), 0), softmax])

    # Create FF
    ff_model = keras.Model(
        inputs=[uid_input, iid_input],
        outputs = [regression, softmax, softmax_argmax, softmax_expectation]
    )

    # Compile MF
    ff_model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    
    return ff_model


# %%
def init_compile_new(params, compilation_params):
    
    # Parameters
    regularizer = params["regularizer"]
    initializer = params["initializer"]
    num_users, num_items = params["num_users"], params["num_items"]
    num_hidden = params["num_hidden"]
    upsilon_id = params["upsilon_id"]
    alpha_beta_adjustment = layers.Lambda(lambda x: x, name="alpha_beta_identity") if "alpha_beta_adjustment" not in params else params["alpha_beta_adjustment"]
    num_beta_samples = params["num_beta_samples"]
    
    optimizer = compilation_params["optimizer"]
    loss = compilation_params["loss"]
    metrics = compilation_params["metrics"] 
    
    upsilons = {
        1: layers.Lambda(lambda x: layers.Multiply()([tf.norm(x[0], axis=1, keepdims=True), tf.norm(x[1], axis=1, keepdims=True)]), name="upsilon1"),
        2: layers.Lambda(lambda x: tf.norm(tf.tensordot(x[0], x[1]), axis=-1, keepdims=True), name="upsilon2"),
        3: layers.Lambda(lambda x: tf.norm(x[0]+x[1], axis=1, keepdims=True), name="upsilon3")
    }
    
    # Get each input to the model
    uid_input, iid_input = keras.Input(shape=(), name="uid"), keras.Input(shape=(), name="iid")

    # Embed each uid and iid
    uid_features = layers.Embedding(num_users, num_hidden, initializer, activity_regularizer=regularizer, name="uid_features")(uid_input)
    iid_features = layers.Embedding(num_items, num_hidden, initializer, activity_regularizer=regularizer, name="iid_features")(iid_input)

    # Forward steps
    dot = layers.Dot(axes=(1,1), name="dot")([uid_features, iid_features])  # u·i
    norm_layer = layers.Lambda(lambda x: tf.norm(x, axis=1, keepdims=True), name="norm_layer")
    uid_norm, iid_norm = norm_layer(uid_features), norm_layer(iid_features)  # ||u||, ||i||
    len_prod = layers.Multiply(name="len_prod")([uid_norm, iid_norm])  # ||u||·||i||
    
    mu = layers.Lambda(lambda x: 0.5+0.5*x[0]/x[1], name="mu")([dot, len_prod])
    upsilon = upsilons[upsilon_id]([uid_features, iid_features])
    
    alpha = layers.Multiply(name="alpha")([mu, upsilon])
    beta = layers.Subtract(name="beta")([upsilon, alpha])  # Equivalent to (1-mu)*upsilon
    alpha, beta = alpha_beta_adjustment([alpha, beta])  # Postprocess alpha, beta
    
    # Calculate outputs
    
    # Beta
    beta_samples = BetaSampling(num_beta_samples, name="beta_samples")([alpha, beta])  # (None, num_beta_samples)
    beta_mean = BetaPoint(name="beta_mean")([alpha, beta])  # (None,)
    
    # Kumaraswamy
    kw_bins_mass = KumaraswamyDiscretizedOutputs(name="kw_bins_mass")([alpha, beta])  # (None, num_bins, 2)
    kw_discretized_expectation = layers.Lambda(lambda x: tf.reduce_sum(x[:,:,0]*x[:,:,1], axis=-1), name="kw_discretized_expectation")(kw_bins_mass)
    kw_discretized_mode = KumaraswamyDiscretizedMode(name="kw_discretized_mode")(kw_bins_mass)
    kw_mode = KumaraswamyMode(name="kw_mode")([alpha, beta])
    kw_median = KumaraswamyMedian(name="kw_median")([alpha, beta])

    outputs = [beta_samples, beta_mean, kw_bins_mass, kw_discretized_expectation, kw_discretized_mode, kw_mode, kw_median, alpha, beta]
    # outputs = [kw_bins_mass, kw_discretized_expectation]
    new_model = keras.Model(inputs=[uid_input, iid_input], outputs=outputs)
    
    # Compile MF
    new_model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    
    return new_model

# %%
