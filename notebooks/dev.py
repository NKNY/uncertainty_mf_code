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
# %load_ext autoreload
# %autoreload 2

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
from modules.models.rec_models import *
from modules.utils import *

# %%
data_df = pd.read_csv("../../data/raw/ml-1m/ratings.dat", sep="::", names=["uid", "iid", "rating", "t"])
observed_data = data_df.drop("t", axis=1).values

# %%
num_observed = len(observed_data)
num_users = observed_data[:,0].max() + 1
num_items = observed_data[:,1].max() + 1

train_mask = np.random.rand(num_observed) < 0.9
test_mask = ~train_mask

train_data = observed_data[train_mask]
test_data = observed_data[test_mask]

# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true jp-MarkdownHeadingCollapsed=true tags=[]
# ##### Matrix Factorisation

# %%
params = {
    "num_users": num_users,
    "num_items": num_items,
    "num_hidden": 64,
    "regularizer": tf.keras.regularizers.L2(l2=0.001),
    "initializer": tf.keras.initializers.RandomNormal(stddev=0.1)
}
compilation_params = {
    "optimizer": keras.optimizers.Adam(0.01),
    "loss": [keras.losses.MeanSquaredError()],
    "metrics": [keras.metrics.RootMeanSquaredError(), keras.metrics.MeanSquaredError()]
}
training_params = {
    "batch_size": 2**12,
    "epochs": 2
}

# %%
mf_model = init_compile_mf(params, compilation_params)
# tf.keras.utils.plot_model(mf_model, show_shapes=True)  # Visualise the model

# %% tags=[]
input_map, output_map = {"uid": 0, "iid": 1},  {"pred": 2}
train_dataset = np_to_tfdataset(train_data, input_map, output_map).shuffle(num_observed).batch(training_params["batch_size"])
test_dataset = np_to_tfdataset(test_data, input_map, output_map).batch(training_params["batch_size"])
mf_model.fit(
    train_dataset, validation_data=test_dataset, epochs=training_params["epochs"],
)

# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true jp-MarkdownHeadingCollapsed=true tags=[] jp-MarkdownHeadingCollapsed=true
# ##### FF

# %%
params = {
    "num_users": num_users,
    "num_items": num_items,
    "num_hidden": [70, 50, 50, 50],
    "regularizer": tf.keras.regularizers.L2(l2=0.001),
    "initializer": tf.keras.initializers.RandomNormal(stddev=0.1)
}
compilation_params = {
    "optimizer": keras.optimizers.Adam(0.001),
    # "loss": {"regression": tf.keras.losses.MeanSquaredError()}, "metrics": {"regression": tf.keras.metrics.MeanSquaredError()},
    "loss": {"softmax": tf.losses.SparseCategoricalCrossentropy()}, "metrics": {"softmax_argmax": tf.keras.metrics.RootMeanSquaredError(), "softmax_expectation": tf.keras.metrics.RootMeanSquaredError()}
}
training_params = {
    "batch_size": 2**12,
    "epochs": 2
}

# %%
ff_model = init_compile_ff(params, compilation_params)

# %%
# input_map, output_map = {"uid": 0, "iid": 1},  {"regression": 2}; output_offset = np.array([[0,0,0]])
input_map, output_map = {"uid": 0, "iid": 1},  {"softmax": 2, "softmax_argmax": 2, "softmax_expectation": 2}; output_offset = np.array([[0,0,1]])
train_dataset = np_to_tfdataset(train_data-output_offset, input_map, output_map).shuffle(num_observed).batch(training_params["batch_size"])
test_dataset = np_to_tfdataset(test_data-output_offset, input_map, output_map).batch(training_params["batch_size"])

# %%
ff_model.fit(
    train_dataset, validation_data=test_dataset, epochs=training_params["epochs"],
)

# %% [markdown]
# #### New

# %%
# NEED TO FIGURE OUT THE CORRECT OUTPUT E.G. 0.5 + pred

# %%
params = {
    "num_users": num_users,
    "num_items": num_items,
    "num_hidden": 2**7,
    "regularizer": tf.keras.regularizers.L2(l2=0.000),
    "initializer": tf.keras.initializers.RandomNormal(stddev=0.1),
    "upsilon_id": 1,
    "num_beta_samples": 10,
    "alpha_beta_adjustment": layers.Lambda(lambda x: (1+x[0], 1.+x[1]), name="ab_adj")
}
compilation_params = {
    "optimizer": keras.optimizers.Adam(0.01),
    "loss": {"beta_samples": SampleMSE()}, "metrics": {"beta_mean": keras.metrics.RootMeanSquaredError()},
    # "loss": {"kw_bins_mass": DiscretizedMSE()}, "metrics": {"kw_discretized_expectation": keras.metrics.RootMeanSquaredError(), "kw_discretized_mode": keras.metrics.RootMeanSquaredError(), "kw_mode": keras.metrics.RootMeanSquaredError(), "kw_median": keras.metrics.RootMeanSquaredError()}
    # "loss": {"kw_bins_mass": AdaptedCrossentropy()}, "metrics": {"kw_discretized_expectation": keras.metrics.RootMeanSquaredError(), "kw_discretized_mode": keras.metrics.RootMeanSquaredError(), "kw_mode": keras.metrics.RootMeanSquaredError(), "kw_median": keras.metrics.RootMeanSquaredError()}
}
training_params = {
    "batch_size": 2**10,
    "epochs": 20
}

# %%
new_model = init_compile_new(params, compilation_params)
# tf.keras.utils.plot_model(new_model, show_shapes=True)

# %%
input_map, output_map = {"uid": 0, "iid": 1},  {"beta_samples": 2, "beta_mean": 2}; output_offset = np.array([[0,0,0]])  # Use this for Beta distribution
# input_map, output_map =  {"uid": 0, "iid": 1},  {"kw_bins_mass": 2, "kw_discretized_expectation": 2, "kw_discretized_mode": 2, "kw_mode": 2, "kw_median": 2}; output_offset = np.array([[0,0,0]])  # Use this for KW distribution
train_dataset = np_to_tfdataset(train_data-output_offset, input_map, output_map).shuffle(num_observed).batch(training_params["batch_size"])
test_dataset = np_to_tfdataset(test_data-output_offset, input_map, output_map).batch(training_params["batch_size"])

# %% tags=[] jupyter={"outputs_hidden": true}
new_model.fit(
    train_dataset, validation_data=test_dataset, epochs=training_params["epochs"],
)

# %% jupyter={"outputs_hidden": true} tags=[]
# Overfit on a bit of training data
compilation_params = {
    "optimizer": keras.optimizers.Adam(0.01),
    # "loss": {"beta_samples": SampleMSE()}, "metrics": {"beta_mean": keras.metrics.RootMeanSquaredError()},
    # "loss": {"kw_bins_mass": DiscretizedMSE()}, "metrics": {"kw_discretized_expectation": keras.metrics.RootMeanSquaredError(), "kw_discretized_mode": keras.metrics.RootMeanSquaredError(), "kw_mode": keras.metrics.RootMeanSquaredError(), "kw_median": keras.metrics.RootMeanSquaredError()}
    "loss": {"kw_bins_mass": AdaptedCrossentropy()}, "metrics": {"kw_discretized_expectation": keras.metrics.RootMeanSquaredError(), "kw_discretized_mode": keras.metrics.RootMeanSquaredError(), "kw_mode": keras.metrics.RootMeanSquaredError(), "kw_median": keras.metrics.RootMeanSquaredError()}
}
new_model = init_compile_new(params, compilation_params)

# input_map, output_map = {"uid": 0, "iid": 1},  {"beta_samples": 2, "beta_mean": 2}; output_offset = np.array([[0,0,0]])  # Use this for Beta distribution
input_map, output_map =  {"uid": 0, "iid": 1},  {"kw_bins_mass": 2, "kw_discretized_expectation": 2, "kw_discretized_mode": 2, "kw_mode": 2, "kw_median": 2}; output_offset = np.array([[0,0,0]])  # Use this for KW distribution
train_dataset = np_to_tfdataset(train_data-output_offset, input_map, output_map).shuffle(num_observed).batch(training_params["batch_size"])
test_dataset = np_to_tfdataset(test_data-output_offset, input_map, output_map).batch(training_params["batch_size"])

new_model.fit(
    test_dataset.take(1).repeat(100), validation_data=test_dataset.take(1), epochs=training_params["epochs"],
)

# %%
# Figure out what model learns
# Say there are 10 item classes
num_users = 1000
num_items = 1000
user_class_scale = np.array([0.2, 0.3, 0.4, 0.5, 0.6])-0.1;
item_class_scale = np.array([0.6, 0.5, 0.4, 0.3, 0.2])-0.1;
num_features = len(user_class_scale)
global_var = 1.
user_var = np.random.uniform(0,0.5, num_users)*0
item_var = np.random.uniform(0,0.5, num_items)*0
user_propensity = np.ones(num_users)
item_propensity = np.random.beta(1, 4, num_items)
user_features = np.random.exponential(user_class_scale, size=(num_users, num_features))
item_features = np.random.exponential(item_class_scale, size=(num_items, num_features))
R = np.tanh(user_features.dot(item_features.T))*5 + np.random.randn(num_users, num_items)*np.sqrt(global_var)\
    + np.random.randn(num_users, num_items)*np.sqrt(user_var[:,None]) + np.random.randn(num_users, num_items)*np.sqrt(item_var[None,:])# r= clip(tanh(u*i)*5 + N(0,global_var), 1, 5)
R = np.clip(np.round(R), 1, 5).astype(int)

data = np.concatenate([
    np.repeat(np.arange(1, num_users+1, dtype=int), num_items)[:,None], np.tile(np.arange(1, num_items+1, dtype=int), num_users)[:,None], R.flatten()[:,None]
], axis=-1)  # Combine as uid-iid-rating

user_labels, item_labels = user_features > 0.5, item_features > 0.5  # labels for visualisation

# O = np.random.rand(num_users*num_items) < (data[:,2]/5 - 0.15)  # Temporarily assume positivity bias
# O = np.ones(num_users*num_items, dtype=bool)
O = np.random.rand(num_users*num_items) < user_propensity[:,None].dot(item_propensity[None,:]).flatten()

num_observed = O.sum()
observed_data = data[O]

train_idx = np.random.rand(num_observed) < 0.9
test_idx = ~train_idx
train_data = observed_data[train_idx]
test_data = observed_data[test_idx]

num_users += 1  # Increase by 1 because internally there is also uid/iid=0
num_items += 1

# %%
params = {
    "num_users": num_users,
    "num_items": num_items,
    "num_hidden": 3,#2**1,
    "regularizer": tf.keras.regularizers.L2(l2=0.001),
    "initializer": tf.keras.initializers.RandomNormal(stddev=0.1),
    "upsilon_id": 1,
    "num_beta_samples": 10,
    "alpha_beta_adjustment": layers.Lambda(lambda x: (1+x[0], 1.+x[1]), name="ab_adj")
}
compilation_params = {
    "optimizer": keras.optimizers.Adam(0.01),
    "loss": {"beta_samples": SampleMSE()}, "metrics": {"beta_mean": keras.metrics.RootMeanSquaredError()},
    # "loss": {"kw_bins_mass": DiscretizedMSE()}, "metrics": {"kw_discretized_expectation": keras.metrics.RootMeanSquaredError(), "kw_discretized_mode": keras.metrics.RootMeanSquaredError(), "kw_mode": keras.metrics.RootMeanSquaredError(), "kw_median": keras.metrics.RootMeanSquaredError()}
    # "loss": {"kw_bins_mass": AdaptedCrossentropy()}, "metrics": {"kw_discretized_expectation": keras.metrics.RootMeanSquaredError(), "kw_discretized_mode": keras.metrics.RootMeanSquaredError(), "kw_mode": keras.metrics.RootMeanSquaredError(), "kw_median": keras.metrics.RootMeanSquaredError()}
}
training_params = {
    "batch_size": 2**10,
    "epochs": 20
}

# %%
new_model = init_compile_new(params, compilation_params)
# tf.keras.utils.plot_model(new_model, show_shapes=True)

# %%
input_map, output_map = {"uid": 0, "iid": 1},  {"beta_samples": 2, "beta_mean": 2}; output_offset = np.array([[0,0,0]])  # Use this for Beta distribution
# input_map, output_map =  {"uid": 0, "iid": 1},  {"kw_bins_mass": 2, "kw_discretized_expectation": 2, "kw_discretized_mode": 2, "kw_mode": 2, "kw_median": 2}; output_offset = np.array([[0,0,0]])  # Use this for KW distribution
train_dataset = np_to_tfdataset(train_data-output_offset, input_map, output_map).shuffle(num_observed).batch(training_params["batch_size"])
test_dataset = np_to_tfdataset(test_data-output_offset, input_map, output_map).batch(training_params["batch_size"])

# %% tags=[]
new_model.fit(
    train_dataset, validation_data=test_dataset, epochs=training_params["epochs"],
)

# %%
# Set up a logs directory, so Tensorboard knows where to look for files.
log_dir='/Users/nknyazev/Documents/Radboud/Research/Projects/uncertainty_mf/logs/tensorboard/new/toy_example/embeddings/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Save user and item labels, line-by-line
uid_metadata = np.concatenate([np.arange(1, num_users)[:,None], user_labels, user_var[:,None], user_propensity[:,None]], axis=-1)
iid_metadata = np.concatenate([np.arange(1, num_items)[:,None], item_labels, item_var[:,None], item_propensity[:,None]], axis=-1)
uid_metadata_df = pd.DataFrame(uid_metadata, columns=["uid"]+list(range(num_features))+["var", "prop"])
iid_metadata_df = pd.DataFrame(iid_metadata, columns=["iid"]+list(range(num_features))+["var", "prop"])
uid_metadata_df.to_csv(os.path.join(log_dir, "uid_metadata.tsv"), sep="\t", index=None)
iid_metadata_df.to_csv(os.path.join(log_dir, "iid_metadata.tsv"), sep="\t", index=None)
 
uid_weights = tf.Variable(new_model.layers[2].get_weights()[0][1:])#- tf.reduce_mean(new_model.layers[2].get_weights()[0][1:], axis=0))
# uid_weights = uid_weights - tf.reduce_mean(uid_weights, axis=0)
iid_weights = tf.Variable(new_model.layers[3].get_weights()[0][1:])# - tf.reduce_mean(new_model.layers[3].get_weights()[0][1:], axis=0))
# iid_weights = iid_weights - tf.reduce_mean(iid_weights, axis=0)

checkpoint = tf.train.Checkpoint(uid_embedding=uid_weights, iid_embedding=iid_weights)

checkpoint.save(os.path.join(log_dir, "embedding.ckpt"))

# Set up config
config = projector.ProjectorConfig()
# User data
embedding = config.embeddings.add()
embedding.tensor_name = "uid_embedding/.ATTRIBUTES/VARIABLE_VALUE"
embedding.metadata_path = 'uid_metadata.tsv'
# Item data
embedding = config.embeddings.add()
embedding.tensor_name = "iid_embedding/.ATTRIBUTES/VARIABLE_VALUE"
embedding.metadata_path = 'iid_metadata.tsv'
# Combine
projector.visualize_embeddings(log_dir, config)


# %% [markdown]
# # Old stuff

# %%
class PrintLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        for i in inputs:
            tf.print("max", i.name, tf.reduce_max(i))
        # tf.print("max", tf.reduce_max(inputs))
        return inputs    


# %%
from surprise import SVD
from surprise import Dataset
from surprise import accuracy
from surprise.reader import Reader
from surprise.model_selection import cross_validate, train_test_split

# data = Dataset.load_builtin('ml-1m')
reader = Reader()
data = Dataset.load_from_df(pd.DataFrame(observed_data), reader)

trainset, testset = train_test_split(data, test_size=.1, shuffle=True, random_state=1)

# We'll use the famous SVD algorithm.
algo = SVD(n_factors=100, biased=False, verbose=True)

# Train the algorithm on the trainset, and predict ratings for the testset
algo.fit(trainset)
predictions = algo.test(testset)

# # Then compute RMSE
accuracy.rmse(predictions)

# %%
train_data = np.array(list(trainset.all_ratings()), dtype=int)
test_data = np.array(testset, dtype=int)

# %%
raw2inner_id_items = {k:v for k,v in trainset._raw2inner_id_items.items()}
raw2inner_id_users = {k:v for k,v in trainset._raw2inner_id_users.items()}
t = [[raw2inner_id_users[x[0]], x[1], x[2]] if x[0] in raw2inner_id_users else x for x in test_data]
t = [[x[0], raw2inner_id_items[x[1]], x[2]] if x[1] in raw2inner_id_items else x for x in t]
test_data = np.array(t)

# %%
num_observed = len(observed_data)
num_users = observed_data[:,0].max() + 1
num_items = observed_data[:,1].max() + 1

# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true jp-MarkdownHeadingCollapsed=true tags=[] jp-MarkdownHeadingCollapsed=true
# ##### NeuMF

# %% tags=[]
# Parameters
batch_size = 2**12
epochs = 20
embedding_size = {"mf": 64, "mlp": 64}
num_hidden = {"mlp":[int(x/1) for x in [128, 64, 32]]}
optimizer = keras.optimizers.Adam(0.001)
embeddings_regularizer = tf.keras.regularizers.L2(l2=0.001)
embeddings_initializer = tf.keras.initializers.RandomNormal(stddev=0.01)
validation_split=0.05

# Get each input to the model
uid_input = keras.Input(shape=(), name="uid")
iid_input = keras.Input(shape=(), name="iid")

# Embed each uid and iid
uid_mf_features = layers.Embedding(
    num_users, embedding_size["mf"], name="uid_mf_features", embeddings_regularizer=embeddings_regularizer, 
    embeddings_initializer=embeddings_initializer
)(uid_input)
iid_mf_features = layers.Embedding(
    num_items, embedding_size["mf"], name="iid_mf_features", embeddings_regularizer=embeddings_regularizer,
    embeddings_initializer=embeddings_initializer
)(iid_input)
uid_mlp_features = layers.Embedding(
    num_users, embedding_size["mlp"], name="uid_mlp_features", embeddings_regularizer=embeddings_regularizer, 
    embeddings_initializer=embeddings_initializer
)(uid_input)
iid_mlp_features = layers.Embedding(
    num_items, embedding_size["mlp"], name="iid_mlp_features", embeddings_regularizer=embeddings_regularizer,
    embeddings_initializer=embeddings_initializer
)(iid_input)

# MF
mf_output = layers.Multiply()([uid_mf_features, iid_mf_features])

# MLP
mlp_output = layers.Concatenate()([uid_mlp_features, iid_mlp_features])
for _num_hidden in num_hidden["mlp"]:
    mlp_output = layers.Dense(_num_hidden, activation="relu")(mlp_output)

# Output
suboutputs = layers.Concatenate()([mf_output, mlp_output])
pred = layers.Dense(1, name="pred")(suboutputs)
# sm_inputs = layers.Dense(2, name="sm_inputs")(suboutputs)
# sm = layers.Softmax(axis=-1, name="sm")(sm_inputs)
# pred = layers.Lambda(lambda x: tf.argmax(x, axis=-1)+1, name="pred")(sm_inputs)

# Create FF
neumf_model = keras.Model(
    inputs=[uid_input, iid_input],
    outputs=[pred]
    # outputs = [sm, pred]
)

# Compile MF
neumf_model.compile(
    optimizer=optimizer, 
    loss = [keras.losses.MeanSquaredError()
    ], metrics = [keras.metrics.MeanSquaredError()]
    # loss = [keras.losses.SparseCategoricalCrossentropy(), None],
    # metrics = [None, keras.metrics.MeanSquaredError()]
)

# Visualise the model
tf.keras.utils.plot_model(neumf_model, show_shapes=True)


# %% tags=[]
# Train FF
train_dataset = tf.data.Dataset.from_tensor_slices(({"uid": train_data[:,0], "iid": train_data[:,1]}, {"pred": train_data[:,2], "sm": train_data[:,2]-1}))\
.shuffle(num_observed)\
.batch(batch_size)

neumf_model.fit(
    train_dataset,
    epochs=epochs,
    validation_data = [{"uid": test_data[:,0], "iid": test_data[:,1]}, {"pred": test_data[:,2], "sm": test_data[:,2]-1}],
    validation_batch_size=batch_size,
)

# %%
# Eval MF
np.sqrt(neumf_model.evaluate({"uid": test_data[:,0], "iid": test_data[:,1]}, {"pred": test_data[:,2]}, batch_size=batch_size))


# %%
neumf_outputs = neumf_model.predict({"uid": observed_data[:,0], "iid": observed_data[:,1]}, batch_size=batch_size)

# %% [markdown]
# ### Embedding vis

# %%
# Set up a logs directory, so Tensorboard knows where to look for files.
log_dir='/Users/nknyazev/Documents/Radboud/Research/Projects/uncertainty_mf/logs/tensorboard/new/embeddings/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Save user and item labels, line-by-line
np.savetxt(os.path.join(log_dir, "uid_metadata.tsv"), np.unique(observed_data[:,0]).astype(int), fmt="%d")
# np.savetxt(os.path.join(log_dir, "iid_metadata.tsv"), np.unique(observed_data[:,1]).astype(int), fmt="%d")  # Saving this earlier already
    
uid_weights = tf.Variable(new_model.layers[2].get_weights()[0][1:])#- tf.reduce_mean(new_model.layers[2].get_weights()[0][1:], axis=0))
# uid_weights = uid_weights - tf.reduce_mean(uid_weights, axis=0)
iid_weights = tf.Variable(new_model.layers[3].get_weights()[0][1:])# - tf.reduce_mean(new_model.layers[3].get_weights()[0][1:], axis=0))
# iid_weights = iid_weights - tf.reduce_mean(iid_weights, axis=0)

checkpoint = tf.train.Checkpoint(uid_embedding=uid_weights, iid_embedding=iid_weights)

checkpoint.save(os.path.join(log_dir, "embedding.ckpt"))

# Set up config
config = projector.ProjectorConfig()
# User data
embedding = config.embeddings.add()
embedding.tensor_name = "uid_embedding/.ATTRIBUTES/VARIABLE_VALUE"
embedding.metadata_path = 'uid_metadata.tsv'
# Item data
embedding = config.embeddings.add()
embedding.tensor_name = "iid_embedding/.ATTRIBUTES/VARIABLE_VALUE"
embedding.metadata_path = 'iid_metadata.tsv'
# Combine
projector.visualize_embeddings(log_dir, config)


# %%
