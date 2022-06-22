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


# %%
def np_to_tfdataset(a, input_map, output_map):
    inputs = {k: a[:,v] for k,v in input_map.items()}
    outputs = {k: a[:,v] for k,v in output_map.items()}
    return tf.data.Dataset.from_tensor_slices((inputs, outputs))
