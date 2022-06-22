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

import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

# %%
sys.path.append("/Users/nknyazev/Documents/Radboud/Research/Projects/uncertainty_mf/code")

# %% [markdown]
# #### Process movie metadata
# Associate each iid with its title and genre(s), represented by binary values in respective columns.
# Fill in titles and genres for iid's not in the data file

# %%
# Process raw data
movies = pd.read_csv("../../data/raw/ml-1m/movies.dat", sep="::", encoding="iso-8859-1", names=["iid", "title", "genre"])

movies_df = []  # Initialise output data as list

# Some item id's are missing (nrow != max_id+1), so have to fill those in in the output df
missing_iids = set(range(movies["iid"].max()))-set(movies["iid"]) - {0}

i=0
j=0
while i < movies["iid"].max():
    if i+1 in missing_iids:
        movies_df.append([i+1, "N/A", "N/A"])  # Fill in missing id titles and genres
    else:
        movies_df.append(movies.loc[j].values)
        j += 1
    i += 1
    
movies_df = pd.DataFrame(movies_df, columns=movies.columns)  # Convert output list to df

# Convert genre strings into binary yes-no vectors (one movie can have multiple genres). 
# "Film noir" category ends up having a "film" and "noir" column but debugging is not worth the effort
tf_vectorizer = CountVectorizer(stop_words=None, token_pattern='(?u)\\b\\w\\w\\w+\\b')
vectorized_data = tf_vectorizer.fit_transform(movies_df.genre)

# Combine and export
movies_df = pd.concat([movies_df.drop("genre", axis=1), pd.DataFrame(vectorized_data.toarray(),columns=tf_vectorizer.get_feature_names())], axis=1)
movies_df.to_csv(os.path.join(log_dir, "iid_metadata.tsv"), sep="\t", index=None)
