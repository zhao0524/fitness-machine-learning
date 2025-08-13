import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sys
sys.path.append('c:/Users/david/OneDrive/Desktop/data-science-template-main')
from src.models.LearningAlgorithms import ClassificationAlgorithms
import seaborn as sns
import itertools
from sklearn.metrics import accuracy_score, confusion_matrix


# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2


# --------------------------------------------------------------
# Create a training and test set
# --------------------------------------------------------------
df = pd.read_pickle("../../data/interim/03_data_features.pkl")


# --------------------------------------------------------------
# Split feature subsets
# --------------------------------------------------------------

df_train = df.drop(columns=["participant", "category", "set"], axis = 1)

X = df_train.drop("label", axis = 1)
Y = df_train["label"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42, stratify=Y)


fig, ax = plt.subplots(figsize=(10, 5))
df_train["label"].value_counts().plot(kind="bar", ax=ax, color = "lightblue", label = "Total")
Y_train.value_counts().plot(kind="bar", ax=ax, color = "dodgerblue", label = "Train")
Y_test.value_counts().plot(kind="bar", ax=ax, color = "royalblue", label = "Test")
plt.legend()
plt.show()

# --------------------------------------------------------------
# Perform forward feature selection using simple decision tree
# --------------------------------------------------------------



# --------------------------------------------------------------
# Grid search for best hyperparameters and model selection
# --------------------------------------------------------------


# --------------------------------------------------------------
# Create a grouped bar plot to compare the results
# --------------------------------------------------------------


# --------------------------------------------------------------
# Select best model and evaluate results
# --------------------------------------------------------------


# --------------------------------------------------------------
# Select train and test data based on participant
# --------------------------------------------------------------


# --------------------------------------------------------------
# Use best model again and evaluate results
# --------------------------------------------------------------


# --------------------------------------------------------------
# Try a simpler model with the selected features
# --------------------------------------------------------------