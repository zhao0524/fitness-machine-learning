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

basic_features = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
square_features = ["acc_r", "gyr_r"]
pca_features = ["pca1", "pca2", "pca3"]
time_features = [f for f in df_train.columns if "_temp_" in f]
freq_features = [f for f in df_train.columns if ("_freq" in f )or ("_pse" in f)]
cluster_feature = ["cluster"]

print("basic_features:", len(basic_features))
print("square_features:", len(square_features))
print("pca_features:", len(pca_features))
print("time_features:", len(time_features))
print("freq_features:", len(freq_features))
print("cluster_feature:", len(cluster_feature))

feature_set_1 = list(set(basic_features))
feature_set_2 = list(set(basic_features + square_features + pca_features))
feature_set_3 = list(set(feature_set_2 + time_features))
feature_set_4 = list(set(feature_set_3 + freq_features + cluster_feature))

# --------------------------------------------------------------
# Create a grouped bar plot to compare the results
# --------------------------------------------------------------

learner = ClassificationAlgorithms()

max_feature = 10
select_features, ordered_features, ordered_scores = learner.forward_selection(
    max_feature, X_train, Y_train
    )

select_features = [
    'pca_1',
    'duration',
    'gyr_r_freq_0.0_Hz_ws_14',
    'pca_3',
    'gyr_z_freq_0.0_Hz_ws_14',
    'acc_y_freq_weighted',
    'gyr_y_freq_weighted',
    'gyr_x_freq_weighted',
    'gyr_z_temp_mean_ws_5',
    'acc_x_freq_0.714_Hz_ws_14'
]

plt.figure(figsize=(10, 5))
plt.plot(np.arange(1, max_feature + 1, 1), ordered_scores)
plt.xlabel("Number of Features")
plt.ylabel("Accuracy")
plt.xticks(np.arange(1, max_feature + 1, 1))
plt.grid()
plt.show()
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