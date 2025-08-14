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

print(X_train.columns)

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
pca_features = ["pca_1", "pca_2", "pca_3"]
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
# Grid search for best hyperparameters and model selection
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

ordered_scores = [0.8872802481902792,
 0.9758703895208549,
 0.9972423302309549,
 0.9993105825577387,
 0.9996552912788693,
 0.9996552912788693,
 0.9996552912788693,
 0.9996552912788693,
 0.9996552912788693,
 0.9996552912788693]


plt.figure(figsize=(10, 5))
plt.plot(np.arange(1, max_feature + 1, 1), ordered_scores)
plt.xlabel("Number of Features")
plt.ylabel("Accuracy")
plt.xticks(np.arange(1, max_feature + 1, 1))
plt.grid()
plt.show()

possible_feature_sets = [feature_set_1, feature_set_2, feature_set_3, feature_set_4, select_features]

feature_names = ["Feature Set 1", "Feature Set 2", "Feature Set 3", "Feature Set 4", "Selected Features"]

iterations = 1
score_df = pd.DataFrame()


for i, f in zip(range(len(possible_feature_sets)), feature_names):
    print("Feature set:", i)
    selected_train_X = X_train[possible_feature_sets[i]]
    selected_test_X = X_test[possible_feature_sets[i]]

    # First run non deterministic classifiers to average their score.
    performance_test_nn = 0
    performance_test_rf = 0

    for it in range(0, iterations):
        print("\tTraining neural network,", it)
        (
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = learner.feedforward_neural_network(
            selected_train_X,
            Y_train,
            selected_test_X,
            gridsearch=False,
        )
        performance_test_nn += accuracy_score(Y_test, class_test_y)

        print("\tTraining random forest,", it)
        (
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = learner.random_forest(
            selected_train_X, Y_train, selected_test_X, gridsearch=True
        )
        performance_test_rf += accuracy_score(Y_test, class_test_y)

    performance_test_nn = performance_test_nn / iterations
    performance_test_rf = performance_test_rf / iterations

    # And we run our deterministic classifiers:
    print("\tTraining KNN")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.k_nearest_neighbor(
        selected_train_X, Y_train, selected_test_X, gridsearch=True
    )
    performance_test_knn = accuracy_score(Y_test, class_test_y)

    print("\tTraining decision tree")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.decision_tree(
        selected_train_X, Y_train, selected_test_X, gridsearch=True
    )
    performance_test_dt = accuracy_score(Y_test, class_test_y)

    print("\tTraining naive bayes")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.naive_bayes(selected_train_X, Y_train, selected_test_X)

    performance_test_nb = accuracy_score(Y_test, class_test_y)

    # Save results to dataframe
    models = ["NN", "RF", "KNN", "DT", "NB"]
    new_scores = pd.DataFrame(
        {
            "model": models,
            "feature_set": f,
            "accuracy": [
                performance_test_nn,
                performance_test_rf,
                performance_test_knn,
                performance_test_dt,
                performance_test_nb,
            ],
        }
    )
    score_df = pd.concat([score_df, new_scores])
# --------------------------------------------------------------
# Create a grouped bar plot to compare the results
# --------------------------------------------------------------
score_df.sort_values(by="accuracy", ascending=False)

plt.figure(figsize=(10, 10))
sns.barplot(x="model", y="accuracy", hue="feature_set", data=score_df)
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.ylim(0.7, 1)
plt.legend(loc="lower right")
plt.show()

# --------------------------------------------------------------
# Select best model and evaluate results
# --------------------------------------------------------------

(
class_train_y,
class_test_y,
class_train_prob_y,
class_test_prob_y,
) = learner.random_forest(
X_train[feature_set_4], Y_train, X_test[feature_set_4], gridsearch=True
)

accuracy = accuracy_score(Y_test, class_test_y)

classes = class_test_prob_y.columns
cm = confusion_matrix(Y_test, class_test_y, labels=classes)

# create confusion matrix for cm
plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = cm.max() / 2.0
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(
        j,
        i,
        format(cm[i, j]),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black",
    )
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.grid(False)
plt.show()
# --------------------------------------------------------------
# Select train and test data based on participant
# --------------------------------------------------------------


# --------------------------------------------------------------
# Use best model again and evaluate results
# --------------------------------------------------------------


# --------------------------------------------------------------
# Try a simpler model with the selected features
# --------------------------------------------------------------