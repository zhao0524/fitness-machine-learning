import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction


# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df = pd.read_pickle("../../data/interim/02_outliers_removed_chauvenets.pkl")

predictor_columns = list(df.columns[:6])

plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2

# --------------------------------------------------------------
# Dealing with missing values (imputation)
# --------------------------------------------------------------

for col in predictor_columns:
    df[col] = df[col].interpolate()


df.info()
# --------------------------------------------------------------
# Calculating set duration
# --------------------------------------------------------------
# making the graph smooth instead of having sharp edges to see the big movement pattern


df[df["set"] == 25]["gyr_y"].plot()

duration = df[df["set"] == 1].index[-1] - df[df["set"] == 1].index[0]

duration.seconds

for s in df["set"].unique():
   start = df[df["set"] == s].index[0]
   stop = df[df["set"] == s].index[-1]

   duration = stop - start
   df.loc[(df["set"] == s), "duration"] = duration.seconds

duration_df = df.groupby(["category"])["duration"].mean()

duration_df.iloc[0]/5
duration_df.iloc[1]/10


# --------------------------------------------------------------
# Butterworth lowpass filter
# --------------------------------------------------------------

df_lowpass = df.copy()

Lowpass = LowPassFilter()

fs = 1000/200
cutoff = 1.3

df_lowpass = Lowpass.low_pass_filter(df_lowpass, "acc_y", fs, cutoff, order = 5)


subset = df_lowpass[df_lowpass["set"] == 55]
print(subset["label"][0])

fig, ax = plt.subplots(nrows=2, sharex= True, figsize=(20, 10))
ax[0].plot(subset["acc_y"].reset_index(drop=True), label = "raw_data")
ax[1].plot(subset["acc_y_lowpass"].reset_index(drop = True), label = "butterworth filter")
ax[0].legend(loc= "upper center", bbox_to_anchor = (0.5, 1.15), fancybox=True, shadow=True)
ax[1].legend(loc= "upper center", bbox_to_anchor = (0.5, 1.15), fancybox=True, shadow=True)

for col in predictor_columns:
    df_lowpass = Lowpass.low_pass_filter(df_lowpass, col, fs, cutoff, order = 5)
    df_lowpass[col] = df_lowpass[col + "_lowpass"]
    del df_lowpass[col + "_lowpass"]
    
    
# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------

df_pca = df_lowpass.copy()
PCA = PrincipalComponentAnalysis()

# determine the optimal amount of components
pc_values = PCA.determine_pc_explained_variance(df_pca, predictor_columns)

plt.figure(figsize=(10, 10))
plt.plot(range(1, len(predictor_columns) + 1), pc_values)
plt.xlabel("principle component number")
plt.ylabel("explained variance")
plt.show()

# 3 is the optimal number of components 

df_pca = PCA.apply_pca(df_pca, predictor_columns, 3)

# visualize them
subset = df_pca[df_pca["set"] == 25]

subset[["pca_1", "pca_2", "pca_3"]].plot()

# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------
df_squared = df_pca.copy()

acc_r = df_squared["acc_x"]**2 +df_squared["acc_y"]**2 +df_squared["acc_z"]**2
gyr_r = df_squared["gyr_x"]**2 +df_squared["gyr_y"]**2 +df_squared["gyr_z"]**2

df_squared["acc_r"] = np.sqrt(acc_r)
df_squared["gyr_r"] = np.sqrt(gyr_r)

subset = df_squared[df_squared["set"] == 25]

subset[["acc_r", "gyr_r"]].plot(subplots = True)
# comboined the acc gyr data into scalar

df_squared
# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------


# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------


# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------


# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------