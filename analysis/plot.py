import pandas as pd
import matplotlib.pyplot as plt
import os

# df_final_orig      = pd.read_csv("airsim_original.csv", delim_whitespace=True)


df_final_orig      = pd.read_csv("airsim_original.csv")
# df_final_orig = df_final_orig.iloc[: , -4:-1]
# df_final_orig      = pd.read_csv("log_sample2.csv")


print(df_final_orig.act_yaw.values)
print(len(df_final_orig.act_yaw))

plt.figure()
plt.hist(df_final_orig.act_yaw.values, bins=30)
plt.savefig("airsim_original.png")


df_final_flip      = pd.read_csv("airsim_augmented.csv")

#print(train_df.act_yaw.describe())
print(len(df_final_flip.act_yaw))
plt.figure()
plt.hist(df_final_flip.act_yaw.values, bins=30)
plt.savefig("airsim_augmented.png")
# print(len(final))

df_augmented = pd.read_csv("airsim_augmented_undersampled.csv")
print(len(df_augmented.act_yaw))
plt.figure()
plt.hist(df_augmented.act_yaw.values, bins=30)
plt.savefig("airsim_augmented_undersampled.png")

