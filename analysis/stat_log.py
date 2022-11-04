'''
This script generates CSV files for performing statistical analysis 
of AirSim Data
'''

import pandas as pd
import glob
import os

# setting the path for joining multiple files
files = os.path.join("log*.csv")

# list of merged files returned
files = glob.glob(files)

print("Resultant CSV after joining all CSV files at a particular location...")

# joining files with concat and read_csv

def flip_without_sampling(csv_path):
    '''
    Performs flipping of control commands without performing 
    undersampling
    '''

    train_df = pd.read_csv(csv_path)
    new_nz_yaw = train_df.copy()
    new_nz_yaw['act_roll'] = train_df['act_roll'].apply(lambda x: x*-1)
    new_nz_yaw['act_yaw'] = train_df['act_yaw'].apply(lambda x: x*-1)
    return new_nz_yaw


def func(csv_path):
    '''
    Aggregate non-zero yaw states and performs undersampling
    of zero-yaw states. No flipping of commands
    '''
    
    train_df = pd.read_csv(csv_path)
    non_zero_yaw = train_df[train_df["act_yaw"] != 0.0]
    zero_yaw = train_df.query("act_yaw == 0.0").sample(frac=0.10)
    final_df = pd.concat([zero_yaw, non_zero_yaw])
    return final_df


def func_flip(csv_path):
    '''
    Performs flipping of control commands with undersampling 
    of zero-yaw states
    '''
    train_df = pd.read_csv(csv_path)
    non_zero_yaw = train_df[train_df["act_yaw"] != 0.0]
    zero_yaw = train_df.query("act_yaw == 0.0").sample(frac=0.10)

    new_nz_yaw = non_zero_yaw.copy()
    new_nz_yaw['act_roll'] = non_zero_yaw['act_roll'].apply(lambda x: x*-1)
    new_nz_yaw['act_yaw'] = non_zero_yaw['act_yaw'].apply(lambda x: x*-1)

    final_df = pd.concat([zero_yaw, new_nz_yaw])
    return final_df

df_orig = pd.concat(map(pd.read_csv, files), ignore_index=True)
df_orig.to_csv("../augmented_log/airsim_original.csv")

df_flip_without_samp = pd.concat(map(flip_without_sampling, files), ignore_index=True)
augmented_df_without_sampling = pd.concat([df_orig, df_flip_without_samp])
augmented_df_without_sampling.to_csv("../augmented_log/airsim_augmented.csv")


df_flip = pd.concat(map(func_flip, files), ignore_index=True)
df     = pd.concat(map(func, files), ignore_index=True)
augmented_df = pd.concat([df_flip, df])
augmented_df.to_csv("../augmented_log/airsim_augmented_undersampled.csv")



# augmented_df.to_csv("../augmented_log/augmented_log.csv", sep='\t') Try this!


