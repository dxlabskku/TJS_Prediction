import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from imblearn.over_sampling import ADASYN
from tqdm import tqdm
from sklearn.datasets import make_classification
from multiprocessing import Pool, cpu_count
import seaborn as sns
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim.lr_scheduler import StepLR
import copy
from collections import Counter

def get_feature_columns(df, exclude_list):
    """
    Return a list of feature columns from df excluding any columns in 'exclude_list' 
    or whose name contains a substring in 'exclude_list'.
    """
    return [
        col for col in df.columns 
        if col not in ['aaa']  # 'aaa' here is an example of a column name to exclude
        and not any(exclude in col for exclude in exclude_list)
    ]

def replace_outliers_47sigma_with_nan_median(df):
    """
    Replace values outside ±4.7 standard deviations from the median with NaN for columns
    after 'diff_ax_CU'. Then drop rows that have NaN.
    """
    start_index = df.columns.get_loc('diff_ax_CU')
    for col in df.columns[start_index:]:
        median = df[col].median()
        std_dev = df[col].std()
        a = 4.7
        df[col] = np.where(
            (df[col] < median - a * std_dev) | (df[col] > median + a * std_dev),
            np.nan,
            df[col]
        )
    # Drop rows containing NaN
    df_cleaned = df.dropna()
    return df_cleaned

def drop_columns(df, columns_to_drop):
    """
    Drop the specified columns from a DataFrame.
    """
    df = df.copy()
    df.drop(labels=columns_to_drop, axis=1, inplace=True)
    return df

def data_split(X, y, random):
    """
    Split the data (X, y) into train/valid/test sets in a 60:20:20 ratio,
    then apply MinMax scaling and reshape for a single-channel CNN input.

    Args:
        X (pd.DataFrame or np.ndarray): Feature matrix.
        y (pd.Series or np.ndarray): Targets.
        random (int): Random seed for reproducibility.

    Returns:
        tuple: (X_train, X_valid, X_test, y_train, y_valid, y_test)
    """
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=random)
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=random)

    # Keep unscaled versions if needed
    X_train_pre = X_train
    X_test_pre  = X_test

    # MinMax scaling
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    X_test_scaled  = scaler.transform(X_test)

    # Reshape for CNN (batch_size, 1, num_features)
    X_train = X_train_scaled[:, np.newaxis, :]
    X_valid = X_valid_scaled[:, np.newaxis, :]
    X_test  = X_test_scaled[:,  np.newaxis, :]

    print(f'X_train shape: {X_train.shape} / X_test shape: {X_test.shape}')
    return X_train, X_valid, X_test, y_train, y_valid, y_test

def preprocess():

    # 1) Load CSV
    data = pd.read_csv('.../final_df.csv')

    # 2) Filter by 'target' == 1
    data = data[data['target'] == 1]

    # 3) Setup exclude list (if needed)
    exclude_list = []

    # 4) Select columns not in exclude list
    data = data[get_feature_columns(data, exclude_list)]

    # 5) Filter new_before_tj < 550
    data = data[data['new_before_tj'] < 550]

    # 6) Drop columns from 'diff_estimated_ba_using_speedangle_CH' to 'diff_launch_speed_SL'
    data = data.drop(data.loc[:, 'diff_estimated_ba_using_speedangle_CH':'diff_launch_speed_SL'].columns, axis=1)

    # 7) Remove outliers with custom 4.7-sigma function
    data_removed = replace_outliers_47sigma_with_nan_median(data)
    data_removed.reset_index(drop=True, inplace=True)

    # 8) Filter again for target == 1 only
    data = data_removed[data_removed['target'] == 1]
    data.drop(labels=['target'], axis=1, inplace=True)

    # 10) Drop some columns
    data = drop_columns(data, ['player_name', 'pitcher', 'height', 'weight', 'bmi', 'player_name'])

    # 11) Prepare X, y
    x = data.drop(labels=['new_before_tj'], axis=1)

    def group_new_before_tj(value):
        """
        Group new_before_tj into bins. 
        Example: 
          < 220 => group in multiples of 5,
          < 550 => group in multiples of 10,
          else => group in multiples of 15.
        """
        if value < 220:
            return (value // 5) * 5
        elif value < 550:
            return (value // 10) * 10
        else:
            return (value // 15) * 15

    y = data['new_before_tj'].apply(group_new_before_tj)

    # Reset indices
    X = x.reset_index(drop=True)
    y = y.reset_index(drop=True)

    return X, y
