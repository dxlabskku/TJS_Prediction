
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch

def replace_outliers_47sigma_with_nan_median(df):
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
    return df

def fill_missing_values(group):
    # Remove duplicate indices and combine all columns with mean (numeric data only)
    group_numeric = group.select_dtypes(include=[np.number])
    group_non_numeric = group.select_dtypes(exclude=[np.number])

    group_numeric = group_numeric.groupby('new_before_tj_group').mean().reset_index()

    # Reindex to fill values in 5-day intervals
    group_numeric = group_numeric.set_index('new_before_tj_group').reindex(range(0, 1311, 5))
    group_numeric = group_numeric.ffill().bfill().reset_index()

    # Fill missing values using linear interpolation
    group_numeric = group_numeric.interpolate(method='linear')

    # Fill non-numeric data and 'game_date' with the first value in the group
    for col in group_non_numeric.columns:
        if col != 'new_before_tj_group':
            group_numeric[col] = group_non_numeric[col].iloc[0]

    return group_numeric

def drop_columns(df, columns_to_drop):
    df = df.copy()
    df.drop(labels=columns_to_drop, axis=1, inplace=True)
    return df

def data_split(X, y, random_state):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=random_state, stratify=y
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp
    )

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_valid_scaled = scaler.transform(X_valid.reshape(-1, X_valid.shape[-1])).reshape(X_valid.shape)
    X_test_scaled  = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    #data transform to pytorch tensor
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_valid_tensor = torch.tensor(X_valid_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32).unsqueeze(1)  # (samples, 1)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    print(f' train : {X_train_tensor.shape} / valid : {X_valid_tensor.shape} / test : {X_test_tensor.shape}')
    print(f' train : {y_train_tensor.shape} / valid : {y_valid_tensor.shape} / test : {y_test_tensor.shape}')


    return X_train_tensor, X_valid_tensor, X_test_tensor, y_train_tensor, y_valid_tensor, y_test_tensor


# ----------------------------------------------------------------------------

def preprocess():
    # Load the data
    data = pd.read_csv('.../final_df.csv')
    data_removed = replace_outliers_47sigma_with_nan_median(data)
    data_removed.reset_index(drop=True, inplace=True)
    data_removed['new_before_tj_group'] = (data_removed['new_before_tj'] // 5) * 5
    # Group by 5-day intervals, fill missing
    grouped = data_removed.groupby(['player_name','pitcher','target']).apply(fill_missing_values).reset_index(drop=True)

    # Example of filtering to a certain range
    grouped_1290 = grouped[
        (grouped['new_before_tj_group'] < 1220) & (grouped['new_before_tj_group'] > 99)
    ]

    # Example drop columns
    df_1290 = drop_columns(
        grouped_1290,
        ['new_before_tj','player_name','height','weight','bmi']
    )

    # Sort, identify target=1 cases
    df_1290.sort_values(by=['target','pitcher','new_before_tj_group'],ascending=False, inplace=True)
    df_1290.reset_index(drop=True, inplace=True)
    df_1290.loc[df_1290['target'] == 1, 'pitcher'] *= -1

    #Preprocessing(remove unnecessary columns2, we removed the data that are about hitters' record)
    df_1290 = df_1290.drop(df_1290.loc[:, 'diff_estimated_ba_using_speedangle_CH':'diff_launch_speed_SL'].columns, axis=1)

    # Final selection
    X = df_1290.drop(columns=['target'])
    y = df_1290[['pitcher','target']]
    feature_columns = [c for c in X.columns.tolist() if c!='pitcher']

    #check the data
    print(X.pitcher.value_counts())
    # This shows us that each sequence contains 128 datapoints

    # confirming this
    (X.pitcher.value_counts() == 620).sum() == len(y)


    # Convert grouping to arrays
    person_group = X.groupby('pitcher')
    person_group_y = y.groupby('pitcher')
    X_list, y_list = [], []
    for name, group in person_group:
        X_list.append(group[feature_columns].values)
    for name, group in person_group_y:
        y_list.append(group['target'].values[0])

    X_array = np.array(X_list)
    y_array = np.array(y_list)

    X = X_array
    y = y_array

    print(X.shape)
    print(y.shape)
    return X, y
