
# Install and import required libraries
!pip install pybaseball

import pandas as pd
import numpy as np
from pybaseball import statcast
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from tqdm import tqdm
import os

# -----------------------------
# 1. Parallel Data Fetch
# -----------------------------
def fetch_data(start, end):
    """
    Fetches Statcast data for the specified date range using pybaseball.statcast.
    Returns a DataFrame or None in case of error.
    """
    try:
        return statcast(start_dt=start, end_dt=end)
    except Exception as e:
        print(f"Error fetching data for {start} to {end}: {e}")
        return None

date_ranges = [
    ("2016-03-01", "2016-11-30"),
    ("2017-03-01", "2017-11-30"),
    ("2018-03-01", "2018-11-30"),
    ("2019-03-01", "2019-11-30"),
    ("2020-03-01", "2020-11-30")
    ("2021-03-01", "2021-11-30"),
    ("2022-03-01", "2022-11-30"),
    ("2023-03-01", "2023-11-30")
]

results = []
# Fetch data in parallel
with ThreadPoolExecutor(max_workers=12) as executor:
    futures = {
        executor.submit(fetch_data, start, end): (start, end)
        for start, end in date_ranges
    }
    for future in as_completed(futures):
        result = future.result()
        if result is not None and not result.empty:
            results.append(result)

# Merge all fetched data
if results:
    data = pd.concat(results, ignore_index=True)
    data['game_date'] = pd.to_datetime(data['game_date'])
    data = data.convert_dtypes()
else:
    print("No data was fetched.")
    data = pd.DataFrame()

# -----------------------------
# 2. Data Filtering
# -----------------------------
# Filter only for regular season games
data_r = data[data['game_type'] == 'R']

# Select relevant columns
select_columns = [
    'player_name', 'game_type', 'home_team', 'pitch_type', 'game_date',
    'pitcher', 'release_speed', 'release_pos_x', 'release_pos_z', 'game_year',
    'pfx_x', 'pfx_z', 'plate_x', 'plate_z', 'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az',
    'effective_speed', 'release_spin_rate', 'release_extension', 'spin_axis'
]
data_selected = data_r[select_columns]

# Convert data to object and replace missing with np.nan for pivot operations
data_selected = data_selected.astype('object').replace({pd.NA: np.nan})

# -----------------------------
# 3. Pivot Table (Averaging Pitch Attributes by Pitch Type)
# -----------------------------
aggregations = {
    'release_speed': 'mean',
    'release_pos_x': 'mean',
    'release_pos_z': 'mean',
    'pfx_x': 'mean',
    'pfx_z': 'mean',
    'plate_x': 'mean',
    'plate_z': 'mean',
    'vx0': 'mean',
    'vy0': 'mean',
    'vz0': 'mean',
    'ax': 'mean',
    'ay': 'mean',
    'az': 'mean',
    'effective_speed': 'mean',
    'release_spin_rate': 'mean',
    'release_extension': 'mean',
    'spin_axis': 'mean'
}

pivot_data = data_selected.pivot_table(
    index=['player_name', 'home_team', 'game_date', 'pitcher', 'game_year'],
    columns=['pitch_type'],
    values=list(aggregations.keys()),
    aggfunc='mean'
)

pivot_data.reset_index(inplace=True)

# Rename pivoted columns (flatten MultiIndex)
pivot_data.columns = [
    '_'.join(col).strip() if col[1] else col[0]
    for col in pivot_data.columns.values
]

# -----------------------------
# 4. Data Quality Check
#    (Remove columns with >90% missing OR only one unique value)
# -----------------------------
final_join_pre = pivot_data.astype('object').replace({pd.NA: np.nan})

col = []
missing = []
level = []
for name in final_join_pre.columns:
    missper = final_join_pre[name].isnull().sum() / final_join_pre.shape[0]
    missing.append(round(missper, 4))
    lel = final_join_pre[name].dropna()
    level.append(len(list(set(lel))))
    col.append(name)

summary = pd.concat([
    pd.DataFrame(col, columns=['name']),
    pd.DataFrame(missing, columns=['Missing Percentage']),
    pd.DataFrame(level, columns=['Level'])
], axis=1)

summary['Missing Percentage'] = pd.to_numeric(summary['Missing Percentage'], errors='coerce')
drop_col = summary['name'][
    (summary['Level'] <= 1) | (summary['Missing Percentage'] >= 0.90)
]
final_join_remove = final_join_pre.drop(columns=drop_col)

# -----------------------------
# 5. Merge with Tommy John Surgery Data
# -----------------------------
tj_list = pd.read_csv('.../list of TJ.csv')
tj_list_join = tj_list[['mlbamid', 'TJ Surgery Date', 'Year of TJ']]
tj_list_join = tj_list_join.sort_values(by=['mlbamid', 'Year of TJ'])
tj_list_join['count'] = tj_list_join.groupby('mlbamid').cumcount() + 1

# Pivot to handle multiple surgeries per pitcher
tj_list_pivot = tj_list_join.pivot_table(
    index='mlbamid',
    columns='count',
    values=['TJ Surgery Date', 'Year of TJ'],
    aggfunc=lambda x: ' '.join(map(str, x))
)
tj_list_pivot.columns = [
    '_'.join([str(i) for i in col]).strip()
    for col in tj_list_pivot.columns.values
]
tj_list_pivot.columns = [col.replace('.0', '') for col in tj_list_pivot.columns]
tj_list_pivot.reset_index(inplace=True)

# Merge final data with pivoted Tommy John data
result = pd.merge(
    final_join_remove,
    tj_list_pivot,
    left_on='pitcher',
    right_on='mlbamid',
    how='left'
).drop(columns='mlbamid')

# Create columns for multiple Tommy John surgery checks
result['TJ Surgery Date_1'] = pd.to_datetime(result['TJ Surgery Date_1'])
result['TJ Surgery Date_2'] = pd.to_datetime(result['TJ Surgery Date_2'])
result['TJ Surgery Date_3'] = pd.to_datetime(result['TJ Surgery Date_3'])
result['TJ Surgery Year_1'] = result['TJ Surgery Date_1'].dt.year
result['TJ Surgery Year_2'] = result['TJ Surgery Date_2'].dt.year
result['TJ Surgery Year_3'] = result['TJ Surgery Date_3'].dt.year
result['season_before_tj_1'] = result['TJ Surgery Year_1'] - result['game_year']
result['season_before_tj_2'] = result['TJ Surgery Year_2'] - result['game_year']
result['season_before_tj_3'] = result['TJ Surgery Year_3'] - result['game_year']

# Calculate date differences
result['TJ Surgery Date_1'] = pd.to_datetime(result['TJ Surgery Date_1'], errors='coerce')
result['TJ Surgery Date_2'] = pd.to_datetime(result['TJ Surgery Date_2'], errors='coerce')
result['TJ Surgery Date_3'] = pd.to_datetime(result['TJ Surgery Date_3'], errors='coerce')
result['game_date'] = pd.to_datetime(result['game_date'], errors='coerce')
result['diff_date_1'] = (result['TJ Surgery Date_1'] - result['game_date']).dt.days
result['diff_date_2'] = (result['TJ Surgery Date_2'] - result['game_date']).dt.days
result['diff_date_3'] = (result['TJ Surgery Date_3'] - result['game_date']).dt.days

def get_closest_surgery_date(row):
    """
    Returns the closest valid (diff > 0) TJ surgery date among
    up to three possible surgeries.
    """
    diffs_dates = [
        (row['diff_date_1'], row['TJ Surgery Date_1']),
        (row['diff_date_2'], row['TJ Surgery Date_2']),
        (row['diff_date_3'], row['TJ Surgery Date_3'])
    ]
    valid_dates = [date for diff, date in diffs_dates if diff > 0]
    if valid_dates:
        return min(valid_dates)
    return pd.NaT

result['TJ Surgery Date'] = result.apply(get_closest_surgery_date, axis=1)
result['TJ Surgery Year'] = result['TJ Surgery Date'].dt.year

# Combine 'season_before_tj' columns to find the minimum valid non-negative
def min_non_negative(row):
    values = [
        row['season_before_tj_1'],
        row['season_before_tj_2'],
        row['season_before_tj_3']
    ]
    valid_values = [v for v in values if v >= 0]
    return min(valid_values) if valid_values else pd.NA

result['season_before_tj'] = result.apply(min_non_negative, axis=1)

# -----------------------------
# 6. Condition-Based Filtering
#    (Identify pitchers who meet certain Tommy John surgery conditions)
# -----------------------------
df = result.copy()

def check_conditions(group):
    """
    1) condition_1: The pitcher has a TJ surgery season (season_before_tj == 0)
       and at least 2 unique 'season_before_tj' records > 0 before that season.
    2) condition_2: No TJ surgery season in the data, but at least 3 unique
       'season_before_tj' > 0.
    Exclude if min_season_before_tj >= 2 for the group.
    """
    group['condition_1'] = False
    group['condition_2'] = False
    group['exclude'] = False

    min_season_before_tj = group['season_before_tj'].min()
    if min_season_before_tj >= 2:
        group['exclude'] = True
        return group

    tj_surgery_season = group[group['season_before_tj'] == 0]
    has_tj_surgery_season = not tj_surgery_season.empty
    pre_surgery_seasons_count = len(set(group[group['season_before_tj'] > 0]['season_before_tj']))

    if has_tj_surgery_season and pre_surgery_seasons_count >= 2:
        group['condition_1'] = True
    if not has_tj_surgery_season and pre_surgery_seasons_count >= 3:
        group['condition_2'] = True
    return group

df = df.groupby(['pitcher', 'TJ Surgery Year'], dropna=False).apply(check_conditions)
df['condition'] = df['condition_1'] | df['condition_2']

# Filter out excluded pitchers
filtered_result = df[(df['condition'] | pd.isna(df['TJ Surgery Date'])) & ~df['exclude']]

# Remove data beyond 'season_before_tj' >= 4
filtered_result = filtered_result[
    (filtered_result['season_before_tj'] < 4) | (pd.isna(filtered_result['season_before_tj']))
].reset_index(drop=True)

# -----------------------------
# 7. Handedness Adjustment
#    (Flip certain columns if pitcher is LHP)
# -----------------------------
pitcher_hand = pd.read_csv('.../pitcher_hand.csv')
result_hand = pd.merge(filtered_result, pitcher_hand, left_on='pitcher', right_on='player_id', how='left').drop(columns=['player_id'])

def adjust_for_hand(row):
    """
    Flip the sign of certain pitch-related columns for LHP
    and adjust spin axis for left-handers.
    """
    for col in row.index:
        if 'ax_' in col or 'pfx_x_' in col or 'release_pos_x_' in col or 'vx0_' in col:
            if row['hand'] == 'lhp':
                row[col] = -row[col]
        if 'spin_axis_' in col:
            if row['hand'] == 'lhp':
                # 360 - spin_axis for left-handers
                row[col] = 360 - row[col] if pd.notnull(row[col]) else np.nan
    return row

filtered_result_hand = result_hand.apply(adjust_for_hand, axis=1)

# -----------------------------
# 8. Time-to-Surgery Features
# -----------------------------
filtered_result_hand['game_date'] = pd.to_datetime(filtered_result_hand['game_date'])
filtered_result_hand['TJ Surgery Date'] = pd.to_datetime(filtered_result_hand['TJ Surgery Date'], errors='coerce')

# Calculate days before TJ, mark out-of-range with 9999
filtered_result_hand['days_before_tj'] = (
    filtered_result_hand['TJ Surgery Date'] - filtered_result_hand['game_date']
).dt.days
filtered_result_hand['days_before_tj'].fillna(9999, inplace=True)
filtered_result_hand['days_before_tj'] = np.where(
    (filtered_result_hand['days_before_tj'] < 0) | (filtered_result_hand['days_before_tj'] > 1290),
    9999,
    filtered_result_hand['days_before_tj']
)

# Minimum days_before_tj per pitcher
filtered_result_hand['min_days_before_tj'] = (
    filtered_result_hand.groupby('pitcher')['days_before_tj'].transform('min')
)

def adjust_days(df):
    """
    Create 'adjust_days_before_tj' based on the difference
    from each pitcher's minimum non-9999 days_before_tj.
    """
    df['adjust_days_before_tj'] = np.where(
        (df['days_before_tj'] == 9999),
        9999,
        df['days_before_tj'] - df['min_days_before_tj']
    )
    return df

final_result = adjust_days(filtered_result_hand)

# Remove one specific player
final_result = final_result[final_result['player_name'] != 'Axford, John']
final_result.sort_values(by=['player_name','game_date'], inplace=True)

# -----------------------------
# 9. Fill Missing Pitch-Type Columns via Interpolation
# -----------------------------
df = final_result.copy()

columns_to_calculate = [
    'ax_CH', 'ax_CU', 'ax_FC', 'ax_FF','ax_SI', 'ax_SL',
    'ay_CH', 'ay_CU', 'ay_FC', 'ay_FF','ay_SI', 'ay_SL',
    'az_CH', 'az_CU', 'az_FC', 'az_FF', 'az_SI', 'az_SL',
    'effective_speed_CH', 'effective_speed_CU', 'effective_speed_FC', 'effective_speed_FF',
    'effective_speed_SI', 'effective_speed_SL',
    'pfx_x_CH', 'pfx_x_CU', 'pfx_x_FC', 'pfx_x_FF', 'pfx_x_SI', 'pfx_x_SL',
    'pfx_z_CH', 'pfx_z_CU', 'pfx_z_FC', 'pfx_z_FF', 'pfx_z_SI', 'pfx_z_SL',
    'plate_x_CH', 'plate_x_CU', 'plate_x_FC', 'plate_x_FF', 'plate_x_SI', 'plate_x_SL',
    'plate_z_CH', 'plate_z_CU', 'plate_z_FC', 'plate_z_FF','plate_z_SI', 'plate_z_SL',
    'release_extension_CH', 'release_extension_CU', 'release_extension_FC', 'release_extension_FF',
    'release_extension_SI', 'release_extension_SL',
    'release_pos_x_CH', 'release_pos_x_CU', 'release_pos_x_FC', 'release_pos_x_FF',
    'release_pos_x_SI', 'release_pos_x_SL',
    'release_pos_z_CH', 'release_pos_z_CU', 'release_pos_z_FC', 'release_pos_z_FF',
    'release_pos_z_SI', 'release_pos_z_SL',
    'release_speed_CH', 'release_speed_CU', 'release_speed_FC', 'release_speed_FF',
    'release_speed_SI', 'release_speed_SL',
    'release_spin_rate_CH', 'release_spin_rate_CU', 'release_spin_rate_FC',
    'release_spin_rate_FF', 'release_spin_rate_SI', 'release_spin_rate_SL',
    'spin_axis_CH', 'spin_axis_CU', 'spin_axis_FC', 'spin_axis_FF','spin_axis_SI', 'spin_axis_SL',
    'vx0_CH', 'vx0_CU', 'vx0_FC', 'vx0_FF', 'vx0_SI', 'vx0_SL',
    'vy0_CH', 'vy0_CU', 'vy0_FC', 'vy0_FF', 'vy0_SI', 'vy0_SL',
    'vz0_CH', 'vz0_CU', 'vz0_FC', 'vz0_FF', 'vz0_SI', 'vz0_SL'
]

# Interpolate or fill with global mean if all values are NaN for that pitcher
total_tasks = len(df['pitcher'].unique()) * len(columns_to_calculate)
progress_bar = tqdm(total=total_tasks, desc="Processing", unit="task")

for pitcher in df['pitcher'].unique():
    pitcher_data = df[df['pitcher'] == pitcher]
    for column in columns_to_calculate:
        if pitcher_data[column].isnull().all():
            # If the entire column for this pitcher is null, fill with overall mean
            overall_mean = df[column].mean()
            df.loc[df['pitcher'] == pitcher, column] = overall_mean
        else:
            # Otherwise, linear interpolation for missing values
            df.loc[df['pitcher'] == pitcher, column] = pitcher_data[column].interpolate(
                method='linear',
                limit_direction='both'
            )
        progress_bar.update(1)

progress_bar.close()

# -----------------------------
# 10. Combine TJ and non-TJ Groups
# -----------------------------
# This step locates pitchers who have no TJ but have multiple consecutive seasons.
# Then merges them with the injury (TJ) group.

# Filter pitchers with no TJ in the data
no_tj_pitchers = df[df['TJ Surgery Year'].isnull()]

def has_consecutive_seasons(pitcher_data):
    """
    Checks if a pitcher has at least 3 consecutive years (out of 4 total).
    """
    years = sorted(pitcher_data['game_year'].unique())
    for i in range(len(years) - 2):
        if years[i+1] == years[i]+1 and years[i+2] == years[i]+2:
            return True
    return False

eligible_pitchers = no_tj_pitchers.groupby('pitcher').filter(has_consecutive_seasons)

# Collect at least 4 consecutive seasons
def get_consecutive_seasons(data):
    years = sorted(data['game_year'].unique())
    consecutive_years = []
    for i in range(len(years) - 3):
        if (years[i+1] == years[i]+1 and
            years[i+2] == years[i]+2 and
            years[i+3] == years[i]+3):
            consecutive_years = [
                years[i], years[i+1], years[i+2], years[i+3]
            ]
            break
    return data[data['game_year'].isin(consecutive_years)]

# Subset only those pitchers with 4 consecutive seasons
consecutive_seasons_data = eligible_pitchers.groupby('pitcher').apply(
    lambda x: get_consecutive_seasons(x)
).reset_index(drop=True)

# Data for TJ group (days_before_tj < 1500)
injury_pitcher_data = df[df['adjust_days_before_tj'] < 1500]
injury_pitcher_data['new_before_tj'] = injury_pitcher_data['adjust_days_before_tj']

# Combine
merge_df = pd.concat([injury_pitcher_data, consecutive_seasons_data])

# Remove certain players/pitchers
merge_df = merge_df[~merge_df['player_name'].isin(['Rogers, Tyler', 'Hudson, Dakota', 'Clase, Emmanuel'])]
merge_df = merge_df[~merge_df['pitcher'].isin([458584])]

# -----------------------------
# 11. Drop Unused Columns
# -----------------------------
columns_to_drop = [
    'last_game_date', 'TJ Surgery Date_1', 'TJ Surgery Date_2', 'TJ Surgery Date_3',
    'Year of TJ_1', 'Year of TJ_2', 'Year of TJ_3',
    'TJ Surgery Year_1', 'TJ Surgery Year_2', 'TJ Surgery Year_3',
    'season_before_tj_1', 'season_before_tj_2', 'season_before_tj_3',
    'diff_date_1', 'diff_date_2', 'diff_date_3',
    'exclude', 'condition',
    # [FIX] Columns below may not exist if pitch_count pivot was never performed:
    # 'count_CH','count_CU','count_FC','count_FF','count_SI','count_SL',
    'season_before_tj','min_days_before_tj','adjust_days_before_tj',
    'hand','days_before_tj','home_team','age','game_date'
]
# [FIX] If any of these columns are not present, this drop may raise an error.
# Make sure these columns actually exist before dropping.
for col_to_drop in columns_to_drop:
    if col_to_drop in merge_df.columns:
        merge_df.drop(columns=col_to_drop, inplace=True)

# -----------------------------
# 12. Set Target (TJ vs. Non-TJ)
# -----------------------------
merge_df['target'] = np.where(merge_df['TJ Surgery Year'].notnull(), 1, 0)

# -----------------------------
# 13. Finalize Data for Classification
# -----------------------------
# Sort for consistent ordering
merge_df.sort_values(
    by=['player_name','target','new_before_tj'],
    inplace=True
)

# We create a new DataFrame 'final_df' with final columns
cal_diff_df = merge_df.copy()

# 13.1 (Optional) You can reorder columns if needed
# Just ensure 'new_before_tj', 'height', 'weight', 'bmi', and 'target' remain.

# 13.2 Compute differences from overall mean for pitch metrics
# If you want the difference columns, define them here:

columns_for_diff = [
    'ax_CH', 'ax_CU', 'ax_FC', 'ax_FF','ax_SI', 'ax_SL',
    'ay_CH', 'ay_CU', 'ay_FC', 'ay_FF','ay_SI', 'ay_SL',
    'az_CH', 'az_CU', 'az_FC', 'az_FF', 'az_SI', 'az_SL',
    'effective_speed_CH', 'effective_speed_CU', 'effective_speed_FC',
    'effective_speed_FF','effective_speed_SI', 'effective_speed_SL',
    'pfx_x_CH', 'pfx_x_CU', 'pfx_x_FC', 'pfx_x_FF', 'pfx_x_SI', 'pfx_x_SL',
    'pfx_z_CH', 'pfx_z_CU', 'pfx_z_FC', 'pfx_z_FF', 'pfx_z_SI', 'pfx_z_SL',
    'plate_x_CH', 'plate_x_CU', 'plate_x_FC', 'plate_x_FF', 'plate_x_SI', 'plate_x_SL',
    'plate_z_CH', 'plate_z_CU', 'plate_z_FC', 'plate_z_FF','plate_z_SI', 'plate_z_SL',
    'release_extension_CH', 'release_extension_CU', 'release_extension_FC',
    'release_extension_FF','release_extension_SI', 'release_extension_SL',
    'release_pos_x_CH', 'release_pos_x_CU', 'release_pos_x_FC', 'release_pos_x_FF',
    'release_pos_x_SI', 'release_pos_x_SL',
    'release_pos_z_CH', 'release_pos_z_CU', 'release_pos_z_FC', 'release_pos_z_FF',
    'release_pos_z_SI', 'release_pos_z_SL',
    'release_speed_CH', 'release_speed_CU', 'release_speed_FC', 'release_speed_FF',
    'release_speed_SI', 'release_speed_SL',
    'release_spin_rate_CH', 'release_spin_rate_CU', 'release_spin_rate_FC',
    'release_spin_rate_FF', 'release_spin_rate_SI', 'release_spin_rate_SL',
    'spin_axis_CH', 'spin_axis_CU', 'spin_axis_FC', 'spin_axis_FF','spin_axis_SI', 'spin_axis_SL',
    'vx0_CH', 'vx0_CU', 'vx0_FC', 'vx0_FF', 'vx0_SI', 'vx0_SL',
    'vy0_CH', 'vy0_CU', 'vy0_FC', 'vy0_FF', 'vy0_SI', 'vy0_SL',
    'vz0_CH', 'vz0_CU', 'vz0_FC', 'vz0_FF', 'vz0_SI', 'vz0_SL'
]

# Group by pitcher and target to get means
mean_df = cal_diff_df.groupby(['pitcher', 'target'])[columns_for_diff].mean().reset_index()
for col in columns_for_diff:
    mean_df.rename(columns={col: f'mean_{col}'}, inplace=True)

cal_diff_df = pd.merge(cal_diff_df, mean_df, on=['pitcher', 'target'], how='left')

# Compute the difference from the group means
for col in columns_for_diff:
    cal_diff_df[f'diff_{col}'] = cal_diff_df[col] - cal_diff_df[f'mean_{col}']

# Remove a couple of specific players
cal_diff_df = cal_diff_df[~cal_diff_df['player_name'].isin(['Sobotka, Chad','Williams, Trevor'])]

# Construct the final set of columns for classification
final_cols = (
    ['player_name', 'pitcher', 'new_before_tj', 'target'] +
    [col for col in cal_diff_df.columns if col.startswith('diff_')]
)

final_df = cal_diff_df[final_cols].sort_values(by=['player_name','new_before_tj'])

# -----------------------------
# 14. Save Final Preprocessed Data
# -----------------------------
final_df.to_csv(
    '.../final_df.csv',
    index=False
)
