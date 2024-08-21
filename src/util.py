import joblib
import pandas as pd
import numpy as np
import configparser

from datetime import datetime, date
from sklearn.preprocessing import MinMaxScaler
from configparser import ConfigParser


def read_config(config_path):
    config = configparser.ConfigParser()
    config.read(config_path)
    return config

# Method for filling the null values
def fill_null_values(df):
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column].fillna('unknown', inplace=True)
        elif df[column].dtype in ['int64', 'float64']:
            df[column].fillna(0, inplace=True)
            df[column][np.isnan(df[column])] = 0
    return df

# Method for reducing the memory usage
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

#Merging date information as a date column
def merge_date_columns(df):

    # Convert 'arrival_date_month' to a numerical representation
    month_mapping = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
                     'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}

    df['arrival_date_month'] = df['arrival_date_month'].map(month_mapping)

    # Create a new 'arrival_date' column by applying a lambda function
    df['arrival_date'] = df.apply(lambda row: pd.to_datetime(f"{row['arrival_date_year']}-{row['arrival_date_month']}-{row['arrival_date_day_of_month']}", errors='coerce'), axis=1)

    return df

# Method for normalizing the values with Min Max Scaler
def normalize_data(dataframe, columns):

    normalized_dataframe = dataframe.copy()

    # Initialize MinMaxScaler
    scaler = MinMaxScaler()

    # Check if the specified columns exist in the DataFrame
    existing_columns = [col for col in columns if col in normalized_dataframe.columns]

    if existing_columns:
        # Fit and transform the specified columns using MinMaxScaler
        normalized_dataframe[existing_columns] = scaler.fit_transform(normalized_dataframe[existing_columns])

    return normalized_dataframe

# Method for bins convertion
def replace_with_bins(dataframe, column_name, bin_ranges):

    df_copy = dataframe.copy()
    df_copy[column_name] = pd.cut(df_copy[column_name], bins=bin_ranges, include_lowest=True).cat.codes
    return df_copy

def replace_columns_with_bins(dataframe, column_list):

    df_copy = dataframe.copy()

    for column_name in column_list:
      #print(column_name)
      if column_name in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy[column_name]):
          bin_ranges = [df_copy[column_name].min(), df_copy[column_name].mean(),
                        df_copy[column_name].mean() + df_copy[column_name].std(),
                        df_copy[column_name].max()]
          df_copy[column_name] = pd.cut(df_copy[column_name],
                                        bins=bin_ranges,
                                        include_lowest=True,
                                        duplicates='drop').cat.codes
    return df_copy

# Method for creating automated dictionary
def create_dict_with_incremental_integers(df, column_name):
    unique_values = df[column_name].unique()
    value_mapping = {value: index + 1 for index, value in enumerate(unique_values)}
    return value_mapping

# Method for daya difference between date and today
def calculate_day_difference(df, column_name):
    print('Day diff fonksiyonun icindeyim')
    today = datetime.today().date()
    #df[column_name] = pd.to_datetime(df[column_name])

    try:
        df[column_name] = pd.to_datetime(df[column_name]).dt.date
        #df[column_name] = pd.to_datetime(df[column_name])
        df[column_name] = (today - df[column_name]).dt.days

    except (TypeError, ValueError):
        print(f"Error: Unable to convert column '{column_name}' to datetime.")

    return df

# Method for ordering the data
def move_target_to_end(df, target_column):
    """
    Move the target column to the end of the DataFrame.

    Parameters:
    - df: pandas DataFrame
    - target_column: str, the name of the target column

    Returns:
    - df: pandas DataFrame, updated DataFrame
    """

    # Ensure the target column is in the DataFrame
    if target_column not in df.columns:
        print(f"Error: Target column '{target_column}' not found in the DataFrame.")
        return df

    # Reorder columns to move the target column to the end
    new_order = [col for col in df.columns if col != target_column] + [target_column]
    df = df[new_order]

    return df

# Columns missing control
def add_missing_cols(list1, list2):
    for element in list2:
        if element not in list1:
            list1.append(element)
    return list1

# Method for correlation control
def drop_highly_correlated_columns(df, threshold):
    # Create correlation matrix
    corr_matrix = df.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find features with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > float(threshold))]

    # Drop features
    df.drop(to_drop, axis=1, inplace=True)
    return df  # Return None if no high correlation is found

# Method for saving the model
def dump_model(model, model_name):
    # Get the current date
    today_date = date.today().strftime("%Y-%m-%d")

    # Define the filename with the current date
    filename = f"models/{model_name}_{today_date}.joblib"

    # Save the model to a file
    joblib.dump(model, filename)
    return today_date

# Method for reading the pre_trained models
def load_joblib_model(model, last_date):
    """
    Load a joblib model from the specified file path.

    Parameters:
    - file_path (str): The path to the joblib file containing the saved model.

    Returns:
    - model: The loaded model.
    """
    # Define the filename with the current date
    filename = f"models/{model}_{last_date}.joblib"

    try:
        # Load the model from the joblib file
        loaded_model = joblib.load(filename)
        return loaded_model
    except Exception as e:
        print(f"Error loading the joblib model: {e}")
        return None
"""
# Method for scoring the predictions
def calc_cancellation_score(final_df, random_forest_accuracy, calibrated_random_forest_accuracy, naive_bias_accuracy, isotonic_calibrated_naive_bias_accuracy, sigmoid_calibrated_naive_bias_accuracy):
  final_df['cancelation_Score'] = final_df['Random_Forest_Probability'] * random_forest_accuracy
  + final_df['Calibrated_Random_Forest_Probability'] * calibrated_random_forest_accuracy
  + final_df['Naive_Bias_Probability'] * naive_bias_accuracy
  + final_df['Isotonic_Calibrated_Naive_Bias_Probability'] * isotonic_calibrated_naive_bias_accuracy
  + final_df['Sigmoid_Calibrated_Naive_Bias_Probability'] * sigmoid_calibrated_naive_bias_accuracy
  def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

  final_df['cancelation_Score'] = NormalizeData(final_df['cancelation_Score'])
  return final_df"""

# Method for saving the model creation date
def write_today_date_to_config(parameter_name, parameter_value):
    """
    Write today's date into the 'config.ini' file.

    Returns:
    - success (bool): True if the operation was successful, False otherwise.
    """
    try:
        # Create a ConfigParser object
        config = ConfigParser()

        # Read the existing 'config.ini' file
        config.read('config.ini')

        # Check if the 'section_name' section exists, if not, create it
        if not config.has_section('date'):
            config.add_section('date')

        # Check if the parameter_name already exists, if not, add it
        if not config.has_option('date', parameter_name):
            config.set('date', parameter_name, str(parameter_value))

            # Write the updated config to 'config.ini'
            with open('config.ini', 'w') as config_file:
                config.write(config_file)

        print(f"Today's date ({parameter_name}) successfully written to 'config.ini'.")
        return True
    except Exception as e:
        print(f"Error writing to 'config.ini': {e}")
        return False

# Method for scoring the predictions
def calc_cancellation_score(final_df, config):

    # Get values from the config file
    random_forest_accuracy = config.get("accuracy", "random_forest_accuracy")
    calibrated_random_forest_accuracy = config.get("accuracy", "calibrated_random_forest_accuracy")
    naive_bias_accuracy = config.get("accuracy", "naive_bias_accuracy")
    isotonic_calibrated_naive_bias_accuracy = config.get("accuracy", "isotonic_calibrated_naive_bias_accuracy")
    sigmoid_calibrated_naive_bias_accuracy = config.get("accuracy", "sigmoid_calibrated_naive_bias_accuracy")

    final_df['cancelation_Score'] = final_df['Random_Forest_Probability'] * float(random_forest_accuracy) + \
                                    final_df['Calibrated_Random_Forest_Probability'] * float(calibrated_random_forest_accuracy) + \
                                    final_df['Naive_Bias_Probability'] * float(naive_bias_accuracy) + \
                                    final_df['Isotonic_Calibrated_Naive_Bias_Probability'] * float(isotonic_calibrated_naive_bias_accuracy) + \
                                    final_df['Sigmoid_Calibrated_Naive_Bias_Probability'] * float(sigmoid_calibrated_naive_bias_accuracy)

    def NormalizeData(data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    final_df['cancelation_Score'] = NormalizeData(final_df['cancelation_Score'])

    return final_df