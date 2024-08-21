from datetime import datetime
import pandas as pd
import util

def main(df):
    config_path = "config.ini"
    config = util.read_config(config_path)

    # Get values from the config file
    drop_cols = config.get("params", "drop_cols").split(', ')
    get_dummy_cols = config.get("params", "get_dummy_cols").split(', ')
    normalized_cols = config.get("params", "normalized_cols").split(', ')
    bin_cols = config.get("params", "bin_cols").split(', ')
    cat_conversion_cols = config.get("params", "cat_conversion_cols").split(', ')
    day_diff_cols = config.get("params", "day_diff_cols").split(', ')

    threshold = config.get("model","threshold")

    print(" df columns = ", df.columns)

    results_df = df[['hotel', 'agent', 'company', 'country', 'market_segment',
                     'distribution_channel', 'customer_type']].copy()
    df2 = util.merge_date_columns(df)
    df2 = df2.drop(columns=drop_cols)
    print("Drop pass")

    df2 = util.reduce_mem_usage(df2)
    print("Reduce Mem_usage pass")

    df2 = util.fill_null_values(df2)
    print("Fill Null_values pass")

    # Get dummy variables for the 'Category' columns
    for col in get_dummy_cols:
        df2 = pd.concat([df2, pd.get_dummies(df2[col], prefix=col + "_cat")], axis=1)
        df2.drop(columns=[col], axis=1, inplace=True)
    print("Get Dummies pass")

    # Normalize values
    df_normalized = util.normalize_data(df2, normalized_cols)
    df3 = df2.copy()
    df3[normalized_cols] = df_normalized[normalized_cols].copy()
    print("Normalization pass")

    # Convert values as bins
    df4 = util.replace_columns_with_bins(df3, bin_cols)
    print("Replace columns pass")


    df5 = df4.copy()

    # Create dictionaries with the values
    meal_list = util.create_dict_with_incremental_integers(df5, 'meal')
    country_list = util.create_dict_with_incremental_integers(df5, 'country')
    market_segment_list = util.create_dict_with_incremental_integers(df5, 'market_segment')
    distribution_channel_list = util.create_dict_with_incremental_integers(df5, 'distribution_channel')
    reserved_room_type_list = util.create_dict_with_incremental_integers(df5, 'reserved_room_type')
    assigned_room_type_list = util.create_dict_with_incremental_integers(df5, 'assigned_room_type')
    print("Create Dict methods pass")


    df5["meal"] = df5["meal"].replace(meal_list.keys(), meal_list.values())
    df5["country"] = df5["country"].replace(country_list.keys(), country_list.values())
    df5["market_segment"] = df5["market_segment"].replace(market_segment_list.keys(), market_segment_list.values())
    df5["distribution_channel"] = df5["distribution_channel"].replace(distribution_channel_list.keys(),
                                                                      distribution_channel_list.values())
    df5["reserved_room_type"] = df5["reserved_room_type"].replace(reserved_room_type_list.keys(),
                                                                  reserved_room_type_list.values())
    df5["assigned_room_type"] = df5["assigned_room_type"].replace(assigned_room_type_list.keys(),
                                                                  assigned_room_type_list.values())

    df6 = df5.copy()

    # Calculate day difference between dates and today
    df6 = util.calculate_day_difference(df6, 'reservation_status_date')
    df6 = util.calculate_day_difference(df6, 'arrival_date')
    print("Calculate Day_diff pass")

    df7 = df6.copy()
    df_date_normalized = util.normalize_data(df6, ['reservation_status_date', 'arrival_date'])
    df7[['reservation_status_date', 'arrival_date']] = df_date_normalized[['reservation_status_date',

                                                                          'arrival_date']].copy()
    # Reduce memory usage
    df7 = util.reduce_mem_usage(df7)

    #Put target data at the end of the dataframe
    df7 = util.move_target_to_end(df7, 'is_canceled')
    print("Move Target end pass")


    X = df7.drop(columns=['is_canceled'])  # features (independent variables)
    y = df7['is_canceled']  # target (dependent variable)
    X = util.drop_highly_correlated_columns(X, threshold)

    X = X.drop(columns=['company', 'agent'])

    X.replace(0, 0.001, inplace=True)

    print("Preprocess completed")

    return X, y, results_df

# Main method
if __name__ == '__main__':
    main()