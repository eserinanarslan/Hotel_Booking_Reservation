import pandas as pd
import util

#Split data into two group: train and test
def split_data(df):

    train = df.sample(frac=0.8, random_state=42)
    predict = df.drop(train.index)

    print("Train shape : ", train.shape)
    print("Test shape : ", predict.shape)

    train.to_csv(train_out_path, index=False)
    predict.to_csv(predict_out_path, index=False)


if __name__ == '__main__':
    # Read configuration from the config file
    config_path = "config.ini"
    config = util.read_config(config_path)

    # Get values from the config file
    data_path = config.get("Settings", "data_path")
    train_out_path = config.get("Settings", "train_out_path")
    predict_out_path = config.get("Settings", "predict_out_path")

    df = pd.read_csv(data_path)
    split_data(df)
