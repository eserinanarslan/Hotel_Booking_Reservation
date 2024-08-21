# Press the green button in the gutter to run the script.
import numpy as np
import pandas as pd
import sqlite3 as sql

from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, classification_report, roc_curve, roc_auc_score

import util, preprocess, train_model

def random_forest_classifier(x_test, y_test, config):

    # Get values from the config file
    last_date = config.get("date", "today_date")
    model = "random_forest_classifier"
    CV_clf_rf = util.load_joblib_model(model, last_date)
    y_test_predict_random = CV_clf_rf.predict_proba(x_test)[:, 1]
    yhat_random = CV_clf_rf.predict(x_test)
    fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_test_predict_random, n_bins=10)

    # Evaluate the accuracy
    accuracy = accuracy_score(y_test, yhat_random)
    print(f"Accuracy: {accuracy}")

    print(classification_report(y_test, yhat_random))

    return y_test_predict_random


def calibrated_classifier(x_test, y_test, config):

    # Get values from the config file
    last_date = config.get("date", "today_date")
    model = "sigmoid_random_forest_classifier"
    clf_sigmoid = util.load_joblib_model(model, last_date)

    y_test_predict_random_calibrated = clf_sigmoid.predict_proba(x_test)[:, 1]
    yhat_calibrated_random = clf_sigmoid.predict(x_test)
    fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_test_predict_random_calibrated, n_bins=10)

    print(classification_report(y_test, yhat_calibrated_random))

    return y_test_predict_random_calibrated

def gaussian_naive_bias(x_test, y_test, config):

    # Get values from the config file
    last_date = config.get("date", "today_date")
    nb_model = "gaussian_naive_bias_classifier"
    clf_nb = util.load_joblib_model(nb_model, last_date)

    y_test_predict_nb = clf_nb.predict_proba(x_test)[:, 1]
    yhat_nb = clf_nb.predict(x_test)
    fraction_of_positives_nb, mean_predicted_value_nb = calibration_curve(y_test, y_test_predict_nb, n_bins=10)

    print(classification_report(y_test, yhat_nb))

    # Sigmoid Calibrated
    sigmoid_model = "sigmoid_calibrated_naive_bias_classifier"
    clf_sigmoid_nb = util.load_joblib_model(sigmoid_model, last_date)
    y_test_predict_nb_calib = clf_sigmoid_nb.predict_proba(x_test)[:, 1]

    yhat_calibrated_nb = clf_sigmoid_nb.predict(x_test)

    fraction_of_positives_nb_calib, mean_predicted_value_nb_calib = calibration_curve(y_test, y_test_predict_nb_calib,
                                                                                      n_bins=10)

    print(classification_report(y_test, yhat_calibrated_nb))

    # Isotonic Calibrated
    isotonic_model = "isotonic_calibrated_naive_bias_classifier"
    clf_sigmoid_nb_calib_sig = util.load_joblib_model(isotonic_model, last_date)

    y_test_predict_nb_calib_platt = clf_sigmoid_nb_calib_sig.predict_proba(x_test)[:, 1]
    yhat_calibrated_platt = clf_sigmoid_nb_calib_sig.predict(x_test)

    fraction_of_positives_nb_calib_platt, mean_predicted_value_nb_calib_platt = calibration_curve(y_test,
                                                                                                  y_test_predict_nb_calib_platt,
                                                                                                  n_bins=10)
    # plt.plot(mean_predicted_value_nb_calib_platt, fraction_of_positives_nb_calib_platt, 's-', color='orange', label='Calibrated (Platt)')

    print(classification_report(y_test, yhat_calibrated_platt))

    return y_test_predict_nb, y_test_predict_nb_calib, y_test_predict_nb_calib_platt


def predict_cancellation(x_test, y_test, config):
    y_test_predict_random = random_forest_classifier(x_test, y_test, config)
    y_test_predict_random_calibrated = calibrated_classifier(x_test, y_test, config)
    y_test_predict_nb, y_test_predict_nb_calib, y_test_predict_nb_calib_platt = gaussian_naive_bias(x_test, y_test, config)

    result_df = pd.DataFrame()
    result_df['Random_Forest_Probability'] = y_test_predict_random
    result_df['Calibrated_Random_Forest_Probability'] = y_test_predict_random_calibrated
    result_df['Naive_Bias_Probability'] = y_test_predict_nb
    result_df['Isotonic_Calibrated_Naive_Bias_Probability'] = y_test_predict_nb_calib
    result_df['Sigmoid_Calibrated_Naive_Bias_Probability'] = y_test_predict_nb_calib_platt

    result_df2 = util.calc_cancellation_score(result_df, config)
    return result_df2


def main():
    # Read configuration from the config file
    config_path = "config.ini"
    config = util.read_config(config_path)

    # Get values from the config file
    data_path = config.get("Settings", "predict_out_path")
    df = pd.read_csv(data_path)

    # Prepare training data for model
    print(" Prediction data is ready for preprocess")
    #process = 'prediction'
    x_test, y_test, results_df = preprocess.main(df)

    # Call training models
    print(" Data is ready for prediction")
    print(" x_test columns = ", x_test.columns)

    df2 = predict_cancellation(x_test, y_test, config)
    df2['prediction'] = np.where(df2['cancelation_Score'] > 0.5, 1, 0)
    df2['actual_value'] = y_test.copy()

    print("df2 shape = ", df2.shape)
    print("y_test shape = ", y_test.shape)
    print("results_df shape = ", results_df.shape)

    print("df2 columns = ", df2.columns)
    print("results_df columns = ", results_df.columns)

    results_df = pd.concat([results_df, df2], axis=1, join='inner')

    # Compute the false positive rate (FPR)
    # and true positive rate (TPR) for different classification thresholds
    fpr, tpr, thresholds = roc_curve(results_df['actual_value'], results_df['cancelation_Score'], pos_label=1)

    # Compute the ROC AUC score
    roc_auc = roc_auc_score(results_df['actual_value'], results_df['cancelation_Score'])

    auc_rate = config.get("model", "auc_rate")


    if (roc_auc < float(auc_rate)):
        print("Model success is under the ROC/AUC over the prediction")
        print("Model is re-training")
        train_model.main()
        print("Model re-trained, please call the predict method")
    else:
        print('!!! roc_auc = ', round(roc_auc, 2), 'and model success criteria is acceptable !!!')

        # Get values from the config file
        result_path = config.get("Settings", "results_path")
        results_db = config.get("Settings", "results_db")

        results_df.to_csv(result_path, index=False)

        conn = sql.connect(results_db + ".db")
        results_df.to_sql("results_db", conn, if_exists='replace')

        print("!!!  Results were written to the db file !!!")


if __name__ == '__main__':
    main()

