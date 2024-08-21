import pandas as pd
import util
import preprocess

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline



def random_forest_classifier(x_train, y_train):
    print("random_forest_classifier started")
    # Read configuration from the config file
    config_path = "config.ini"
    config = util.read_config(config_path)

    # Get values from the config file
    """
    n_estimators = config.get("model", "n_estimators")
    max_depth = config.get("model", "max_depth")
    min_samples_split = config.get("model", "min_samples_split")
    min_samples_leaf = config.get("model", "min_samples_leaf")
    """
    hyper_random = {"n_estimators": [300, 700],
                    "max_depth": [10, 8],
                    "min_samples_split": [10, 5],
                    "min_samples_leaf": [5, 3]}

    clf_rf_tuned = GridSearchCV(RandomForestClassifier(), hyper_random,
                                cv=5, verbose=1,
                                n_jobs=-1)
    clf_rf_tuned.fit(x_train, y_train)

    best_params_random = clf_rf_tuned.best_params_
    print("best_params_random = ", best_params_random)

    cv_clf_rf = RandomForestClassifier(max_depth=best_params_random["max_depth"],
                                       min_samples_leaf=best_params_random["min_samples_leaf"],
                                       min_samples_split=best_params_random["min_samples_split"],
                                       n_estimators=best_params_random["n_estimators"])

    cv_clf_rf.fit(x_train, y_train)

    #Dump model
    today_date = util.dump_model(cv_clf_rf, 'random_forest_classifier')
    util.write_today_date_to_config("today_date", today_date)


    return cv_clf_rf

def calibrated_classifier(x_train, y_train, cv_clf_rf):
    print("calibrated_classifier started")

    # Create a corrected classifier.
    clf_sigmoid = CalibratedClassifierCV(cv_clf_rf, cv=10, method='sigmoid')
    clf_sigmoid.fit(x_train, y_train)

    #Dump model
    util.dump_model(clf_sigmoid, 'sigmoid_random_forest_classifier')

def gaussian_naive_bias(x_train, y_train):
    print("gaussian_naive_bias_classifier started")

    # Read configuration from the config file
    config_path = "config.ini"
    config = util.read_config(config_path)

    # Get values from the config file
    nb_var_smoothing = config.get("model", "nb_var_smoothing")
    nb_priors = config.get("model", "nb_priors")

    # Create a pipeline with only Gaussian Naive Bayes
    pipeline = Pipeline([
        ('nb', GaussianNB())
    ])

    # Define the parameters you want to tune
    parameters = {
        'var_smoothing': [1e-9, 1e-8, 1e-7],  # Example parameter for Gaussian Naive Bayes
        'priors': [None, [0.2, 0.8], [0.5, 0.5]],  # Another parameter for Gaussian Naive Bayes
        'class_prior' : [0.3, 0.7]
        # Add more parameters as needed
    }
    # Create a GridSearchCV object for Uncalibrated
    #clf_nb = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1)
    clf_nb = GaussianNB()

    # Fit the data to find the best parameters
    clf_nb.fit(x_train, y_train)

    #Dump model
    util.dump_model(clf_nb, 'gaussian_naive_bias_classifier')

    # Calibrated
    clf_sigmoid_nb = CalibratedClassifierCV(clf_nb, cv=10, method='isotonic')

    clf_sigmoid_nb.fit(x_train, y_train)

    #Dump model
    util.dump_model(clf_sigmoid_nb, 'isotonic_calibrated_naive_bias_classifier')

    # Calibrated, Platt
    clf_sigmoid_nb_calib_sig = CalibratedClassifierCV(clf_nb, cv=10, method='sigmoid')
    clf_sigmoid_nb_calib_sig.fit(x_train, y_train)

    #Dump model
    util.dump_model(clf_sigmoid_nb_calib_sig, 'sigmoid_calibrated_naive_bias_classifier')

def train_model(x_train, y_train):
    cv_clf_rf = random_forest_classifier(x_train, y_train)
    calibrated_classifier(x_train, y_train, cv_clf_rf)
    gaussian_naive_bias(x_train, y_train)

def main():
    # Read configuration from the config file
    config_path = "config.ini"
    config = util.read_config(config_path)

    # Get values from the config file
    data_path = config.get("Settings", "train_out_path")
    df = pd.read_csv(data_path)

    # Prepare training data for model
    print(" Data is ready for preprocess")
    #process = 'training'
    x_train, y_train, results_df = preprocess.main(df)

    # Call training models
    print(" Data is ready for model")
    print(" x_train columns = ", x_train.columns)

    train_model(x_train, y_train)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()