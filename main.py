import pandas as pd
import sqlite3 as sql
import flask
from flask import jsonify
import configparser
import json
import warnings

from sklearn.metrics import roc_auc_score


warnings.filterwarnings("ignore")
pd.set_option('display.float_format', '{:.4f}'.format)

config = configparser.ConfigParser()

try:
    config.read('config.ini')
except Exception as e:
    print(f"Error reading config file: {e}")
    # Handle the error as needed, e.g., exit the program or set default values

# Initialize SQLite connection
try:
    con = sql.connect("data/results.db")
except Exception as e:
    print(f"Error connecting to SQLite: {e}")
    # Handle the error as needed, e.g., exit the program or set default values

# Read data from SQLite
try:
    data = pd.read_sql_query("SELECT * from results_db", con)
except Exception as e:
    print(f"Error reading data from SQLite: {e}")
    # Handle the error as needed, e.g., exit the program or set default values

# Fill NaN values
data.fillna("NA", inplace=True)

def column_format(data):
    try:
        data['Random_Forest_Probability'] = data['Random_Forest_Probability'].apply(lambda x: '{:.5f}'.format(x))
        data['Calibrated_Random_Forest_Probability'] = data['Calibrated_Random_Forest_Probability'].apply(lambda x: '{:.5f}'.format(x))
        data['Naive_Bias_Probability'] = data['Naive_Bias_Probability'].apply(lambda x: '{:.5f}'.format(x))
        data['Isotonic_Calibrated_Naive_Bias_Probability'] = data['Isotonic_Calibrated_Naive_Bias_Probability'].apply(lambda x: '{:.5f}'.format(x))
        data['Sigmoid_Calibrated_Naive_Bias_Probability'] = data['Sigmoid_Calibrated_Naive_Bias_Probability'].apply(lambda x: '{:.5f}'.format(x))
        data['cancelation_Score'] = data['cancelation_Score'].apply(lambda x: '{:.5f}'.format(x))
    except Exception as e:
        print(f"Error formatting columns: {e}")
        # Handle the error as needed, e.g., exit the program or set default values

    return data

# Apply column formatting
data = column_format(data)

# Convert DataFrame to JSON
try:
    df = data.to_json(orient="records")
    df = json.loads(df)

except Exception as e:
    print(f"Error converting DataFrame to JSON: {e}")
    # Handle the error as needed, e.g., exit the program or set default values

app = flask.Flask(__name__)
app.config["DEBUG"] = True

# Define API endpoint
@app.route('/all_results', methods=['GET'])
def total_api():
    return jsonify(df[:100])

# Run the Flask app
try:
    app.run(host=config["Service"]["Host"], port=int(config["Service"]["Port"]), debug=True)
except Exception as e:
    print(f"Error running Flask app: {e}")
    # Handle the error as needed, e.g., exit the program or set default values
