# CANCEL - PREDICTION

The task is to predict the cancellation of booking for each case

This project was prepared with two steps. The first step consists of data pre-processing, feature engineering, model training and prediction, while the second step is the service API. The output of machine learning algorithms were written in both sqlite as a database and csv as a file.
SQLite was chosen as the database because this project did not have a very complex data set and focused on that was paid to the ease of installation. Compared to other databases, SQLite performs lower performance, but high performance is not expected from a dataset in this project.

In this solution case, you can execute training and prediction independently. After prediction step, there is a success measurement control according to the ROC/AUC score. If the accuracy is lower than threshold, train model method calls again.

In this solution, 5 different calibrated versions of 2 main machine learning algorithms were used. After the predictions, a scoring algorithm was developed that was weighted according to the accuracy rates of each model. This score, recorded as 'cancelation_Score' in the result table, is a probability score and is currently set at 70%. This threshold can be changed in the config file if desired.

Again after prediction, you can create rest api to see results. "main.py" folder was created for rest service. In this step for easy and fast execution, I prefer to dockerize the service. For dockerization, you have to run below commands on terminal.

*** For model training, you have to run "python train_model.py" on terminal
*** For model prediction, you have to run "python prediction.py" on terminal
*** For model service, you have to run "python main.py" on terminal

But I highly recommend to use dockerize flask service version with help of below shell scripts

1) docker build --tag cancel-prediction-app:1.0 .
2) docker run -p 1001:1001 --name cancel-prediction-app cancel-prediction-app:1.0

After this process, you can use Postman to test. You can find postman file under "collection" file. You have to import that json file to the Postman. 

**Service:**

(get_all_results) : This service return probability value for every transaction. This method doesn't need any parameter. 

Services return dataframe as a json message.

**Architecture:**

![cancel_prediction.jpg](architecture%2Fcancel_prediction.jpg)