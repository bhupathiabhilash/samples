# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Predicting Iris Flower Class (XGBoost 0.81) 
# MAGIC 
# MAGIC ## Problem Statement:
# MAGIC Given Sepal and Petal lengths and width predict the class of Iris flower
# MAGIC 
# MAGIC <img src="https://miro.medium.com/max/1400/1*7bnLKsChXq94QjtAiRn40w.png" alt="Image" border="0">
# MAGIC 
# MAGIC 
# MAGIC ## Dataset Source: 
# MAGIC https://gist.github.com/netj/8836201
# MAGIC 
# MAGIC ## Attribute Information:
# MAGIC    1. sepal length in cm
# MAGIC    2. sepal width in cm
# MAGIC    3. petal length in cm
# MAGIC    4. petal width in cm
# MAGIC    5. class: 
# MAGIC       * Iris Setosa
# MAGIC       * Iris Versicolour
# MAGIC       * Iris Virginica

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Install Packages

# COMMAND ----------

#Python packages
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
import json
import uuid

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ## Scenarios & Configs
# MAGIC 
# MAGIC  - List all configs in one box
# MAGIC  - Ease of automation and tracking

# COMMAND ----------

# All configurations 

# Scenario 1 - 
test_size_ratio = 0.33
max_depth_val = 2
eta_val = 1                  #learning_rate
num_round = 10
raw_data_version = "2"

# Scenario 2 - 
# test_size_ratio = 0.20
# max_depth_val = 4
# eta_val = 1
# num_round = 20
# raw_data_version = "2"

# Scenario 3 - 
# test_size_ratio = 0.10
# max_depth_val = 4
# eta_val = 0.5
# num_round = 10
# raw_data_version = "2"

# Scenario 4 - 
# test_size_ratio = 0.40
# max_depth_val = 2
# eta_val = 1
# num_round = 10
# raw_data_version = "2"

## End of configurations

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Load Raw Data
# MAGIC 
# MAGIC Raw input data is stored in Delta Table --> dsmi.iris_raw

# COMMAND ----------

def load_raw_data(table_name, version):
  raw_query= "select * from <TABLE> VERSION AS OF <VERSION>"
  prepared_query = raw_query.replace("<VERSION>",version).replace("<TABLE>",table_name)
  df = spark.sql(prepared_query)
  return df
raw_input_df = load_raw_data("dsmi.iris_raw",raw_data_version)
raw_input = raw_input_df.toPandas()
print(raw_input)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Visualize raw data 

# COMMAND ----------

import seaborn as sns
sns.set(style="white")

df = sns.load_dataset("iris")
g = sns.PairGrid(df, diag_sharey=False)
g.map_lower(sns.kdeplot)
g.map_diag(sns.kdeplot, lw=3)

g.map_upper(sns.regplot)

display(g.fig)


# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ## Transform Raw Data

# COMMAND ----------

renamed_input = raw_input.rename(columns = {"sepal.length":"sepal length", "sepal.width":"sepal width", "petal.length":"petal length", "petal.width":"petal width","variety":"class"}) 
transformed_input = renamed_input.apply(pd.to_numeric, errors='ignore') 
transformed_input["class"] = transformed_input["class"].astype('category')
transformed_input["classIndex"] = transformed_input["class"].cat.codes
print(transformed_input)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Experiment Data Preparation
# MAGIC 
# MAGIC Create a unique dataset everytime for an experiment. Steps -
# MAGIC 1.  Create a unique run-id.
# MAGIC 2.  Create a copy of input data at following location 'prep' location- 
# MAGIC   * s3://bucket/application_name/prep/run-id/input/module-name/
# MAGIC   * s3://bucket/application_name/prep/run-id/test/
# MAGIC   * s3://bucket/application_name/prep/run-id/train/

# COMMAND ----------

application_name = 'dsmi-iris-category'
uniqueKeyForRun = uuid.uuid1().hex
print("Preparing data for experiment with run-id: " + uniqueKeyForRun)
bukcet_name = 'spg-use1-databricks-dev-source'
out_base_path ='dsmi/' + application_name + '/prep/' + uniqueKeyForRun + '/'
train_path = out_base_path + 'train/train.csv'
test_path = out_base_path + 'test/test.csv' 

from io import StringIO # python3; python2: BytesIO 
import boto3

def push_to_s3(bucket, file_path, df):
  csv_buffer = StringIO()
  df.to_csv(csv_buffer)
  s3_resource = boto3.resource('s3')
  s3_resource.Object(bucket, file_path).put(Body=csv_buffer.getvalue())


# COMMAND ----------

# Split to train/test
training_df, test_df = train_test_split(transformed_input,test_size=test_size_ratio)
push_to_s3(bukcet_name, train_path, training_df )
push_to_s3(bukcet_name, test_path, test_df )
print("We have %d training examples and %d test examples." % (training_df.shape[0], test_df.shape[0]))


# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC Prepare XGBoost Model

# COMMAND ----------

dtrain = xgb.DMatrix(training_df[["sepal length","sepal width", "petal length", "petal width"]], label=training_df["classIndex"])
param = {'max_depth': max_depth_val, 'eta': eta_val, 'silent': 1, 'objective': 'multi:softmax'}
param['nthread'] = 4
param['eval_metric'] = 'auc'
param['num_class'] = 6

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ## Train & Evaluate model in MLFlow Experiment context

# COMMAND ----------

import mlflow
import mlflow.sklearn

with mlflow.start_run() as run:
  # Log a parameter (key-value pair)
  mlflow.log_param("number of rounds", num_round)
  mlflow.log_param("max_depth_val", max_depth_val)
  mlflow.log_param("test_size_ratio", test_size_ratio)
  mlflow.log_param("eta_val", eta_val)
  mlflow.log_param("raw_data_version", raw_data_version)
  mlflow.log_param("test_data_path", test_path)
  mlflow.log_param("training_data_path", train_path)
  mlflow.log_param("run-id",uniqueKeyForRun)
  
  bst = xgb.train(param, dtrain, num_round)
  dtest = xgb.DMatrix(test_df[["sepal length","sepal width", "petal length", "petal width"]])
  ypred = bst.predict(dtest)
  pre_score = precision_score(test_df["classIndex"],ypred, average='micro')
  print("xgb_pre_score:",pre_score)
  
  # Log a metric; metrics can be updated throughout the run
  mlflow.log_metric("accuracy", pre_score, step=1)
  # TODO - Add more metrics
  
  with open("output.txt", "w") as f:
      f.write("Trained model with following params:")
      f.write(json.dumps(param))
  
  # Store related artifacts in MLflow  
  mlflow.log_artifact("output.txt")
  # Log model
  mlflow.sklearn.log_model(bst, "Xgboost model")
  runID = run.info.run_uuid
  experimentID = run.info.experiment_id
  
  print("Inside MLflow Run with run_id {} and experiment_id {}".format(runID, experimentID))
  
  #mlflow.log_artifacts('s3://'+bukcet_name+'/'+out_base_path)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### MLfLow's Autolog API -
# MAGIC No need of writing code for logging individual params and metrics. Ml Libs like Keras, Tensorflow's are integrated by MLflow tracking API through autolog()!!!
# MAGIC 
# MAGIC Example - 
# MAGIC import mlflow.keras
# MAGIC 
# MAGIC mlflow.keras.autolog() # This is all you need!
# MAGIC 
# MAGIC It will AUTOMATICALLY log --> the layer count, optimizer name, learning rate and epsilon value as parameters; loss and accuracy at each step of the training and validation stages; the model summary, as a tag; and finally, the model checkpoint as an artifact.
# MAGIC 
# MAGIC More details --> https://databricks.com/blog/2019/08/19/mlflow-tensorflow-open-source-show.html