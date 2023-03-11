import pandas as pd
import numpy as np

from sklearn.preprocessing import RobustScaler,StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer

from xgboost import XGBRegressor

import os
# Load data
df = pd.read_csv("data/train.csv",index_col="id")

# Create X and y
X = df.drop("Strength",axis=1)
y = df["Strength"]

# Train and val dataset
X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.2)

# Load X_test
X_test = pd.read_csv("data/test.csv").drop("id",axis=1)
id = pd.read_csv("data/test.csv")[["id"]]

# Create pipeline for preprocessing
def create_pipe():
    standard = ["CementComponent","WaterComponent","CoarseAggregateComponent","FineAggregateComponent","AgeInDays"]
    minmax = ["BlastFurnaceSlag","FlyAshComponent","SuperplasticizerComponent"]

    col_trans = ColumnTransformer([("stdscaler",StandardScaler(with_mean=True, with_std=True),standard),
                                ("minmaxscaler",MinMaxScaler(),minmax)])

    model = XGBRegressor(gamma= 0.66,
                        learning_rate= 0.08,
                        max_depth= 3,
                        n_estimators= 950,
                        reg_alpha= 0,
                        reg_lambda= 0.22,
                        subsample= 0.5,
                        colsample_bytree=0.8)


    return make_pipeline(col_trans,model)

def train(x,y):
    pipeline = create_pipe()

    pipeline.fit(x,y)

    return pipeline


def eval(model,x,y):
    y_pred = model.predict(x)
    loss = np.sqrt(mean_squared_error(y,y_pred))
    print("RMSE for model is : ",round(loss,2))

def make_prediction(model,x,id):
    y_pred = model.predict(x)
    X_test.index
    test = pd.concat([id, pd.DataFrame(y_pred, columns=['Strength'])], axis=1)
    test.to_csv("submission.csv",index=False)



if __name__ == "__main__":
    model = train(X_train,y_train)
    eval(model,X_val,y_val)
    make_prediction(model,X_test,id)
