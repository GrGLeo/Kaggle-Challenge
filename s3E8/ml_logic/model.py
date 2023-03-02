from preprocessing import create_pipeline
import pandas as pd
from sklearn.model_selection import train_test_split
import os

pipe = create_pipeline()

data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"data/train.csv")
data = pd.read_csv(data_path)

X = data.drop("price",axis=1)
y = data["price"]

X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.2)
print(1)
pipe.fit(X_train,y_train)

print(1)
