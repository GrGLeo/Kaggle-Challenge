from sklearn.preprocessing import Normalizer, OneHotEncoder
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor

def create_pipeline():
    #Scale numerical value
    num_transform = Normalizer()


    #Encore categorical value
    cat_transfrom = OneHotEncoder(handle_unknown="ignore")


    preprocessor = ColumnTransformer([("num_scaler",num_transform,["carat","depth","table","x","y","z"]),
                                      ("cat_encoder",cat_transfrom,["cut","color","clarity"])
                                      ])

    model = XGBRegressor(colsample_bytree= 0.2967090532371497,
                            gamma= 2.6591286863761523,
                            learning_rate= 0.06567763787532808,
                            max_depth= 7,
                            n_estimators= 619,
                            reg_alpha= 0.06119370172660288,
                            reg_lambda= 9.211907840629712,
                            subsample= 0.8912096887033312)
    pipeline = make_pipeline(preprocessor,model)

    return pipeline
