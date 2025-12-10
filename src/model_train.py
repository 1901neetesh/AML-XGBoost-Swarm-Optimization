import xgboost as xgb
from sklearn.model_selection import train_test_split

def prepare_train_test(df):
    X = df.drop("Is_Laundering", axis=1)
    y = df["Is_Laundering"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    return X_train, X_test, y_train, y_test, dtrain, dtest
