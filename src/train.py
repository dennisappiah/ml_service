import pandas as pd
from src import config
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


def train(data: pd.DataFrame) -> None:
    df = pd.read_csv(data)

    MODELS = {
        'linearregression': LinearRegression(),
        'randomforest': RandomForestRegressor(n_estimators=200),
        'svr': SVR(kernel='rbf')
    }

    # select entire rows and entire columns minus the last column
    X = df.iloc[:, :-1]
    y = df.price

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10, test_size=0.3)

    model = MODELS[config.MODEL]
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)

    print(config.MODEL, " r2_score:  ", r2_score(y_test, y_predict))


# todo train parameters missing.
train(config.TRAIN_PROCESSED_DATA)
