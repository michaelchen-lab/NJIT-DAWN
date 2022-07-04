import pandas as pd

def split():
    X = pd.read_csv("dataset.csv")
    X_train, X_test = X.iloc[:int(len(X) * 0.8)], X.iloc[int(len(X) * 0.8):]
    X_test = X_test[X_test['fullVisitorId'].isin(list(X_train['fullVisitorId']))]  # only returning visitors
    return X, X_train, X_test



