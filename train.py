import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pandas as pd

from deepctr.models import DAWN
from deepctr.feature_column import SparseFeat, VarLenSparseFeat, DenseFeat, get_feature_names

from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
from time import perf_counter

def split():
    X = pd.read_csv("dataset.csv")
    X_train, X_test = X.iloc[:int(len(X) * 0.8)], X.iloc[int(len(X) * 0.8):]
    X_test = X_test[X_test['fullVisitorId'].isin(list(X_train['fullVisitorId']))]  # only returning visitors
    return X, X_train, X_test

def get_xy_fd(X, X_all, maxlength = 4):
    '''
    :param X: PandasDataFrame, Train Data
    :return: prepared input data
    '''
    behavior_feature_list = ["item_id", "cate_id", 'author_id']
    uid = np.array(X['fullVisitorId'])
    country_id = np.array(X['country'])
    OS_id = np.array(X['operatingSystem'])
    browser_id = np.array(X['browser'])
    source_id = np.array(X['source'])
    medium_id = np.array(X['medium'])
    iid = np.array(X['intended_page'])  # 0 is mask value
    cate_id = np.array(X['inferred_category'])  # 0 is mask value
    author_id = np.array(X['author'])
    popularity = np.array(X['popularity'])
    freshness = np.array(X['freshness'])
    visitNumber = np.array(X['visitNumber'])


    hist_iid = np.array(list(X['hist_intended_page'].apply(lambda x: x.strip("[]").split(", ")))).astype(int)
    hist_cate_id = np.array(list(X['hist_inferred_category'].apply(lambda x: x.strip("[]").split(", ")))).astype(int)
    hist_author_id = np.array(list(X['hist_author'].apply(lambda x: x.strip("[]").split(", ")))).astype(int)
    hist_dt = np.array(list(X['hist_dwell_time'].apply(lambda x: x.strip("[]").split(", ")))).astype(int)
    hist_ht = np.array(list(X['hist_hits'].apply(lambda x: x.strip("[]").split(", ")))).astype(int)
    hist_wl = np.array(list(X['hist_whitelist'].apply(lambda x: x.strip("[]").split(", ")))).astype(int)

    feature_columns = [SparseFeat('user', X_all['fullVisitorId'].nunique() + 1, embedding_dim=10),
                       SparseFeat('country', X_all['country'].nunique() + 1, embedding_dim=8),
                       SparseFeat('OS', X_all['operatingSystem'].nunique() + 1, embedding_dim=8),
                       SparseFeat('browser', X_all['browser'].nunique() + 1, embedding_dim=8),
                       SparseFeat('source', X_all['source'].nunique() + 1, embedding_dim=8),
                       SparseFeat('medium', X_all['medium'].nunique() + 1, embedding_dim=8),
                       SparseFeat('item_id', X_all['intended_page'].nunique() + 1, embedding_dim=10),
                       SparseFeat('cate_id', X_all['inferred_category'].nunique() + 1, embedding_dim=5),
                       SparseFeat('author_id', X_all['author'].nunique() + 1, embedding_dim=10),
                       DenseFeat('popularity', 1),
                       DenseFeat('freshness', 1),
                       DenseFeat('visitNumber', 1)]
    feature_columns += [
        VarLenSparseFeat(
            SparseFeat('hist_item_id', vocabulary_size=X_all['intended_page'].nunique() + 1, embedding_dim=10,
                       embedding_name='item_id'),
            maxlen=maxlength),
        VarLenSparseFeat(
            SparseFeat('hist_cate_id', vocabulary_size=X_all['inferred_category'].nunique() + 1, embedding_dim=5,
                       embedding_name='cate_id'),
            maxlen=maxlength),
        VarLenSparseFeat(
            SparseFeat('hist_author_id', vocabulary_size=X_all['author'].nunique() + 1, embedding_dim=10,
                       embedding_name='author_id'),
            maxlen=maxlength),
        VarLenSparseFeat(
            SparseFeat('hist_dt', vocabulary_size=X_all['dwell_time'].nunique() + 1, embedding_dim=10, embedding_name='dt'),
            maxlen=maxlength),
        VarLenSparseFeat(
            SparseFeat('hist_ht', vocabulary_size=X_all['hits'].nunique() + 1, embedding_dim=10, embedding_name='ht'),
            maxlen=maxlength),
        VarLenSparseFeat(
            SparseFeat('hist_wl', vocabulary_size=X_all['whitelist'].nunique() + 1, embedding_dim=5, embedding_name='wl'),
            maxlen=maxlength)

        ]

    feature_dict = {'user': uid, 'country': country_id,
                    'OS': OS_id, 'browser': browser_id,
                    'source': source_id, 'medium': medium_id,
                    'item_id': iid, 'cate_id': cate_id, 'author_id': author_id,
                    'popularity': popularity,
                    'freshness': freshness,
                    'visitNumber': visitNumber,
                    'hist_item_id': hist_iid,
                    'hist_cate_id': hist_cate_id,
                    'hist_author_id': hist_author_id,
                    'hist_dt': hist_dt,
                    'hist_ht': hist_ht,
                    'hist_wl': hist_wl
                    }
    x = {name:feature_dict[name] for name in get_feature_names(feature_columns)}
    y1 = np.array(X['whitelist'])
    y2 = np.array(X['dwell_time'])
    y = [y1, y2]
    return x, y, feature_columns, behavior_feature_list




if __name__ == "__main__":
    maxlength = 6
    X, X_train, X_test = split()
    x, y, feature_columns, behavior_feature_list = get_xy_fd(X_train, X, maxlength=maxlength)
    dt_weight = 0.001
    start = perf_counter()

    model = DAWN(feature_columns, behavior_feature_list, dnn_hidden_units=(40, 15), att_hidden_size=(30, 15))
    losses = {
        "wl_output": "binary_crossentropy",
        "dt_output" : "mean_squared_error"
    }
    lossWeights = {"wl_output" : 1.0, "dt_output" : dt_weight}
    model.compile('adam', loss=losses, loss_weights=lossWeights)
    history = model.fit(x, y, verbose=1, epochs=2, validation_split=0.1, batch_size=128)
    x_test, y_test, feature_columns, behavior_feature_list = get_xy_fd(X_test, X, maxlength=maxlength)
    y_pred = model.predict(x_test)[0]
    y_test = y_test[0]

    # measure the performance
    ll = round(log_loss(y_test, y_pred), 4)
    auc = round(roc_auc_score(y_test, y_pred), 4)
    end = perf_counter()
    print(f"time cost {end - start}")
    print(f"AUC: {auc}; Log Loss: {ll}")
