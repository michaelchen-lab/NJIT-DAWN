# -*- coding:utf-8 -*-
"""
created for adblocking project
"""

import tensorflow as tf
from ..feature_column import SparseFeat, VarLenSparseFeat, DenseFeat, build_input_features
from ..inputs import create_embedding_matrix, embedding_lookup, get_dense_input, varlen_embedding_lookup, \
    get_varlen_pooling_list
from ..layers.core import DNN, PredictionLayer
from ..layers.sequence import AttentionSequencePoolingLayer
from ..layers.utils import concat_func, NoMask, combined_dnn_input
from tensorflow.keras.layers import Multiply

def DAWN(dnn_feature_columns, history_feature_list, dnn_use_bn=False,
        dnn_hidden_units=(200, 80), dnn_activation='relu', att_hidden_size=(80, 40), att_activation="dice",
        att_weight_normalization=False, l2_reg_dnn=0, l2_reg_embedding=1e-6, dnn_dropout=0, seed=1024,
        task='binary'):
    """Instantiates the Deep Interest Network architecture.

    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param history_feature_list: list,to indicate  sequence sparse field
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in deep net
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of deep net
    :param dnn_activation: Activation function to use in deep net
    :param att_hidden_size: list,list of positive integer , the layer number and units in each layer of attention net
    :param att_activation: Activation function to use in attention net
    :param att_weight_normalization: bool.Whether normalize the attention score of local activation unit.
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param seed: integer ,to use as random seed.
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return: A Keras model instance.

    """

    features = build_input_features(dnn_feature_columns)

    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), dnn_feature_columns)) if dnn_feature_columns else []
    dense_feature_columns = list(
        filter(lambda x: isinstance(x, DenseFeat), dnn_feature_columns)) if dnn_feature_columns else []
    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat), dnn_feature_columns)) if dnn_feature_columns else []

    history_feature_columns = []
    sparse_varlen_feature_columns = []  # ['hist_dt', 'hist_ht', 'hist_wl']
    history_fc_names = list(map(lambda x: "hist_" + x, history_feature_list))
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        if feature_name in history_fc_names:
            history_feature_columns.append(fc)
        else:
            sparse_varlen_feature_columns.append(fc)

    inputs_list = list(features.values())
    # print("sparse_varlen_feature_columns", sparse_varlen_feature_columns)
    # print("history_fc_names", history_fc_names)
    embedding_dict = create_embedding_matrix(dnn_feature_columns, l2_reg_embedding, seed, prefix="")

    query_emb_list = embedding_lookup(embedding_dict, features, sparse_feature_columns, history_feature_list,
                                      history_feature_list, to_list=True)
    keys_emb_list = embedding_lookup(embedding_dict, features, history_feature_columns, history_fc_names,
                                     history_fc_names, to_list=True)
    dnn_input_emb_list = embedding_lookup(embedding_dict, features, sparse_feature_columns,
                                          mask_feat_list=history_feature_list, to_list=True)
    dense_value_list = get_dense_input(features, dense_feature_columns)
    sequence_embed_dict = varlen_embedding_lookup(embedding_dict, features, sparse_varlen_feature_columns)
    sequence_embed_list = get_varlen_pooling_list(sequence_embed_dict, features, sparse_varlen_feature_columns,
                                                  to_list=True)
    sequence_embed_actions = embedding_lookup(embedding_dict, features, sparse_varlen_feature_columns,
                                     mask_feat_list=history_fc_names, to_list=True) # hist actions

    dnn_input_emb_list += sequence_embed_list
    keys_emb = concat_func(keys_emb_list, mask=True)
    actions_emb = concat_func(sequence_embed_actions, mask=True)
    keys_emb = Multiply()([keys_emb, actions_emb])  # element-wise product of hist page features and action features
    deep_input_emb = concat_func(dnn_input_emb_list)
    query_emb = concat_func(query_emb_list, mask=True)
    hist = AttentionSequencePoolingLayer(att_hidden_size, att_activation,
                                         weight_normalization=att_weight_normalization, supports_masking=True)([
        query_emb, keys_emb])

    action_pooling = tf.reduce_sum(actions_emb, 1, keepdims=True)
    # residual connection
    deep_input_emb = tf.keras.layers.Concatenate()([NoMask()(deep_input_emb), hist, action_pooling])
    deep_input_emb = tf.keras.layers.Flatten()(deep_input_emb)
    dnn_input = combined_dnn_input([deep_input_emb], dense_value_list)
    wl_output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn,
                 dnn_dropout, dnn_use_bn, seed)(dnn_input)
    wl_final_logit = tf.keras.layers.Dense(1, use_bias=False,
                                        kernel_initializer=tf.keras.initializers.glorot_normal(seed))(wl_output)
    dt_output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn,
                    dnn_dropout, dnn_use_bn, seed)(dnn_input)
    dt_final_logit = tf.keras.layers.Dense(1, use_bias=False,
                                           kernel_initializer=tf.keras.initializers.glorot_normal(seed))(dt_output)

    wl_output = PredictionLayer(task = 'binary', name='wl_output')(wl_final_logit)
    dt_output = PredictionLayer(task = 'regression', name = 'dt_output')(dt_final_logit)
    model = tf.keras.models.Model(inputs=inputs_list, outputs=[wl_output, dt_output])
    return model
