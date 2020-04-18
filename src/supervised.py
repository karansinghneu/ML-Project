import numpy as np, pandas as pd
import ast
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')
import spacy
from nltk import Tree

import xgboost as xgb
from nltk.stem.lancaster import LancasterStemmer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

en_nlp = spacy.load('en')
st = LancasterStemmer()


def load_data():
    data = pd.read_csv("train_detect_sent.csv").reset_index(drop=True)

    print(data.shape)

    print(data.head(3))

    print(ast.literal_eval(data["sentences"][0]))

    data_total = data[data["sentences"].apply(lambda x: len(ast.literal_eval(x))) < 11].reset_index(drop=True)
    return data_total


def create_features(data):
    train = pd.DataFrame()

    for k in range(len(data["euclidean_dis"])):
        dis = ast.literal_eval(data["euclidean_dis"][k])
        for i in range(len(dis)):
            train.loc[k, "column_euc_" + "%s" % i] = dis[i]

    print("Finished")

    for k in range(len(data["cosine_sim"])):
        dis = ast.literal_eval(data["cosine_sim"][k].replace("nan", "1"))
        for i in range(len(dis)):
            train.loc[k, "column_cos_" + "%s" % i] = dis[i]

    train["target"] = data["target"]
    return train


def create_concatenated(training):
    training.apply(max, axis=0)

    subset1 = training.iloc[:, :10].fillna(60)
    subset2 = training.iloc[:, 10:].fillna(1)

    print(subset1.head(3))

    print(subset2.head(3))

    # train2 = pd.concat([subset1, subset2], axis=1, join_axes=[subset1.index])
    train2 = pd.concat([subset1, subset2], axis=1, join='outer')

    print(train2.head(3))

    train2.apply(max, axis=0)
    return train2


def log_reg_fit(training, training_standardised):
    ### Fitting Multinomial Logistic Regression

    ### Standardize

    scaler = MinMaxScaler()
    X = scaler.fit_transform(training_standardised.iloc[:, :-1])

    print(X)

    train_x, test_x, train_y, test_y = train_test_split(X,
                                                        training.iloc[:, -1], train_size=0.8, random_state=5)

    mul_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg')
    mul_lr.fit(train_x, train_y)

    print("Multinomial Logistic regression Train Accuracy : ", metrics.accuracy_score(train_y, mul_lr.predict(train_x)))
    print("Multinomial Logistic regression Test Accuracy : ", metrics.accuracy_score(test_y, mul_lr.predict(test_x)))


def get_columns_from_root(train):
    for i in range(train.shape[0]):
        if len(ast.literal_eval(train["root_match_idx"][i])) == 0:
            pass

        else:
            for item in ast.literal_eval(train["root_match_idx"][i]):
                train.loc[i, "column_root_" + "%s" % item] = 1
    return train

## For Model with root matching
def log_reg_root(predicted1, train2):
    # ### Logistic-Regression with Root Match feature
    # predicted = pd.read_csv("train_detect_sent.csv").reset_index(drop=True)
    #
    # predicted = predicted[predicted["sentences"].apply(lambda x: len(ast.literal_eval(x))) < 11].reset_index(drop=True)

    print(predicted1.shape)
    predicted_new = get_columns_from_root(predicted1)

    print(predicted_new.head(3).transpose())

    subset3 = predicted_new[
        ["column_root_0", "column_root_1", "column_root_2", "column_root_3", "column_root_4", "column_root_5", \
         "column_root_6", "column_root_7", "column_root_8", "column_root_9"]]

    subset3.fillna(0, inplace=True)

    train3 = pd.concat([subset3, train2], axis=1, join='outer')

    print(train3.head(3).transpose())

    train3 = train3[
        ["column_root_0", "column_root_1", "column_root_2", "column_root_3", "column_root_4", "column_root_5", \
         "column_root_6", "column_root_7", "column_root_8", "column_root_9", "column_cos_0", "column_cos_1", \
         "column_cos_2", "column_cos_3", "column_cos_4", "column_cos_5", \
         "column_cos_6", "column_cos_7", "column_cos_8", "column_cos_9", "target"]]

    train_x, test_x, train_y, test_y = train_test_split(train3.iloc[:, :-1],
                                                        train3.iloc[:, -1], train_size=0.8, random_state=5)

    mul_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg')
    mul_lr.fit(train_x, train_y)

    print("Multinomial Logistic regression Train Accuracy : ", metrics.accuracy_score(train_y, mul_lr.predict(train_x)))
    print("Multinomial Logistic regression Test Accuracy : ", metrics.accuracy_score(test_y, mul_lr.predict(test_x)))
    ### Random Forest

    rf = RandomForestClassifier(min_samples_leaf=8, n_estimators=60)
    rf.fit(train_x, train_y)

    print("Random Forest Train Accuracy : ", metrics.accuracy_score(train_y, rf.predict(train_x)))
    print("Random Forest Test Accuracy : ", metrics.accuracy_score(test_y, rf.predict(test_x)))

    ### XgBoost

    model = xgb.XGBClassifier()
    param_dist = {"max_depth": [3, 5, 10],
                  "min_child_weight": [1, 5, 10],
                  "learning_rate": [0.07, 0.1, 0.2],
                  }

    # run randomized search
    grid_search = GridSearchCV(model, param_grid=param_dist, cv=3,
                               verbose=5, n_jobs=-1)
    grid_search.fit(train_x, train_y)

    print(grid_search.best_estimator_)

    xg = xgb.XGBClassifier(max_depth=5)
    xg.fit(train_x, train_y)

    print("XG Boost Train Accuracy : ", metrics.accuracy_score(train_y, xg.predict(train_x)))
    print("XG Boost Test Accuracy : ", metrics.accuracy_score(test_y, xg.predict(test_x)))


data_usage = load_data()
train_set = create_features(data_usage)

# del data

print(train_set.head(3))

# train.fillna(10000, inplace=True)

print(train_set.head(3).transpose())
training_standardised = create_concatenated(train_set)
log_reg_fit(train_set, training_standardised)

## To run Model with root matching
log_reg_root(data_usage, training_standardised)
