import math

import numpy as np, pandas as pd
import ast
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import warnings

from sklearn.utils import resample

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

features_csv_path = "./train_detect_sent.csv"
features_csv_with_root_matching_path = "data/train_detect_sent_root_matching.csv"

save_models = True


def load_data(path):
    data = pd.read_csv(path).reset_index(drop=True)

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
    skf = KFold(n_splits=10)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(training_standardised.iloc[:, :-1])
    acc_array = np.zeros((10, 100))
    ind = 0
    for train_index, test_index in skf.split(X, training_standardised.iloc[:, -1]):
        mul_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg')
        mul_lr.fit(X[train_index], training_standardised.iloc[:, -1][train_index])
        model_name = "log_reg_fit_"+train_index+"_"+test_index+".pickle"
        save_models and pickle.dump(mul_lr, open( model_name, "wb" ))
        for m in range(100):
            new_data, new_output = resample(X[test_index], training_standardised.iloc[:, -1][test_index], replace=True)
            accuracy = metrics.accuracy_score(new_output, mul_lr.predict(new_data))
            acc_array[ind][m] = accuracy
        ind += 1
    final_acc_array = np.mean(acc_array, axis=0)
    sample_mean = np.average(final_acc_array)
    sum_std_err = 0
    for each in final_acc_array:
        sum_std_err += (each - sample_mean) ** 2
    std_error = math.sqrt(sum_std_err / (len(final_acc_array) - 1))

    print('Multinimoal Logistic Regression Accuracy is', sample_mean)
    print('Multinimoal Logistic Regression Standard Error is', std_error)
    print('Multinimoal Logistic Regression Confidence Interval is: ', sample_mean - std_error, ' to: ',
          sample_mean + std_error)


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
    # predicted = pd.read_csv(features_csv_with_root_matching_path).reset_index(drop=True)
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

    # train_x, test_x, train_y, test_y = train_test_split(train3.iloc[:, :-1],
    #                                                     train3.iloc[:, -1], train_size=0.8, random_state=5)
    # mul_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg')
    # mul_lr.fit(train_x, train_y)
    # print("Multinomial Logistic regression Train Accuracy : ", metrics.accuracy_score(train_y, mul_lr.predict(train_x)))
    # print("Multinomial Logistic regression Test Accuracy : ", metrics.accuracy_score(test_y, mul_lr.predict(test_x)))

    dataset = train3.iloc[:, :-1].to_numpy()
    output = train3.iloc[:, -1].to_numpy()
    skf = KFold(n_splits=10)
    acc_array_log = np.zeros((10, 100))
    acc_array_random = np.zeros((10, 100))
    acc_array_xg = np.zeros((10, 100))
    ind = 0
    for train_index, test_index in skf.split(dataset, output):

        # train_x, test_x, train_y, test_y = train_test_split(train3.iloc[:, :-1],
        #                                                     train3.iloc[:, -1], train_size=0.8, random_state=5)
        mul_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg')
        mul_lr.fit(dataset[train_index], output[train_index])

        model_name = "log_reg_root_lr"+train_index+"_"+test_index+".pickle"
        save_models and pickle.dump(mul_lr, open( model_name, "wb" ))


        rf = RandomForestClassifier(min_samples_leaf=8, n_estimators=60)
        rf.fit(dataset[train_index], output[train_index])

        model_name = "log_reg_root_rf"+train_index+"_"+test_index+".pickle"
        save_models and pickle.dump(rf, open( model_name, "wb" ))

        model = xgb.XGBClassifier()
        param_dist = {"max_depth": [3, 5, 10],
                      "min_child_weight": [1, 5, 10],
                      "learning_rate": [0.07, 0.1, 0.2],
                      }

        # run randomized search
        grid_search = GridSearchCV(model, param_grid=param_dist, cv=3,
                                   verbose=5, n_jobs=-1)
        grid_search.fit(dataset[train_index], output[train_index])

        print('Best Estimator', grid_search.best_estimator_)

        xg = xgb.XGBClassifier(max_depth=5)
        xg.fit(dataset[train_index], output[train_index])


        model_name = "log_reg_root_xg"+train_index+"_"+test_index+".pickle"
        save_models and pickle.dump(xg, open( model_name, "wb" ))

        for m in range(100):
            new_data, new_output = resample(dataset[test_index], output[test_index], replace=True)
            accuracy_log = metrics.accuracy_score(new_output, mul_lr.predict(new_data))
            accuracy_random = metrics.accuracy_score(new_output, rf.predict(new_data))
            accuracy_xg = metrics.accuracy_score(new_output, xg.predict(new_data))
            acc_array_log[ind][m] = accuracy_log
            acc_array_random[ind][m] = accuracy_random
            acc_array_xg[ind][m] = accuracy_xg
        ind += 1
    final_acc_array_log = np.mean(acc_array_log, axis=0)
    final_acc_array_random = np.mean(acc_array_random, axis=0)
    final_acc_array_xg = np.mean(acc_array_xg, axis=0)
    sample_mean_log = np.average(final_acc_array_log)
    sample_mean_random = np.average(final_acc_array_random)
    sample_mean_xg = np.average(final_acc_array_xg)
    sum_std_err_log = 0
    sum_std_err_random = 0
    sum_std_err_xg = 0
    for each in final_acc_array_log:
        sum_std_err_log += (each - sample_mean_log) ** 2
    std_error_log = math.sqrt(sum_std_err_log / (len(final_acc_array_log) - 1))
    for each in final_acc_array_random:
        sum_std_err_random += (each - sample_mean_random) ** 2
    std_error_random = math.sqrt(sum_std_err_random / (len(final_acc_array_random) - 1))
    for each in final_acc_array_xg:
        sum_std_err_xg += (each - sample_mean_xg) ** 2
    std_error_xg = math.sqrt(sum_std_err_xg / (len(final_acc_array_xg) - 1))

    print('Multinimoal Logistic Regression Accuracy with root matching is: ', sample_mean_log)
    print('Multinimoal Logistic Regression Standard Error with root matching is: ', std_error_log)
    print('Multinimoal Logistic Regression Confidence Interval with root matching is: ',
          sample_mean_log - std_error_log, ' to: ',
          sample_mean_log + std_error_log)

    print('Random Forest Accuracy with root matching is: ', sample_mean_random)
    print('Random Forest Standard Error with root matching is: ', std_error_random)
    print('Random Forest Confidence Interval with root matching is: ',
          sample_mean_random - std_error_random, ' to: ',
          sample_mean_random + std_error_random)

    print('XG Boost Accuracy with root matching is: ', sample_mean_xg)
    print('XG Boost Standard Error with root matching is: ', std_error_xg)
    print('XG Boost Confidence Interval with root matching is: ',
          sample_mean_xg - std_error_xg, ' to: ',
          sample_mean_xg + std_error_xg)

data_usage = load_data(features_csv_with_root_matching_path)
train_set = create_features(data_usage)

# del data

print(train_set.head(3))

# train.fillna(10000, inplace=True)

print(train_set.head(3).transpose())
training_standardised = create_concatenated(train_set)
log_reg_fit(train_set, training_standardised)

## To run Model with root matching
log_reg_root(data_usage, training_standardised)