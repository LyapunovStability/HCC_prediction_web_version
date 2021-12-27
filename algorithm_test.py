# several models conduct on clinical data

# Input:
# patientID1 attribute1 attribute2 attribute3
# patientID2 attribute1 attribute2 attribute3

# Output：
# Prob: the probability of risk of liver cancer

# Model:
# Logistic regression
# ridge regression
# adaboost
# decision tree
# random forest

###
# default

#cmd:python algorithm.py --model 0 --atts -1 --seed 1



import pdb
import copy
import csv
import numpy as np
import argparse
import random
import pandas as pd
from sklearn.metrics import roc_auc_score,confusion_matrix
from sklearn.model_selection import train_test_split

from sklearn.ensemble import AdaBoostClassifier
# from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer as Imputer

from sklearn.metrics import roc_curve, auc

from sklearn.model_selection import train_test_split


index_show= {'lower_sensitivity':0.0,
           'lower_specificity':0.0,
            'lower_ppv':0.0,
            'lower_npv':0.0,
            'lower_cutoff':0.0,
            'upper_sensitivity':0.0,
            'upper_specificity':0.0,
            'upper_ppv':0.0,
            'upper_npv':0.0,
            'upper_cutoff':0.0,
            'lower_percent':'',
            'upper_percent':'',
            'num_parameter':0.0,
            'num_sample':0,
            'auroc':0.0
            }

task = {'train_lower_sensitivity': "0.0",
                  'train_lower_specificity': "0.0",
                  'train_lower_ppv': "0.0",
                  'train_lower_npv': "0.0",
                  'train_lower_cutoff': "0.0",
                  'train_upper_sensitivity': "0.0",
                  'train_upper_specificity': "0.0",
                  'train_upper_ppv': "0.0",
                  'train_upper_npv': "0.0",
                  'train_upper_cutoff': "0.0",
                  'train_lower_percent': "0.0",
                  'train_upper_percent': "0.0",
                  'train_num_parameter': "0.0",
                  'train_num_sample': "0.0",
                  'train_auroc': "0.0",
                  'test_lower_sensitivity': "0.0",
                  'test_lower_specificity': "0.0",
                  'test_lower_ppv': "0.0",
                  'test_lower_npv': "0.0",
                  'test_lower_cutoff': "0.0",
                  'test_upper_sensitivity': "0.0",
                  'test_upper_specificity': "0.0",
                  'test_upper_ppv': "0.0",
                  'test_upper_npv': "0.0",
                  'test_upper_cutoff': "0.0",
                  'test_lower_percent': "0.0",
                  'test_upper_percent': "0.0",
                  'test_num_parameter': "0.0",
                  'test_num_sample': "0.0",
                  'test_auroc': "0.0"
                  }




def copy_result(index, mode="train_"):
    for key in index.keys():
        if "percent" in key:
            task[mode + key] = index[key]
        else:
            task[mode + key] = "{0:.3f}".format(index[key])





def dual_cutoffs(y_test, test_preds):
    fpr, tpr, thresholds = roc_curve(y_test, test_preds)
    sensitivity = tpr
    specification = 1 - fpr
    lower_cutoff = None
    upper_cutoff = None
    index1 = np.argwhere(sensitivity > 0.9)
    index2 = np.argwhere(specification <= 0.9)

    if len(index1) != 0:
        lower_cutoff = thresholds[index1[0, 0]]
        if lower_cutoff <= 0.01:
            lower_cutoff = 0.01
    if len(index2) != 0:
        if index2[0, 0]>0:
            index =index2[0, 0] - 1
            upper_cutoff = thresholds[index]
        else:
            upper_cutoff = thresholds[index2[0, 0]]

        if upper_cutoff <= 0.01:
            upper_cutoff = 0.01

    index_show['lower_cutoff'] = lower_cutoff
    index_show['upper_cutoff'] = upper_cutoff

    if lower_cutoff is not None:
        conf_matrix(y_test, test_preds, cutoff=lower_cutoff, lower=True)
    if upper_cutoff is not None:
        conf_matrix(y_test, test_preds, cutoff=upper_cutoff, lower=False)




def conf_matrix(y_test, test_preds, cutoff=0.5, lower=True):
    pred = test_preds >= cutoff
    print("cut_off: ", cutoff)

    tn,fp,fn,tp = confusion_matrix(y_test, pred).ravel()

    sensitivity = tp/(tp + fn)
    specificity = tn/(tn + fp)
    ppv = tp/(tp + fp)
    npv = tn/(tn + fn)

    print("conf: ",tn, fp, fn, tp)
    print("ppv: ",ppv)

    if lower == True:
        num = len(pred)-np.count_nonzero(pred)
        sum = len(pred)
        print("<lower cutoffs: {0}({1:.2f}) ".format(num, num/sum))
        index_show['lower_sensitivity']=sensitivity
        index_show['lower_specificity']=specificity
        index_show['lower_ppv']=ppv
        index_show['lower_npv']=npv
        index_show['lower_percent']="{0}({1:.2f})".format(num, num/sum)

    elif lower == False:
        num = np.count_nonzero(pred)
        sum = len(pred)
        print(">higher cutoffs:{0}({1})".format(num, num/sum))
        index_show['upper_sensitivity']=sensitivity
        index_show['upper_specificity']=specificity
        index_show['upper_ppv']=ppv
        index_show['upper_npv']=npv
        index_show['upper_percent']="{0}({1:2f})".format(num, num/sum)

    index_show['num_sample']=len(pred)






def read_data_from_file(path):
    f = open(path, 'r')
    csv_reader = csv.reader(f)
    data = []
    for ii, line in enumerate(csv_reader):
        if ii == 0:
            continue
        data.append(line[1:])
    data = np.array(data)
    return data


# return DataFrame format
def pandas_read_data_from_file(path, atts):
    temp_data = pd.read_csv(path, header=0, nrows=1)
    n_atts = temp_data.shape[1]

    new_atts = [ii for ii in range(1, n_atts)]
    data = pd.read_csv(path, dtype=np.float32, header=0, usecols=new_atts)
    data_id = pd.read_csv(path, dtype=np.float32, header=0, usecols=[0])
    print('number of attributes: ' + str(len(new_atts) - 2))
    index_show['num_parameter'] = len(new_atts) - 2
    return data,data_id


# parameters selection
# impute missing value
def preprocess_data(raw_data, predicting_year=5):
    def standarize_y_label(predicting_year, y1, y2):
        temp1 = (y1 <= predicting_year)
        temp2 = (y2 > 0)
        y = np.array(temp1 * temp2).astype(np.float32)
        return y

    def standarize_y_vector(predicting_year, y1, y2):
        pass

    def impute_missing_value(data):
        imp = Imputer(missing_values=np.nan, strategy='mean')
        pdata = imp.fit_transform(data)
        pdata.astype(np.float)
        return pdata

    def select_samples(data, predicting_year):
        years = data[:, -2].astype(np.float)
        liver_cancer = data[:, -1].astype(np.float)
        index1 = years < predicting_year
        index2 = liver_cancer == 0
        index = ~(index1 * index2)
        select_data = data[index, :]
        return select_data

    def exclude_na_smaple(data):
        # time_to_hcc, hcc, hbv_period, hcv_period, hbv_hcv_period

        years = data[:, -5].astype(np.float)
        liver_cancer = data[:, -4].astype(np.float)
        hbv_period = data[:, -3].astype(np.float)
        index1 = years < 0
        index2 = np.isnan(years)
        index3 = np.isnan(liver_cancer)
        index4 = np.isnan(hbv_period)
        index = ~(index1 + index2 + index3 + index4)
        select_data = data[index, :-3]

        return select_data

    print('observation year is: ' + str(predicting_year))
    # no selection, keep all the data!
    # whether remove the smaple, whose observation year is small than predicting year, and do not have hcc cancer
    # data=select_samples(raw_data,predicting_year)

    # exclude NA samples
    data = exclude_na_smaple(raw_data)

    if data.shape[0] == 0:
        print('select no samples')
    else:
        print('select ' + str(data.shape[0]) + ' samples from all ' + str(raw_data.shape[0]))

    all_X = data[:, :-2]
    years = data[:, -2].astype(np.float)
    print('The longest year is ' + str(np.max(years)))
    liver_cancer = data[:, -1].astype(np.float)

    imputed_X = impute_missing_value(all_X)
    print("x[0][0]: ", imputed_X[0][0])
    Y = standarize_y_label(predicting_year, years, liver_cancer)
    return imputed_X, Y


def split_data_for_training_and_testing(X, Y, seed=1, ratio=0.2):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=ratio, random_state=seed)
    return x_train, x_test, y_train, y_test


def count_positive_negative(gts):
    num = len(gts)
    positive = np.sum(gts)
    negative = num - positive
    return positive, negative


def calculate_auroc(pred, gt):
    score = roc_auc_score(gt, pred)
    return score


class Model(object):
    def __init__(self, classifier_type):

        self.model_type = classifier_type
        if classifier_type == 'adaboost':
            self.classifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=10)
        elif classifier_type == 'decision_tree':
            self.classifier = DecisionTreeClassifier(random_state=1, max_depth=10)
        elif classifier_type == 'random_forest':
            self.classifier = RandomForestClassifier(n_estimators=20, random_state=0)
        elif classifier_type == 'logistic_regression':
            self.classifier = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
        #            self.classifier=LogisticRegression(random_state=1, solver='saga',multi_class='ovr')
        #            self.classifier=LogisticRegression(random_state=1000)
        elif classifier_type == 'ridge_regression':
            self.classifier = Ridge(alpha=0.5, normalize=True)

    def train(self, x_train, y_train):
        est = self.classifier.fit(x_train, y_train)
        print('summary of estimator is: ')
        # print(est.summary2())
        print('training done!')

    def test(self, x_test):
        if self.model_type == "ridge_regression" or self.model_type == "random_forest":
            predictions = self.classifier.predict(x_test)
        else:
            predictions = self.classifier.predict_proba(x_test)[:, 1]
        return predictions

    def confidence_upper_lower(self, y_pred, y_test):
        n_booststraps = 1000
        rng_seed = 42
        bootstrapped_scores = []
        rng = np.random.RandomState(rng_seed)
        for i in range(n_booststraps):
            indices = rng.randint(0, len(y_pred) - 1, len(y_pred))
            if len(np.unique(y_test[indices])) < 2:
                continue
            score = roc_auc_score(y_test[indices], y_pred[indices])
            bootstrapped_scores.append(score)
        sorted_scores = np.array(bootstrapped_scores)
        sorted_scores.sort()
        confidence_lower = sorted_scores[int(0.025) * len(sorted_scores)]
        confidence_upper = sorted_scores[int(0.975) * len(sorted_scores)]

        print(str(confidence_lower) + '-' + str(confidence_upper))

    def condidence_interval(self, preds, gts, auc, confidence_level=0.95):
        if confidence_level == 0.8:
            z = 1.28
        elif confidence_level == 0.9:
            z = 1.645
        elif confidence_level == 0.95:
            z = 1.96
        elif confidence_level == 0.98:
            z = 2.33
        elif condidence_level == 0.99:
            z = 2.58
        num1 = np.sum(gts)
        num = len(gts)
        num2 = num - num1
        q1 = auc / (2 - auc)
        q2 = 2 * auc * auc / (1 + auc)
        se_auc = np.sqrt(
            (auc * (1 - auc) + (num1 - 1) * (q1 - auc * auc) + (num2 - 1) * (q2 - auc * auc)) / (num1 * num2))
        z_alpha_se = z * se_auc
        return z_alpha_se

    def cross_val(self, x_train, y_train, cv=5):
        score = cross_val_score(self.classifier, x_train, y_train, cv=cv, scoring='roc_auc')
        return score


def exe_model(model=None, data_path='BUCS_maindata_template.csv', ratio=0.3, predicting_year=5):
    MODEL_CHOICE = ['adaboost', 'decision_tree', 'random_forest', 'logistic_regression', 'ridge_regression']
    LIVER_DATA_PATH = data_path
    data_frame, patient_id= pandas_read_data_from_file(LIVER_DATA_PATH, -1)
    data = data_frame.values

    X, Y = preprocess_data(data, predicting_year)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=ratio, shuffle=True,
                                                        random_state=1)

    my_classifier = Model(MODEL_CHOICE[model])
    my_classifier.train(x_train, y_train)

    train_preds = my_classifier.test(x_train)
    test_preds = my_classifier.test(x_test)
    train_score = roc_auc_score(y_train, train_preds)
    test_score = roc_auc_score(y_test, test_preds)
    train_interval = my_classifier.condidence_interval(train_preds, y_train, train_score)
    test_interval = my_classifier.condidence_interval(test_preds, y_test, test_score)
    my_classifier.confidence_upper_lower(train_preds, y_train)
    my_classifier.confidence_upper_lower(test_preds, y_test)
    train_pos, train_neg = count_positive_negative(y_train)
    test_pos, test_neg = count_positive_negative(y_test)

    print('---' * 10)
    print('train the model with ' + MODEL_CHOICE[model])
    print('num of training samples is: ' + str(train_pos + train_neg))
    print('num of positive and negative samples is ' + str(train_pos) + ' : ' + str(train_neg))
    print('AUROC for train: ' + str(train_score))
    print('+-' + str(train_interval))

    print('--' * 3)
    print('num of testing samples is: ' + str(test_pos + test_neg))
    print('num of positive and negative samples is ' + str(test_pos) + ' : ' + str(test_neg))
    print('AUROC for test: ' + str(test_score))
    print('+-' + str(test_interval))

    print('y_test')
    print(y_test)

    print('test_preds')
    print(test_preds)

    dual_cutoffs(y_train, train_preds)
    index_show_train = copy.deepcopy(index_show)
    index_show_train['auroc'] = train_score

    dual_cutoffs(y_test, test_preds)
    index_show_test = copy.deepcopy(index_show)
    index_show_test['auroc'] = test_score

    copy_result(index_show_test, mode="test_")
    copy_result(index_show_train, mode="train_")



    #-------------output final result:
    Y_pred = my_classifier.test(X)

    patient_id = (patient_id.values)[:,0]

    output = [patient_id,Y_pred,Y]
    output = np.array(output).T.tolist()
    df = pd.DataFrame(data=output)
    df.to_csv("./Test.csv", encoding="utf-8-sig", mode="a", header=False, index=False)

    return task




