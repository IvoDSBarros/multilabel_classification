'''
Rock News NLP
Multi-label Classification: hyperparameter tuning on different estimators using a multi-output classification approach
Created on Mon Jan  1 14:16:06 2024
@author: IvoBarros
'''
import pandas as pd
import numpy as np
import os
import os.path
from ast import literal_eval
from time import time
import sys
import rock_news_nlp_multi_label_utilities as utils_multi_label
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.model_selection import iterative_train_test_split
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, hamming_loss, make_scorer
from sklearn.model_selection import GridSearchCV
## PROBLEM TRANSFORMATION APPROACHES
from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.problem_transform import ClassifierChain
from skmultilearn.problem_transform import LabelPowerset
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
## MULTI OUTPUT CLASSIFIER
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import RidgeClassifier

def grid_search_multi_output_classifier(estimator, tfidf_vectorizer, param_grid, X_train, y_train):
    """
    To select optimal hyperparameters different estimators using a multi-output classification approach

    Args:
        estimator : estimator object
        tfidf_vectorizer : feature_extraction.text.TfidfVectorizer
        param_grid : dict
        X_train : list
        y_train : Array of int32

    Returns:
        dict, float64
    """
    pipe_temp = Pipeline([('tfidf', tfidf_vectorizer),
                          ('clf', MultiOutputClassifier(estimator()))])

    param_grid_temp = {'clf__estimator': [estimator()]} if estimator in [AdaBoostClassifier,GradientBoostingClassifier] else {'clf': [estimator()]}
    param_grid_temp.update(param_grid)
    param_grid_temp = [param_grid_temp]

    # CUSTOM SCORER: MICRO-AVERAGE F1-SCORE
    f1_scorer = make_scorer(f1_score, average='micro')

    grid_search_temp = GridSearchCV(pipe_temp, param_grid_temp, scoring=f1_scorer, cv=3, n_jobs=1, verbose=0)
    grid_search_temp.fit(X_train, y_train)

    best_params = grid_search_temp.best_params_
    best_score = grid_search_temp.best_score_

    return best_params, best_score

print("The script is running...")
t_start = time()

path = os.getcwd()
path_parent_dir = os.path.dirname(os.getcwd())
path_data = f'{path_parent_dir}\\data'
path_data_multi_label_dataset = f'{path_data}\\multi_label_dataset'
path_output = f'{path_parent_dir}\\output'
path_output_csv = f'{path_output}\\csv'

## LOAD DATA: SUBSET OF TRAINING DATA
X_train = pd.read_csv(f'{path_output_csv}/X_train_subset.csv',sep=';')
y_train = pd.read_csv(f'{path_output_csv}/y_train_subset.csv',sep=';')
X_test = pd.read_csv(f'{path_output_csv}/X_test_subset.csv',sep=';')
y_test = pd.read_csv(f'{path_output_csv}/y_test_subset.csv',sep=';')

X_train = X_train['0'].to_list()
y_train = y_train.to_numpy()
X_test = X_test['0'].to_list()
y_test = y_test.to_numpy()

## TFIDF VECTORIZER
tfidf_vectorizer = TfidfVectorizer(max_df=0.7, max_features=10000)

## EXPLORE AND SELECT OPTIMAL HYPERPARAMETERS
dict_param_grid = {'DecisionTreeClassifier': {'clf__criterion': ['entropy', 'gini'],
                                              'clf__max_depth': np.arange(1, 21).tolist()[0::2] + [None],
                                              'clf__min_samples_split': np.arange(2, 11).tolist()[0::2],
                                              'clf__max_leaf_nodes': np.arange(3, 26).tolist()[0::2] + [None]},
                    'RandomForestClassifier': {'clf__bootstrap': [True, False], 
                                              'clf__max_depth': [5, 10, None], 
                                              'clf__max_features': ['sqrt', 'log2', None], 
                                              'clf__n_estimators': [10, 50, 100],
                                              'clf__max_leaf_nodes': [3, 6, 9, None]},                
                    'AdaBoostClassifier': {'clf__estimator__n_estimators': [20, 30, 50],
                                          'clf__estimator__learning_rate': [(0.97 + x / 100) for x in range(0, 8)], 
                                          'clf__estimator__algorithm': ['SAMME', 'SAMME.R']},                 
                    'ExtraTreesClassifier': {'clf__n_estimators': [20,30,50],
                                            'clf__criterion': ['gini', 'entropy']}                 
                    }

df_best_params_multioutput_class = pd.DataFrame()

for k, v in dict_param_grid.items():
    best_params, best_score = grid_search_multi_output_classifier(eval(k),tfidf_vectorizer,v,X_train,y_train)
    df_best_params = utils_multi_label.create_df_best_params(best_params,k,"MultiOutputClassifier")   
    df_best_params_multioutput_class = pd.concat([df_best_params_multioutput_class,df_best_params])

df_best_params_multioutput_class = df_best_params_multioutput_class.loc[~(df_best_params_multioutput_class['Param'].isin(['clf','clf__estimator']))].copy()
df_best_params_multioutput_class['Param'] = df_best_params_multioutput_class['Param'].str.replace("clf__estimator__","",regex=True).replace("clf__","",regex=True)
df_best_params_multioutput_class.to_csv(f'{path_output_csv}/best_params_multioutput_classifier.csv', header=True, index=False, encoding='utf-8',sep=';')
print("...it has been successfully executed in %0.1fs." % (time() - t_start))