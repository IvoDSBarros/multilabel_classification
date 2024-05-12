"""
Rock News NLP
Multi-label Classification: hyperparameter tuning on multi-label classifiers combined with logistic regression
Created on Mon Jan  1 14:16:06 2024
@author: IvoBarros
"""
import pandas as pd
import numpy as np
import os
import os.path
from time import time
import rock_news_nlp_multi_label_utilities as utils_multi_label
from skmultilearn.model_selection import iterative_train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, make_scorer
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV
from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.problem_transform import ClassifierChain
from sklearn.linear_model import LogisticRegression

def binary_relevance_log_reg_hyp_param_tuning(C_values,penalty_values,solver_values,max_iter_values,tfidf_vectorizer,X_train,y_train,X_test,y_test):
    """
    To select optimal hyperparameters regarding the Binary Relevance multi-label classifier combined with a logistic regression
 
    Args:
        C_values : list
        penalty_values : list
        solver_values : list
        max_iter_values : list
        X_train : list
        y_train : Array of int32
        X_test : list
        y_test : Array of int32
    
    Returns:
        dict
    """
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    
    best_params = {}
    
    for C in C_values:
        for penalty in penalty_values:
            for solver in solver_values:
                for max_iter in max_iter_values:
                    classifier = LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=max_iter)
                    br_classifier = BinaryRelevance(classifier=classifier)
    
                    br_classifier.fit(X_train_tfidf, y_train)
                    y_pred = br_classifier.predict(X_test_tfidf)
    
                    f1_micro = f1_score(y_test, y_pred, average='micro')
    
                    if f1_micro > 0.0:
                        best_params = {'C': C, 'penalty': penalty, 'solver': solver, 'max_iter': max_iter}
                        best_score = f1_micro                        
                    else:
                        best_params = {'C': 0, 'penalty': None, 'solver': None, 'max_iter': 0}
                        best_score = 0
    return best_params, best_score

def classifier_chain_log_reg_hyp_param_tuning(C_values,penalty_values,solver_values,max_iter_values,tfidf_vectorizer,X_train, y_train):
    """
    To select optimal hyperparameters regarding the multi-label model Classifier Chain combined with a logistic regression

    Args:
        C_values : list
        penalty_values : list
        solver_values : list
        max_iter_values : list
        X_train : list
        y_train : Array of int32

    Returns:
        dict
    """
    class ClassifierChainWrapper(ClassifierMixin, BaseEstimator):
        def __init__(self, base_classifier):
            self.base_classifier = base_classifier
            self.chain_ = ClassifierChain(self.base_classifier)

        def fit(self, X, y):
            self.chain_.fit(X, y)
            return self

        def predict(self, X):
            return self.chain_.predict(X)

        @property
        def classes_(self):
            return np.array(self.chain_.classifiers_[-1].classes_)

    pipe_lr = Pipeline([('tfidf', TfidfVectorizer(max_df=0.7, max_features=10000)),
                        ('clf', ClassifierChainWrapper(LogisticRegression()))])

    lr_param_grid = [{'clf__base_classifier__C': C_values,
                      'clf__base_classifier__penalty': penalty_values,
                      'clf__base_classifier__solver': solver_values,
                      'clf__base_classifier__max_iter': max_iter_values}]

    f1_scorer = make_scorer(f1_score, average='micro')

    lr_grid_search = GridSearchCV(pipe_lr,lr_param_grid,cv=10,scoring=f1_scorer,n_jobs=-1)
    lr_grid_search.fit(X_train, y_train)

    return lr_grid_search.best_params_,  lr_grid_search.best_score_

print("The script is running...")
t_start = time()

path = os.getcwd()
path_parent_dir = os.path.dirname(os.getcwd())
path_data = f'{path_parent_dir}\\data'
path_data_multi_label_dataset = f'{path_data}\\multi_label_dataset'
path_output = f'{path_parent_dir}\\output'
path_output_csv = f'{path_output}\\csv'

## LOAD DATA
X_train_to_balance = pd.read_csv(f'{path_output_csv}/X_train.csv',sep=';')
X_train_ref_to_balance = pd.read_csv(f'{path_output_csv}/X_train_ref.csv',sep=';')
y_train_to_balance = pd.read_csv(f'{path_output_csv}/y_train.csv',sep=';')
X_train_to_balance = X_train_to_balance.rename(columns={'0': 'title_clean'})
X_train_ref_to_balance = X_train_ref_to_balance.rename(columns={'0': 'full_pk'})

df_to_balance = pd.concat([X_train_to_balance,X_train_ref_to_balance,y_train_to_balance], axis=1)

## BALANCED SUBSET OF DATA
feature = 'title_clean'
feature_ref = 'full_pk'
labels = [col for col in df_to_balance.columns if col not in [feature,feature_ref]]
sample_size = 0.5
df_ref_full_pk, X_balanced_subset, y_balanced_subset = utils_multi_label.balanced_subset(df_to_balance,feature,feature_ref,labels,sample_size)

## MULTILABEL DATA STRATIFICATION
X_train_ref, y_train, X_test_ref, y_test = iterative_train_test_split(X_balanced_subset, y_balanced_subset, test_size = 0.2)
X_train = utils_multi_label.train_test_split_support_x(df_ref_full_pk, X_train_ref,feature,feature_ref)
X_test = utils_multi_label.train_test_split_support_x(df_ref_full_pk, X_test_ref,feature,feature_ref)

## TFIDF VECTORIZER
tfidf_vectorizer = TfidfVectorizer(max_df=0.7, max_features=10000)
 
# HYPERPARAMETER COMBINATIONS
C_values = [pow(10,-i) for i in range(1,3)]
penalty_values = ['l2']
solver_values = ['sag']
max_iter_values = [100,500,1000]

## EXPLORE AND SELECT OPTIMAL HYPERPARAMETERS
bin_rel_best_params, bin_rel_best_score = binary_relevance_log_reg_hyp_param_tuning(C_values,penalty_values,solver_values,max_iter_values,tfidf_vectorizer,X_train,y_train,X_test,y_test)
class_chain_best_params, class_chain_best_score = classifier_chain_log_reg_hyp_param_tuning(C_values,penalty_values,solver_values,max_iter_values,tfidf_vectorizer,X_train,y_train)

df_bin_rel_best_params = utils_multi_label.create_df_best_params(bin_rel_best_params,"LogisticRegression","BinaryRelevance")
df_class_chain_best_params = utils_multi_label.create_df_best_params(class_chain_best_params,"LogisticRegression","ClassifierChain")

## SAVE DATA
for k, v in {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}.items(): 
    pd.DataFrame(v).to_csv(f'{path_output_csv}/{k}_subset.csv', header=True, index=False, encoding='utf-8',sep=';')

df_best_params_class_lr = pd.concat([df_bin_rel_best_params, df_class_chain_best_params],axis=0).reset_index(drop=True)
df_best_params_class_lr.to_csv(f'{path_output_csv}/best_params_class_lr.csv', header=True, index=False, encoding='utf-8',sep=';')
print("...it has been successfully executed in %0.1fs." % (time() - t_start))
