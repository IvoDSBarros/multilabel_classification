"""
Rock News NLP
Multi-label Classification: Model Development and Evaluation 
Created on Mon Jan  1 14:16:06 2024
@author: IvoBarros
"""
import pandas as pd
import os
import os.path
from time import time
import csv
import rock_news_nlp_multi_label_utilities as utils_multi_label

from skmultilearn.model_selection import iterative_train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, hamming_loss
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
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.linear_model import RidgeClassifier

print("The script is running...")
t_start = time()

path = os.getcwd()
path_parent_dir = os.path.dirname(os.getcwd())
path_data = f'{path_parent_dir}\\data'
path_data_multi_label_dataset = f'{path_data}\\multi_label_dataset'
path_output = f'{path_parent_dir}\\output'
path_output_csv = f'{path_output}\\csv'

#==============================================================================
# MODEL DEVELOPMENT AND EVALUATION
#==============================================================================
## LOAD DATA
df_multi_label = pd.read_csv(f'{path_data_multi_label_dataset}/rock_news_multi_label_dataset.csv',sep=';')

## MULTILABEL DATA STRATIFICATION
feature = 'title_clean'
feature_ref = 'full_pk'
labels = [col for col in df_multi_label.columns if col not in ['title',feature,feature_ref]]
test_size = 0.2
X_train, y_train, X_test, y_test, X_train_ref, y_train, X_test_ref = utils_multi_label.custom_multilabel_data_strat(df_multi_label,feature,feature_ref,labels,test_size)

## TRAIN AND EVALUATE 
class_evaluation = {'classifier': [], 'estimator': [], 'accuracy_score': [], 'macro_f1': [], 'micro_f1': [], 'hamming_loss': []}

dict_classifiers = {## PROBLEM TRANSFORMATION APPROACHES
                    'PTC_BR_LR': ['BinaryRelevance',"LogisticRegression(penalty='l2',solver='sag')"],
                    'PTC_BR_MNB': ['BinaryRelevance', 'MultinomialNB()'],
                    'PTC_BR_GNB': ['BinaryRelevance', 'GaussianNB()'],
                    'PTC_CC_LR': ['ClassifierChain', "LogisticRegression(penalty='l2',solver='sag')"],
                    'PTC_CC_MNB': ['ClassifierChain', 'MultinomialNB()'],
                    'PTC_CC_GNB': ['ClassifierChain', 'GaussianNB()'],
                    ## MULTIOUTPUT CLASSIFIER
                    'MOC_DT': ['MultiOutputClassifier', 'DecisionTreeClassifier()'],
                    'MOC_DT_bal': ['MultiOutputClassifier', "DecisionTreeClassifier(class_weight='balanced')"],
                    'MOC_ET': ['MultiOutputClassifier', 'ExtraTreeClassifier()'],
                    'MOC_RF': ['MultiOutputClassifier', 'RandomForestClassifier()'],
                    'MOC_ETS': ['MultiOutputClassifier', 'ExtraTreesClassifier()'],
                    'MOC_AB': ['MultiOutputClassifier', 'AdaBoostClassifier()'],
                    'MOC_GB': ['MultiOutputClassifier', 'GradientBoostingClassifier()'],
                    'MOC_RC': ['MultiOutputClassifier', 'RidgeClassifier()']}

for k, v in dict_classifiers.items():
    ## SET PIPELINE OF TRANSFORMS
    pipeline = Pipeline([
                        ('tfidf', TfidfVectorizer(max_df=0.7, max_features=10000)),
                        ('clf', eval(v[0])(eval(v[1])))
                        ])
    pipeline.fit(X_train, y_train)
    prediction = pipeline.predict(X_test)
    
    class_evaluation['classifier'].append(v[0])    
    class_evaluation['accuracy_score'].append(accuracy_score(y_test, prediction))
    class_evaluation['macro_f1'].append(f1_score(y_test, prediction, average='macro'))
    class_evaluation['micro_f1'].append(f1_score(y_test, prediction, average='micro'))
    class_evaluation['hamming_loss'].append(hamming_loss(y_test, prediction))

    if k=='MOC_DT_bal':
        class_evaluation['estimator'].append('DecisionTreeClassifier_Bal')
    else:    
        class_evaluation['estimator'].append(v[1].split('(')[0])

df_class_evaluation = pd.DataFrame(class_evaluation)

for k, v in {"X_train": X_train, "X_test": X_test, 
             "y_train": y_train, "y_test": y_test,
             "X_train_ref": X_train_ref,  "X_test_ref": X_test_ref}.items():
    pd.DataFrame(v).to_csv(f'{path_output_csv}/{k}.csv', header=True, index=False, encoding='utf-8',sep=';')

df_class_evaluation.to_csv(f'{path_output_csv}/classifiers_evaluation.csv', header=True, index=False, encoding='utf-8',sep=';')
print("...it has been successfully executed in %0.1fs." % (time() - t_start))
