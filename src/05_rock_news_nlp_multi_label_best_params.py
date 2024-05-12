"""
Rock News NLP
Multi-label Classification: Model Development and Evaluation of high performance models with best parameters
Created on Mon Jan  1 14:16:06 2024
@author: IvoBarros
"""
import pandas as pd
import numpy as np
import os
import os.path
from time import time
import rock_news_nlp_multi_label_utilities as utils_multi_label

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, hamming_loss
## PROBLEM TRANSFORMATION APPROACHES
from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.problem_transform import ClassifierChain
from sklearn.linear_model import LogisticRegression
## MULTI OUTPUT CLASSIFIER
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

print("The script is running...")
t_start = time()

path = os.getcwd()
path_parent_dir = os.path.dirname(os.getcwd())
path_data = f'{path_parent_dir}\\data'
path_data_multi_label_dataset = f'{path_data}\\multi_label_dataset'
path_output = f'{path_parent_dir}\\output'
path_output_csv = f'{path_output}\\csv'

## LOAD DATA
df_best_params_class_lr = pd.read_csv(f'{path_output_csv}/best_params_class_lr.csv',sep=';')
df_best_params_multioutput_classifier = pd.read_csv(f'{path_output_csv}/best_params_multioutput_classifier.csv',sep=';')
df_multi_label = pd.read_csv(f'{path_data_multi_label_dataset}/rock_news_multi_label_dataset.csv',sep=';')

## PARAMETERS DATA PREPARATION
df_best_params_multioutput_classifier["Value"] = df_best_params_multioutput_classifier["Value"].fillna('None')
df_best_params = pd.concat([df_best_params_class_lr,df_best_params_multioutput_classifier])
df_best_params['param_val'] = df_best_params.apply(lambda x: f"{x.Param}='{x.Value}'" if (any(c.isalpha() for c in x.Value)==True 
                                                                                          and x.Value!='None' and x.Value!='True' and x.Value!='False') 
                                                                                      else f"{x.Param}={x.Value}", axis=1)
df_best_params = df_best_params.drop(['Param','Value'],axis=1) 
df_best_params = df_best_params.groupby(['Classifier','Estimator'])['param_val'].agg(pd.Series.tolist).reset_index()
df_best_params['parameters'] = df_best_params.apply(lambda x: f"{x.Estimator}({', '.join([i for i in x.param_val])})", axis=1)

conditions_classifiers = [(df_best_params['Classifier']=='BinaryRelevance') & (df_best_params['Estimator']=='LogisticRegression'),
                          (df_best_params['Classifier']=='ClassifierChain') & (df_best_params['Estimator']=='LogisticRegression'),
                          (df_best_params['Classifier']=='MultiOutputClassifier') & (df_best_params['Estimator']=='DecisionTreeClassifier'),
                          (df_best_params['Classifier']=='MultiOutputClassifier') & (df_best_params['Estimator']=='ExtraTreesClassifier'),
                          (df_best_params['Classifier']=='MultiOutputClassifier') & (df_best_params['Estimator']=='RandomForestClassifier'),
                          (df_best_params['Classifier']=='MultiOutputClassifier') & (df_best_params['Estimator']=='AdaBoostClassifier')]

actions_classifiers = ['PTC_BR_LR','PTC_CC_LR','MOC_DT','MOC_ET','MOC_RF','MOC_AB'] 
df_best_params['dict_key'] = np.select(conditions_classifiers,actions_classifiers,default='N/A')
df_best_params['dict_values'] = df_best_params[['Classifier','parameters']].values.tolist()

dict_best_params = dict(zip(df_best_params['dict_key'],df_best_params['dict_values']))

## MULTILABEL DATA STRATIFICATION
feature = 'title_clean'
feature_ref = 'full_pk'
labels = [col for col in df_multi_label.columns if col not in ['title',feature,feature_ref]]
test_size = 0.2
X_train, y_train, X_test, y_test, X_train_ref, y_train, X_test_ref = utils_multi_label.custom_multilabel_data_strat(df_multi_label,feature,feature_ref,labels,test_size)

## TRAIN AND EVALUATE 
class_evaluation = {'classifier': [], 'estimator': [], 'accuracy_score': [], 'macro_f1': [], 'micro_f1': [], 'hamming_loss': []}

for k, v in dict_best_params.items():
    # if k=='MOC_ET':        
    ## SET PIPELINE OF TRANSFORMS
    pipeline = Pipeline([
                        ('tfidf', TfidfVectorizer(max_df=0.7, max_features=10000)),
                        ('clf', eval(v[0])(eval(v[1])))
                        ])
    pipeline.fit(X_train, y_train)
    prediction = pipeline.predict(X_test)
    
    class_evaluation['classifier'].append(v[0])    
    class_evaluation['estimator'].append(v[1].split('(')[0])
    class_evaluation['accuracy_score'].append(accuracy_score(y_test, prediction))
    class_evaluation['macro_f1'].append(f1_score(y_test, prediction, average='macro'))
    class_evaluation['micro_f1'].append(f1_score(y_test, prediction, average='micro'))
    class_evaluation['hamming_loss'].append(hamming_loss(y_test, prediction))
       
df_class_evaluation = pd.DataFrame(class_evaluation)

df_class_evaluation.to_csv(f'{path_output_csv}/best_params_classifiers_evaluation.csv', header=True, index=False, encoding='utf-8',sep=';')
print("...it has been successfully executed in %0.1fs." % (time() - t_start))