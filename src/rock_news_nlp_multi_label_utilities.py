"""
Rock News NLP: Multi-label classification utilities
Created on Sun Feb 11 17:32:53 2024
@author: IvoBarros
"""
import pandas as pd
import numpy as np
import math
from skmultilearn.model_selection import iterative_train_test_split
from sklearn.utils import shuffle

def train_test_split_support_x(df_ref,x_ref,feature,feature_ref):
    """
    To return the headlines of the text corpus (based on the primary key) as list
 
    Args:
        df_ref : DataFrame
        x_ref : DataFrame
        feature : str
        feature_ref : str
    
    Returns:
        list
    """ 
    x_ref = df_ref.merge(pd.DataFrame(x_ref, columns=[feature_ref]),how='inner',on=feature_ref)
    return x_ref[feature].tolist()

def custom_multilabel_data_strat(df,feature,feature_ref,labels,test_size):
    """
    To perform the iterative_train_test_split and preserve the relationship between
    the headlines of the text corpus and the original dataset
 
    Args:
        df : DataFrame
        feature : str
        feature_ref : str 
        labels : list
        test_size : float
    
    Returns:
        list, array of object, array of int64 
    """
    df[feature] = df[feature].fillna('')
    df_ref_full_pk = df[[feature_ref,feature]]
    
    X = np.array(df[feature_ref]).reshape(-1,1)
    y = df[labels].copy().to_numpy()
    
    X_train_ref, y_train, X_test_ref, y_test = iterative_train_test_split(X, y, test_size = test_size)
    X_train = train_test_split_support_x(df_ref_full_pk, X_train_ref,feature,feature_ref)
    X_test = train_test_split_support_x(df_ref_full_pk, X_test_ref,feature,feature_ref)
    
    return X_train, y_train, X_test, y_test, X_train_ref, y_train, X_test_ref

def balanced_subset(df,feature,feature_ref,labels,sample_size):
    """
    To create a balanced subset of data
    
    Args:
        df : DataFrame
        feature : str
        feature_ref : str 
        labels : list
        sample_size : float
    
    Returns:
        DataFrame
    """
    df[feature] = df[feature].fillna('')
    df_ref_full_pk = df[[feature_ref,feature]]
    label_combinations = df[labels].astype(str).agg('_'.join, axis=1)
    df['id_combination'] = pd.factorize(label_combinations)[0]
    proportions = df['id_combination'].value_counts(normalize=True)
    # ADJUSTED SAMPLE SIZE FOR EACH GROUP AND ROUND UP
    adjusted_sample_sizes = {group_id: math.ceil(sample_size * len(df) * proportion) for group_id, proportion in proportions.items()}

    balanced_samples = []
    for group_id, group_size in adjusted_sample_sizes.items():
        group = df[df['id_combination'] == group_id]
        group = shuffle(group, random_state=42)
        balanced_samples.append(group.head(group_size))

    balanced_subset = pd.concat(balanced_samples, ignore_index=True)
    balanced_subset = balanced_subset.drop(columns=['id_combination'])
    X_balanced_subset = np.array(balanced_subset[feature_ref]).reshape(-1,1)
    y_balanced_subset = balanced_subset[(labels)].copy().to_numpy()

    return df_ref_full_pk, X_balanced_subset, y_balanced_subset

def create_df_best_params(dict_best_params,estimator,classifier):
    """
    To create a dataframe on the best parameters of a given classifier
    
    Args:
        dict_best_params : dict
        estimator : str
        classifier : str
    
    Returns:
        DataFrame
    """
    list_param = list(dict_best_params.keys())
    list_value = list(dict_best_params.values())
    
    df_best_params = pd.DataFrame({'Param': list_param, 'Value': list_value})    
    df_best_params.insert(0, "Estimator", estimator, True)
    df_best_params.insert(0, "Classifier", classifier, True)

    return df_best_params
