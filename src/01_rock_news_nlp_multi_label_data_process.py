"""
Rock News NLP
Multi-label Classification: Data Process 
Created on Mon Jan  1 14:16:06 2024
@author: IvoBarros
"""
import pandas as pd
import os
import os.path
from ast import literal_eval
from time import time
import sys
from sklearn.preprocessing import MultiLabelBinarizer

path_to_class = os.path.dirname(os.path.abspath(sys.argv[0]))
class_directory = 'https://github.com/IvoDSBarros/rock-is-not-dead_nlp-experiments-on-rock-news-articles/blob/main/src'

class_file_dir = os.path.join(path_to_class, class_directory)
sys.path.append(class_file_dir)
from rock_news_nlp_class_text_preprocessing import text_preprocessing as tpp

print("The script is running...")
t_start = time()

path = os.getcwd()
path_parent_dir = os.path.dirname(os.getcwd())
path_data = f'{path_parent_dir}\\data'
path_data_web_scraper = f'{path_data}\\web_scraper'
path_data_rule_based_text_class = f'{path_data}\\rule_based_text_class'
path_data_multi_label_dataset = f'{path_data}\\multi_label_dataset'

## LOAD DATA
df_rock_news = pd.read_csv(f'{path_data_web_scraper}/rock_news.csv',sep=';',usecols=['title','full_pk'])
df_rock_news_category_tags = pd.read_csv(f'{path_data_rule_based_text_class}/rock_news_nlp_rock_news_category_tags.csv',
                                          sep=';', usecols=['full_pk', 'keywords'],
                                          converters={'keywords': literal_eval})
df_rock_news_category_tags['keywords'] = df_rock_news_category_tags['keywords'].apply(lambda j: ['n/a'] if bool(j)==False else j)

## ONE HOT ENCODING
mlb = MultiLabelBinarizer()
df_multi_label = pd.DataFrame(mlb.fit_transform(df_rock_news_category_tags['keywords']),columns=mlb.classes_)
df_multi_label['full_pk'] = df_rock_news_category_tags['full_pk']
df_multi_label = df_multi_label.merge(df_rock_news, how='inner', on='full_pk')

## TEXT PREPROCESSING
corpus_title = df_multi_label['title'].to_list()
corpus_title_clean = tpp.text_preprocessing_multi_label_class(corpus_title)
df_multi_label['title_clean'] = corpus_title_clean

## SAVE DATA
df_multi_label.to_csv(f'{path_data_multi_label_dataset}\\rock_news_multi_label_dataset.csv', header=True, index=False, encoding='utf-8',sep=';')
print("...it has been successfully executed in %0.1fs." % (time() - t_start))
