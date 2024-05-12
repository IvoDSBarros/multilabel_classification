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
                    ## MULTI OUTPUT CLASSIFIER
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

###############################################################################

## MULTILABEL DATA STRATIFICATION
# df_multi_label['title_clean'] = df_multi_label['title_clean'].fillna('')
# df_ref_full_pk = df_multi_label[['full_pk','title_clean']]

# X = np.array(df_multi_label['full_pk'] ).reshape(-1,1)
# y = df_multi_label[([col for col in df_multi_label.columns if col not in ['title','title_clean','full_pk']])].copy().to_numpy()

# X_train_ref, y_train, X_test_ref, y_test = iterative_train_test_split(X, y, test_size = 0.2)
# X_train = utils_multi_label.train_test_split_support_x(df_ref_full_pk, X_train_ref)
# X_test = utils_multi_label.train_test_split_support_x(df_ref_full_pk, X_test_ref)


# from sklearn.multioutput import MultiOutputClassifier
# from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
# from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier



# # pipeline = Pipeline([
# #                     ('count', CountVectorizer(analyzer='word', min_df=10)),
# #                     ('clf', ClassifierChain(LogisticRegression(penalty='l2',solver='sag'))),
# #                     ])



# pipeline = Pipeline([
#                     ('tfidf', TfidfVectorizer(max_df=0.7, max_features=10000)),
#                     ('clf', MultiOutputClassifier(DecisionTreeClassifier()))
#                     ])



# # def select_class_problem_transf(classifier, estimator, *args, **kwargs):
# #     """
# #     classifier_problem_transf : str {'BR': 'BinaryRelevance','CC': 'ClassifierChain','LP': 'LabelPowerset'} 
# #     """

# #     dict_classifier = {'BR': 'BinaryRelevance','CC': 'ClassifierChain','LP': 'LabelPowerset'}
# #     dict_estimator = {'LR': 'LogisticRegression','MNB': 'MultinomialNB', 'GNB': 'GaussianNB'}
# #     classifier = dict_classifier.get(classifier)
# #     estimator = dict_estimator.get(estimator)
    
# #     return Pipeline([
# #                     ('count', CountVectorizer(analyzer='word', min_df=10)),
# #                     ('clf', eval(classifier)(eval(estimator)(*args, **kwargs))),
# #                     ]) 




# pipeline.fit(X_train, y_train)
# prediction = pipeline.predict(X_test)

# ######
# # df_prediction = pd.DataFrame(prediction.todense())
# # test = pd.DataFrame.sparse.from_spmatrix(predictions)
# df_prediction = pd.DataFrame(prediction)
# #####


# df_output = pd.concat([pd.DataFrame(X_test_ref, columns=['full_pk']), df_prediction], axis=1)
# df_output = df_output.merge(df_ref_full_pk,how='inner',on='full_pk')
# df_output = df_output.iloc[:,np.r_[0,-1,1:(len(df_output.columns)-1)]]
# df_output.columns = ['full_pk','title_clean'] + list_col_names_classes

# test_2 = df_output[df_output['full_pk'].str.contains('REHAB')==True]







# print('accuracy_score: ', accuracy_score(y_test, prediction))
# print('macro_f1: ', f1_score(y_test, prediction, average='macro'))
# print('micro_f1: ', f1_score(y_test, prediction, average='micro'))
# print('hamming_loss: ', hamming_loss(y_test, prediction))




"""
http://scikit.ml/api/skmultilearn.html
Problem Transformation approaches¶
The skmultilearn.problem_transform module provides classifiers that follow the problem transformation approaches to multi-label classification.
The problem transformation approach to multi-label classification converts multi-label problems to single-label problems: single-class or multi-class.


https://scikit-learn.org/stable/modules/multiclass.html


from skmultilearn.adapt import MLkNN
from scipy.sparse import csr_matrix, lil_matrix
https://towardsdatascience.com/journey-to-the-center-of-multi-label-classification-384c40229bff

https://github.com/binanhiasm/MultiLabel

https://stackoverflow.com/questions/57148129/randomforestclassifier-in-multi-label-problem-how-it-works
https://stackoverflow.com/questions/44107236/example-of-multi-label-multi-classmulti-output-randomforest-nearest-neighbo
https://stackoverflow.com/questions/76912377/can-i-run-the-cuml-randomforestclassifier-with-a-sklearn-multioutputclassifier-w

http://scikit.ml/stratification.html


"""


# ## PROBLEM TRANSFORMATION APPROACHES
# problem_transf_class_eval = {'classifier': [], 'estimator': [], 'accuracy_score': [], 'macro_f1': [], 'micro_f1': [], 'hamming_loss': []}
# # dict_problem_transf_class = {'BR': 'BinaryRelevance','CC': 'ClassifierChain','LP': 'LabelPowerset'}
# # dict_problem_transf_estimator = {'LR': "LogisticRegression(penalty='l2',solver='sag')", 'MNB': 'MultinomialNB()', 'GNB': 'GaussianNB()', 'SVC': 'SVC()'}

# dict_problem_transf_class = {'BR': 'BinaryRelevance','CC': 'ClassifierChain'}
# dict_problem_transf_estimator = {'LR': "LogisticRegression(penalty='l2',solver='sag')", 'MNB': 'MultinomialNB()', 'GNB': 'GaussianNB()'}

# for k_1, v_1 in dict_problem_transf_class.items():
#     for k_2, v_2 in dict_problem_transf_estimator.items():
#         ## SET PIPELINE OF TRANSFORMS
#         pipeline = Pipeline([
#                             ('tfidf', TfidfVectorizer(max_df=0.7, max_features=10000)),
#                             ('clf', eval(v_1)(eval(v_2)))
#                             ])
#         pipeline.fit(X_train, y_train)
#         prediction = pipeline.predict(X_test)
        
#         problem_transf_class_eval['classifier'].append(v_1)
#         problem_transf_class_eval['estimator'].append(v_2)
#         problem_transf_class_eval['accuracy_score'].append(accuracy_score(y_test, prediction))
#         problem_transf_class_eval['macro_f1'].append(f1_score(y_test, prediction, average='macro'))
#         problem_transf_class_eval['micro_f1'].append(f1_score(y_test, prediction, average='micro'))
#         problem_transf_class_eval['hamming_loss'].append(hamming_loss(y_test, prediction))
        
# df_problem_transf_class_eval = pd.DataFrame(problem_transf_class_eval)

# ## MULTI OUTPUT CLASSIFIER
# multi_output_class_eval = {'classifier': [], 'estimator': [], 'accuracy_score': [], 'macro_f1': [], 'micro_f1': [], 'hamming_loss': []}

# dict_multi_output_class_estimator = {'DT': 'DecisionTreeClassifier()',
#                                       'DT_bal': "DecisionTreeClassifier(class_weight='balanced')",
#                                       'ET': 'ExtraTreeClassifier()',
#                                       'RF': 'RandomForestClassifier()',
#                                       'ETS': 'ExtraTreesClassifier()',
#                                       'AB': 'AdaBoostClassifier()',
#                                       'GB': 'GradientBoostingClassifier()',
#                                       'RC': 'RidgeClassifier()'}

# #Problem Transformation approaches

# # dict_class_problem_transf = {'BR': 'BinaryRelevance','CC': 'ClassifierChain','LP': 'LabelPowerset'}
# # dict_classifier = {'LR': 'LogisticRegression','MNB': 'MultinomialNB', 'GNB': 'GaussianNB'}

# # print(dict_class_problem_transf.get('BR'))


# # def select_class_problem_transf(classifier, estimator, *args, **kwargs):
# #     """
# #     classifier_problem_transf : str {'BR': 'BinaryRelevance','CC': 'ClassifierChain','LP': 'LabelPowerset'} 
# #     """

# #     dict_classifier = {'BR': 'BinaryRelevance','CC': 'ClassifierChain','LP': 'LabelPowerset'}
# #     dict_estimator = {'LR': 'LogisticRegression','MNB': 'MultinomialNB', 'GNB': 'GaussianNB'}
# #     classifier = dict_classifier.get(classifier)
# #     estimator = dict_estimator.get(estimator)
    
# #     return Pipeline([
# #                     ('count', CountVectorizer(analyzer='word', min_df=10)),
# #                     ('clf', eval(classifier)(eval(estimator)(*args, **kwargs))),
# #                     ]) 
    
# # pipeline = select_class_problem_transf('CC','LR', penalty='l2',solver='sag')



# # pipeline = Pipeline([
# #     ('tfidf', CountVectorizer(analyzer='word', min_df=10)),
# #     ('clf', MultiOutputClassifier(ExtraTreesClassifier()))
# #     ])

# # def select_multi_output_estim(freq_count, estimator, *args, **kwargs):
# #     """
# #     To select the estimator of the MultiOutputClassifier and apply a list of transforms 
# #     namely word frequencies and the MultiOutputClassifier

# #     Args:
# #         freq_count : str {'CountVectorizer', 'TfidfVectorizer'} 
# #         estimator : str {'DT': 'DecisionTreeClassifier','ET': 'ExtraTreeClassifier',
# #                          'ETS': 'ExtraTreesClassifier','KN': 'KNeighborsClassifier',
# #                          'MPL': 'MLPClassifier','RN': 'RadiusNeighborsClassifier',
# #                          'RF': 'RandomForestClassifier','RC': 'RidgeClassifier',
# #                          'RCCV': 'RidgeClassifierCV'}
        
# #     Returns:
# #         Pipeline
# #     """ 
# #     dict_estimator =  {'DT': 'DecisionTreeClassifier',
# #                        'ET': 'ExtraTreeClassifier',
# #                        'ETS': 'ExtraTreesClassifier',
# #                        'KN': 'KNeighborsClassifier',
# #                        'MPL': 'MLPClassifier',
# #                        'RN': 'RadiusNeighborsClassifier',
# #                        'RF': 'RandomForestClassifier',
# #                        'RC': 'RidgeClassifier',
# #                        'RCCV': 'RidgeClassifierCV'}
# #     estimator = dict_estimator.get(estimator)
    
# #     return Pipeline([('count', eval(freq_count)(*args, **kwargs)),
# #                      ('clf', MultiOutputClassifier(eval(estimator)(*args, **kwargs)))]) 
    

# # pipeline = select_multi_output_estim('CountVectorizer','DT',analyzer='word', min_df=10)



# # pipeline = Pipeline([
# #                     ('count', CountVectorizer(analyzer='word', min_df=10)),
# #                     ('clf', ClassifierChain(LogisticRegression(penalty='l2',solver='sag'))),
# #                     ])


# # pipeline = Pipeline([
# #                     ('tfidf', TfidfVectorizer()),
# #                     ('clf', BinaryRelevance(MultinomialNB())),
# #                     ])

# # pipeline = Pipeline([
# #                 ('tfidf', TfidfVectorizer()),
# #                 ('clf', BinaryRelevance(LogisticRegression(solver='sag'))),
# #             ])

# # pipeline = Pipeline([
# #                 ('tfidf', TfidfVectorizer()),
# #                 ('clf', ClassifierChain(LogisticRegression(penalty=,solver='sag'))),
# #             ])

# # pipeline = Pipeline([
# #                 ('count', CountVectorizer(analyzer='word', min_df=10)),
# #                 ('clf', ClassifierChain(LogisticRegression(solver='sag'))),
# #             ])



# # pipeline = Pipeline([('count', CountVectorizer(analyzer='word', min_df=10,)),
# #                      ('clf', ClassifierChain(LogisticRegression(solver='sag')))
# #                  ])

# # pipeline = Pipeline([
# #                     ('count', CountVectorizer(analyzer='word', min_df=10,)),
# #                     ('clf', BinaryRelevance(MultinomialNB()))
# #                     ])



# # pipeline = Pipeline([
# #                 ('tfidf', TfidfVectorizer()),
# #                 ('clf', ClassifierChain(MultinomialNB())),
# #             ])

# # pipeline = Pipeline([
# #                 ('tfidf', TfidfVectorizer()),
# #                 ('clf',  LabelPowerset(LogisticRegression())),
# #             ])


# # pipeline = Pipeline([('count', CountVectorizer(analyzer='word', min_df=10)),
# #                       ('clf', ClassifierChain(LogisticRegression(solver='sag')))
# #                   ])


# # pipeline = Pipeline([
# #                     ('tfidf', TfidfVectorizer()),
# #                     ('clf', BinaryRelevance(MultinomialNB())),
# #                     ])

# x_train = corpus_title_train_set
# y_train = train_set_category_tags.iloc[:,np.r_[4:len(train_set_category_tags.columns)]]


# # y_train = train_set_category_tags.iloc[:,np.r_[4:10,31:35]]

# # # .astype(float)

# # print(y_train.dtypes)

# x_test = corpus_title_test_set
# y_test = test_set_category_tags_exploded.iloc[:,3:]



# pipeline.fit(x_train, y_train)
# predictions = pipeline.predict(x_test)

# ######
# # df_predictions = pd.DataFrame(predictions.todense())
# # test = pd.DataFrame.sparse.from_spmatrix(predictions)
# ######

# df_predictions = pd.DataFrame(predictions)

# df_output = pd.concat([test_set_category_tags, df_predictions], axis=1)

# list_cols = [i for i in train_set_category_tags.columns]

# # list_cols = [i for i in train_set_category_tags.iloc[:,0:10].columns]

# 

# # from sklearn.metrics import accuracy_score
# print('accuracy_score: ', accuracy_score(y_test, predictions))
# print('macro_f1: ', f1_score(y_test, predictions, average='macro'))
# print('micro_f1: ', f1_score(y_test, predictions, average='micro'))
# print('hamming_loss: ', hamming_loss(y_test, predictions))


# df_output.columns = list_cols
# print("...it has been successfully executed in %0.1fs." % (time() - t_start))



# test.insert(0,'title',corpus_title_test_set)

# #                         & 
# #                         (df_multi_label['album cover']==0)
# #                         & 
# #                         (df_multi_label['album cover art']==0)
# #                         & 
# #                         (df_multi_label['album release']==0)
# #                         &
# #                         (df_multi_label['album review']==0)
# #                         &
# #                         (df_multi_label['new album']==0)
# #                         &
# #                         (df_multi_label['album related']==1)
# #                         ]

# # conditions_cat_tour = [(df_multi_label['tour announcement']==0) 
# #                         & 
# #                         (df_multi_label['tour dates']==0)
# #                         &
# #                         (df_multi_label['tour reschedule']==0)
# #                         & 
# #                         (df_multi_label['new tour']==0)
# #                         & 
# #                         (df_multi_label['tour related']==1)
# #                         ]


# # df_multi_label['album related'] = np.select(conditions_cat_album,[1],0)
# # df_multi_label['tour related'] = np.select(conditions_cat_tour,[1],0)



# # conditions_album = (
# #                     (df_multi_label['album announcement']==1) 
# #                     | 
# #                     (df_multi_label['album cover']==1)
# #                     | 
# #                     (df_multi_label['album cover art']==1)
# #                     | 
# #                     (df_multi_label['album release']==1)
# #                     |
# #                     (df_multi_label['album review']==1)
# #                     | 
# #                     (df_multi_label['album related']==1)
# #                     | 
# #                     (df_multi_label['new album']==1)
# #                     |
# #                     (df_multi_label['tour announcement']==1) 
# #                     | 
# #                     (df_multi_label['tour dates']==1)
# #                     |
# #                     (df_multi_label['tour reschedule']==1)
# #                     | 
# #                     (df_multi_label['tour related']==1)
# #                     | 
# #                     (df_multi_label['new tour']==1)
# #                     )
                    

# # df_multi_label = df_multi_label[conditions_album]


# # from nltk.tokenize import word_tokenize
# # list_test = ['a new tour has begun','new single', 'album release']
# # set_keyword_stem = {'release','begun'}

# # aaa = [' '.join([j for j in word_tokenize(i) if j in set_keyword_stem]) for i in list_test]

# # for i in list_test:
# #     for j in word_tokenize(i):
# #         print(j)
# #     # for j in i:
# #     #     print(j)
# #     print(i)


# #         lst_txt = [' '.join([j for j in i if j in set_keyword_stem]) for i in lst_txt_temp] 

# # from scipy import sparse
# # combined = sparse.hstack([predictions.astype(float), test_set])

# list_test = []

# for i in [1,4,7,10]:
#     list_test+=[i for i in range(i,i+3)]
    
    
# # from sklearn.metrics import accuracy_score
# # print('Accuracy = ', accuracy_score(y_test,predictions))
# # print('F1 score is ',f1_score(y_test, predictions, average="micro"))
# # print('Hamming Loss is ', hamming_loss(y_test, predictions))





# from sklearn.preprocessing import MultiLabelBinarizer

# mlb = MultiLabelBinarizer()
# df_2 = pd.DataFrame(mlb.fit_transform(df_rock_news_category_tags['sub_category_tags']),columns=mlb.classes_)


# # df1 = pd.read_csv(key_1, sep=';', converters={'Tags_Title': literal_eval,'Tags_Desc': literal_eval,
# #                                                                   'RockArtist_Tags': literal_eval}) 


# X = df_rock_news_category_tags['title']
 
# y = df_rock_news_category_tags['sub_category_tags']
 
# y = MultiLabelBinarizer().fit_transform(y)


# # df = df['col'].str.strip('[]').str.get_dummies(', ')
# # df.columns = df.columns.str.strip("'")


# # test_01 = df_rock_news_category_tags.set_index('full_pk')['sub_category_tags'].str.get_dummies().max(level=0).reset_index()
# # test_02 = df_rock_news_category_tags.pivot_table(index='full_pk', columns='sub_category_tags', 
# #                      aggfunc=any, fill_value=False).astype(int).reset_index()
# # test_03 = pd.crosstab(df_rock_news_category_tags['full_pk'],df_rock_news_category_tags['sub_category_tags']).astype(int).reset_index()

# # test_02 = df_rock_news_category_tags.pivot_table(index='full_pk', columns='sub_category_tags', aggfunc='size', fill_value=0).reset_index()


# # text_corpus = utils_tpp.load_text_corpus(path_data_data_subsets, 'rock_news_test_set.csv', ';', None, 'title')
# # lda_model = utils_tpp.load_py_object('lda_model', path_output_pickled_obj, 'sklearn_train_lda_model.pickle')
# # count_vec = utils_tpp.load_py_object('count_vec', path_output_pickled_obj, 'sklearn_train_vectorizer.pickle')

# # #==============================================================================
# # # 2. EVALUATE THE MODEL
# # #==============================================================================
# # text_corpus_clean, word_freq_count_pred, lda_array, dict_topics = lda_sklearn.topic_prediction(text_corpus, lda_model, count_vec)
# # lda_sklearn_test_subset_output = lda_sklearn.df_output(lda_array, dict_topics)
# # lda_sklearn_test_subset_output.insert(0,'title',text_corpus)
# # lda_sklearn_test_subset_output.to_csv(f'{path_output_csv}/rock_news_nlp_lda_sklearn_test_subset_output.csv', header=True, index=False, encoding='utf-8',sep=';')
# # model_perplexity = lda_model.perplexity(word_freq_count_pred)
# # log_likelihood_score = lda_model.score(word_freq_count_pred)
# # utils_tpp.print_lda_model_topics_stats(dict_topics,model_perplexity,log_likelihood_score)
# # print("...it has been successfully executed in %0.1fs." % (time() - t_start))

 
# # # """
# # # # https://gist.github.com/ululh/c3edda2497b8ff9d4f70e63b0c9bd78c
# # # """

# lst = ['Metallica are touring Europe this summer', 
#         'Slash has confirmed a new GNR album',
#         'A new drummer joined Foo Fighters',
#         "New rock bands don't succeed as in the past",
#         '']

# from nltk.tokenize import word_tokenize

# aaa = ['n/a' if bool(i)==False else i for i in lst]


# [f(x) if x is not None else '' for x in xs]

# [text_preprocessing.ps.stem(word) for word in tokens if len(text_preprocessing.ps.stem(word))>2]

# for j in lst:
#     # for i.word_tokenize in j:
#     print(bool(word_tokenize(j)))
        
#     if len((word_tokenize(j)))==0:
#         j=='Not assigned'


# df_multi_label['category_tags'].apply(lambda j: ['diverse topics'] if bool(j)==False else j)

# # # clean_lst = tpp.text_preprocessing_to_sklearn(lst)
# # # vec_data = count_vec.transform(clean_lst)
# # # predict = lda_model.transform(vec_data)


# # #     def topic_prediction(unseen_text_corpus, lda_model, word_freq_count):
# # #         """
# # #         To predict topics on unseen text corpus
        
# # #         Args:
# # #             unseen_text_corpus : list
# # #             lda_model : decomposition._lda.LatentDirichletAllocation
# # #             word_freq_count : sparce.csr.csr_matrix
        
# # #     	Returns:
# # #     		Array of float64, list
# # #         """
# # #         text_corpus_clean = tpp.text_preprocessing_to_sklearn(unseen_text_corpus)    
# # #         text_corpus_clean = word_freq_count.transform(text_corpus_clean)
# # #         topic_probability_scores = lda_model.transform(text_corpus_clean)
# # #         top_words = [[word_freq_count.get_feature_names()[i] for i in j.argsort()[-5:][::-1]] for i, j in enumerate(lda_model.components_)]    
# # #         return topic_probability_scores, top_words