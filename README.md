# Multilabel classification task on rock news articles
# Overview
Based upon a previous rule-based text classification model, an hybrid multilabel classifier was developed to assign topic labels to a dataset of rock news headlines, aiming to explore this variant of the classification problem and enhance its accuracy. This repository presents the steps implemented to develop the multilabel classification task. Several classifiers were tested including the ones following the problem transformation approach and the MultiOutputClassifier.

# Exploratory data analysis (EDA)
+ The number of labels for which a headline can be assigned ranges from 1 to 7 (see Chart 1).
+ Two-thirds of the headlines are assigned to a single topic label, while nearly one-fourth are tagged with two topic labels (see Chart 1).
+ The cumulative percentage of headlines assigned to more than three labels is not significant (see Chart 1).
+ The text corpus shows high imbalance (see Chart 2). 
+ Nearly one-third of the headlines (6,397 out of 20,000) are tagged with the class 'diverse', indicating topics other than the 35 predefined labels in this classification task (see Chart 2).
+ Core topic labels include: 'announce', 'release', 'album', 'tour', 'song', 'show', 'watch', 'video', 'single', 'death', 'play' and 'cover' (see Chart 2).
+ Most topic labels tend to co-occur with another label rather than being associated with multiple labels or appearing as a single label (see Chart 2).
+ Exceptions to this pattern include 'song', 'death' or 'cover', which tend to appear as single labels, and 'video' and 'single' which are more associated with multiple labels (see Chart 2).
+ Strong correlations are observed among pairs of labels such as ['tour', 'announce'], ['album', 'announce'], ['album', 'release'], ['single', 'release'] and ['video', 'release'] (see Chart 3).

![](https://github.com/IvoDSBarros/multilabel_classification/blob/e98a5697d1cf6451b7c01bb0d69bac152d5f0fcf/png/eda_histogram.png)

![](https://github.com/IvoDSBarros/multilabel_classification/blob/9e292a11864c84c5a8d289b6ea6f2e7b26ac8334/png/eda_bar.png)

![](https://github.com/IvoDSBarros/multilabel_classification/blob/c23aff2ac9a5af021db93f38360b504db29c9041/png/eda_heatmap.png)
