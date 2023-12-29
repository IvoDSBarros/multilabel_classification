# Multilabel classification task on rock news articles
# Overview
Based upon a previous rule-based text classification model, an hybrid multilabel classifier was developed to assign topic labels to a dataset of rock news headlines, aiming to explore this variant of the classification problem and enhance its accuracy. This repository presents the steps implemented to develop the multilabel classification task. Several classifiers were tested including the ones following the problem transformation approach and the MultiOutputClassifier.

# Exploratory data analysis
+ The dataset contains 20.000 headlines and the average number of labels per headline stands at 1.45 (see Table 1).
+ 36 predefined labels were derived from rule-based text classification model (see Table 1).
+ The number of labels for which a headline can be assigned ranges from 1 to 7 (see Figure 1).
+ Two-thirds of the headlines are assigned to a single topic label, while nearly one-fourth are tagged with two topic labels (see Figure 1).
+ The cumulative percentage of headlines assigned to more than three labels is not significant (see Figure 1).
+ The text corpus shows high imbalance (see Figure 2). 
+ Nearly one-third of the headlines (6,397 out of 20,000) are tagged with the class 'diverse', indicating topics other than the 35 predefined labels in this classification task (see Figure 2).
+ Core topic labels include: 'announce', 'release', 'album', 'tour', 'song', 'show', 'watch', 'video', 'single', 'death', 'play' and 'cover' (see Figure 2).
+ Most topic labels tend to co-occur with another label rather than being associated with multiple labels or appearing as a single label (see Figure 2).
+ Exceptions to this pattern include 'song', 'death' or 'cover', which tend to appear as single labels, and 'video' and 'single' which are more associated with multiple labels (see Figure 2).
+ Strong correlations are observed among pairs of labels such as ['tour', 'announce'], ['album', 'announce'], ['album', 'release'], ['single', 'release'] and ['video', 'release'] (see Figure 3).
<br>

**Table 1: Dataset descriptive statistics**

![](https://github.com/IvoDSBarros/multilabel_classification/blob/74c3c828ef2d12a7ff073af97560fb6c05dcc2a2/png/eda_statistics.PNG)

<br>

**Figure 1: Distribution of the number of topic labels**

<br>

![](https://github.com/IvoDSBarros/multilabel_classification/blob/30adfe717ba17bf6d367985d50097785d17851fb/png/eda_histogram.png)

<br>

**Figure 2: Frequency of topic labels and respective co-occurrence**

<br>

![](https://github.com/IvoDSBarros/multilabel_classification/blob/06f8f39003d93c910868948cf6f3f6f32f7e5714/png/eda_bar.png)

<br>

**Figure 3: Co-occurrence of topic labels**
![](https://github.com/IvoDSBarros/multilabel_classification/blob/4bf018bf4d9fd4b22ac773e0d95a3e6944e8832d/png/eda_heatmap.png)
