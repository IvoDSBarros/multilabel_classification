# Multilabel classification task on rock news articles
# Overview
Based upon a previous rule-based text classification model, an hybrid multilabel classifier was developed to assign topic labels to a dataset of rock news headlines, aiming to explore this variant of the classification problem and enhance its accuracy. This repository presents the steps implemented to develop the multilabel classification task. Several classifiers were tested including the ones following the problem transformation approach and the MultiOutputClassifier.

<details>
<summary> Table of Contents </summary>

1. [Exploratory data analysis](#1-exploratory-data-analysis)
2. [About the methodology](#2-about-the-metodology)
3. [Topic modelling experiments](#topic-modelling-experiments)
    + [LDA model using Scikit-learn](#1-lda-model-using-scikit-learn)
    + [LDA model using Gensim](#2-lda-model-using-gensim)
4. [Rule-based text classification Vs. Machine Learning classification: final thoughts and further research](#rule-based-text-classification-vs-machine-learning-classification-final-thoughts-and-further-research)
6. [References](#references)

</details>

# 1. Exploratory data analysis
+ The dataset contains 20.000 headlines and the average number of labels per headline stands at 1.45 (see Table 1).
+ 36 predefined labels were derived from the rule-based text classification model (see Table 1).
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

![](https://github.com/IvoDSBarros/multilabel_classification/blob/e4edd9fd2adc45ca5ea99e508ed139459756da99/png/eda_desc_statistics.PNG)

<br>

**Figure 1: Distribution of the number of topic labels**

<br>

![](https://github.com/IvoDSBarros/multilabel_classification/blob/30adfe717ba17bf6d367985d50097785d17851fb/png/eda_histogram.png)

<br>

**Figure 2: Frequency distribution of topic labels and respective co-occurrence**

<br>

![](https://github.com/IvoDSBarros/multilabel_classification/blob/06f8f39003d93c910868948cf6f3f6f32f7e5714/png/eda_bar.png)

<br>

**Figure 3: Co-occurrence of topic labels**
![](https://github.com/IvoDSBarros/multilabel_classification/blob/4bf018bf4d9fd4b22ac773e0d95a3e6944e8832d/png/eda_heatmap.png)

<br>

<div align = "right">    
  <a href="#overview">(back to top)</a>
</div>

## 2. About the methodology
+ The multilabel classification task was built from a rule-based text classification model with the purpose of identifying keywords and assign both topic labels and publication type categories (details about the rule-based text classification model can be found [here](https://github.com/IvoDSBarros/rock-is-not-dead_nlp-experiments-on-rock-news-articles/blob/main/README.md#rule-based-text-classification)). Actually, the keywords of the manual rule-based model were the foundation to assign topic labels to headlines. Therefore, instead of using directly the derived topic labels, the multilabel classifier is based on the keywords.
+ In order to provide a "well-balanced distribution of (...) label relations", an interactive stratification was implemented to split data into train(80)/test(20) sets .

<div align = "right">    
  <a href="#overview">(back to top)</a>
</div>

## References
+ [Szymański, P., Kajdanowicz, T. (2016) A scikit-based Python environment for performing multi-label classification. Journal of Machine Learning Research, 1, 1-15.](https://arxiv.org/abs/1702.01460)



+ [Chang, J., Boyd-Graber, J., Gerrish, S., Wang, C., Blei, D. (2009, December) Reading Tea Leaves: How Humans Interpret Topic Models. NIPS'09: Proceedings of the 22nd International Conference on Neural Information Processing Systems, 288–296.](https://proceedings.neurips.cc/paper/2009/file/f92586a25bb3145facd64ab20fd554ff-Paper.pdf)
+ [Kelechava, M. (2019) Using LDA Topic Models as a Classification Model Input. Predicting Future Yelp Review Sentiment. Towards Data Science.](https://towardsdatascience.com/unsupervised-nlp-topic-models-as-a-supervised-learning-input-cf8ee9e5cf28)
+ [Řehůřek, R. (2022a) models.ensembelda – Ensemble Latent Dirichlet Allocation. https://radimrehurek.com/gensim/models/ensemblelda.html](https://radimrehurek.com/gensim/models/ensemblelda.html)
+ [Řehůřek, R. (2022b) Ensemble LDA. https://radimrehurek.com/gensim/auto_examples/tutorials/run_ensemblelda.html](https://radimrehurek.com/gensim/auto_examples/tutorials/run_ensemblelda.html)
+ [Roeder, M., Both, A., Hinneburg, A. (2015, February). Exploring the Space of Topic Coherence Measures. WSDM '15: Proceedings of the Eighth ACM International Conference on Web Search and Data Mining, 399–408. https://doi.org/10.1145/2684822.2685324](http://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf)
+ [Roeder, M. (2018) Not being able to replicate coherence scores from paper #13](https://github.com/dice-group/Palmetto/issues/13)
+ [Sievert, C., Shirley, K. (2014, June) LDAvis: A method for visualizing and interpreting topic. Proceedings of the Workshop on Interactive Language Learning, Visualization, and Interfaces, 63–70.](https://nlp.stanford.edu/events/illvi2014/papers/sievert-illvi2014.pdf) 
+ [Trenquier, H. (2018) Improving Semantic Quality of Topic Models for Forensic Investigations. University of Amsterdam, MSc System and Network Engineering Research Project 2.](https://rp.os3.nl/2017-2018/p76/report.pdf)
+ [Yadav, A., Patel, A., Shah, M. (2021) A comprehensive review on resolving ambiguities in natural language processing. AI Open, Volume 2, Pages 85-92](https://www.sciencedirect.com/science/article/pii/S2666651021000127)

<div align = "right">    
  <a href="#overview">(back to top)</a>
</div>
