# Multilabel classification task on rock news articles
# Overview
Based upon a previous rule-based text classification model, an hybrid multilabel classifier was developed to assign topic labels to a dataset of rock news headlines, aiming to explore this variant of the classification problem and enhance its accuracy. This repository presents the steps implemented to develop the multilabel classification task. Several classifiers were tested including the ones following the problem transformation approach and the MultiOutputClassifier.

<details>
<summary> Table of Contents </summary>

1. [Exploratory data analysis](#1-exploratory-data-analysis)
2. [About the methodology](#2-about-the-methodology)
3. [Results](#results)
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
+ In order to provide a *"well-balanced distribution of (...) label relations"*, an iteractive stratification was implemented to split the dataset into train/test sets (Szymański, P., Kajdanowicz, T., 2016). The test size was set at 0.2.
+ No re-sampling or re-weighting methods were adopted to deal with class imbalance as both *"result in oversampling of common labels"* (Huang, Y. et al., 2021).
+ Several inherently robust algorithms to imbalanced datasets were tested including tree-based and ensemble methods (Ganganwar, 2012; Mahani, 2022; Mulugeta, G. et al., 2023).
+ A cost-sensitive learning experiment was implemented by adjusting the "class_weight" parameter of a tree-based classifier with no accuracy improvement.
+ Emphasis was placed on more informative evaluation metrics for imbalanced datasets such as Micro-average F1-score.

<div align = "right">    
  <a href="#overview">(back to top)</a>
</div>

## 3. Results


<div align = "right">    
  <a href="#overview">(back to top)</a>
</div>

## References
+ [Ganganwar, V. (2012) An overview of classification algorithms for imbalanced datasets. International Journal of Emerging Technology and Advanced Engineering. ISSN 2250-2459, Volume 2, Issue 4.](https://www.researchgate.net/profile/Vaishali-Ganganwar/publication/292018027_An_overview_of_classification_algorithms_for_imbalanced_datasets/links/58c7707a458515478dc4c68b/An-overview-of-classification-algorithms-for-imbalanced-datasets.pdf)
+ [Huang, Y., Giledereli, B., Köksal, A., Özgür, A., Ozkirimli, E. (2021) Balancing Methods for Multi-label Text Classification with Long-Tailed Class Distribution.](https://arxiv.org/abs/2109.04712)
+ [Mahani, A. (2022) Classification in Multi-Label Datasets in Information Systems Management.](https://www.intechopen.com/chapters/85471)
+ [Mulugeta, G., Zewotir, T., Tegegne, A., Juhar, L., Muleta, M. (2023) Classification of imbalanced data using machine learning algorithms to predict the risk of renal graft failures in Ethiopia. BMC Medical Informatics and Decision Making, Vol. 23, Nr. 98.](https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-023-02185-5)
+ [Szymański, P., Kajdanowicz, T. (2016) A scikit-based Python environment for performing multi-label classification. Journal of Machine Learning Research, 1, 1-15.](https://arxiv.org/abs/1702.01460)






<div align = "right">    
  <a href="#overview">(back to top)</a>
</div>
