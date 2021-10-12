<!-- # ACOS

We are making the final preparations for the release of our data and code. They will be coming soon. -->

# Aspect-Category-Opinion-Sentiment (ACOS) Quadruple Extraction

This repo contains the data sets and source code of our paper: 

Aspect-Category-Opinion-Sentiment Quadruple Extraction with Implicit Aspects and Opinions [[ACL 2021]](https://aclanthology.org/2021.acl-long.29.pdf).
- We introduce a new ABSA task, named Aspect-Category-Opinion-Sentiment Quadruple (ACOS) Extraction, to extract fine-grained ABSA Quadruples from product reviews;
- We construct two new datasets for the task, with ACOS quadruple annotations, and benchmark the task with four baseline systems;
- Our task and datasets provide a good support for discovering implicit opinion targets and implicit opinion expressions in product reviews.

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/aspect-category-opinion-sentiment-quadruple/aspect-category-opinion-sentiment-quadruple-1)](https://paperswithcode.com/sota/aspect-category-opinion-sentiment-quadruple-1?p=aspect-category-opinion-sentiment-quadruple)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/aspect-category-opinion-sentiment-quadruple/aspect-category-opinion-sentiment-quadruple)](https://paperswithcode.com/sota/aspect-category-opinion-sentiment-quadruple?p=aspect-category-opinion-sentiment-quadruple)

## Task
The Aspect-Category-Opinion-Sentiment (ACOS) Quadruple Extraction aims to extract all aspect-category-opinion-sentiment quadruples in a review sentence and provide full support for aspect-based sentiment analysis with implicit aspects and opinions.

![Alt text](img/figure1.PNG?raw=true "Example")

## Datasets
Two new datasets, Restaurant-ACOS and Laptop-ACOS are constructed for the ACOS Quadruple Extraction task. Following are the comparison between the sizes of our two ACOS Quadruple datasets and existing representative ABSA datasets:

![Alt text](img/stat.PNG?raw=true "stat")

## Method
Overview of our Extract-Classify-ACOS method. The first step performs aspect-opinion co-extraction, and the second step predicts category-sentiment given the extracted aspect-opinion pairs.

![Alt text](img/method.PNG?raw=true "method")

## Citation
If you use the data and code in your research, please cite our paper as follows:
```
@inproceedings{cai2021aspect,
  title={Aspect-Category-Opinion-Sentiment Quadruple Extraction with Implicit Aspects and Opinions},
  author={Cai, Hongjie and Xia, Rui and Yu, Jianfei},
  booktitle={Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)},
  pages={340--350},
  year={2021}
}
```
