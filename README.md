# Lung Cancer Classification using Computerized Tomography (CT) Data

This project implements a machine learning model for lung cancer classification using CT scan data. The main goal is to accurately distinguish between malignant and benign lung nodules, which can assist in early cancer detection and treatment. Given the challenge of limited labeled medical data, this project incorporates semi-supervised learning methods, specifically Label Spreading, to leverage both labeled and unlabeled data for better classification performance.

## Project Overview

Early detection of lung cancer significantly improves treatment outcomes. However, labeling medical imaging data, such as CT scans, is time-consuming and often requires specialized expertise. To address this, semi-supervised learning methods allow us to make use of a large amount of unlabeled data alongside a smaller labeled dataset.

This project:
- Utilizes CT scan data and proffessionals annotations to identify and classify lung nodules.
- Incorporates semi-supervised learning with the Label Spreading algorithm to maximize the value of both labeled and unlabeled data.
- Demonstrates the effectiveness of semi-supervised learning in the context of medical image classification, particularly in cases where labeled data is scarce.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset preparation](#modeling-approach)
- [Semi-Supervised Learning with Label Spreading](#Semi-Supervised-Learning-with-Label-Spreading)
- [Results](#Results)
- [Discussion and Future Work](#discussion-and-future-work)



## Dataset Preparation
1. **Dataset**: The dataset used in this project contains CT scan images of lung nodules and radiologists annotations, with labels indicating whether each nodule is malignant or benign,according to the proffesional. A significant portion of the data is unlabeled, which makes it an ideal candidate for semi-supervised learning methods. For privacy and ethical reasons, medical datasets are not included in this repository; however, publicly available datasets, such as those from the [LIDC-IDRI dataset](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI), are suitable for similar experiments.
2. **Data Preprocessing**: CT images are preprocessed with standard techniques such as normalization, HU capping, and augmentation to enhance the model's robustness.
3. **Exploratory Data Analysis**: The **LIDC-IDRI** dataset consists of over 1,000 thoracic CT scans from patients, with a primary focus on detecting and analyzing lung nodules. A unique feature of this dataset is the detailed nodule annotations provided by four experienced thoracic radiologists. Each radiologist independently marked and classified nodules, allowing for up to four distinct annotations per nodule. Only nodules larger than 3 mm were included, as these are clinically significant for lung cancer screening. The annotations capture important information for training machine learning models, such as the nodule’s location, volume, sphericity, texture, and other relevant characteristics.
The *PyLidc* module facilitates the processing of these annotations, which are stored in XML format, and provides tools for visualizing the data, making it easier to extract features and prepare it for model development.
4. **Dataset Preparation**: After conducting a thorough analysis of the data, we will proceed with the construction of the dataset, which will serve as the foundation for training machine learning models. Our primary objective is to **predict the benignity of a nodule** based on features annotated by radiologists, as well as characteristics extracted from the images. Examples of features that can be extracted include texture measures, shape descriptors, and statistical metrics such as intensity histograms.
To achieve this, we will utilize statistical summaries of the annotations and the CT slices, primarily focusing on metrics like the mean and mode, which will aid in estimating the location of the nodule and feature extraction. This estimation will be crucial for generating the corresponding mask and for determining its associated features.


## **Semi-Supervised Learning with Label Spreading**
Given the limited labeled data, we use Label Spreading and other clustering methods to propagate labels from labeled samples to unlabeled ones based on feature similarity. This method assumes that similar samples are more likely to share the same label, which can be effective for medical imaging data.
## **Results**
After comparing all algorithms, both unsupervised and semi-supervised, and their performance in terms of generalization to new data, we have observed that **none seem to outperform the original dataset** resulting from the aforementioned radiologists' consensus algorithm. Although no improvements were found, it is important to highlight the **generalization capacity of label spreading** and the added robustness to the models by incorporating over 1,900 examples into the dataset, achieving an accuracy above 90% and +80% recall, and F1 score  in the hold-out of the original data.
## **Discussion and Future Work**
In this project, we have undertaken a comprehensive data science and engineering endeavor aimed at detecting cancerous nodules in biomedical images of the lungs. Our approach involved processing of images alongside radiologist annotations to develop accurate and robust machine learning (ML) models. A key focus of our methodology was the prioritization of metrics such as recall when selecting the most effective models. This choice was driven by the need to minimize the occurrence of false negatives, which is critical given the serious implications for patient care and outcomes.

Looking ahead, future work could include using wavelet transforms for better feature extraction and exploring 3D volumetric metrics for deeper insights into nodule characteristics. We may also consider leveraging advanced deep learning classification models, which could improve accuracy and reliability.

It is essential to recognize that these models should be viewed as tools to assist radiologists and medical professionals rather than definitive sources of truth. The integration of ML models in clinical practice necessitates careful consideration of their role in decision-making processes. Ultimately, these systems are designed to augment the expertise of healthcare providers, facilitating more informed and timely diagnoses.

Therefore, we can conclude that **both models**, logistic regression with the original data and the semi-supervised label-spreading data, are robust and effective for the project's objective: predicting cancer in lung nodules showing over 95% of accuracy and recall.


## Requirements

- Python 3.8+
- `numpy`
- `pandas`
- `scikit-learn`
- `scikit-learn-extra`
- `xgboost`
- `seaborn`
- `ast`
- `PIL`
- `pywt`
- `pyradiomics`
- `pathlib`
- `json`
- `pylidc` 
- `scikit-image`
- `matplotlib`
- Jupyter Notebook (for running the included notebook)

To install the required packages, run:
```bash
pip install -r requirements.txt
