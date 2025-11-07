# COS711 Assignment 3: Automatically Labelling Radio Sources with Deep Learning

![Astrophysics](https://img.shields.io/badge/Field-Radio%20Astronomy-blue)
![Deep Learning](https://img.shields.io/badge/Method-Deep%20Learning-orange)
![Python](https://img.shields.io/badge/Language-Python-brightgreen)

## 🚀 Project Overview

This project focuses on developing a deep learning pipeline to automatically classify radio source morphologies in images produced by the MeerKAT telescope as part of the MGCLS survey. The core challenge is to leverage a small, partially-labelled dataset provided by the University of Pretoria's radio astronomy group to accurately predict labels for a much larger set of unlabelled sources.

The primary goal is to build a robust, multi-label classifier that can handle class imbalance, identify rare "exotic" sources, and effectively use semi-supervised techniques like pseudo-labelling to enhance performance.

---

## 👥 Team Members

* Lerato Letsepe
* Aidan Govender
* Ettienne van Zyl

---

## 💾 Dataset

The dataset is sourced from the MeerKAT Galaxy Cluster Legacy Survey (MGCLS) and is split into three main subsets:

* **Typical Sources (`typ.zip`):** 2049 images of common radio sources like FR I and FR II galaxies.
* **Exotic Sources (`exo.zip`):** 59 images of rare and scientifically interesting sources, such as X-shaped and Z-shaped galaxies.
* **Unlabelled Sources (`unl.zip`):** 13,821 images from the same data distribution but without any human-assigned labels.

This is a **multi-label classification** task, as some sources can have multiple valid labels (e.g., "FR I" and "Bent").

---

## 🛠️ Methodology

Our approach will be structured in the following phases:

1.  **Data Preparation & Augmentation:**
    * Implementing a robust coordinate-matching algorithm to link labels with images.
    * Analyzing class distribution and devising a strategy for handling severe class imbalance.
    * Applying data augmentation techniques (rotations, flips, etc.) to expand the small training set and prevent overfitting.

2.  **Baseline Model Development:**
    * Establishing a baseline using a pre-trained Convolutional Neural Network (CNN) via **transfer learning**.
    * Fine-tuning the model on the human-labelled dataset.

3.  **Classification Enhancement:**
    * Implementing a **pseudo-labelling** pipeline to leverage the 13,821 unlabelled images. This involves using the baseline model to generate labels, filtering by confidence, and re-training on the combined human- and pseudo-labelled dataset.

4.  **Evaluation & Analysis:**
    * Evaluating the model using appropriate multi-label classification metrics (e.g., F1-score, precision, recall).
    * Analyzing model performance on both common and rare classes.

---

## ⚙️ Getting Started

### Prerequisites

* Python 3.8+
* Jupyter Notebook in Google Colab (preferred) or your favorite IDE
* Required libraries listed in `requirements.txt`

### Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/lerato1ofwan/radio-morphology-deep-learning-classification.git](https://github.com/lerato1ofwan/radio-morphology-deep-learning-classification.git)
    cd radio-morphology-deep-learning-classification

    ```

2.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3.  (Instructions on how to run your main script/notebook will go here)

---

## 📁 Project Structure

TBC
