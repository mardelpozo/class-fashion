# Fashion-MNIST Image Classification: CNN vs Random Forest

A comparative computer vision project evaluating a Convolutional Neural Network (CNN) and a Random Forest classifier on the Fashion-MNIST dataset. The project analyzes classification performance, training cost, and class-level failure modes in order to recommend an appropriate model under different operational constraints.

---

## Table of Contents
* [Project Overview](#project-overview)
* [Dataset](#dataset)
* [Methodology](#methodology)
* [Results](#results)
* [Error Analysis](#error-analysis)
* [Recommendations](#recommendations)
* [Repository Structure](#repository-structure)
* [Reproducibility](#reproducibility)
* [References](#references)

---

## Project Overview
This project evaluates two approaches for classifying the Fashion-MNIST dataset into 10 clothing categories:
1) Convolutional Neural Network (tensorflow/keras)
2) Random Forest Classifier (scikit-learn).

The goal is to compare performance, training time, and failure modes, to later recommend a model for production use.

**Objective:**  
To evaluate both approaches in terms of:
- classification accuracy
- training time
- error analysis per class

---

## Dataset
- **Dataset:** Fashion-MNIST  
- **Samples:**  
  - Training: 60,000 images  
  - Test: 10,000 images  
- **Image format:** 28×28 grayscale  
- **Classes (10):**  
  T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot

---

## Methodology

### Models

**1. Convolutional Neural Network (CNN)**  
- Framework: TensorFlow / Keras  
- Architecture:
  - Two convolutional blocks (`Conv2D` + `MaxPooling`)
  - Dense classifier  
- Input: normalized images with shape `(28, 28, 1)` 
- Optimizer: Adam  
- Loss: Sparse Categorical Crossentropy  
- Epochs: 10  

**2. Random Forest Classifier**  
- Framework: scikit-learn  
- Input: flattened normalized pixel values (784 features)  
- Parameters: 
  - `n_estimators=100`
  - `criterion='entropy'`
  - `max_depth=100`

### Evaluation

The evaluation strategy comprised the following:
- Test accuracy
- Confusion matrices
- Per-class precision and recall
- Training time

---

## Results

- **CNN**
  - Test accuracy: **91.32%**
  - Training time: **216.95 s**
- **Random Forest**
  - Test accuracy: **87.53%**
  - Training time: **17.33 s**

The CNN achieves higher accuracy, particularly for visually similar garment categories, at the cost of significantly longer training time.

---

## Error Analysis

The most frequent misclassifications for both models occur among upper-body garments with similar silhouettes:
- Shirt 
- T-shirt/top  
- Pullover  
- Coat

Classes with distinct silhouette shapes (such as footwear, bags, and trousers) exhibit higher precision and recall across both models.

---

## Recommendations

- **CNN:** Recommended when classification accuracy is the primary objective, especially for visually similar products.  
- **Random Forest:** Suitable for environments where computational resources are limited or where fast training and simpler deployment are priority.

---

## Repository Structure

This repository contains a **single Jupyter notebook** that implements the complete workflow:

```text
├── fashion.ipynb
├── environment.yml
└── README.md
```

---

## Reproducibility
### Environment
This project was developed using:
- `python=3.11`
- `tensorflow=2.20.0`
- `keras=3.13.2`
- `scikit-learn
- CPU-only (no GPU)

To recreate the environment:
```
conda env create -f environment.yml
conda activate class-fashion
```

### Setup
Launch Jupyter and run `fashion.ipynb`:
```
jupyter lab
```

## References

For a complete list of references, please consult the **Bibliography** section of the corresponding project report.

This project was conducted as part of the **Project: Computer Vision** course at IU International University of Applied Sciences.

---
*Author: [Mariana Del Pozo Patrón](https://github.com/mardelpozo)*
