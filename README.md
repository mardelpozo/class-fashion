# Fashion-MNIST Image Classification: CNN vs Random Forest

A comparative study evaluating a convolutional neural network (CNN) and a Random Forest classifier on the Fashion-MNIST dataset.  

The project analyzes differences in classification accuracy, training time, and class-level error behavior under a controlled experimental setup.


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
1) A **Convolutional Neural Network (CNN)**, designed to utilize spatial structures in image data.
2) A **Random Forest (RF)** classifier, trained on feature vector flattened pixel representations.

Both models are trained and evaluated on identical data splits using the same evaluation framework to ensure a fair comparison.

---

## Dataset
- **Dataset:** Fashion-MNIST  
- **Images:** 70,000 grayscale images  
- **Resolution:** 28 × 28 pixels  
- **Split:**  
  - 60,000 training images  
  - 10,000 test images  
- **Classes (10):**  
  T-Shirt/Top, Trouser, Pullover, Dress, Coat, Sandals, Shirt, Sneaker, Bag, Ankle boots

The dataset presents particular difficulty for visually similar upper-body garments (e.g., shirt, pullover, coat), making it suitable for evaluating fine-grained visual classification performance.

---

## Methodology

### Models

**1. Convolutional Neural Network (CNN)**  
- Framework: **TensorFlow / Keras**
- Architecture:
  - Two convolutional layers:
    - `Conv2D` with ReLU activation 
    - `MaxPooling`
  - Fully connected dense layers for classification  
- Input: normalized images with shape `(28, 28, 1)` 
- Optimizer: Adam  
- Loss: Sparse Categorical Crossentropy  
- Epochs: **10**

**2. Random Forest Classifier**  
- Framework: **scikit-learn**  
- Input: flattened normalized pixel values (784-dimensional feature vectors)  
- Parameters: 
  - `n_estimators=100`
  - `criterion='entropy'`
  - `max_depth=100`

### Evaluation

Model performance is evaluated using:
- Overall classification accuracy
- Training time
- Confusion matrices (training and test sets)
- Per-class precision and recall

All experiments are executed in the same computational environment to ensure comparability.

---

## Results

- **CNN**
  - Test accuracy: **91.32%**
  - Training time: **216.95 s**
- **Random Forest**
  - Test accuracy: **87.53%**
  - Training time: **17.33 s**

Key observations:
- The CNN achieves higher classification accuracy, particularly for visually similar garment categories.
- The Random Forest trains substantially faster but exhibits stronger class confusion on the test set.
- The Random Forest achieves perfect accuracy on the training set, indicating overfitting.

Detailed results, including confusion matrices and classification reports, are documented in the notebook.

---

## Error Analysis

The most frequent misclassifications for both models occur among upper-body garments with similar silhouettes:
- Shirt 
- T-Shirt/Top  
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
├── fashion.ipynb #Reproducible notebook
├── environment.yml #Conda environment file
└── README.md #This file
```

---

## Reproducibility

To reproduce results:

1. Clone the repository:
```
git clone https://github.com/mardelpozo/class-fashion.git
```
2. Recreate the environment:
```
conda env create -f environment.yml
conda activate class-fashion
```

3. Launch Jupyter and run `fashion.ipynb`:
```
jupyter lab
```

### Environment
This project was developed using:
- `python=3.11`
- `tensorflow=2.20.0`
- `keras=3.13.2`
- `scikit-learn`
- Jupyter Lab
- CPU-only (no GPU)

## References

For a complete list of references, please consult the **Bibliography** section of the corresponding project report.

This project was conducted as part of the **Project: Computer Vision** course at IU International University of Applied Sciences.

---
*Author: [Mariana Del Pozo Patrón](https://github.com/mardelpozo)*
