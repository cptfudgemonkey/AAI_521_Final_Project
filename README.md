
# **AAI 521 Final Project: Image Enhancement Tasks with Deep Learning**

## **Overview**
This project focuses on enhancing images using a tailored U-Net architecture for four key computer vision tasks: **denoising**, **super-resolution**, **colorization**, and **inpainting**. The CIFAR-100 dataset is utilized for all experiments, and model performance is evaluated using **PSNR**, **SSIM**, **MSE**, and **MAE** metrics.

The repository includes preprocessing, fine-tuning, evaluation, and application scripts for an end-to-end pipeline. Due to file size limitations, the **preprocessed data** and **trained model weights** are hosted on Google Drive.

---

## **Table of Contents**
1. [Repository Structure](#repository-structure)
2. [Key Features](#key-features)
3. [Setup Instructions](#setup-instructions)
    - [Clone the Repository](#1-clone-the-repository)
    - [Install Dependencies](#2-install-dependencies)
    - [Access Shared Data and Models](#3-access-shared-data-and-models)
4. [Running the Project](#running-the-project)
    - [Step 1: Preprocess Data](#step-1-preprocess-data)
    - [Step 2: Fine-Tune Models](#step-2-fine-tune-models)
    - [Step 3: Evaluate Models](#step-3-evaluate-models)
    - [Step 4: Launch Application](#step-4-launch-application)
5. [Results](#results)
6. [Future Work](#future-work)
7. [Acknowledgments](#acknowledgments)

---

## **Repository Structure**
```
├── AAI_521_Project_Preprocessing.ipynb       # Preprocess CIFAR-100 dataset
├── AAI_521_Project_Fine_Tuning.ipynb         # Fine-tune models for each task
├── AAI_521_Project_Evaluation_Metrics.ipynb  # Evaluate models using metrics
├── AAI_521_Project_Application.ipynb         # Gradio application for real-time image enhancement
├── README.md                                 # Project overview and instructions
```

---

## **Key Features**
- **Denoising:** Removes noise from images.
- **Super-Resolution:** Enhances resolution of low-quality images.
- **Colorization:** Converts grayscale images to color.
- **Inpainting:** Fills missing regions in images.

---

## **Setup Instructions**

### 1. Clone the Repository
```bash
git clone https://github.com/cptfudgemonkey/AAI_521_Final_Project.git
cd AAI_521_Final_Project
```

### 2. Install Dependencies
Run the following command in your Python environment:
```bash
pip install torch torchvision opencv-python tqdm scikit-image matplotlib gradio
```

### 3. Access Shared Data and Models
- **Preprocessed Data:** [Download Preprocessed Data](https://drive.google.com/uc?id=1JgwTV5N7Lm0MKR5gdMAXLsroNtMff-K5)
- **Trained Models:** [Download Trained Models](https://drive.google.com/uc?id=1-XVnUaJBYYcH8nGQ-FA4Cg8NXo0yxTUe)

Save these files in the root directory and unzip them:
```bash
unzip preprocessed_data.zip -d preprocessed_data
unzip trained_models.zip -d trained_models
```

---

## **Running the Project**

### **Step 1: Preprocess Data**
Run the preprocessing notebook to prepare the CIFAR-100 dataset for all tasks:
```bash
jupyter notebook AAI_521_Project_Preprocessing.ipynb
```

### **Step 2: Fine-Tune Models**
Train and fine-tune the U-Net models for all tasks:
```bash
jupyter notebook AAI_521_Project_Fine_Tuning.ipynb
```

### **Step 3: Evaluate Models**
Evaluate the trained models and generate visualizations:
```bash
jupyter notebook AAI_521_Project_Evaluation_Metrics.ipynb
```

### **Step 4: Launch Application**
Run the Gradio-powered application for real-time image enhancement:
```bash
jupyter notebook AAI_521_Project_Application.ipynb
```

In the interface:
1. Select a task (e.g., "Denoising").
2. Upload an image.
3. View the enhanced output.

---

## **Results**
| Task               | PSNR   | SSIM   | MSE    | MAE    |
|--------------------|--------|--------|--------|--------|
| **Denoising**      | 20.42  | 0.6582 | 0.0091 | 0.0752 |
| **Super-Resolution** | 21.94  | 0.8154 | 0.0185 | 0.0487 |
| **Colorization**   | 23.71  | 0.9085 | 0.0103 | 0.0600 |
| **Inpainting**     | 18.73  | 0.8444 | 0.0179 | 0.0293 |

---

## **Future Work**
1. Incorporate advanced model architectures like Transformers.
2. Extend to larger datasets with real-world images.
3. Optimize for computational efficiency and scalability.

---

## **Acknowledgments**
This project was developed as part of the AAI 521 course (Introduction to Computer Vision). The CIFAR-100 dataset and Google Drive were used for data sharing and model storage.

---

