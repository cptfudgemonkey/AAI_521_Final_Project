AAI 521 Final Project: Image Enhancement Tasks with Deep Learning
Overview

This project focuses on enhancing images using a tailored U-Net architecture for four key computer vision tasks: denoising, super-resolution, colorization, and inpainting. The CIFAR-100 dataset is utilized for all experiments, and model performance is evaluated using PSNR, SSIM, MSE, and MAE metrics.

The repository includes preprocessing, fine-tuning, evaluation, and application scripts for an end-to-end pipeline. Due to file size limitations, the preprocessed data and trained model weights are hosted on Google Drive.
Repository Structure

├── AAI_521_Project_Preprocessing.ipynb       # Preprocess CIFAR-100 dataset
├── AAI_521_Project_Fine_Tuning.ipynb         # Fine-tune models for each task
├── AAI_521_Project_Evaluation_Metrics.ipynb  # Evaluate models using metrics
├── AAI_521_Project_Application.ipynb         # Gradio application for real-time image enhancement
├── README.md                                 # Project overview and instructions

Key Features

    Denoising: Removes noise from images.
    Super-Resolution: Enhances resolution of low-quality images.
    Colorization: Converts grayscale images to color.
    Inpainting: Fills missing regions in images.

Setup Instructions
1. Clone the Repository

git clone https://github.com/cptfudgemonkey/AAI_521_Final_Project.git
cd AAI_521_Final_Project

2. Install Dependencies

Run the following command in your Python environment:

pip install torch torchvision opencv-python tqdm scikit-image matplotlib gradio

3. Access Shared Data and Models

    Preprocessed Data: Download preprocessed_data.zip
    Trained Models: Download trained_models.zip

Save these files in the root directory and unzip them:

unzip preprocessed_data.zip -d preprocessed_data
unzip trained_models.zip -d trained_models

Running the Project
Step 1: Preprocess Data

Run AAI_521_Project_Preprocessing.ipynb to preprocess the CIFAR-100 dataset for all tasks. This will normalize, augment, and organize the dataset for training and validation.
Step 2: Fine-Tune Models

Execute AAI_521_Project_Fine_Tuning.ipynb to train and fine-tune the U-Net models for the four tasks. Trained models are saved in the trained_models folder.
Step 3: Evaluate Models

Run AAI_521_Project_Evaluation_Metrics.ipynb to compute PSNR, SSIM, MSE, and MAE metrics for each task. Summary statistics, comparative analyses, and visualizations are generated.
Step 4: Launch Application

Use the Gradio-powered interface for real-time image enhancement:

jupyter notebook AAI_521_Project_Application.ipynb

Open the Gradio interface in the notebook and follow these steps:

    Select a task (e.g., "Denoising").
    Upload an image.
    View the enhanced output.

Results
Task	PSNR	SSIM	MSE	MAE
Denoising	20.42	0.6582	0.0091	0.0752
Super-Resolution	21.94	0.8154	0.0185	0.0487
Colorization	23.71	0.9085	0.0103	0.0600
Inpainting	18.73	0.8444	0.0179	0.0293
Future Work

    Incorporate advanced model architectures like Transformers.
    Extend to larger datasets with real-world images.
    Optimize for computational efficiency and scalability.

Acknowledgments

This project was developed as part of the AAI 521 course (Introduction to Computer Vision). The CIFAR-100 dataset and Google Drive were used for data sharing and model storage.
