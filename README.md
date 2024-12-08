AAI 521 Final Project: Image Enhancement Tasks with Deep Learning
Overview

This project explores four key computer vision tasks—denoising, super-resolution, colorization, and inpainting—using the CIFAR-100 dataset and advanced deep learning techniques. Each task is optimized to enhance specific image characteristics, leveraging tailored models to address unique challenges. Key metrics such as Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index Measure (SSIM) were used to evaluate performance.

The repository contains preprocessing, fine-tuning, and evaluation notebooks for a comprehensive end-to-end pipeline, from data preparation to model performance analysis.
Repository Structure

├── AAI_521_Project_Preprocessing.ipynb    # Preprocesses CIFAR-100 data for all tasks

├── AAI_521_Project_Fine_Tuning.ipynb      # Fine-tunes models for each task

├── AAI_521_Project_Evaluation_Metrics.ipynb # Evaluates models using various metrics

├── README.md                              # Overview of the project

Tasks
1. Denoising

    Removes noise from images to improve visual quality.
    Metric Highlights:
        Consistency: Most stable across the dataset.
        Achieved PSNR: ~20.42 dB
        SSIM: 0.658

2. Super-Resolution

    Upscales low-resolution images to higher resolutions.
    Metric Highlights:
        Achieved PSNR: ~21.94 dB
        SSIM: 0.815
        High variability in performance.

3. Colorization

    Converts grayscale images to color.
    Metric Highlights:
        Achieved the highest SSIM: ~0.91
        Instances of perfect reconstruction.

4. Inpainting

    Fills missing parts in images seamlessly.
    Metric Highlights:
        PSNR: ~18.73 dB
        SSIM: 0.844
        Most effective at seamless blending.

Methodology

    Preprocessing:
        Implemented in AAI_521_Project_Preprocessing.ipynb.
        Tasks-specific augmentations:
            Noise addition for denoising.
            Downsampling and upsampling for super-resolution.
            Grayscale conversion for colorization.
            Random masks for inpainting.

    Model Training:
        Implemented in AAI_521_Project_Fine_Tuning.ipynb.
        Used task-specific models tailored to enhance image characteristics.

    Evaluation:
        Conducted in AAI_521_Project_Evaluation_Metrics.ipynb.
        Metrics include PSNR, SSIM, MSE, and MAE.
        Statistical significance tests and confidence intervals provide additional insights.

Key Insights

    Colorization demonstrated the highest structural similarity, excelling in perceptual quality.
    Denoising was the most consistent across metrics.
    Inpainting achieved perfect reconstruction in specific cases.
    Super-resolution displayed variability, indicating opportunities for optimization.

Results Visualization

    Confidence Intervals:
        Confidence intervals for PSNR, SSIM, MSE, and MAE highlight significant differences between tasks.

    Metric Correlations:
        Heatmaps visualize correlations between PSNR, SSIM, MSE, and MAE for each task.

    Distributions:
        Violin plots illustrate metric distributions across tasks.

How to Run

    Clone this repository:

    git clone https://github.com/cptfudgemonkey/AAI_521_Final_Project.git

    Open and run the notebooks in the following order:
        AAI_521_Project_Preprocessing.ipynb
        AAI_521_Project_Fine_Tuning.ipynb
        AAI_521_Project_Evaluation_Metrics.ipynb
    Ensure all dependencies are installed (e.g., tensorflow, torch, opencv-python, etc.).

Future Work

    Incorporate advanced model architectures (e.g., transformers for super-resolution).
    Expand the dataset to include more challenging real-world scenarios.
    Optimize models further for computational efficiency.

Acknowledgments

This project is part of the AAI 521 (Introduction to Computer Vision) course on advanced AI applications. The CIFAR-100 dataset was used for all experiments.
