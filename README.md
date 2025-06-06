# pathloss-map-prediction-unet
# Pathloss Map Prediction with U-Net and Radial Sampling  This repository contains two Jupyter notebooks developed for the MLSP2025 pathloss map prediction challenge. The overall goal is to accurately estimate indoor radio signal propagation losses (pathloss maps) using deep learnnig
## Overview

- **Task 1: U-Net Model for Pathloss Map Prediction**  
  In Task_1_Participante.ipynb, a U-Net architecture is implemented in TensorFlow/Keras to train on raw, high-resolution pathloss image patches. This notebook covers:
  1. Data loading and preprocessing (reading images and ground-truth pathloss maps from Google Drive).
  2. Definition of the U-Net model (encoder/decoder blocks, skip connections).
  3. Training loop (with mean squared error loss, validation metrics, and early stopping callbacks).
  4. Evaluation on a held-out validation set and visualization of predicted vs. actual pathloss maps.
  5. Tips for adjusting batch size, modifying convolutional layers, and improving edge performance.

- **Task 2: Radial Sampling Method**  
  In Task_2_Participante.ipynb, a custom radial sampling strategy is applied to reduce the data’s spatial redundancy before feeding it to the model. This notebook covers:
  1. Definition of a radial sampling function that selects pixels along concentric circles (to reflect antenna radiation patterns).
  2. Generation of downsampled input-output pairs (features and ground truth) using the radial mask.
  3. Integration of the radial sampling into the U-Net training pipeline (showing how to load only sampled points instead of full grids).
  4. Comparison of performance between standard grid-based training and radial sampling.

## Results

- Training time per epoch is approximately 1–2 minutes (depending on GPU/CPU), with per-image validation at ~0.8 ms.  
- The chosen U-Net model achieved competitive MSE on the validation set (MSE ≈ … dB²), ranking in the top 10 % of participants.  
- Radial sampling reduces memory load and can improve convergence speed; further hyperparameter tuning (e.g., edge-aware loss functions or atrous convolutions) may yield even better edge accuracy.

## How to Use This Repository

1. **Clone this repo**  
   ```bash
   git clone https://github.com/<your-username>/pathloss-map-prediction.git
   cd pathloss-map-prediction
