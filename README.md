# TJS_Prediction

This repository contains the Pytorch implementation code for the paper "Data-Driven Approaches for Predicting Tommy John Surgery Risk in Major League Baseball Pitchers"

Using MLB pitching
data (2016–2023), the classification model detects injury risk up to 100 days in advance with high prediction performance of 0.73 F1-score, while the regression model estimates the time remaining until the player’s last pre-surgery game with R2 of 0.78. Additionally, for enhance our model’s applicability, we employ an Explainable AI (XAI) technique to analyze the impacting mechanical features, such as a lowered four-seam fastball release point, that accelerate UCL deterioration, increasing TJS risk. 

# Dependencies
CPU : Intel(R) Xeon(R) CPU @ 2.20 GHz.

GPU :  NVIDIA L4 GPU (22.5 GB memory)

CUDA : 12.5

Python : 3.11.11

pytorch : 2.5.1

# 1. Architectures
![github classification framework](https://github.com/user-attachments/assets/c85e18ea-cb82-4e6e-997a-af6faac398ce)

![github regression framework](https://github.com/user-attachments/assets/e876bd52-0027-453a-88b4-0f1ab5c8567e)
# 2. Preprocessing
In this study, we utilized MLB pitching data from the 2016 to 2023 seasons, characterized by the extensive use of ‘Trackman’ technology to capture detailed pitch metrics. The data was gathered through the Pybaseball [14] package in Python, resulting in a dataset of 5,537,981 pitches with 94 distinct attributes, including metrics such as pitch velocity, spin rate, release angle, and pitch type. The dataset includes only regular-season games.

For detailed information on preprocessing, please refer to section “3. Data Source” in the paper, and for the data preprocessed before training, see the “final_df.csv” file.

# 3. Classification Framework
The dataset used in this study consisted of multivariate time-series data, with each pitcher’s game schedule varying, leading to inherently irregular sampling intervals. While we resampled the data to regular intervals, some irregularities persisted due to differences in player availability and event frequency. To address the temporal nature of this data, we first employed well-established time-series models—LSTM, CNN-LSTM, and Transformer-Encoder. Specifically, LSTM targets long-term dependencies, CNN-LSTM combines local and temporal features, and the Transformer-Encoder’s parallel attention mechanism captures extended interactions within the pitching metrics. Considering the findings of prior research [33], which demonstrated that vision-based models can outperform traditional methods in handling irregularly sampled time-series data, we extended our approach to incorporate ResNet and ViT. For these models, we transformed the time-series data into single-channel images, with time on one axis and pitching metrics on the other. This transformation allows the vision models to interpret the data as a two-dimensional ‘feature-time space,’ enabling them to capture complex cross-feature interactions and subtle temporal patterns that may be difficult to detect using purely sequential architectures. 

For all the models used in the classification task, please refer to the "Classification" file.

# 4. Regression Framework
Given that each player’s pitch data is irregularly sampled, we initially tested LSTM, CNN+LSTM, and Transformer for the regression task, using interpolation to handle missing time points (as in our classification approach). However, after 10 runs with different random seeds, none of these sequential models achieved an R2 above 0.1, indicating a fundamental limitation under our dataset conditions. We also considered vision-based architectures (ResNet, ViT) referenced from our classification framework, but deemed them unsuitable for many-to-many regression scenarios. Drawing on insights from Bai et al.[41], which highlight the stability of convolutional approaches for sequence modeling, we therefore adopted a single-channel 1D-CNN. Rather than modeling each player’s time series separately, this method aggregates all pre-injury data from injured players to capture generalizable patterns as the injury date approaches. By predicting the number of days remaining until injury, our 1D-CNN offers a more robust quantitative understanding of how subtle variations in pitching metrics can signal an impending injury event.

For the models used in the Regression Task, please refer to the model in the "Regression file."

# 5. Results on Classification
![image](https://github.com/user-attachments/assets/625ebd1e-2522-4530-85be-1f80b7e522b7)

# 6. Results on Regression
![image](https://github.com/user-attachments/assets/c65cee4e-73bd-43ce-99e3-ca1b2bac6750)




