# Cardiac Arrhythmias Classification using Photoplethysmographic Signals

## Premise
This project was not initially organized on GitHub. As a result, this README does not include instructions on how to use the code, since such documentation was not originally created during the project's development. However, it thoroughly explains all the decisions made and the results obtained throughout the study.

## Overview

Cardiac arrhythmias, particularly premature atrial contractions (PACs) and premature ventricular contractions (PVCs), are common irregular heartbeats that can serve as early indicators of more serious cardiac conditions. PACs are premature heartbeats originating from the atria, while PVCs arise from the ventricles. Although often benign, frequent PACs and PVCs can lead to more serious arrhythmias, such as atrial fibrillation or ventricular tachycardia, making their detection and monitoring critical.

Traditionally, electrocardiography (ECG) has been the gold standard for detecting PACs and PVCs due to its ability to capture detailed electrical activity of the heart [3]. However, photoplethysmography (PPG) advent as a non-invasive, cost-effective alternative has opened new possibilities for continuous cardiac monitoring in various settings, thanks to the widespread distribution of wearable devices.

### Goal of the Project

The goal of this study is to explore the efficacy of PPG in detecting PACs and PVCs, utilizing both traditional machine learning models and advanced deep learning techniques. The study first addresses a binary classification problem to distinguish between normal and abnormal beats. Following this, a three-class classification is conducted to differentiate among normal beats, PACs, and PVCs, aiming to enhance the precision of arrhythmia detection.

## Dataset

The dataset employed for this task comprises recordings from 105 subjects, each associated with a PPG signal, annotated beat peak positions, and corresponding labels. In particular, the labels are denoted as:
- "N", normal beats
- "S", premature atrial contractions (PAC)
- "V", premature ventricular contractions (PVC)

The PPG recordings exhibited non-uniform sampling frequencies, with: 
- 62 recordings sampled at 128 Hz
- 43 recordings sampled at 256 Hz
All recordings were subject to high-frequency noise, likely due to motion artifacts from the subjects.

Moreover, the dataset presented a significant class imbalance between beats types:
- 92.8% as "N"
- 3.9% as "S"
- 3.3% as "V"

## Data Preprocessing

Different steps were followed in the data preprocessing.

### Exclusion of Subjects with Exclusively "N" Beats

To mitigate class imbalance and reduce redundancy, subjects exhibiting only "N" type beats were excluded from the dataset. This resulted in the removal of 14 signals, all of which were recorded at a sampling rate of 250 Hz.

### Downsampling of 250 Hz Signals

To ensure consistency in the analysis, it was necessary to standardize the sampling frequencies of the signals. The decision was made to downsample the 250 Hz signals to 128 Hz, based on several considerations:
- Themajority of signals in the dataset were sampled at 128 Hz.
- A lower sampling rate reduces the volume of data, resulting in faster processing times andlower computational costs.
- Reduced data size contributes to more efficientstorage utilization.
- A sampling frequency of 128 Hz is sufficient tocapture all essential physiological information.
Before downsampling, the Fast Fourier Transform (FFT) of each 250 Hz signal was analyzed to confirm the absence of relevant frequency components above 64 Hz, in accordance with the Nyquist-Shannon sampling theorem.

### Signal Filtering

Bandpass Butterworth filtering was then applied to all the signals. The analysis of the FFT of the signals previously performed, also helped to determine the cut-off frequencies of the filter. In order to remove the DC component of the signal, a high pass cut-off of 0.5 Hz was chosen. Instead, considering the low frequency content of the signals, a 5 Hz low pass cut-off was chosen.

### Signal Normalization

To standardize the PPG signals across different subjects, a Z-score normalization was applied. This method adjusts each signal to have a mean of zero and a standard deviation of one, ensuring that the data is centered and scaled uniformly.

### Beat Segmentation

Individual beats were extracted using a dynamic windowing technique centered around systolic peaks. This method adapts window size based on inter-peak intervals, accounting for heart rate variability and ensuring complete waveform capture. After preprocessing, 91 signals remained, with a final distribution of 91.8% normal beats, 4.5% PACs, and 3.7% PVCs, totaling 215,350 beats.

## Feature Extraction

To effectively classify beats, 28 features from each signal were extracted, and were grouped into four main categories:
- Temporal Features: Metrics such as peak amplitude, rise time, and fall time.
- Morphological Features: Structural characteristics of the beat, including amplitude range, root mean square (RMS), and energy.
- Statistical Features: Measures like skewness, kurtosis, and entropy to quantify signal distribution.
- Frequency Features: Derived from the power spectral density and FFT components.

## Outlier Removal and Feature Selection

To enhance the reliability of the dataset, we applied the Z-score method to identify and remove outliers based on key features such as peak amplitude, frequency components, and entropy. Any beat with a Z-score exceeding a predefined threshold was considered an outlier and removed. This step helped eliminate anomalies that could skew model performance and ensured a more balanced distribution of beats across categories.

Furthermore, after studying the correlation between features, several features were found to be highly correlated with others, which could introduce redundancy and reduce the effectiveness of the model. As a result, the following features were removed from the final dataset: beat duration, band power, rms, range amplitude, and std amplitude.

## Classification Approaches

Two primary classification tasks were explored:
- Binary Classification, where the goal was to distinguish between Normal and Abnormal beats
- Three-Class Classification, where the goal was to distinguish between Normal, PAC, and PVC beats
For these tasks, both machine learning and deep learning models were tested.

### Binary Classification
For this task, multiple classification models were tested:
- Machine Learning Models: KNN, Decision Tree, Naive Bayes, Logistic Regression, and XGBoost
- Deep Learning Model: a custom Feedforward Neural Network (FFNN) incorporating dropout and batch normalization layers to improve robustness
Among these models, the FFNN delivered the best results, outperforming traditional approaches in terms of precision and recall

### Three-Class Classification
For multi-class classification, a similar approach was employed:
- Machine Learning Models: The same models used for binary classification.
- Deep Learning Model: A hybrid model combining an LSTM network with feature-based inputs, designed to capture both temporal and morphological characteristics of the beats.
This task proved more challenging, particularly in distinguishing supraventricular beats (S). To improve classification accuracy, class weights were manually adjusted, leading to better results.

## Evaluation Metrics
Given the diagnostic context in which these models will be applied, a custom evaluation metric was developed to balance the detection of abnormal beats while minimizing false positives. The metric incorporates:
- Precision for Normal Beats: The function first calculates the precision for class 0. A low precision would indicate a high number of false negatives, meaning that many abnormal beats are incorrectly classified as normal
- Recall for Abnormal Beats: Then, it calculates the recall for class 1 in binary classification, and for both class 1 and class 2 in the three class scenario. In this context, recall measures the model’s ability to correctly identify abnormal beats
- Threshold and Penalty Mechanism: If the precision for normal beats falls below a specified
threshold, the recall for abnormal beats is penalized. This penalty is set to the precision of class 0, but can be customized
- Output: The function returns the recall for abnormal beats, adjusted by the penalty if necessary. In the three-class classification problem, the metric returns the average recall between `class 1` and `class 2`

```python

from sklearn.metrics import precision_score, recall_score

def custom_metric_binary(y_true, y_pred, precision_threshold=0.9, penalty=None):
    """
    Custom metric for binary classification.
    """
    
    # Calculate precision for class 0 (Normal) and recall for class 1 (Abnormal)
    precision_class_0 = precision_score(y_true, y_pred, pos_label=0)
    recall_class_1 = recall_score(y_true, y_pred, pos_label=1)

    # If the precision for 'Normal' beats is below the threshold, apply a penalty to recall
    if precision_class_0 < precision_threshold:
        if penalty is None:
            penalty = precision_class_0
        return round(recall_class_1 * penalty, 2)

    # Otherwise, return the recall for 'Abnormal' beats
    return recall_class_1
```

```python
from sklearn.metrics import precision_score, recall_score

def custom_metric_threeclass(y_true, y_pred, precision_threshold=0.9, penalties=None):
    
    Custom metric for three-class classification. Balances precision for the Normal class
    with recall for the Supraventricular and Ventricular classes.
    
    # Calculate precision and recall for each class
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    
    # Set default penalties if not provided
    if penalties is None:
        penalties = [precision[0], recall[1], recall[2]]
    
    # Initialize weighted score
    score = 0
    
    # Penalize based on thresholds
    if precision[0] < precision_threshold:
        score += recall[1] * penalties[0] # Supraventricular recall penalized
        score += recall[2] * penalties[0] # Ventricular recall penalized
    else:
        score += recall[1]  # Supraventricular recall unpenalized
        score += recall[2]  # Ventricular recall unpenalized
    
    # Average score for the two abnormal classes
    return round(score / 2, 2)
```

## Results
Detailed results, including model performance metrics and analysis, are thoroughly presented in the Final Report available.

## Future Improvements
There are several areas for potential improvement that could enhance the effectiveness of the classification approach:
- Balancing the Dataset More Effectively: one of the main shortcomings in the preprocessing phase was the decision to upsample the minority classes without applying a corresponding downsampling of the majority class (Normal beats). Implementing a more balanced approach could improve model generalization
- Exploring Non-Linear Characteristics: investigating non-linear signal properties may provide deeper insights into distinguishing between different types of beats
- Leveraging Time-Frequency Representations: techniques such as wavelet transforms could be explored to capture additional distinguishing features that may not be evident in traditional feature extraction methods
- Further Hyperparameter Tuning: 0ptimizing the deep learning architectures through more extensive hyperparameter tuning could lead to performance improvements and better model generalization


