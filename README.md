# Smoking vs Non-Smoking Image Classification with CNN

This project aims to classify images into "smoking" and "non-smoking" categories using a Convolutional Neural Network (CNN). I start with a simple CNN model and later enhance the performance using the InceptionV3 architecture via transfer learning.

## Dataset

- **Source**: [Smoking Images Dataset on Kaggle](https://www.kaggle.com/datasets/sujaykapadnis/smoking)
- **Description**: The dataset contains labeled images of people smoking and not smoking. It’s used here for binary classification.

## Models Used

### 1. Basic CNN Model
A straightforward CNN architecture is implemented as a baseline model with:
- Convolutional layers followed by pooling layers
- Dense layers for classification

### 2. InceptionV3 Model
For improved accuracy, we apply transfer learning using the **InceptionV3** architecture, pre-trained on the ImageNet dataset. By fine-tuning this model on our data, we achieve better feature extraction and classification accuracy.

## Training Process

1. **Data Splitting**: The dataset is split into training, validation, and test sets.
2. **Data Augmentation**: Applied to the training dataset to improve model generalization.
3. **Model Training**: Each model is trained using the training and validation data, with early stopping to avoid overfitting.

## Evaluation and Results

- **Validation Accuracy**: The InceptionV3 model achieved higher validation accuracy compared to the basic CNN model, demonstrating the effectiveness of transfer learning.
- **Test Accuracy**: We use the test set for final evaluation, ensuring the model generalizes well to unseen data.

## Exporting the Model

To make the model more versatile, we save it in multiple formats:

1. **SavedModel**: TensorFlow's standard format for easy deployment.
2. **TF-Lite**: Optimized for mobile and embedded devices.
3. **TensorFlow.js**: For running in web applications.
