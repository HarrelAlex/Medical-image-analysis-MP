# Prognosis Using Medical Imaging (PMI)

### IDEA:

Develop a system that analyzes medical images to assist in diagnosing conditions. Users could upload medical scans, and the system could provide insights or connect them with relevant medical resources.

## Steps:

### 1. Problem definition: 

   Web app that takes Mri, X-ray scans as input and train a model that can predict the disease and provide resources relating to  it

### 2. Data Collection:

   Acquire a diverse and representative dataset for your chosen medical imaging task. Ensure that the dataset includes sufficient examples of both normal and abnormal cases.

### 3. Data Preprocessing:

   Clean and preprocess the data. Common preprocessing steps include resizing images, normalizing pixel values, handling missing data, and applying data augmentation to increase the dataset size.

### 4. Model Selection/Transfer Learning:

   Choose a machine learning model suitable for your task. For medical image analysis, convolutional neural networks (CNNs) are often preferred due to their ability to capture spatial features.

   Consider leveraging pre-trained models for medical image analysis tasks. Pre-trained models on large datasets can be fine-tuned for your specific task, saving time and resources.

### 5. Data Splitting:

   Divide your dataset into training, validation, and testing sets. The training set is used to train the model, the validation set helps tune hyperparameters, and the testing set assesses the model's performance on unseen data.

### 6. Feature Extraction and Model Training:

   Extract relevant features from medical images, especially if using traditional machine learning models. For deep learning models like CNNs, feature extraction is typically performed automatically during training.

   Train your chosen model using the training dataset. Adjust hyperparameters based on the model's performance on the validation set.

### 7. Model Evaluation:

   Assess the model's performance on the testing set using appropriate evaluation metrics (e.g., accuracy, precision, recall, F1 score). Consider using a confusion matrix for a detailed analysis.

### 8. Interpreting  and Providing Insights: 

   Interpret the decisions made by your model. Provide necessary medical resources or indicate danger level as interpreted by the model. Can also provide various treatment methods and advices if necessary 

## Framework

1.Image scanning 

2.Reverse image search 

3.Prediction generation

4.Model training

5.Simple UI
