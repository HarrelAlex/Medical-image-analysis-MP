## Methodology

CAD is fundamentally based on highly complex pattern recognition. X-ray or other types of images are scanned for suspicious structures. 
Normally a few thousand images are required to optimize the algorithm. 
Digital image data are copied to a CAD server in a DICOM-format and are prepared and analyzed in several steps.

-> Data Acquisition:
* Gather medical images relevant to your task (e.g., MRI scans for disease detection).

-> Data Preprocessing:
* Preprocess the images (e.g., noise reduction, normalization, resizing) to ensure consistency and improve training efficiency.

-> Select Pre-trained Model:
* Choose a pre-trained CNN model on a large image dataset like ImageNet. (Common choices include VGG16, ResNet50, etc.)

-> Model Architecture:

1. Import pre-trained model: Load the pre-trained model architecture, excluding the final classification layers.
2. Freeze pre-trained layers: Set the weights (parameters) of the pre-trained layers to non-trainable (frozen) to prevent them from changing during training. These layers act as the feature extractor.
3. Add new layers: Design and add new fully-connected layers specific to your classification task (e.g., disease vs. healthy tissue).

-> Data Splitting:

* Divide your preprocessed medical image data into training, validation, and (optional) testing sets.
* The training set is used to train the model, the validation set is used to monitor performance and prevent overfitting, and the testing set (if used) is for final evaluation after training.

-> Transfer Learning Training:
* Train the modified CNN model on the training set:
* The model propagates the input image through the frozen pre-trained layers (feature extractor).
* New features specific to your task are extracted in the newly added layers.
* The final layers perform classification based on the learned features.
* Use the validation set to monitor training progress and adjust hyperparameters (learning rate, optimizer) if needed to prevent overfitting.

-> Evaluation:
* Evaluate the trained model's performance on the held-out testing set (if available) or a separate validation set.
* Assess metrics like accuracy, precision, recall, and F1 score to measure the model's effectiveness for your specific task.

-> (Optional) Fine-tuning:
* If performance is not satisfactory, consider fine-tuning the pre-trained layers by allowing them to update their weights slightly during further training. This can be helpful when your task is closely related to the original training data of the pre-trained model.
-> (Optional) Save Model:
* If satisfied with the model's performance, save the trained model for future use in image classification or prediction tasks.


