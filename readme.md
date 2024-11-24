# Potato Disease Classification (Intro to Machine Learning Project)


#### **Project Overview**
This project aims to classify potato leaf diseases using machine learning techniques, specifically Convolutional Neural Networks (CNNs). The dataset includes images of potato leaves categorized as healthy or affected by diseases like Early Blight and Late Blight. The goal of the project is to train a model capable of accurately classifying these leaf images into the appropriate disease categories.

#### **Dataset**
- **Source:** [PlantVillage Dataset on Kaggle](https://www.kaggle.com/arjuntejaswi/plant-village)

- **Categories:**
  - Healthy
  - Early Blight
  - Late Blight

---

#### **Key Features**
1. **Data Preprocessing:**
   - Resizing: All images are resized to 256x256 pixels to standardize the input size for the model.
   - Normalization: Image pixel values are scaled to a range of 0 to 1 for improved training.
   - Data Augmentation: Random flips and rotations are applied to the training data to increase model robustness.

2. **Model Architecture:**
   - The project uses a custom CNN, consisting of multiple convolutional and max-pooling layers to extract features from the images.
   - The model ends with a fully connected layer followed by a softmax activation function to classify the images into different categories.

3. **Model Training:**
   - The model is trained for 50 epochs using the Adam optimizer and sparse categorical cross-entropy loss function, optimizing for accuracy.
   - The training data is split into three subsets: 70% for training, 20% for validation, and 10% for testing.

4. **Evaluation:**
   - The performance of the model is evaluated based on accuracy and loss metrics.
   - Training and validation accuracy/loss are visualized using graphs to assess overfitting and underfitting.

5. **Inference:**
   - After training, the model is used to predict disease labels for test images.
   - The results show the actual label, predicted label, and prediction confidence for each test image.

---

#### **Setup Instructions**

1. **Install Dependencies:**
   Ensure you have Python 3.8 or higher installed, then install the required libraries by running:
   ```bash
   pip install tensorflow matplotlib
   ```

2. **Dataset Setup:**
   - Download the dataset from [Kaggle](https://www.kaggle.com/arjuntejaswi/plant-village).
   - After extracting the dataset, place it in the project directory with the following structure:
     ```
     /PlantVillage
       /Healthy
         image1.jpg
         image2.jpg
       /EarlyBlight
         image3.jpg
         image4.jpg
       /LateBlight
         image5.jpg
         image6.jpg
     ```

3. **Run the Notebook:**
   Open the `potato-disease-classification-model.ipynb` it is inside training folder file in Jupyter Notebook or Google Colab and run each cell in sequence.

---

#### **Project Results**
- The model successfully classifies potato leaf diseases into categories like healthy, early blight, and late blight.
- The training and validation accuracy/loss graphs show the model's learning progress over epochs.
- The model's performance on test data is evaluated, providing insight into its generalization capabilities.

---
#### **Future Improvements**
- **Transfer Learning:** Implement pre-trained models such as VGG16 or ResNet for better performance on a complex problem.
- **Hyperparameter Tuning:** Experiment with different batch sizes, learning rates, or layer configurations to improve accuracy.
- **Model Explainability:** Use techniques like Grad-CAM to visualize the areas of the images the model is focusing on when making predictions.

---

#### **Acknowledgments**
- Dataset from [Kaggle](https://www.kaggle.com/arjuntejaswi/plant-village).
- Machine Learning framework: TensorFlow and Keras.

---