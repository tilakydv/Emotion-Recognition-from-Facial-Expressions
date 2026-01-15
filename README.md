Emotion Recognition from Facial Expressions
This project implements a custom Convolutional Neural Network (CNN) to classify human emotions from grayscale images. Using the FER-2013 dataset, the system categorizes facial expressions into seven distinct emotional states.
+1

📊 Dataset Overview
The model utilizes the FER-2013 dataset, which consists of 48x48 pixel grayscale images of faces.

Registration: Faces are automatically centered and occupy a consistent amount of space in each image.

Emotion Categories:

0: Angry

1: Disgust

2: Fear

3: Happy

4: Sad

5: Surprise

6: Neutral

Data Split:

Training Set: 28,709 examples.

Public Test Set: 3,589 examples.

🛠️ Tech Stack

Language: Python 


Deep Learning: TensorFlow, Keras 


Computer Vision: OpenCV 


Data Processing: NumPy, Matplotlib 

🚀 Technical Implementation
To achieve high model stability and performance, the following features were implemented:


CNN Architecture: A multi-layer Convolutional Neural Network built for feature extraction and classification.


Data Augmentation: Applied rotation, zooming, and horizontal flipping to enhance model generalization.

Model Optimization:


Early Stopping: Monitored validation loss to prevent overfitting.


Learning-Rate Scheduling: Dynamically adjusted the learning rate during training for better convergence.


Checkpointing: Saved the best-performing model weights during the training process.

📈 Evaluation
The project includes comprehensive visualization of the training process:

Accuracy & Loss Curves: Graphical representation of model performance over epochs.

Confusion Matrix: Detailed breakdown of classification performance across all seven categories.
