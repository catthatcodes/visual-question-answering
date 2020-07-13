# Visual Question Answering
Application of Natural Language Processing in Computer Vision. Uses the easy-vqa dataset to train a VQA model that combines image and text features to answer open-ended questions.


## Use Case

Question answering (QA) is a computer science discipline within the fields of information retrieval and natural language processing (NLP), which is concerned with building systems that automatically answer questions posed by humans in a natural language. VQA is the application of QA to images, combining visual and language understanding. Given an image, the system should be able answer open-ended questions regarding the image. The model first extracts features from both the entered question and the image and then compares those to result in a cogent answer. The question is generally posed to comprise of three sections: type of question, subject and context. The type refers to ‘what’ and ‘whether’ questions. The image features are compared against the subject and the context to derive an answer. The usability of such a model poses great advancements in forensics, education. Some of the current research is to aid the blind and visually impaired.


## Solution

1. Build - Build the image and text model which uses their respective features. Merge both image and question vectors, so the reusltant now contains information derived from both.

2. Setup - Extract the training and testing set for the images and the questions and process them seperately to extarct their seperate features.

3. Train - This model is trained on the easy-vqa dataset.

4. Analyse -  The model is the tested to analyze it's performance.
  

## Tools and Frameworks Used

Data Set: 
  1. easy-vqa - Easy Visual Questions Answering dataset
  
Model Development:
  1. TensorFlow – Deep Learning Library
  
  2. Keras – Deep Learning Library
  
  3. NumPy – Scientific numerical calculations library
  
  4. Scikit-learn – Machine learning algorithms tools

Development Environment:
  1. PyCharm IDE – Python program development environment

  2. Jupyter Notebooks – web application for interactive data science and scientific computing 

Libraries Used:
  1. numpy
  
  2. keras

  3. easy_vqa


## easy-vqa Dataset

The dataset consits of:

1. train images = 4000

2. test images = 1000

3. train questions = 38575

4. test questions = 9673

5. possibles answers = 13

6. training questions that are yes/no = 28407

7. testing questions that are yes/no = 7136


Sample Images: 

   <img src="https://victorzhou.com/media/vqa-post/examples.png" width ="500" height ="500"/>


Sample Questions:

1. "What shape is blue?"

2. "Does the image contain a square?"

3. "What is the color of the triangle?"


Possible Answers:

1. Yes/No: Yes, No

2. Shapes: Cicrle, Rectangle, Triangle

3. Colors: Red, Green, Blue, Black, Gray, Teal, Brown, Yellow


## Image Model

A Convolutional Neural Network(CNN) is used to extract information from the input images. Since the dataset relatively lacks complexity, a simple model is designed as follows:

   <img src="https://victorzhou.com/media/vqa-post/cnn.svg" width ="700" height ="450"/>

1. image size is set to 64x64

2. Convolutional layer with eight 3x3 filters using "same" padding. Resultant volume is 64x64x8.

3. Standard max pooling layer. Resultant volume is 32x32x16.

4. Convolutional layer with sixteen filters. Resultant volume is 32x32x16.

5. Standard max pooling layer. Resultant volume is 16x16x16.

6. FLatten layer. Resultant layers has 4096 nodes.


## Question Model

   <<img src="https://victorzhou.com/media/vqa-post/feedforward.svg" width ="700" height ="250"/>

1. vectorize every question using Bag of Words approach(BOW).

2. Input the above feature vector to a standard neural network consisting of 2 fully connected layers.


## Merged Model

Using element wise multiplication as available in the Merge layer in keras, the image and question vecotrs are combined together.

1. Multiply Layer

2. Softmax to turn output values into probabilities so each answer can be quantified.

## Result

      Epoch 1/8
      loss: 0.8887 - accuracy: 0.6480 - val_loss: 0.7504 - val_accuracy: 0.6838
      Epoch 2/8
      loss: 0.7443 - accuracy: 0.6864 - val_loss: 0.7118 - val_accuracy: 0.7095
      Epoch 3/8
      loss: 0.6419 - accuracy: 0.7468 - val_loss: 0.5659 - val_accuracy: 0.7780
      Epoch 4/8
      loss: 0.5140 - accuracy: 0.7981 - val_loss: 0.4720 - val_accuracy: 0.8138
      Epoch 5/8
      loss: 0.4155 - accuracy: 0.8320 - val_loss: 0.3938 - val_accuracy: 0.8392
      Epoch 6/8
      loss: 0.3078 - accuracy: 0.8775 - val_loss: 0.3139 - val_accuracy: 0.8762
      Epoch 7/8
      loss: 0.1982 - accuracy: 0.9286 - val_loss: 0.2202 - val_accuracy: 0.9212
      Epoch 8/8
      loss: 0.1157 - accuracy: 0.9627 - val_loss: 0.1883 - val_accuracy: 0.9378
      
      
