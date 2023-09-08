# :robot: CNN :car:

## :arrow_forward: Introduction
Image recongnitions represent a tough process even for human brain. This project consists of different models that will try to classify images in 7 possible classes.

## :memo: Description
It can be quite challenging due to its complexity. Image classifications is often used for multiple applications such as object recognition, cybersecurity, medicine or autonomous vehicles.
Deep learning is a subset of machine learning, which is essentially a neural network with three or more layers. These neural networks attempt to simulate the behavior of the human brain.

The evaluation measure is the classification accuracy computed on the test set. The classification accuracy is given by the number of correctly predicted labels divided by the total number of test samples.


## :computer: Implementation
Dataset consists of training data and test data. The submission file contains two columns: id and label. The id is the file name of an image sample. The label is the class label, 0 to 7, predicted for the corresponding data sample.
Models are imported from sklearn and from tensorflow.keras.models
Accuracy can be interogated using confusion matrics 
For more accuracy, the CNN model implements EarlyStopping imported from keras.callbacks
Routes are designed using Flask and represents a defined way of communication with endpoint 

## :exclamation: Instructions
The application can be started using 'flask --app main run' command




