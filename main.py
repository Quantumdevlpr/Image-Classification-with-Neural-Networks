import cv2 as cv 
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
training_images , testing_images = training_images / 255 , testing_images / 255

class_names = ['Plane' , 'Car' , 'Bird' , 'Cat' , 'Deer' , 'Dog', 'Frog', 'Horse' , 'Ship' , 'Truck']


for i in range(16):
    plt.subplot(4, 4, i + 1) #this says we are taking 4*4 grid with each iteration we are choosing one image
    plt.xticks([]) #this removes the x ticks
    plt.yticks([]) #this removes the y ticks
    plt.imshow(training_images[i] ,cmap=plt.cm.binary) #this shows the image in binary color) 
    plt.xlabel(class_names[training_labels[i][0]])
 #this shows the label of the image
    
plt.show() #this displays the image 
training_images = training_images[:20000];
training_labels = training_labels[:20000];
testing_images = testing_images[:4000];
testing_labels = testing_labels[:4000];

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (32, 32, 3))) #this adds a convolutional layer with 32 filters of size 3*3 and relu activation function
model.add(layers.MaxPooling2D((2, 2))) #this adds a max pooling layer with size 2*2
model.add(layers.Conv2D(64, (3, 3), activation = 'relu')) #this adds a convolutional layer with 64 filters of size 3*3 and relu activation function
model.add(layers.MaxPooling2D((2, 2))) #this adds a max pooling layer with size 2*2
model.add(layers.Conv2D(64, (3, 3), activation = 'relu')) #this adds a convolutional layer with 64 filters of size 3*3 and relu activation function
model.add(layers.Flatten()) #this flattens the output of the previous layer
model.add(layers.Dense(64, activation = 'relu')) #this adds a dense layer with 64 neurons and relu activation function
model.add(layers.Dense(10, activation = 'softmax')) #this adds a dense layer with 10 neurons and softmax activation function
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy']) #this compiles the model with adam optimizer and sparse categorical crossentropy loss function


model.fit(training_images, training_labels, epochs = 10, validation_data = (testing_images , testing_labels)) #this fits the model on the training data with 10 epochs and batch size of 64 and validation split of 0.2

loss, accuracy = model.evaluate(testing_images, testing_labels) #this evaluates the model on the testing data
print('Loss: ', loss) #this prints the loss of the model
print('Accuracy: ', accuracy) #this prints the accuracy of the model


model.save('cifar10_model.h5') #this saves the model in h5 format
models.load_model('cifar10_model.h5') #this loads the model from h5 format
