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
