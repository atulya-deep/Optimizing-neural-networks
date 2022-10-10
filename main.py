import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import regularizers
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout

# Get current working directory
current_dir = os.getcwd() 

# Append data/mnist.npz to the previous path to get the full path
data_path = os.path.join(current_dir, "/content/mnist.npz") 

# Get only training set
(training_images, training_labels), _ = tf.keras.datasets.mnist.load_data(path=data_path)

def reshape_and_normalize(images):
    
    ### START CODE HERE

    # Reshape the images to add an extra dimension
    images = images.reshape(60000, 28, 28, 1)
    
    # Normalize pixel values
    images = images/255
    
    ### END CODE HERE

    return images

# Reload the images in case you run this cell multiple times
(training_images, _), _ = tf.keras.datasets.mnist.load_data(path=data_path) 

# Apply your function
training_images = reshape_and_normalize(training_images)

print(f"Maximum pixel value after normalization: {np.max(training_images)}\n")
print(f"Shape of training set after reshaping: {training_images.shape}\n")
print(f"Shape of one image after reshaping: {training_images[0].shape}")

"""# **L2 regularization** AND DROP OUT"""

def convolutional_model():
    ### START CODE HERE

   
    mnist = tf.keras.datasets.mnist
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data()

    training_images=training_images.reshape(60000, 28, 28, 1)
    training_images=training_images / 255.0
    test_images = test_images.reshape(10000, 28, 28, 1)
    test_images=test_images/255.0
    model = tf.keras.models.Sequential([
      tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),
      tf.keras.layers.MaxPooling2D(2, 2),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(10,Dropout(0.2)),
      tf.keras.layers.Dense(128,kernel_regularizer=regularizers.l2(0.01),activation='relu'),

      tf.keras.layers.Dense(10,Dropout(0.2)),
      tf.keras.layers.Dense(10, activation='softmax')
        ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

"""**EARLY STOPPING**"""

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

model = convolutional_model()

# Train your model (this can take up to 5 minutes)
history = model.fit(training_images, training_labels, epochs=10,callbacks=[callback])
