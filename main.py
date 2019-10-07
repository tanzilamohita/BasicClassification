# importing Tenserflow and Keras

import tensorflow as tf
from tensorflow import keras

# helper libraries
import numpy as np
import matplotlib.pyplot as plt

# print(tf.__version__)

# importing data set
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# exploring data
print("Train Images Shape:", train_images.shape)
print("Training Set Labels:", len(train_labels))
print("Training Set Labels Integer value:", train_labels)
print("Test Images Shape:",test_images.shape)
print("Test Set Labels:", len(test_labels))

# Pre-processing Data

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(True)
#plt.show()

train_images = train_images / 255.0
test_images = test_images / 255.0

# verifying data into correct format

plt.figure(figsize=(10, 10))
for i in range(28):
    plt.subplot(6, 6, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.grid(False)
    plt.xlabel(class_names[train_labels[i]])
#plt.show()

# building the model
# tf.keras.layers.Flatten ->
# transforms the format of the images from a two-dimensional array (of 28 by 28 pixels)
# to a one-dimensional array (of 28 * 28 = 784 pixels).

# two tf.keras.layers.Dense layers ->
# These are densely connected, or fully connected, neural layers.
# The first Dense layer has 128 nodes (or neurons).
# The second (and last) layer is a 10-node softmax layer that returns an array of 10 probability scores
# that sum to 1.

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# compiling the model
# Optimizer —This is how the model is updated based on the data it sees and its loss function.
# Loss function —This measures how accurate the model is during training.
# Metrics —Used to monitor the training and testing steps.
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# training the model
model.fit(train_images, train_labels, epochs=10)

# Evaluating Accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest Accuracy:', test_acc)
print('\nTest Loss:', test_loss)

# making predictions
predictions = model.predict(test_images)
print(predictions[0])
print(np.argmax(predictions[0]))
print(test_labels[0])

# graph

def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array, true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')


# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()















