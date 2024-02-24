# IMPORTING LIBRARIES

# Data manipulation and analysis
import numpy as np
import pandas as pd

# For data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# For deep learning
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

# For efficient performance by the model 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# For model evaluation
from sklearn.metrics import classification_report, confusion_matrix

# Display plots in the notebook (when we use jupyter Notebook)
#% matplotlib inline

# LOADING AND PREPROCESSING OF DATASET

# Loading CIFAR-10 image dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

print(train_images)

# Let's check the shape of the data
print(train_images.shape)
print(test_images.shape)


print(train_labels)
print(test_labels)

# Let's check the shape of the labels
print(train_labels.shape)
print(test_labels.shape)

# Normalizing the images in the dataset
train_images, test_images = train_images / 255.0, test_images / 255.0

# Now let's see training and test images in matrices form
print(train_images)

# Verify the shape of the loaded data
print("Training images shape:", train_images.shape)
print("Training labels shape:", train_labels.shape)
print("Testing images shape:", test_images.shape)
print("Testing labels shape:", test_labels.shape)


# VISUALIZATION OF THE DATA

# Define class names for CIFAR-10
class_names = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Function to display images with labels
def plot_images(images, labels, class_names, num_images=25):
    plt.figure(figsize=(15, 10))
    for i in range(num_images):
        plt.subplot(5, 5, i + 1)
        plt.imshow(images[i])
        plt.title(class_names[labels[i][0]])
        plt.axis('off')
    plt.show()

# Display sample training images
plot_images(train_images, train_labels, class_names)


# This is the function used to visualize the Class distribution
def plot_class_distribution(labels, dataset_type):
    plt.figure(figsize=(10, 5))
    sns.countplot(x=labels.flatten(), palette='viridis')
    plt.title(f'Class Distribution in {dataset_type} Data')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(ticks=np.arange(10), labels=class_names)
    plt.show()

# Checking class distribution of the training data...
plot_class_distribution(train_labels, 'Training')

# Checking the class distibution for the testing data...
plot_class_distribution(test_labels, 'Testing')

# FEATURE ENGINEERING - ONE-HOT ENCODING

# Let us check the train_labels and test_labels one more time...
print(f'Training labels : {train_labels}')
print(f'Testing labels : {test_labels}')

# Performing one-hot encoding
train_labels_one_hot = to_categorical(train_labels, num_classes=10)
test_labels_one_hot = to_categorical(test_labels, num_classes=10)

# Now let's look the the data...
print("One-hot encoded training labels :", train_labels_one_hot)
print("One-hot encoded testing labels :", test_labels_one_hot)

# Let's see the changes before and after encoding in detail...
print("Training labels shape:", train_labels.shape)
print("Testing labels shape:", test_labels.shape)
print("One-hot encoded training labels shape:", train_labels_one_hot.shape)
print("One-hot encoded testing labels shape:", test_labels_one_hot.shape)

# DEEP LEARNING MODEL : CNN

# Firstly let's create a CNN model
model = Sequential()

# Convolutional layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(1, 1))
model.add(Dropout(0.25))

# Flatten the Output of the convolutional layers
model.add(Flatten())

# Fully Connected Layer
model.add(Dense(512, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(10 ,activation = 'softmax'))

# adding learning rate scheduling and compiling the model

from tensorflow.keras.optimizers.legacy import Adam
METRICS = ['accuracy',tf.keras.metrics.Precision(name='precision'),tf.keras.metrics.Recall(name='recall')]
model.compile(optimizer=Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=METRICS)

# Let's look into our model..
model.summary()

# Define callbacks
checkpoint = ModelCheckpoint("best_model.h5", save_best_only=True)
early_stopping = EarlyStopping(patience=5, restore_best_weights=True)

# To make model more robust and efficient we will use data augumentation technique..
# Use data augmentation for training
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

# And finally model fitting...
epochs = 30
batch_size = 256

history = model.fit(
    datagen.flow(train_images, train_labels_one_hot, batch_size=batch_size),
    steps_per_epoch=len(train_images) // batch_size,
    epochs=epochs,
    validation_data=(test_images, test_labels_one_hot),
    callbacks=[checkpoint, early_stopping]
)

# PERFORMANCE VALIDATION

# Plotting accuracy
plt.figure(figsize=(20, 22))
plt.subplot(4, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Train and Validation Accuracy')
plt.legend()

# Plotting loss
plt.subplot(4, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train and Validation Loss')
plt.legend()

# Plotting precision
plt.subplot(4, 2, 3)
plt.plot(history.history['precision'], label='Train Precision')
plt.plot(history.history['val_precision'], label='Validation Precision')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.title('Train and Validation Precision')
plt.legend()

# Plotting recall
plt.subplot(4, 2, 4)
plt.plot(history.history['recall'], label='Train Recall')
plt.plot(history.history['val_recall'], label='Validation Recall')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.title('Train and Validation Recall')
plt.legend()

plt.show()

# MODEL EVALUATION
evaluation = model.evaluate(test_images, test_labels_one_hot)

# Let's calculate Accurracy and Loss...
print(f"\nTest Accuracy: {evaluation[1]*100:.2f}%")
print(f"Test Loss: {evaluation[0]:.4f}")

# Let's predict against the test_set
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

# For analysis and evaluation purposes let's convert labels (one-hot encoded data ) into original...
true_labels = np.argmax(test_labels_one_hot, axis=1)

# Create classification report for more understanding of model performance...
print("\nClassification Report:")
print(classification_report(true_labels, predicted_labels, target_names=class_names))

# Let's understand the model performance with Confusion Matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)
print("\nConfusion Matrix:")
print(conf_matrix)

# Let's plot this(Confusion_matrix) using graph...

def plot_confusion_matrix(conf_matrix, class_names):
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

plot_confusion_matrix(conf_matrix, class_names)


# PREDICTION

import random

def display_predictions(images, true_labels, class_names, model):
    plt.figure(figsize=(25, 20))
    for i in range(len(images)):
        # Expand the dimensions of the image before prediction
        img = np.expand_dims(images[i], axis=0)
        # Get the predicted probabilities for each class
        predictions = model.predict(img)
        # Get the index of the class with the highest probability
        predicted_label_index = np.argmax(predictions)
        
        plt.subplot(5,5, i+1)
        plt.imshow(images[i])
        plt.title(f"True: {class_names[true_labels[i]]}\nPredicted: {class_names[predicted_label_index]}")
        plt.axis('off')
    plt.show()

# Randomly select 25 images from the test set
random_indices = random.sample(range(len(test_images)), 25)
sample_images = [test_images[i] for i in random_indices]
sample_true_labels = [true_labels[i] for i in random_indices]

# Display the Predictions
display_predictions(sample_images, sample_true_labels, class_names, model)

# SAVING THE CREATED MODEL

from tensorflow.keras.models import load_model

model.save('ImageClassification_.keras')




