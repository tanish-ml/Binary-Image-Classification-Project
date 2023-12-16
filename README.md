
# Binary Image Classification Project

## Overview

This repository contains a binary image classification project using TensorFlow and Keras. The goal is to classify images as either a "cat" or a "dog." The convolutional neural network (CNN) is trained on a dataset of images and tested on a separate dataset.

## Prerequisites

Before running the code, make sure you have the required dependencies installed. You can install them using the following command:

```bash
pip install tensorflow keras
```

## Usage

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/binary-image-classification.git
    ```

2. Navigate to the project directory:

    ```bash
    cd binary-image-classification
    ```

3. Run the image classification script:

    ```bash
    python image_classification.py
    ```

## Dataset

The project uses two datasets: a training set and a test set. Images are preprocessed using the `ImageDataGenerator` from Keras to enhance the model's robustness.
You can use your own dataset to train the model. or email me for the dataset I used

## CNN Architecture

The CNN architecture consists of convolutional layers with max-pooling, followed by fully connected layers. The model is compiled with the Adam optimizer and binary crossentropy loss.

```python
# ... [previous code]
cnn.summary()
```

## Training

The model is trained for 25 epochs on the training set, with checkpoints to save the best model weights.

```python
# ... [previous code]
filename = 'cnn.h5'
checkpoint = ModelCheckpoint(filename, save_best_only=True)
cnn.fit(x=training_set, validation_data=test_set, epochs=25, callbacks=[checkpoint])
```

## Testing

After training, the model is tested on a sample image to demonstrate its predictions.

```python
# ... [previous code]
result = cnn.predict(test_image / 255.0)
training_set.class_indices
# ... [previous code]
