import time
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.datasets import fashion_mnist
from dnn import SequentialNetwork

if __name__ == "__main__":
    
    # Load & preprocessing
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0

    # Build
    model = SequentialNetwork(dropout=0.2, hidden_layers=3)

    # Train & evaluate
    model.train(
        train_images,
        train_labels, 
        test_images,
        test_labels,
        n_epochs=10, 
        batch_size=64, 
        validation_split=0.1, 
        one_cycle=True, 
        early_stopping=True
    )
    