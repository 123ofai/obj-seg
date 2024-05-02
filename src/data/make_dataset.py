# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import tensorflow_datasets as tfds
import tensorflow as tf

def normalize_img(data):
    """Normalizes images: `uint8` -> `float32`."""
    image = data['image']
    mask = data['segmentation_mask']
    image = tf.image.resize(image, [128, 128])
    mask = tf.image.resize(mask, [128, 128], method='nearest')
    image = tf.cast(image, tf.float32) / 255.0
    mask = tf.cast(mask-1, tf.float32)
    return image, mask

def preprocess_dataset(dataset, batch_size, isTrain=1):
    """
    Divide dataset into train & test
    Cache, Batch, Shuffle, Pre-fetch for each
    Processed sizes:
        Image: (batch_size, h, w, 3), float32, {0, 1}
        Mask: (batch_size, h, w, 1), float32, [0, 1, 2]
    """
    dataset = dataset.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.cache()
    if isTrain:
        dataset = dataset.shuffle(1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset
   
def create_dataset():
    dataset = tfds.load('oxford_iiit_pet', shuffle_files=True)
    train_dataset = preprocess_dataset(dataset, batch_size=128, isTrain=1)
    test_dataset = preprocess_dataset(dataset, batch_size=128)
    return train_dataset, test_dataset
