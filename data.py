import tensorflow as tf
from tensorflow import keras

import numpy as np
import os
import cv2
import csv
import sys

from tensorflow.keras.preprocessing.image import ImageDataGenerator


def process_image_file(filepath, size):
    img = cv2.imread(filepath)
    img = cv2.resize(img, size)
    return img


_augmentation_transform = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    brightness_range=(0.9, 1.1),
    zoom_range=(0.85, 1.15),
    fill_mode='constant',
    cval=0.,
)

def apply_augmentation(img):
    img = _augmentation_transform.random_transform(img)
    return img

def _process_csv_file(file):
    with open(file, 'r') as fr:
        files = fr.readlines()
    return files

class Skin_Lesion_Dataset_Balanced(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(
            self,
            data_dir="",
            csv_file="",
            is_training=True,
            batch_size=8,
            malignant_percent=0.65,
            input_shape=(224, 224),
            n_classes=2,
            num_channels=3,
            mapping={
                'benign': 0,
                'malignant': 1
            },
            shuffle=True,
            augmentation=apply_augmentation
    ):
        'Initialization'
        self.datadir = data_dir
        self.dataset = _process_csv_file(csv_file)
        self.is_training = is_training
        self.batch_size = batch_size
        self.N = len(self.dataset)
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.num_channels = num_channels
        self.mapping = mapping
        self.shuffle = True
        self.malignant_percent = malignant_percent
        self.n = 0
        self.augmentation = augmentation

        # datasets[0] is benign, dataset[1] is malignant
        self.datasets = [[], []]
        for sample in self.dataset:
            self.datasets[int(sample.split(',')[1])].append(sample)

        print(len(self.datasets[0]), len(self.datasets[1]))

        self.on_epoch_end()

    def __next__(self):
        # Get one batch of data
        batch_x, batch_y = self.__getitem__(self.n)
        # Batch index
        self.n += 1

        # If we have processed the entire dataset then
        if self.n >= self.__len__():
            self.on_epoch_end()
            self.n = 0

        return batch_x, batch_y

    def __len__(self):
        return int(np.ceil(len(self.datasets[0]) / float(self.batch_size)))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            for v in self.datasets:
                np.random.shuffle(v)

    def __getitem__(self, idx):
        batch_x, batch_y = np.zeros(
            (self.batch_size, self.input_shape[0], self.input_shape[1],
             self.num_channels)), np.zeros(self.batch_size)

        batch_files = self.datasets[0][idx * self.batch_size:(idx + 1) *
                                       self.batch_size]

        # upsample malignant cases
        malignant_size = max(int(len(batch_files) * self.malignant_percent), 1)
        malignant_inds = np.random.choice(np.arange(len(batch_files)),
                                      size=malignant_size,
                                      replace=False)
        malignant_files = np.random.choice(self.datasets[1],
                                       size=malignant_size,
                                       replace=False)
        for i in range(malignant_size):
            batch_files[malignant_inds[i]] = malignant_files[i]

        # The mean and std of the entire ISIC dataset
        global_mean = [0.52559245, 0.56309465, 0.71815116]
        global_std = [0.12765675, 0.10001463, 0.105755]

        for i in range(len(batch_files)):
            sample = batch_files[i].split(',')

            x = process_image_file(os.path.join(self.datadir, sample[0]), self.input_shape)
            x = x.astype('float32')/255.0
            # zero center
            b_chan, g_chan, r_chan = cv2.split(x)
            b_chan = (b_chan - global_mean[0])/global_std[0]
            g_chan = (g_chan - global_mean[1])/global_std[1]
            r_chan = (r_chan - global_mean[2])/global_std[2]
            x = cv2.merge((b_chan, g_chan, r_chan))

            if self.is_training and hasattr(self, 'augmentation'):
                x = self.augmentation(x)

            x = x.astype('float32') / 255.0
            y = sample[1]

            batch_x[i] = x
            batch_y[i] = y

        return batch_x, keras.utils.to_categorical(batch_y, num_classes=self.n_classes) 
