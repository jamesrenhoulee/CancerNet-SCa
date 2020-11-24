import tensorflow as tf
import numpy as np

import csv
import random


def _process_csv_file(file):
    with open(file, 'r') as fr:
        reader = csv.reader(fr)
        data = list(reader)
    return data


class ISIC_Dataset():
    def __init__(
        self,
        data_dir,
        csv_file,
        is_training=True,
        batch_size=4,
        input_shape=(224, 224),
        num_channels=3,
        balance_dataset=True,
        augmentation_prob=0.8,
        n_classes = 2,
        mapping={'0': 'benign', '1': 'malignant'},
        random_seed = 0):

        # Initialization
        self.datadir = data_dir
        self.dataset = _process_csv_file(csv_file)
        self.is_training = is_training
        self.batch_size = batch_size
        self.len_data = len(self.dataset)
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.num_channels = num_channels
        self.balance_dataset = balance_dataset
        self.augmentation_prob = augmentation_prob
        self.mapping = mapping
        self.random_seed = random_seed

        # Seperate into benign and malignant samples and print statistics
        datasets = {'benign': [], 'malignant': []}
        for sample in self.dataset:
            label = sample[1]
            datasets[mapping[label]].append(sample)
        self.datasets = datasets
        self.num_benign = len(self.datasets['benign'])
        self.num_malignant = len(self.datasets['malignant'])
        print('Number of benign samples: ', self.num_benign)
        print('Number of malignant samples: ', self.num_malignant)

    def parse_function_test(self, filename, label):
        img_decoded = tf.image.decode_jpeg(tf.io.read_file(self.datadir + filename), channels=self.num_channels)
        img = tf.image.resize_images(img_decoded, self.input_shape)
        img = tf.image.convert_image_dtype(img, dtype=tf.float32)/255.0
        label = tf.one_hot(tf.strings.to_number(label, out_type=tf.int32), self.n_classes)
        return {'image': img, 'label/one_hot': label}

    def parse_function_train(self, filename, label):
        # Include Random Augmentation when training
        img_decoded = tf.image.decode_jpeg(tf.io.read_file(self.datadir + filename), channels=self.num_channels)
        img = tf.image.resize_images(img_decoded, self.input_shape)
        img = tf.image.convert_image_dtype(img, dtype=tf.float32)/255.0
        label = tf.one_hot(tf.strings.to_number(label, out_type=tf.int32), self.n_classes)
        
        # Chance for augmentation
        if random.random() < self.augmentation_prob:
            which_aug = random.randint(0,3)
            if which_aug == 0:
                crop_size = round(self.input_shape[0]*0.9)
                img = tf.image.random_crop(img, [crop_size, crop_size, self.num_channels])
                img = tf.image.resize_images(img, self.input_shape)
            elif which_aug == 1:
                if random.random() < 0.5:
                    img = tf.image.random_flip_left_right(img)
                else:
                    img = tf.image.random_flip_up_down(img)
            elif which_aug == 2:
                img = tf.image.random_brightness(img, 0.1)
            elif which_aug == 3:
                # Rotation, +10 or -10 degrees
                degrees = random.random() * 20. - 10
                img = tf.contrib.image.rotate(img, degrees * math.pi / 180, interpolation='BILINEAR')
        
        return {'image': img, 'label/one_hot': label}


    def create_tf_dataset(self):
        # Balance dataset by undersampling benign
        if self.balance_dataset:
            np.random.seed(self.random_seed)
            undersample_to_num = min(self.num_benign, self.num_malignant)
            rand_indices = np.random.choice(np.arange(self.num_benign), size=undersample_to_num, replace=False)
            self.datasets['benign'] = np.take(self.datasets['benign'], rand_indices, axis=0)
        
        # Concatenate data back together and extract filenames and labels
        concat_data = np.concatenate((self.datasets['benign'], self.datasets['malignant']), axis=0)
        filenames = concat_data[:, 0]
        labels = concat_data[:, 1]

        # Create TF Datasets
        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
        if self.is_training:
            dataset = dataset.repeat()
            dataset = dataset.shuffle(5000)
            dataset = dataset.map(map_func=self.parse_function_train) # Apply augmentation to data
            dataset = dataset.batch(batch_size=self.batch_size)
        else:
            dataset = dataset.repeat()
            dataset = dataset.map(map_func=self.parse_function_test) # Process images and labels
            dataset = dataset.batch(batch_size=self.batch_size)
        
        return dataset