import tensorflow as tf
from tensorflow import keras
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import os
from keras_deeplab_v3 import model
from datetime import datetime  
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops import math_ops
import keras.backend as K

AUTOTUNE = tf.data.experimental.AUTOTUNE
tf.enable_eager_execution()

import argparse
parser = argparse.ArgumentParser(description='Optional app description')

parser.add_argument('--dataset_dir',
                    help='Reads data from TFRecords file')

parser.add_argument('--checkpoint_path',
                    help='Path to stored checkpoint of the DeepLab trained model.')

parser.add_argument('--num_of_images', type=int,
                    help='Mention number of images for inference.') 

parser.add_argument('--image_height', type=int,
                    help='Image height')

parser.add_argument('--image_width', type=int,
                    help='Image width')  

args = parser.parse_args()

DATASET_DIR = args.dataset_dir
CHECKPOINT_PATH = args.checkpoint_path
NUM_OF_IMAGES = args.num_of_images
IMAGE_HEIGHT = args.image_height
IMAGE_WIDTH = args.image_width 

# Dataset names.
_SFD = 'sfd'

# Max number of entries in the colormap for each dataset.
_DATASET_MAX_ENTRIES = {
    _SFD: 14
}

LABELS_CLASS = 'labels_class'
IMAGE = 'image'
HEIGHT = 'height'
WIDTH = 'width'
IMAGE_NAME = 'image_name'
LABEL = 'label'
ORIGINAL_IMAGE = 'original_image'

print(tf.__version__)

def get_all_files(file_path):
    """Gets all the files to read data from.
    Returns:
      A list of input files.
    """
    image_data_dir = pathlib.Path(file_path)
    image_list_ds = tf.data.Dataset.list_files(str(image_data_dir/'*'))
    return image_list_ds

files = get_all_files(DATASET_DIR)
# print(files)

def parse_function(example_proto):
    """Function to parse the example proto.
    Args:
      example_proto: Proto in the format of tf.Example.
    Returns:
      A dictionary with parsed image, label, height, width and image name.
    Raises:
      ValueError: Label is of wrong shape.
    """

    # Currently only supports jpeg and png.
    # Need to use this logic because the shape is not known for
    # tf.image.decode_image and we rely on this info to
    # extend label if necessary.
    def _decode_image(content, channels):
      return tf.cond(
          tf.image.is_jpeg(content),
          lambda: tf.image.decode_jpeg(content, channels),
          lambda: tf.image.decode_png(content, channels))

    features = {
        'image/encoded':
            tf.io.FixedLenFeature((), tf.string, default_value=''),
        'image/filename':
            tf.io.FixedLenFeature((), tf.string, default_value=''),
        'image/format':
            tf.io.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/height':
            tf.io.FixedLenFeature((), tf.int64, default_value=0),
        'image/width':
            tf.io.FixedLenFeature((), tf.int64, default_value=0),
        'image/segmentation/class/encoded':
            tf.io.FixedLenFeature((), tf.string, default_value=''),
        'image/segmentation/class/format':
            tf.io.FixedLenFeature((), tf.string, default_value='png'),
    }

    parsed_features = tf.io.parse_single_example(example_proto, features)

    image = _decode_image(parsed_features['image/encoded'], channels=3)

    label = _decode_image(
          parsed_features['image/segmentation/class/encoded'], channels=1)

    image_name = parsed_features['image/filename']
    if image_name is None:
      image_name = tf.constant('')

    sample = {
        IMAGE: image,
        IMAGE_NAME: image_name,
        HEIGHT: parsed_features['image/height'],
        WIDTH: parsed_features['image/width'],
    }

    if label is not None:
      if label.get_shape().ndims == 2:
        label = tf.expand_dims(label, 2)
      elif label.get_shape().ndims == 3 and label.shape.dims[2] == 1:
        pass
      else:
        raise ValueError('Input label shape must be [height, width], or '
                         '[height, width, 1].')

      label.set_shape([None, None, 1])
      sample[LABELS_CLASS] = label

    return sample


def load_image_train(dataset):
    input_image = tf.image.resize(dataset['image'], (IMAGE_HEIGHT, IMAGE_WIDTH), method=tf.compat.v2.image.ResizeMethod.NEAREST_NEIGHBOR)
    input_mask = tf.image.resize(dataset['labels_class'], (IMAGE_HEIGHT, IMAGE_WIDTH), method=tf.compat.v2.image.ResizeMethod.NEAREST_NEIGHBOR)
    input_image, input_mask = normalize(input_image, input_mask)
    return input_image, input_mask

def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask = input_mask
    return input_image, input_mask

def display(display_list):
    plt.figure(figsize=(15, 15))
    title = ['Input Image', 'True Mask', 'Predicted Mask']
    for i in range(len(display_list)):
      plt.subplot(1, len(display_list), i+1)
      plt.title(title[i])
      plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
      plt.axis('off')     
    plt.show() 

def create_label_colormap(dataset=_SFD):
    """Creates a label colormap for the specified dataset.
    Args:
      dataset: The colormap used in the dataset.
    Returns:
      A numpy array of the dataset colormap.
    """
    if dataset == _SFD:
      return create_sfd_label_colormap()
    else:
      raise ValueError('Unsupported dataset.')    

def label_to_color_mask(label, dataset=_SFD):
    """Adds color defined by the dataset colormap to the label.
    Args:
      label: A 2D array with integer type, storing the segmentation label.
      dataset: The colormap used in the dataset.
    Returns:
      result: A 2D array with floating type. The element of the array
        is the color indexed by the corresponding element in the input label
        to the dataset color map.
    """
    if label.ndim != 2:
      raise ValueError('Expect 2-D input label. Got {}'.format(label.shape))

    if np.max(label) >= _DATASET_MAX_ENTRIES[dataset]:
      raise ValueError(
          'label value too large: {} >= {}.'.format(
              np.max(label), _DATASET_MAX_ENTRIES[dataset]))

    colormap = create_label_colormap(dataset)
    return colormap[label]   


def create_sfd_label_colormap():
    """Creates a label colormap used in SFD segmentation.
    Returns:
      A colormap for visualizing segmentation results.
    """
    return np.asarray([
      [0, 0, 0],
      [120, 120, 120],
      [180, 120, 120],
      [6, 230, 230],
      [80, 50, 50],
      [4, 200, 3],
      [120, 120, 0],
      [140, 140, 140],
      [204, 5, 255],
      [230, 230, 230],
      [4, 250, 7],
      [112, 9, 255],
      [235, 255, 7],
      [150, 5, 61]
    ])

def create_pred_mask(pred_mask): 
    pred_mask = tf.argmax(pred_mask, axis=-1)
    if pred_mask.ndim != 2:
      pred_mask = np.squeeze(pred_mask)
    pred_mask = label_to_color_mask(pred_mask)
    pred_mask = tf.expand_dims(pred_mask, 0)
    return pred_mask[0]

def create_true_mask(true_mask):
    true_mask = np.squeeze(true_mask)
    true_mask = label_to_color_mask(true_mask)
    true_mask = tf.expand_dims(true_mask, 0)
    return true_mask[0]

def show_predictions(dataset=None, num=1):
    if dataset:
      for image, true_mask in dataset.take(num):  
        pred_mask = model.predict(image) 
        display([image[0], create_true_mask(true_mask), create_pred_mask(pred_mask)])

def meanIOU(true_mask, pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = tf.reshape(tf.squeeze(pred_mask), [-1])
    true_mask = tf.reshape(tf.squeeze(true_mask), [-1])

    confusion_mat = confusion_matrix.confusion_matrix(
        true_mask,
        pred_mask,
        14,
        weights=None)

    sum_row = tf.reduce_sum(confusion_mat, 0) 
    sum_col = tf.reduce_sum(confusion_mat, 1)

    true_pos = tf.matrix_diag_part(confusion_mat)
    denominator = tf.cast(sum_row + sum_col - true_pos, tf.float32)
    num_valid_entries = tf.reduce_sum(tf.cast(tf.math.not_equal(denominator, 0), 'float32'))
    
    iou = tf.math.divide_no_nan(tf.cast(true_pos, tf.float32), denominator)
    mIOU = tf.math.divide_no_nan(
        tf.reduce_sum(iou, name='mean_iou'), num_valid_entries) 
    return mIOU

dataset = tf.data.TFRecordDataset(files)
dataset = dataset.map(parse_function)
dataset = dataset.map(load_image_train)
test_dataset = dataset.batch(1)
test_dataset = test_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

model = model.Deeplabv3(input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3), classes=14)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy', meanIOU])

model.load_weights(CHECKPOINT_PATH)
loss, acc, mIOU  = model.evaluate(test_dataset, verbose=2, steps=2)
# print("Restored model, accuracy: {:5.2f}%".format(100*acc))
show_predictions(test_dataset, NUM_OF_IMAGES)