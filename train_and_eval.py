import tensorflow as tf
from keras_deeplab_v3 import model
from tensorflow import keras
import pathlib
import numpy as np
import os
# from model import Deeplabv3 
from datetime import datetime
import argparse
from tensorflow.python.ops import confusion_matrix
import keras.backend as K
parser = argparse.ArgumentParser(description='Optional app description')

AUTOTUNE = tf.data.experimental.AUTOTUNE
# tf.compat.v1.enable_eager_execution()

parser.add_argument('--batch_size', type=int,
                    help='Batch size for training')

parser.add_argument('--buffer_size', type=int,
                    help='Buffer size for training')                    

parser.add_argument('--train_dataset_dir',
                    help='Where train dataset reside.')

parser.add_argument('--test_dataset_dir',
                    help='Where test dataset reside.')

parser.add_argument('--image_height', type=int,
                    help='Image height')

parser.add_argument('--image_width', type=int,
                    help='Image width')  

parser.add_argument('--num_classes',type=int,
                    help='Number of classes')   

parser.add_argument('--checkpoint_dir',
                    help='Checkpoint directory') 

parser.add_argument('--log_dir',
                    help='Logs directory')

parser.add_argument('--num_of_epochs', type=int,
                    help='Number of epochs') 

parser.add_argument('--checkpoint_freq', type=int,
                    help='Frequency of checkpoints') 

parser.add_argument('--num_samples', type=int,
                    help='Number of samples')                                        

parser.add_argument('--weight_dir',
                    help='Weight directory')

parser.add_argument('--num_val_steps', type=int,
                    help='Number of validation steps')    

parser.add_argument('--val_freq', type=int,
                    help='Number of validation steps')

args = parser.parse_args()

print(tf.__version__)

LABELS_CLASS = 'labels_class'
IMAGE = 'image'
HEIGHT = 'height'
WIDTH = 'width'
IMAGE_NAME = 'image_name'
LABEL = 'label'
ORIGINAL_IMAGE = 'original_image'

BUFFER_SIZE = args.buffer_size
BATCH_SIZE = args.batch_size
TRAIN_DATASET_DIR = args.train_dataset_dir
TEST_DATASET_DIR = args.test_dataset_dir
IMAGE_HEIGHT = args.image_height
IMAGE_WIDTH = args.image_width 
NUM_CLASSES = args.num_classes
CHECKPOINT_DIR = args.checkpoint_dir
LOG_DIR = args.log_dir
EPOCHS = args.num_of_epochs
NUM_SAMPLES = args.num_samples
STEPS_PER_EPOCH = NUM_SAMPLES//BATCH_SIZE
CHECKPOINT_FREQ = args.checkpoint_freq
WEIGHT_DIR = args.weight_dir
NUM_VAL_STEPS = args.num_val_steps
VAL_FREQ = args.val_freq

def get_all_files(file_path):
    """Gets all the files to read data from.
    Returns:
      A list of input files.
    """
    image_data_dir = pathlib.Path(file_path)
    image_list_ds = tf.data.Dataset.list_files(str(image_data_dir/'*'))
    return image_list_ds

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
    input_image = tf.image.resize(dataset['image'], (IMAGE_HEIGHT, IMAGE_WIDTH),
                                    method=tf.compat.v2.image.ResizeMethod.NEAREST_NEIGHBOR)                           
    input_mask = tf.image.resize(dataset['labels_class'], (IMAGE_HEIGHT, IMAGE_WIDTH),
                                    method=tf.compat.v2.image.ResizeMethod.NEAREST_NEIGHBOR)
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    input_image, input_mask = normalize(input_image, input_mask)
    return input_image, input_mask

def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask = input_mask
    return input_image, input_mask

def meanIOU(true_mask, pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = tf.reshape(tf.squeeze(pred_mask), [-1])
    true_mask = tf.reshape(tf.squeeze(true_mask), [-1])

    confusion_mat = confusion_matrix.confusion_matrix(true_mask, pred_mask, 14, weights=None)

    sum_row = tf.reduce_sum(confusion_mat, 0) 
    sum_col = tf.reduce_sum(confusion_mat, 1)

    true_pos = tf.matrix_diag_part(confusion_mat)
    denominator = tf.cast(sum_row + sum_col - true_pos, tf.float32)
    num_valid_entries = tf.reduce_sum(tf.cast(tf.math.not_equal(denominator, 0), 'float32'))
    
    iou = tf.math.divide_no_nan(tf.cast(true_pos, tf.float32), denominator)
    mIOU = tf.math.divide_no_nan(
        tf.reduce_sum(iou, name='mean_iou'), num_valid_entries) 
    return mIOU

train_files = get_all_files(TRAIN_DATASET_DIR)
train_dataset = tf.data.TFRecordDataset(train_files)
train_dataset = train_dataset.map(parse_function)
train_dataset = train_dataset.map(load_image_train)
train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

test_files = get_all_files(TEST_DATASET_DIR)
test_dataset = tf.data.TFRecordDataset(test_files)
test_dataset = test_dataset.map(parse_function)
test_dataset = test_dataset.map(load_image_train)
test_dataset = test_dataset.batch(1)
test_dataset = test_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

deeplab_model = model.Deeplabv3(input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3), classes=NUM_CLASSES) 
deeplab_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                            metrics=['accuracy', meanIOU])

if WEIGHT_DIR != None:
  deeplab_model.load_weights(WEIGHT_DIR)

# For saving checkpoints of the trained model
checkpoint_path = CHECKPOINT_DIR+"cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(CHECKPOINT_DIR)    

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1, period=CHECKPOINT_FREQ)

# For logging the training in TensorBoard
logdir= LOG_DIR + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)   

model_history = deeplab_model.fit(train_dataset, epochs=EPOCHS,
                        steps_per_epoch=STEPS_PER_EPOCH, callbacks=[cp_callback, tensorboard_callback],
                        validation_data=test_dataset, validation_steps=NUM_VAL_STEPS, validation_freq=VAL_FREQ)

# deeplab_model.save('/Users/manasbhardwaj/Desktop/MasterThesis/code/keras_deeplab/keras-deeplab-v3-plus/saved_models/') 


