"""Create the input data pipeline using `tf.data`"""

import tensorflow as tf

from dataset.imagenetvid.imagenet_dataset import get_next_training_batch
from dataset.imagenetvid.imagenet_dataset import get_next_train_eval_batch
from dataset.imagenetvid.imagenet_dataset import get_next_val_eval_batch


def imagenet_train_input_fn(params):
    """Train input function for the ImageNet VID dataset.

    Args:
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """
    dataset = tf.data.Dataset().from_generator(
            get_next_training_batch,
            output_types=(tf.float32, tf.uint8),
            output_shapes=(tf.TensorShape([None, 255, 255, 3]), tf.TensorShape([None])))
    transform_fn_ = lambda image_batch, label_batch: transform_fn(
        image_batch, label_batch, transform=params.preprocess)
    dataset = dataset.map(transform_fn_, num_parallel_calls=params.prefetch_threads)
    dataset = dataset.repeat(params.num_epochs)  # repeat for multiple epochs
    dataset = dataset.prefetch(params.prefetch_threads)  # make sure you always a few batches ready to serve
    return dataset.make_one_shot_iterator().get_next()


def imagenet_train_eval_input_fn(params):
    dataset = tf.data.Dataset().from_generator(
            get_next_train_eval_batch,
            output_types=(tf.float32, tf.uint8),
            output_shapes=(tf.TensorShape([None, 255, 255, 3]), tf.TensorShape([None])))
    transform_fn_ = lambda image_batch, label_batch: transform_fn(
        image_batch, label_batch, transform=params.preprocess)
    dataset = dataset.map(transform_fn_, num_parallel_calls=params.prefetch_threads)
    dataset = dataset.repeat(params.num_epochs)
    dataset = dataset.prefetch(params.prefetch_threads)
    return dataset.make_one_shot_iterator().get_next()


def imagenet_val_eval_input_fn(params):
    dataset = tf.data.Dataset().from_generator(
            get_next_val_eval_batch,
            output_types=(tf.float32, tf.uint8),
            output_shapes=(tf.TensorShape([None, 255, 255, 3]), tf.TensorShape([None])))
    transform_fn_ = lambda image_batch, label_batch: transform_fn(
        image_batch, label_batch, transform=params.preprocess)
    dataset = dataset.map(transform_fn_, num_parallel_calls=params.prefetch_threads)
    dataset = dataset.repeat(params.num_epochs)
    dataset = dataset.prefetch(params.prefetch_threads)
    return dataset.make_one_shot_iterator().get_next()


def transform_fn(image_batch, label_batch, transform=False):
    def img_transform(images):
        # Normalize from [0, 255] to [0.0, 1.0]
        images = tf.cast(images, tf.float32)
        return images / 255.0

    if transform:
        images = img_transform(image_batch)
    return images, label_batch
