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
    transform_fn_ = lambda imgs, lbls: transform_fn(imgs, lbls, params)
    dataset = dataset.map(transform_fn_, num_parallel_calls=params.prefetch_threads)
    dataset = dataset.repeat(params.num_epochs)
    dataset = dataset.prefetch(params.prefetch_threads)
    return dataset.make_one_shot_iterator().get_next()


def imagenet_train_eval_input_fn(params):
    dataset = tf.data.Dataset().from_generator(
            get_next_train_eval_batch,
            output_types=(tf.float32, tf.uint8),
            output_shapes=(tf.TensorShape([None, 255, 255, 3]), tf.TensorShape([None])))
    transform_fn_ = lambda imgs, lbls: transform_fn(imgs, lbls, params)
    dataset = dataset.map(transform_fn_, num_parallel_calls=params.prefetch_threads)
    dataset = dataset.repeat(params.num_epochs)
    dataset = dataset.prefetch(params.prefetch_threads)
    return dataset.make_one_shot_iterator().get_next()


def imagenet_val_eval_input_fn(params):
    dataset = tf.data.Dataset().from_generator(
            get_next_val_eval_batch,
            output_types=(tf.float32, tf.uint8),
            output_shapes=(tf.TensorShape([None, 255, 255, 3]), tf.TensorShape([None])))
    transform_fn_ = lambda imgs, lbls: transform_fn(imgs, lbls, params)
    dataset = dataset.map(transform_fn_, num_parallel_calls=params.prefetch_threads)
    dataset = dataset.repeat(params.num_epochs)
    dataset = dataset.prefetch(params.prefetch_threads)
    return dataset.make_one_shot_iterator().get_next()


def transform_fn(image_batch, label_batch, params):
    def img_transform(images):
        # Resize images
        new_size = tf.constant([params.image_size, params.image_size], tf.int32)
        images = tf.image.resize_images(images, new_size)

        # Normalize from [0, 255] to [-1.0, 1.0]
        images = tf.cast(images, tf.float32)
        return 2 * (images / 255.0) - 1.0

    if params.preprocess:
        image_batch = img_transform(image_batch)
    return image_batch, label_batch
