"""Define the model."""

import functools

import tensorflow as tf
slim = tf.contrib.slim

from model.triplet_loss import batch_all_triplet_loss
from model.triplet_loss import batch_hard_triplet_loss

from model.embeddings.inception_resnet import inception_resnet_v2_arg_scope
from model.embeddings.inception_resnet import inception_resnet_v2_custom

from model.triplet_accuracy import calculate_accuracy, calculate_easier_accuracy


def build_model(is_training, images, params):
    """Compute outputs of the model (embeddings for triplet loss).

    Args:
        is_training: (bool) whether we are training or not
        images: (dict) contains the inputs of the graph (features)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) hyperparameters

    Returns:
        output: (tf.Tensor) output of the model
    """
    out = images
    # Define the number of channels of each convolution
    # For each block, we do: 3x3 conv -> batch norm -> relu -> 2x2 maxpool
    num_channels = params.num_channels
    bn_momentum = params.bn_momentum
    channels = [num_channels, num_channels * 2]
    for i, c in enumerate(channels):
        with tf.variable_scope('block_{}'.format(i+1)):
            out = tf.layers.conv2d(out, c, 3, padding='same')
            if params.use_batch_norm:
                out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
            out = tf.nn.relu(out)
            out = tf.layers.max_pooling2d(out, 2, 2)

    # assert out.shape[1:] == [7, 7, num_channels * 2], out.shape[1:]
    #
    # out = tf.reshape(out, [-1, 7 * 7 * num_channels * 2])
    out = tf.contrib.layers.flatten(out)
    with tf.variable_scope('fc_1'):
        out = tf.layers.dense(out, params.embedding_size)

    return out

def build_inception_resnet_model(is_training, images, params):
    """
    Compute outputs of the model (embeddings for triplet loss) using the
    Inception Resnet V2 model.

    Args:
        is_training: (bool) whether we are training or not
        images: (dict) contains the inputs of the graph (features)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) hyperparameters

    Returns:
        output: (tf.Tensor) output of the model
    """
    @functools.wraps(inception_resnet_v2_custom)
    def embedding_fn(images, is_training):
      with slim.arg_scope(inception_resnet_v2_arg_scope()):
        return inception_resnet_v2_custom(images, is_training=is_training)

    logits, embeddings, _ = embedding_fn(images, is_training)

    embeddings = tf.squeeze(embeddings)

    # print("Logits SIZE: {0}".format(logits.get_shape()))
    # print("Embeds SIZE: {0}".format(embeddings.get_shape()))
    # exit(1)

    # print(images.get_shape())
    # exit(1)

    # exclude = ['InceptionResnetV2/Logits', 'InceptionResnetV2/AuxLogits']
    # variables_to_restore = slim.get_variables_to_restore(exclude = exclude)

    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())

    # if params.pretrained_model != "" and not is_training:
    #     exclude = ['InceptionResnetV2/Logits', 'InceptionResnetV2/AuxLogits']
    #     variables_to_restore = slim.get_variables_to_restore(exclude = exclude)
    #     with tf.Session() as sess:
    #         saver = tf.train.Saver(variables_to_restore)
    #         saver.restore(sess, params.pretrained_model)

    return embeddings

def model_fn(features, labels, mode, params):
    """Model function for tf.estimator

    Args:
        features: input batch of images
        labels: labels of the images
        mode: can be one of tf.estimator.ModeKeys.{TRAIN, EVAL, PREDICT}
        params: contains hyperparameters of the model (ex: `params.learning_rate`)

    Returns:
        model_spec: tf.estimator.EstimatorSpec object
    """
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    images = features
    images = tf.reshape(images, [-1, params.image_size, params.image_size, params.image_channels])
    assert images.shape[1:] == [params.image_size, params.image_size, params.image_channels], "{}".format(images.shape)

    # -----------------------------------------------------------
    # MODEL: define the layers of the model
    # with tf.variable_scope('model'):
    # Compute the embeddings with the model
    if params.model_type == 'basic':
        embeddings = build_model(is_training, images, params)
    elif params.model_type == 'inception_resnet_v2':
        embeddings = build_inception_resnet_model(is_training, images, params)
    else:
        raise ValueError("Invalid model type: {0}".format(model_type))

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'embeddings': embeddings}
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    labels = tf.cast(labels, tf.int64)

    # Define triplet loss
    if params.triplet_strategy == "batch_all":
        loss, fraction = batch_all_triplet_loss(labels, embeddings, margin=params.margin,
                                                squared=params.squared)
    elif params.triplet_strategy == "batch_hard":
        loss = batch_hard_triplet_loss(labels, embeddings, margin=params.margin,
                                       squared=params.squared)
    elif params.triplet_strategy == "tf_batch_semi_hard":
        normalized_embeddings = tf.nn.l2_normalize(embeddings, axis=0)
        loss = tf.contrib.losses.metric_learning.triplet_semihard_loss(labels,
                                normalized_embeddings, margin=params.margin)
    else:
        raise ValueError("Triplet strategy not recognized: {}".format(params.triplet_strategy))

    # -----------------------------------------------------------
    # METRICS AND SUMMARIES
    # Batch metrics
    with tf.variable_scope("batch_metrics"):
        tf.summary.scalar('batch_loss', loss)
        if params.triplet_strategy == "batch_all":
            tf.summary.scalar('fraction_positive_triplets', fraction)

    # Metrics for evaluation using tf.metrics (average over whole dataset)
    with tf.variable_scope("metrics"):
        mean_loss, mean_loss_op = tf.metrics.mean(loss)
        tf.summary.scalar('loss', mean_loss)

    if mode == tf.estimator.ModeKeys.EVAL:
        embedding_mean_norm = tf.reduce_mean(tf.norm(embeddings, axis=1))
        eval_metric_ops = {
            "embedding_mean_norm": tf.metrics.mean(embedding_mean_norm),
            "mean_loss": (mean_loss, mean_loss_op)
        }

        if params.triplet_strategy == "batch_all":
            eval_metric_ops['fraction_positive_triplets'] = tf.metrics.mean(fraction)

        if params.model_type == 'inception_resnet_v2' and params.dataset == 'imagenetvid':
            accuracy = calculate_accuracy(embeddings, labels)
            eval_metric_ops["accuracy_mean"] = tf.metrics.mean(accuracy)
            easier_accuracy = calculate_easier_accuracy(embeddings, labels)
            eval_metric_ops["easier_accuracy_mean"] = tf.metrics.mean(easier_accuracy)

        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)

    # Summaries for training
    tf.summary.image('train_image', images, max_outputs=1)

    # Define training step that minimizes the loss with the Adam optimizer
    optimizer = tf.train.AdamOptimizer(params.learning_rate)
    global_step = tf.train.get_global_step()
    if params.use_batch_norm:
        # Add a dependency to update the moving mean and variance for batch normalization
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op = optimizer.minimize(loss, global_step=global_step)
    else:
        train_op = optimizer.minimize(loss, global_step=global_step)

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
