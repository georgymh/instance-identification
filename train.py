"""Train the model"""

import argparse
import os

import tensorflow as tf

from dataset.mnist.input_fn import mnist_train_input_fn
from dataset.imagenetvid.input_fn import imagenet_train_input_fn
from model.model_fn import model_fn
from model.utils import Params


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/test',
                    help="Experiment directory containing params.json")


if __name__ == '__main__':
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Define the model
    tf.logging.info("Creating the model...")
    config = tf.estimator.RunConfig(tf_random_seed=230,
                                    model_dir=args.model_dir,
                                    save_summary_steps=params.save_summary_steps,
                                    save_checkpoints_steps=1000)
    estimator = tf.estimator.Estimator(model_fn, params=params, config=config)

    # Define the dataset
    tf.logging.info("Defining the dataset...")
    if params.dataset == 'mnist':
        train_input_fn = lambda: mnist_train_input_fn('data/mnist', params)
    elif params.dataset == 'imagenetvid':
        train_input_fn = lambda: imagenet_train_input_fn(params)
    else:
        # Error -- dataset should be defined
        exit(1)

    # Train the model
    tf.logging.info("Starting training for {} epoch(s).".format(params.num_epochs))
    estimator.train(train_input_fn)
