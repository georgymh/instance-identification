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
                                    save_checkpoints_steps=params.save_checkpoints_steps)
    if params.warm_start_from == "":
        estimator = tf.estimator.Estimator(model_fn, params=params, config=config)
    else:
        ws = tf.estimator.WarmStartSettings(
            ckpt_to_initialize_from=params.warm_start_from,
            vars_to_warm_start='.*',
        )
        estimator = tf.estimator.Estimator(model_fn, params=params, config=config,
                                           warm_start_from=ws)

    # Define the dataset
    tf.logging.info("Defining the dataset...")
    assert params.dataset in ['mnist', 'imagenetvid'], \
        "Dataset {} not supported".format(params.dataset)
    if params.dataset == 'mnist':
        train_input_fn = lambda: mnist_train_input_fn('data/mnist', params)
    elif params.dataset == 'imagenetvid':
        train_input_fn = lambda: imagenet_train_input_fn(params)

    # Train the model
    tf.logging.info("Starting training for {} epoch(s).".format(params.num_epochs))
    if params.only_one_step:
        estimator.train(train_input_fn, max_steps=1)
    else:
        estimator.train(train_input_fn)
