"""Evaluate the model"""

import argparse
import os
import time

import tensorflow as tf

from dataset.mnist.input_fn import mnist_train_input_fn
from dataset.mnist.input_fn import mnist_test_input_fn
from dataset.imagenetvid.input_fn import imagenet_train_eval_input_fn
from dataset.imagenetvid.input_fn import imagenet_val_eval_input_fn
from dataset.imagenetvid.input_fn import imagenet_easy_val_eval_input_fn

from model.model_fn import model_fn
from model.utils import Params


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Experiment directory containing params.json")

def run_evaluation(estimator, test_input_fn, eval_type):
    res = estimator.evaluate(test_input_fn, name=eval_type)
    if "accuracy_mean" in res:
        tf.summary.scalar('accuracy_mean', res["accuracy_mean"])
    for key in res:
        print("{}: {}".format(key, res[key]))

if __name__ == '__main__':
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)

    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Define the model
    tf.logging.info("Creating the model...")
    estimator = tf.estimator.Estimator(model_fn, params=params, model_dir=args.model_dir)

    # Define the dataset
    tf.logging.info("Defining the dataset...")
    assert params.dataset in ['mnist', 'imagenetvid'], 'Dataset must be valid!'
    if params.dataset == 'mnist':
        train_eval_input_fn = lambda: mnist_train_input_fn('data/mnist', params)
        val_eval_input_fn = lambda: mnist_test_input_fn('data/mnist', params)
    elif params.dataset == 'imagenetvid':
        train_eval_input_fn = lambda: imagenet_train_eval_input_fn(params)
        val_eval_input_fn = lambda: imagenet_val_eval_input_fn(params)
        easy_val_eval_input_fn = lambda: imagenet_easy_val_eval_input_fn(params)

    # Evaluate model on the test set
    tf.logging.info("Evaluation on the test set.")
    timeout = 0
    checkpoints = set()
    while True:
        if timeout > params.eval_timeout_secs:
            # Stop eval script after doing nothing for eval_timeout_secs seconds.
            break
        if params.eval_run_once:
            # When run_once is true, checkpoint_path should point to the exact
            # checkpoint file.
            print('Evaluating {}...'.format(ckpt.model_checkpoint_path))
            run_evaluation(estimator, train_eval_input_fn, 'train')
            run_evaluation(estimator, val_eval_input_fn, 'val')
            run_evaluation(estimator, easy_val_eval_input_fn, 'easy_val')
            break
        else:
            # When run_once is false, checkpoint_path should point to the directory
            # that stores checkpoint files.
            ckpt = tf.train.get_checkpoint_state(args.model_dir)
            if ckpt and ckpt.model_checkpoint_path:
                if ckpt.model_checkpoint_path in checkpoints:
                    # Do not evaluate on the same checkpoint
                    print('Wait {:d}s for new checkpoints to be saved ... '
                          .format(params.eval_interval_secs))
                    time.sleep(params.eval_interval_secs)
                    timeout += params.eval_interval_secs
                else:
                    # Add checkpoint to set and evaluate
                    checkpoints.add(ckpt.model_checkpoint_path)
                    print('Evaluating {}...'.format(ckpt.model_checkpoint_path))
                    run_evaluation(estimator, train_eval_input_fn, 'train')
                    run_evaluation(estimator, val_eval_input_fn, 'val')
                    run_evaluation(estimator, easy_val_eval_input_fn, 'easy_val')
                    timeout = 0
            else:
                print('No checkpoint file found')
                if not params.eval_run_once:
                    print('Wait {:d}s for new checkpoints to be saved ... '
                          .format(params.eval_interval_secs))
                time.sleep(params.eval_interval_secs)
                timeout += params.eval_interval_secs
