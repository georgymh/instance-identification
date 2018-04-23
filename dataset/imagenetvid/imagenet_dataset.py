import os
import xmltodict
import pickle
import random

from imageio import imread

DATA_PATH_STR = 'data/ILSVRC2015-VID-Curation/Data/VID/'

SUBDIR_MAP = {'ILSVRC2015_VID_train_0000': 'a',
            'ILSVRC2015_VID_train_0001': 'b',
            'ILSVRC2015_VID_train_0002': 'c',
            'ILSVRC2015_VID_train_0003': 'd',
            '': 'e'}

TRAIN_SET_PKL = 'dataset/imagenetvid/train_set.pkl'
TRAIN_EVAL_SET_PKL = 'dataset/imagenetvid/train_eval_set.pkl'
VAL_EVAL_SET_PKL = 'dataset/imagenetvid/val_eval_set.pkl'


def get_next_training_batch():
    yield from _get_next_batch(TRAIN_SET_PKL)


def get_next_train_eval_batch():
    yield from _get_next_batch(TRAIN_EVAL_SET_PKL)


def get_next_val_eval_batch():
    yield from _get_next_batch(VAL_EVAL_SET_PKL)


def _get_next_batch(pkl_file_path):
    def _pickle_generator(pkl_file):
        try:
            while True:
                yield pickle.load(pkl_file)
        except EOFError:
            pass

    with open(pkl_file_path, 'rb') as f:
        for image_paths, labels in _pickle_generator(f):
            images = [imread(img) for img in image_paths]
            yield images, labels
