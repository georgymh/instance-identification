# Instance Identification
*Author: Georgy Marrero*

**Instance Identification** or **Instance ID** is the process of matching one or more instances of an object across frames of a video.

This repository contains a TensorFlow implementation of a **convolutional neural network** with **triplet loss** and **online triplet mining** that attempts to create an embedding where examples of objects of the same category are close, but examples of objects of the same instances are even closer.

**Credits:** The code structure and custom loss functions are adapted from [this blog post](https://omoindrot.github.io/triplet-loss) and [this repository](https://github.com/omoindrot/tensorflow-triplet-loss). Be sure to check them out for an introduction on how triplet loss and triplet mining works. Also, some of the ImageNet VID curation code was ported from [this SiamFC code re-implementation](https://github.com/bilylee/SiamFC-TensorFlow).

## Overview

#### Network Architecture

The model architecture consists of a fresh/pre-trained **Inception-ResNet-V2** with the **batch-hard/batch-all/batch-semi-hard triplet loss function**.

#### Dataset

There are 4 different datasets that this model trains and validates on, all based on **ImageNet VID 2015**.

The training dataset is curated the following way:

- `train`: batches from the training split of size `B=PK`, where `P` is the number of video snippets sampled per batch and `K` is the number of instances of the same object we sample per video snippet.

On the other hand, the evaluation datasets are curated the following way:

- `train_eval`: batches formed by choosing ~10K random video snippets from the training split (can be repeated) and extracting one non-occluded object crop from one frame and all the other non-occluded object crops from another frame, included the same object instance from the first frame.

- `val_eval`: batches formed by choosing ~10K random video snippets from the validation split (can be repeated) and extracting one non-occluded object crop from one frame and all the other non-occluded object crops from another frame, included the same object instance from the first frame.

- `easy_val_eval` or `random_crop`: batches formed by choosing ~10K random video snippets from the validation split (can be repeated) and extracting one non-occluded object crop from one frame, as well as the same object instance and `M` random crops from a second frame.

Note: The `train` batches are pre-mined for efficiency but the model chooses which triplets to train on (it is still online triplet mining because the loss function chooses which triplets from each batch to train on).

#### Loss functions

This repository only has support for **online triplet mining**, and implements support for the following triplet strategies:

- `tf_batch_semi_hard`: built-in TensorFlow strategy that tries to choose negative examples within the margin (semi-hard negative), or the hardest negative if there are no semi-hard ones.

- `batch_all`: custom strategy that selects all the valid triplets, and average the loss on the hard and semi-hard triplets (no easy triplets), generating `PK(K - 1)(PK - K)` triplets per batch.

- `batch_hard`: custom strategy chooses the hardest positive and negative per batch, generating `PK` triplets per batch.

To specify which strategy to use, just set ``"triplet_strategy"`` in `params.json` to be one of the strategy names defined above.


#### Accuracy Metrics

To measure the accuracy of this model, we use two different strategies:

- `general_accuracy`: the average number of positive/matching instances identified (when the anchor-positive distance is strictly less than all of the anchor-negative distances in 2 frames of a video).

- `easy_accuracy`: the average percentage of anchor-negative distances farther than the anchor-positive distance per video snippet.

#### Results

The best results obtained so far are the following:

 | General Accuracy | Easy Accuracy
----------- | ------------ | -------------
**General Validation Set** | 73.28% | 86.7%
**Random Crop Validation Set** | 95.08% | 95.75%

The hyperparameters used for this model can be found in [`experiments/best_model_so_far`](experiments/best_model_so_far).

## Requirements

It's recommended to use python 3 and a virtual environment.

```bash
pip install virtualenv
virtualenv -p python3 venv
source venv/bin/activate venv
```

Then, you will need to install the required pip dependencies by running:

```bash
pip install -r requirements_cpu.txt
```

If you are using a GPU, you will need to install `tensorflow-gpu` by running:
```bash
pip install -r requirements_gpu.txt
```

## Setting up the Datasets

To set up the dataset, you must do the following:

- Download the ImageNet VID 2015 dataset from [kaggle](https://www.kaggle.com/c/imagenet-object-detection-from-video-challenge/data) or [the official site](http://image-net.org/challenges/LSVRC/2015/) (a little harder) and uncompress it.
- (Recommended) Create a soft link to the downloaded dataset inside `data/`:
```bash
$DATASET=/path/to/ILSVRC2015
ln -s $DATASET data/ILSVRC2015
```

- Run the image cropping process:
```bash
python dataset/imagenetvid/scripts/crop_imagenet.py
```

- Curate the training and evaluation triplets:
```bash
python dataset/imagenetvid/scripts/curate_triplets.py
```

## Training on ImageNet VID

You will first need to create a configuration file like this one: [`params.json`](experiments/base_model/params.json). This json file specifies all the hyperparameters for the model.

Optionally, you can download the pre-trained weights for Inception-ResNet-V2 [from this link](http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz). You can copy them into [`model/embeddings/pre-trained`](model/embeddings/pre-trained) and set your `params.json`'s `"warm_start_from"` to be the path to these model parameters.

To run a new experiment called `base_model`, do:
```bash
python train.py --model_dir experiments/base_model
```

All the weights and summaries will be saved in the `model_dir`, so to visualize the training on **tensorboard**, it's sufficient to do:

```bash
tensorboard --logdir experiments/base_model
```


## Evaluating on ImageNet VID

Evaluation is meant to run while training is happening (on a separate GPU for example).

The evaluation script will evaluate only on new model parameters at the frequency specified in the `params.json`. It will synchronously evaluate on the `train_eval`, `val_eval`, and `easy_val_eval` datasets. The script will also terminate after the timeout duration specified in `params.json`.

To do evaluation, you just need to run:

```bash
python evaluate.py --model_dir experiments/base_model
```

Again, these results will be saved in the `model_dir`, so to visualize the training and evaluation on **tensorboard**, it's sufficient to do:

```bash
tensorboard --logdir experiments/base_model
```

## Jupyter Notebooks

There are a few jupyter notebooks doing visual data exploration. These can be accessed from [`notebooks/`](notebooks).


## Test

To run all the tests, run this from the project directory:
```bash
pytest
```

To run a specific test:
```bash
pytest model/tests/test_triplet_loss.py
```


## Resources

- Excellent [blog post][blog] explaining triplet loss
- Source code for the built-in TensorFlow function for semi hard online mining triplet loss: [`tf.contrib.losses.metric_learning.triplet_semihard_loss`][tf-triplet-loss].
- [Facenet paper][facenet] introducing online triplet mining
- Detailed explanation of online triplet mining in [*In Defense of the Triplet Loss for Person Re-Identification*][in-defense]
- Blog post by Brandom Amos on online triplet mining: [*OpenFace 0.2.0: Higher accuracy and halved execution time*][openface-blog].
- The [coursera lecture][coursera] on triplet loss


[blog]: https://omoindrot.github.io/triplet-loss
[triplet-types-img]: https://omoindrot.github.io/assets/triplet_loss/triplets.png
[triplet-loss-img]: https://omoindrot.github.io/assets/triplet_loss/triplet_loss.png
[online-triplet-loss-img]: https://omoindrot.github.io/assets/triplet_loss/online_triplet_loss.png
[embeddings-img]: https://omoindrot.github.io/assets/triplet_loss/embeddings.png
[embeddings-gif]: https://omoindrot.github.io/assets/triplet_loss/embeddings.gif
[openface-blog]: http://bamos.github.io/2016/01/19/openface-0.2.0/
[facenet]: https://arxiv.org/abs/1503.03832
[in-defense]: https://arxiv.org/abs/1703.07737
[tf-triplet-loss]: https://www.tensorflow.org/api_docs/python/tf/contrib/losses/metric_learning/triplet_semihard_loss
[coursera]: https://www.coursera.org/learn/convolutional-neural-networks/lecture/HuUtN/triplet-loss
