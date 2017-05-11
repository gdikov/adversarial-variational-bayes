import urllib2
import os
import struct
import sklearn.datasets.mldata as fetcher
from utils.logger_config import logger
import numpy as np
np.random.seed(7)

# FIXME: make it configurable
PROJECT_DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "..", "data")


def load_mnist(local_data_path=None):
    """
    Load the MNIST dataset from local file or download it if not available.
    
    Args:
        local_data_path: path to the MNIST dataset. Assumes unpacked files and original filenames. 

    Returns:
        A dict with `data` and `target` keys with the MNIST data converted to [0, 1] floats. 
    """
    if local_data_path is None:
        logger.info("Path to locally stored data not provided. Proceeding with downloading the MNIST dataset.")
        mnist_path = os.path.join(PROJECT_DATA_DIR, "MNIST")
        try:
            mnist = fetcher.fetch_mldata("MNIST Original", data_home=mnist_path)
            mnist = {'data': mnist.data, 'target': mnist.target}
        except urllib2.HTTPError:
            logger.warning("Fetching data from mldata.org failed. The server is probably unreachable. "
                           "Proceeding with fetching from Tensorflow.examples.tutorials.mnist.")
            from tensorflow.examples.tutorials.mnist import input_data
            mnist = input_data.read_data_sets(mnist_path, one_hot=False)
            mnist = {'data': np.concatenate((mnist.train.images, mnist.test.images, mnist.validation.images)),
                     'target': np.concatenate((mnist.train.labels, mnist.test.labels, mnist.validation.labels))}
    else:
        logger.info("Loading MNIST dataset from {}".format(local_data_path))
        mnist_imgs, mnist_labels = _load_mnist_from_file(local_data_path)
        mnist = {'data': mnist_imgs, 'target': mnist_labels}

    return mnist


def _load_mnist_from_file(data_dir=None):
    """
    Load the binary files from disk. 
    
    Args:
        data_dir: path to folder containing the MNIST dataset blobs. 

    Returns:
        A numpy array with the images and a numpy array with the corresponding labels. 
    """
    # The files are assumed to have these names and should be found in 'path'
    image_files = ('train-images-idx3-ubyte', 't10k-images-idx3-ubyte')
    label_files = ('train-labels-idx1-ubyte', 't10k-labels-idx1-ubyte')

    def read_labels(fname):
        with open(os.path.join(data_dir, fname), 'rb') as flbl:
            # remove header
            magic, num = struct.unpack(">II", flbl.read(8))
            labels = np.fromfile(flbl, dtype=np.int8)
        return labels

    def read_images(fname):
        with open(os.path.join(data_dir, fname), 'rb') as fimg:
            # remove header
            magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
            images = np.fromfile(fimg, dtype=np.uint8).reshape(num, -1)
        return images

    images = np.concatenate([read_images(fname) for fname in image_files])
    labels = np.concatenate([read_labels(fname) for fname in label_files])

    return images, labels
