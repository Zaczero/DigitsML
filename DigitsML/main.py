import os
import requests
import zipfile
import gzip
import glob
import shutil
import numpy as np
from mnist import MNIST
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import get_keras_model_initialized
from optimize import hp_optimize
from config import MNIST_DIRECTORY, MNIST_DOWNLOAD_URL, IMAGE_SIZE, MODEL_FILE


try:
    # reduce initial gpu memory allocation (faster startup)
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception as e:
    print(e)


def ensure_dataset():
    if os.path.exists(MNIST_DIRECTORY):
        return

    archive_path = os.path.join(MNIST_DIRECTORY, 'mnist.zip')

    print('Downloading dataset')
    archive_response = requests.get(MNIST_DOWNLOAD_URL)
    archive_response.raise_for_status()

    os.mkdir(MNIST_DIRECTORY)

    with open(archive_path, 'wb') as f:
        f.write(archive_response.content)

    print('Extracting (1/2)')
    with zipfile.ZipFile(archive_path, 'r') as f:
        f.extractall(MNIST_DIRECTORY)

    os.unlink(archive_path)

    print('Extracting (2/2)')
    for file_path in glob.glob(os.path.join(MNIST_DIRECTORY, '*.gz')):
        decompressed_path = file_path.replace('.gz', '')

        with gzip.open(file_path, 'rb') as f_in:
            with open(decompressed_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        os.unlink(file_path)

    print('Dataset is ready to use')


def to_categorical(data):
    result = np.zeros((len(data), max(data) + 1))
    result[np.arange(len(data)), data] = 1
    return result


def prepare_data(x, y=None):
    x = np.array(x)
    x = np.reshape(x, (len(x), IMAGE_SIZE, IMAGE_SIZE, 1)) / 255.0

    if y is not None:
        y = np.array(y)
        y = to_categorical(y)

    return x, y


def train_model(x, y):
    model = get_keras_model_initialized()

    callbacks = [
        EarlyStopping(patience=5),
        ModelCheckpoint(MODEL_FILE, save_best_only=True)
    ]

    model.summary()
    model.fit(x, y,
              batch_size=64,
              epochs=100,
              verbose=2,
              callbacks=callbacks,
              validation_split=.2)

    # restore best weights
    model.load_weights(MODEL_FILE)

    return model


ensure_dataset()

mndata = MNIST(MNIST_DIRECTORY)
X_train, y_train = prepare_data(*mndata.load_training())
X_test, y_test = prepare_data(*mndata.load_testing())

# print(hp_optimize(X_train, y_train))

train_model(X_train, y_train)
