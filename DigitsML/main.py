import os
import requests
import zipfile
import gzip
import glob
import shutil
import numpy as np
from hyperopt import hp, fmin, tpe, space_eval
from matplotlib import pyplot as plt
from mnist import MNIST
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout

MNIST_DOWNLOAD_URL = 'https://data.deepai.org/mnist.zip'
MNIST_DIRECTORY = 'mnist'
IMAGE_SIZE = 28
MODEL_FILE = 'model.hdf5'


def install_dataset():
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


def prepare_data(x, y):
    x = np.array(x)
    y = np.array(y)

    x = np.reshape(x, (len(x), IMAGE_SIZE, IMAGE_SIZE, 1)) / 255.0
    y = to_categorical(y)

    return x, y


def get_model(conv_size_1, conv_kern_1,
              conv_size_2, conv_kern_2,
              dense_size_1,
              learning_rate, momentum, nesterov):

    model = Sequential()

    model.add(Conv2D(conv_size_1, conv_kern_1, activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1)))
    model.add(MaxPool2D(2))
    model.add(Dropout(.2))
    model.add(Conv2D(conv_size_2, conv_kern_2, activation='relu'))
    model.add(MaxPool2D(2))
    model.add(Dropout(.2))
    model.add(Flatten())
    model.add(Dense(dense_size_1, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    optimizer = SGD(learning_rate, momentum, nesterov)

    model.compile(optimizer, 'categorical_crossentropy', metrics=['accuracy'])

    return model


def get_model_initialized():
    model = get_model(**{
        'conv_size_1': 8.0,
        'conv_kern_1': 5,

        'conv_size_2': 30.0,
        'conv_kern_2': 3,

        'dense_size_1': 21.0,

        'learning_rate': 0.29421394650707094,
        'momentum': 0.05552671455772779,
        'nesterov': False
    })

    if os.path.exists(MODEL_FILE):
        model.load_weights(MODEL_FILE)

    return model


def train_model():
    model = get_model_initialized()

    callbacks = [
        EarlyStopping(patience=5),
        ModelCheckpoint(MODEL_FILE, save_best_only=True)
    ]

    model.summary()
    model.fit(X_train, y_train,
              batch_size=64,
              epochs=100,
              verbose=2,
              callbacks=callbacks,
              validation_split=.2)

    # restore best weights
    model.load_weights(MODEL_FILE)

    return model


def hp_objective(args):
    model = get_model(**args)

    hist = model.fit(X_train, y_train,
                     batch_size=64,
                     epochs=5,
                     verbose=2,
                     validation_split=.2)

    return hist.history['val_loss'][-1]


def hp_optimize():
    space = {
        'conv_size_1': hp.quniform('conv_size_1', 4, 32, 1),
        'conv_kern_1': hp.choice('conv_kern_1', [3, 5]),

        'conv_size_2': hp.quniform('conv_size_2', 4, 32, 1),
        'conv_kern_2': hp.choice('conv_kern_2', [3, 5]),

        'dense_size_1': hp.quniform('dense_size_1', 4, 32, 1),

        'learning_rate': hp.uniform('learning_rate', 0.001, 1),
        'momentum': hp.uniform('momentum', 0, 1),
        'nesterov': hp.choice('nesterov', [True, False]),
    }

    best = fmin(hp_objective, space, tpe.suggest, 1)

    print(space_eval(space, best))


def show_image(data):
    size = int(np.sqrt(len(data)))
    data = np.reshape(data, (size, size))
    plt.imshow(data)
    plt.show()


if not os.path.exists(MNIST_DIRECTORY):
    install_dataset()

mndata = MNIST(MNIST_DIRECTORY)
X_train, y_train = mndata.load_training()
X_test, y_test = mndata.load_testing()

X_train, y_train = prepare_data(X_train, y_train)
X_test, y_test = prepare_data(X_test, y_test)

# hp_optimize()
train_model()
