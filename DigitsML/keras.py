import os
from config import IMAGE_SIZE, MODEL_FILE
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout


def get_keras_model(conv_size_1, conv_kern_1,
                    conv_size_2, conv_kern_2,
                    dense_size_1,
                    learning_rate, momentum, nesterov,
                    **kwargs):
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


def get_keras_model_initialized():
    model = get_keras_model(**{
        'conv_size_1': 8,
        'conv_kern_1': 5,

        'conv_size_2': 30,
        'conv_kern_2': 3,

        'dense_size_1': 21,

        'learning_rate': 0.29421394650707094,
        'momentum': 0.05552671455772779,
        'nesterov': False
    })

    if os.path.exists(MODEL_FILE):
        model.load_weights(MODEL_FILE)

    return model
