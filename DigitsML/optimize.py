from hyperopt import hp, fmin, tpe, space_eval
from keras import get_keras_model


def hp_objective(args):
    model = get_keras_model(**args)

    hist = model.fit(args['x'], args['y'],
                     batch_size=64,
                     epochs=5,
                     verbose=2,
                     validation_split=.2)

    return hist.history['val_loss'][-1]


def hp_optimize(x, y):
    space = {
        'x': x,
        'y': y,

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
    best_dict = space_eval(space, best)

    del best_dict['x']
    del best_dict['y']

    return best_dict
