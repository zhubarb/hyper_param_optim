import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import model_selection
from functools import partial
from hyperopt import hp, fmin, tpe, Trials  # https://anaconda.org/conda-forge/hyperopt
from hyperopt.pyll.base import scope

# first need to create an optimize function, this remains the same as gp_optimize
def optimize(params, x, y):

    model = RandomForestClassifier(**params)  # feed in all optim params in initialising the model
    kf = model_selection.StratifiedKFold(n_splits=5)
    accuracies = []

    for idx in kf.split(X=x, y=y):
        train_idx, valid_idx = idx[0], idx[1]
        xtrain = x[train_idx]
        ytrain = y[train_idx]

        xval = x[valid_idx]
        yval = y[valid_idx]

        model.fit(xtrain, ytrain)
        preds = model.predict(xval)
        fold_accuracy = metrics.accuracy_score(yval, preds)
        accuracies.append(fold_accuracy)

    # return the negative mean fold accuracy (since wwe minimize)
    return -1.0 * np.array(accuracies).mean()


# Everything needs to be done in a cross vlaidation loop
if __name__ == '__main__':
    # Bayesian (Gaussian Process Optimisation)
    # https://scikit-optimize.github.io/stable/modules/generated/skopt.gp_minimize.html
    df = pd.read_csv('../data/datasets_11167_15520_train.csv')
    print('%i predictive features' % (len(df.columns) - 1))
    X = df.drop('price_range', axis=1).values
    y = df['price_range'].values

    param_space = {
        'max_features': hp.uniform('max_features', 0.1, 1),
        'n_estimators': scope.int(hp.quniform('n_estimators', 100, 1000, 1)),
        'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 5, 25, 1)),
        'criterion': hp.choice('criterion', ['gini', 'entropy'])
    }

    optimization_function = partial(
        optimize,
        x=X,
        y=y
    )

    trials = Trials()  # initialise trials

    result = fmin(fn=optimization_function,
                  space=param_space,
                  max_evals=15,
                  trials=trials,
                  algo=tpe.suggest
                  )

    print(result)

    # Time taken: 3.0806
    # Function valueobtained: -0.9060
    # Current minimum: -0.9060
    # {'max_features': 0.7187929962475232, 'n_estimators': 100, 'min_samples_leaf': 5, 'criterion': 'entropy'}
