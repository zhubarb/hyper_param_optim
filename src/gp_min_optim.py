import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import model_selection
from functools import partial
from skopt import gp_minimize
from skopt import space

# first need to create an optimize function
def optimize(params, param_names, x, y):

    params = dict(zip(param_names, params))
    model = RandomForestClassifier(**params) # feed in all optim params in initialising the model
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
        accuracies. append(fold_accuracy)

    # return the negative mean fold accuracy (since wwe minimize)
    return -1.0 * np.array(accuracies).mean()

# Everything needs to be done in a cross vlaidation loop
if __name__ == '__main__':

    # Bayesian (Gaussian Process Optimisation)
    # https://scikit-optimize.github.io/stable/modules/generated/skopt.gp_minimize.html
    df = pd.read_csv('../data/datasets_11167_15520_train.csv')
    print('%i predictive features'%(len(df.columns)-1))
    X = df.drop('price_range', axis=1).values
    y = df['price_range'].values

    param_space =[
        space.Real(0.1, 1, prior= 'uniform', name='max_features'),
        space.Integer(100,1000, name='n_estimators'),
        space.Integer(5, 25, name='min_samples_leaf'),
        space.Categorical(['gini', 'entropy'], name='criterion')
    ]

    param_names = [
        'max_features',
        'n_estimators',
        'min_samples_leaf',
        'criterion'
    ]

    optimization_function = partial(
        optimize,
        param_names = param_names,
        x=X,
        y=y
    )

    result = gp_minimize(optimization_function,
                         dimensions=param_space,
                         n_calls=15,
                         n_random_starts=10,
                         verbose=10)

    print(
        dict(zip(param_names, result.x))
    )

    # Time taken: 3.0806
    # Function valueobtained: -0.9060
    # Current minimum: -0.9060
    # {'max_features': 0.7187929962475232, 'n_estimators': 100, 'min_samples_leaf': 5, 'criterion': 'entropy'}