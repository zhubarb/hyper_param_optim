import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import model_selection
from functools import partial
import optuna

# first need to create an optimize function, this remains the same as gp_optimize
def optimize(trial, x, y):

    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
    n_estimators = trial.suggest_int('n_estimators', 100, 1000)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 5, 25)
    max_features = trial.suggest_uniform('max_features', 0.1, 1)

    model = RandomForestClassifier(
        n_estimators = n_estimators,
        criterion=criterion,
        max_features=max_features,
        min_samples_leaf= min_samples_leaf
    )  # feed in all optim params in initialising the model
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

    optimization_function = partial(optimize, x=X, y=y)

    study = optuna.create_study(direction = 'minimize')
    study.optimize(optimization_function, n_trials=15)
