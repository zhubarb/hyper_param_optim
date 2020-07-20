import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import model_selection

search = 'BAYESIAN' #'RANDOM' # 'GRID'

# Everyhitng needs to be done in a cross vlaidation loop
if __name__ == '__main__':

    df = pd.read_csv('../data/datasets_11167_15520_train.csv')
    print('%i predictive features'%(len(df.columns)-1))
    X = df.drop('price_range', axis=1).values
    y = df['price_range'].values

    classifier = RandomForestClassifier(n_jobs=-1)

    if search== 'GRID':
        # GRID Search
        param_grid={
            'n_estimators': [200, 400],
            'criterion': ['gini', 'entropy'],
            'min_samples_leaf': [5, 10, 20],
            'max_features': [5, 10]
        }

        model = model_selection.GridSearchCV(
            estimator = classifier,
            param_grid= param_grid,
            scoring = 'accuracy',
            verbose=10,
            n_jobs=1,
            cv=5
        )
    elif search == 'RANDOM': #  More efficient, less expensive than grid search
        # RANDOMISED Search
        param_grid = {
            'n_estimators': np.arange(100, 1000, 100) ,
            'criterion': ['gini', 'entropy'],
            'min_samples_leaf': np.arange(5, 20, 2),
            'max_features': [5, 10]
        }

        model = model_selection.RandomizedSearchCV(
            estimator = classifier,
            param_distributions= param_grid,
            n_iter = 10,
            scoring = 'accuracy',
            verbose=10,
            n_jobs=1,
            cv=5
        )

    model.fit(X,y)
    print(model.best_score_)
    print(model.best_estimator_.get_params())
