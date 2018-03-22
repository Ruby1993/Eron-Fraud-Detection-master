
from sklearn.grid_search import GridSearchCV

def model_tuning(pipe, param,X,y):

    model = GridSearchCV(pipe, param, scoring='f1')
    model.fit(X, y)
    return model.best_estimator_

