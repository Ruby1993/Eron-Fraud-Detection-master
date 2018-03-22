### Feature selection in different ways

#Select k-Best

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import Imputer
import pandas as pd
import numpy as np

def selectKbest(labels, features):
    selector = SelectKBest(f_classif, k='all')
    selector.fit(features, labels)
    scores = selector.scores_
    elect_score_1 = pd.DataFrame({'features': features,
                                  'scores': scores})
    select_score_1.set_index(['features'])
    select_score_1 = select_score_1.sort(['scores'], ascending=0)
