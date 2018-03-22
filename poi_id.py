#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
import tester
from tester import dump_classifier_and_data
import pandas as pd
import numpy as np

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import Imputer
from model_tuning import model_tuning
#model_training_evaluation,

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".  'email_address'
finance_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
                 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
                 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock',
                 'director_fees']
email_features = ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi',
              'shared_receipt_with_poi']

poi_feature=['poi']

feature_list=poi_feature + finance_features + email_features


 # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
data=pd.DataFrame.from_dict(data_dict,orient='index')  # read the data and traverse the columns and rows

### reorder the features
data=data.ix[:,feature_list]

### Data type changes
data_type=[bool]+[np.float]*15+[str]+[np.float]*4
for i in range(len(feature_list)):
    data[feature_list[i]]=data[feature_list[i]].astype(data_type[i])
### fill in nan value

data[email_features] = data[email_features].fillna(data[email_features].median())
data[finance_features] = data[finance_features].fillna(0)


### Task 2: Remove outliers

#check the data distribution, identified the total as outlier, and another two,
# and one of which feature is almost blank and another one is irrelevant to any person.
data=data[(data.index!='TOTAL')]
data=data[(data.index!='LOCKHART EUGENE E')]
data=data[(data.index!='THE TRAVEL AGENCY IN THE PARK')]

# value might be wrong
data.loc['BHATNAGAR SANJAY','restricted_stock']=-data.loc['BHATNAGAR SANJAY','restricted_stock']
data.loc['BELFER ROBERT','total_stock_value']=-data.loc['BELFER ROBERT','total_stock_value']
data.loc['BELFER ROBERT','restricted_stock']=-data.loc['BELFER ROBERT','restricted_stock']


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.


data['received_from_poi_ratio']=data['from_poi_to_this_person']/data['to_messages']
data['sent_to_poi_ratio']=data['from_this_person_to_poi']/data['from_messages']
data['shared_receipt_with_poi_ratio']=data['shared_receipt_with_poi']/data['to_messages']

feature_list=feature_list+['received_from_poi_ratio','sent_to_poi_ratio','shared_receipt_with_poi_ratio']

### Extract features and labels from dataset for local testing
my_dataset = data.drop(['email_address'], axis=1)
feature_list=my_dataset.columns.values
my_dataset=data.to_dict('index')
data2 = featureFormat(my_dataset, feature_list, sort_keys = True)
labels, features = targetFeatureSplit(data2)

# features include the \n in the string, so remove the \n, and transform to array
features2=[]
for i in features:
    features2.append([word for word in i])
features2=np.array(features2)
labels2=np.array(labels)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# In this section, if we use pipeline, we could bing feature preprocessing (null value fill out),
# feature selection and model training in the same place)
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier

#### Naive Bayes based on different method of feature selection
####
####
pipe_NB_SK=make_pipeline(SelectKBest(f_classif),
                        GaussianNB())
param_NB_SK={ 'selectkbest__k':[i for i in xrange(6,20)]}
result_NB_SK= model_tuning(pipe_NB_SK, param_NB_SK, features2,labels2)

pipe_NB_PCA=make_pipeline(Imputer(strategy='median', axis=0),
                         PCA(),
                        GaussianNB())
param_NB_PCA={ 'pca__n_components': [10,12,14,16,18,20]}
result_NB_PCA= model_tuning(pipe_NB_PCA, param_NB_PCA, features2,labels2)

pipe_NB_DT=make_pipeline(Imputer(strategy='median', axis=0),
                         SelectFromModel(ExtraTreesClassifier()),
                        GaussianNB())
param_NB_DT={}
result_NB_DT= model_tuning(pipe_NB_DT, param_NB_DT, features2,labels2)
#print result_NB_DT
####
###

#### Logistic Regression based on different method of feature selection
###
###
pipe_LR_SK=make_pipeline(Imputer(strategy='median', axis=0),
                         SelectKBest(f_classif),
                        LogisticRegression())
param_LR_SK={ 'logisticregression__penalty':('l1', 'l2'),
    'logisticregression__C':[0.05, 0.5, 1, 10, 10**2, 10**3, 10**5, 10**10, 10**15],
              'selectkbest__k': [i for i in xrange(6, 20)]}
result_LR_SK= model_tuning(pipe_LR_SK, param_LR_SK, features2,labels2)


pipe_LR_PCA=make_pipeline(Imputer(strategy='median', axis=0),
                    PCA(),
                    LogisticRegression())
param_LR_PCA={ 'logisticregression__penalty':('l1', 'l2'),
    'logisticregression__C':[0.05, 0.5, 1, 10, 10**2, 10**3, 10**5, 10**10, 10**15],
               'pca__n_components': [10,12,14,16,18,20]}
result_LR_PCA = model_tuning(pipe_LR_PCA, param_LR_PCA, features2,labels2)


pipe_LR_DT=make_pipeline(Imputer(strategy='median', axis=0),
                         SelectFromModel(ExtraTreesClassifier()),
                   LogisticRegression())
param_LR_DT={ 'logisticregression__penalty':('l1', 'l2'),
    'logisticregression__C':[0.05, 0.5, 1, 10, 10**2, 10**3, 10**5, 10**10, 10**15]
              }
result_LR_DT = model_tuning(pipe_LR_DT, param_LR_DT, features2,labels2)
###
###

#### K nearest neighbor based on different method of feature selection
###
###
pipe_KNN_SK=make_pipeline(Imputer(strategy='median', axis=0),
                         SelectKBest(f_classif),
                          KNeighborsClassifier())
param_KNN_SK={ 'kneighborsclassifier__n_neighbors':[3,5,7,10],
               'selectkbest__k': [i for i in xrange(6, 20)]}

result_KNN_SK= model_tuning(pipe_KNN_SK, param_KNN_SK, features2,labels2)


pipe_KNN_PCA=make_pipeline(Imputer(strategy='median', axis=0),
                    PCA(),
                    KNeighborsClassifier())
param_KNN_PCA={ 'kneighborsclassifier__n_neighbors':[3,5,7,10],
               'pca__n_components': [10,12,14,16,18,20]}
result_KNN_PCA = model_tuning(pipe_KNN_PCA, param_KNN_PCA, features2,labels2)


pipe_KNN_DT=make_pipeline(Imputer(strategy='median', axis=0),
                         SelectFromModel(ExtraTreesClassifier()),
                          KNeighborsClassifier())
param_KNN_DT={ 'kneighborsclassifier__n_neighbors':[3,5,7,10]
               }
result_KNN_DT = model_tuning(pipe_KNN_DT, param_KNN_DT, features2,labels2)

###
###

#### Random Forest based on different method of feature selection
###
###
pipe_RF_SK=make_pipeline(Imputer(strategy='median', axis=0),
                         SelectKBest(f_classif),
                         RandomForestClassifier())
param_RF_SK={ 'randomforestclassifier__n_estimators':[5,8,10,12,15,20],
'randomforestclassifier__criterion': ('gini','entropy'),
'randomforestclassifier__bootstrap': (True,False),
'randomforestclassifier__min_samples_leaf': [1, 2, 3, 4, 5],
'randomforestclassifier__min_samples_split': [2, 3, 4, 5],
               'selectkbest__k': [i for i in xrange(6, 20)]}

result_RF_SK= model_tuning(pipe_RF_SK, param_RF_SK, features2,labels2)


pipe_RF_PCA=make_pipeline(Imputer(strategy='median', axis=0),
                    PCA(),
                    RandomForestClassifier())
param_RF_PCA={ 'randomforestclassifier__n_estimators':[5,8,10,12,15,20],
'randomforestclassifier__criterion': ('gini','entropy'),
'randomforestclassifier__bootstrap': (True,False),
               'pca__n_components': [10,12,14,16,18,20],
               'randomforestclassifier__min_samples_leaf': [1, 2, 3, 4, 5],
               'randomforestclassifier__min_samples_split': [2, 3, 4, 5]
               }
result_RF_PCA = model_tuning(pipe_RF_PCA, param_RF_PCA, features2,labels2)


pipe_RF_DT=make_pipeline(Imputer(strategy='median', axis=0),
                         SelectFromModel(ExtraTreesClassifier()),
                         RandomForestClassifier())
param_RF_DT={ 'randomforestclassifier__n_estimators':[5,8,10,12,15,20],
'randomforestclassifier__criterion': ('gini','entropy'),
'randomforestclassifier__bootstrap': (True,False),
'randomforestclassifier__min_samples_leaf': [1, 2, 3, 4, 5],
'randomforestclassifier__min_samples_split': [2, 3, 4, 5]}
result_RF_DT = model_tuning(pipe_RF_DT, param_RF_DT, features2,labels2)

###
###

#### AdaBoost based on different method of feature selection
###
###

pipe_AB_SK=make_pipeline(Imputer(strategy='median', axis=0),
                         SelectKBest(f_classif),
                         AdaBoostClassifier())
param_AB_SK={'adaboostclassifier__n_estimators': [5,8,10,12,15,20],
              'adaboostclassifier__algorithm': ['SAMME', 'SAMME.R'],
               'selectkbest__k': [i for i in xrange(6, 20)]
}
result_AB_SK= model_tuning(pipe_AB_SK, param_AB_SK, features2,labels2)


pipe_AB_PCA=make_pipeline(Imputer(strategy='median', axis=0),
                    PCA(),
                    AdaBoostClassifier())
param_AB_PCA={'adaboostclassifier__n_estimators': [5,8,10,12,15,20],
              'adaboostclassifier__algorithm': ['SAMME', 'SAMME.R'],
               'pca__n_components': [10,12,14,16,18,20]}
result_AB_PCA = model_tuning(pipe_AB_PCA, param_AB_PCA, features2,labels2)


pipe_AB_DT=make_pipeline(Imputer(strategy='median', axis=0),
                         SelectFromModel(ExtraTreesClassifier()),
                         AdaBoostClassifier())
param_AB_DT={
              'adaboostclassifier__n_estimators': [5,8,10,12,15,20],
              'adaboostclassifier__algorithm': ['SAMME', 'SAMME.R']
}
result_AB_DT = model_tuning(pipe_AB_DT, param_AB_DT, features2,labels2)

###
###

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
#from sklearn.model_selection import StratifiedShuffleSplit

#sss = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=0)

tester.test_classifier(result_RF_SK, my_dataset, feature_list)
tester.test_classifier(result_NB_PCA, my_dataset, feature_list)
tester.test_classifier(result_NB_DT, my_dataset, feature_list)

tester.test_classifier(result_LR_SK, my_dataset, feature_list)
tester.test_classifier(result_LR_PCA, my_dataset, feature_list)
tester.test_classifier(result_LR_DT, my_dataset, feature_list)

tester.test_classifier(result_KNN_SK, my_dataset, feature_list)
tester.test_classifier(result_KNN_PCA, my_dataset, feature_list)
tester.test_classifier(result_KNN_DT, my_dataset, feature_list)

tester.test_classifier(result_RF_SK, my_dataset, feature_list)
tester.test_classifier(result_RF_PCA, my_dataset, feature_list)
tester.test_classifier(result_RF_DT, my_dataset, feature_list)

tester.test_classifier(result_AB_SK, my_dataset, feature_list)
tester.test_classifier(result_AB_PCA, my_dataset, feature_list)
tester.test_classifier(result_AB_DT, my_dataset, feature_list)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(result_AB_DT, my_dataset, feature_list)