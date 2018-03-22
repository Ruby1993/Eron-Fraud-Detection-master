import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data,test_classifier
import pandas as pd
import numpy as np

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import Imputer
from model_tuning import model_tuning
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
from sklearn.preprocessing import StandardScaler

#### Naive Bayes based on different method of feature selection
####
####


pipe_AB_SK=make_pipeline(Imputer(strategy='median', axis=0),
                         SelectKBest(f_classif),
                         StandardScaler(),
                         AdaBoostClassifier())
param_AB_SK={'adaboostclassifier__n_estimators': [5,8,10,12,15,20],
              'adaboostclassifier__algorithm': ['SAMME', 'SAMME.R'],
               'selectkbest__k': [10,15,20]
}
result_AB_SK= model_tuning(pipe_AB_SK, param_AB_SK, features2,labels2)

test_classifier(result_AB_SK,my_dataset, feature_list)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(test_classifier(result_AB_SK,my_dataset, feature_list), my_dataset, feature_list)