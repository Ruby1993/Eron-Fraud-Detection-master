## Project Summary

#### Project Background
In 2000, Enron was one of the largest companies in the United States.
By 2002, it had collapsed into bankruptcy due to widespread corporate fraud.
In the resulting Federal investigation, a significant amount of typically
confidential information entered into the public record, including tens of
thousands of emails and detailed financial data for top executives.

In this project, I will apply machine learning techniques to build a person of
interest identifier based on financial and email data made public as a result of
the Enron scandal, which could help us to identify any potential person in the
fraud case based on the available data we collected.

In order to wrap up this detective work, we do have a list of person that are
already tagged with 'fraud' as they were addicted, reached a settlement or
deal with the government, or testified in exchange for prosecution immunity.

#### Problem Simplification
Identify persons who are in the corporate fraud based on the available data
within financial features, email features, and available 'poi' labels.

For the whole process, it could be broken down into parts below,
Feature preprocessing:
     - Outliers detection and removal
     - Fill in the nan value by considering features in different groups
     - Standardize the feature
Feature Creation
Modeling Creation/Validation:
    - Built up the pipelines and include the naive bayes, decision tree, logistic
    Regression, support vector machine, KNN, random forest, Adaboost, and Gradientboost.
    - Used grid search to find the optimal parameters based on the test result.
    - Adopted the stratified shuffle split cross validation to fully make use
     of our dataset, and pick the best model we built.

#### Dataset Overview

For the dataset we had, there are 146 records and 21 variables.

The 21 variables including financial features, email features,poi labels, which I listed below,

financial features(14)
['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees'] (all units are in US dollars)

Email features(6)
['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'] (units are generally number of emails messages; notable exception is ‘email_address’, which is a text string)

POI Feature-POI label(1)
[‘poi’] (boolean, represented as integer)

#### Outliers Issue
Based on data exploration, there are several outliers I found and will remove,
- TOTAL: This row is the sum of each variable, which need to removed.
- LOCKHART EUGENE E: Except the POI label(false), there is no data in all other
variables.
- THE TRAVEL AGENCY IN THE PARK: Not related to the person based on the name.


#### Feature selection process

- New Feature engineered
Before feature selection, I added three features related to ratio to make the data
more reasonable, which are listed below,

received_from_poi_ratio: Received from poi/all message the person received
(to_messages: the person was included in the to part)
sent_to_poi_ratio: Sent to poi /all message the person sent
(from_messages: the person was included in the from part)
shared_receipt_with_poi_ratio: 'shared_receipt_with_poi'/'to_messages'
[only one record is over 1, which might be a typero, here I just assumed it is 1]

In feature selection part in Project - Identify Fraud from Enron Email-Revised
ipython notebook, based on the correlation calculation, we could see the three
new-engineered features do get the higher correlation (positive) with poi compare to their
single feature, which are the good features to be included in the model. From
feature selection (either KBest or Treebased model), sent_to_poi_ratio and shared_receipt_with_poi_ratio have the higher rank which need to be included in
the model.

- KBest/TreeBased Model

As the number scale in different features are different, I used the MinmaxScaler to
map all the values to 0-1, although tree-based model do not require that. Then
I used two methods one is treebased model and one is SelectKBest I tried in the
ipython nodetebook. The detaild score was embeded in the project doc, and what i
focused on is the common lowest scored features which i decide to exclude them
in the model, which I listed below,

restricted_stock_deferred
from_message
to_message
deferral_payments
director_fees
from_this_person_to_poi

Below is the KBest result,

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>features</th>
      <th>scores</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9</th>
      <td>exercised_stock_options</td>
      <td>24.815080</td>
    </tr>
    <tr>
      <th>7</th>
      <td>total_stock_value</td>
      <td>24.179972</td>
    </tr>
    <tr>
      <th>4</th>
      <td>bonus</td>
      <td>20.792252</td>
    </tr>
    <tr>
      <th>0</th>
      <td>salary</td>
      <td>18.289684</td>
    </tr>
    <tr>
      <th>20</th>
      <td>sent_to_poi_ratio</td>
      <td>12.818278</td>
    </tr>
    <tr>
      <th>6</th>
      <td>deferred_income</td>
      <td>11.458477</td>
    </tr>
    <tr>
      <th>11</th>
      <td>long_term_incentive</td>
      <td>9.922186</td>
    </tr>
    <tr>
      <th>12</th>
      <td>restricted_stock</td>
      <td>8.828679</td>
    </tr>
    <tr>
      <th>2</th>
      <td>total_payments</td>
      <td>8.772778</td>
    </tr>
    <tr>
      <th>21</th>
      <td>shared_receipt_with_poi_ratio</td>
      <td>7.744506</td>
    </tr>
    <tr>
      <th>18</th>
      <td>shared_receipt_with_poi</td>
      <td>7.385691</td>
    </tr>
    <tr>
      <th>3</th>
      <td>loan_advances</td>
      <td>7.184056</td>
    </tr>
    <tr>
      <th>8</th>
      <td>expenses</td>
      <td>6.094173</td>
    </tr>
    <tr>
      <th>15</th>
      <td>from_poi_to_this_person</td>
      <td>4.225818</td>
    </tr>
    <tr>
      <th>10</th>
      <td>other</td>
      <td>4.187478</td>
    </tr>
    <tr>
      <th>17</th>
      <td>from_this_person_to_poi</td>
      <td>2.187071</td>
    </tr>
    <tr>
      <th>13</th>
      <td>director_fees</td>
      <td>2.126328</td>
    </tr>
    <tr>
      <th>19</th>
      <td>received_from_poi_ratio</td>
      <td>1.677382</td>
    </tr>
    <tr>
      <th>14</th>
      <td>to_messages</td>
      <td>0.866376</td>
    </tr>
    <tr>
      <th>1</th>
      <td>deferral_payments</td>
      <td>0.224611</td>
    </tr>
    <tr>
      <th>16</th>
      <td>from_messages</td>
      <td>0.189798</td>
    </tr>
    <tr>
      <th>5</th>
      <td>restricted_stock_deferred</td>
      <td>0.065500</td>
    </tr>
  </tbody>
</table>

TreeBased model is below,

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>features</th>
      <th>scores</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7</th>
      <td>total_stock_value</td>
      <td>0.097293</td>
    </tr>
    <tr>
      <th>4</th>
      <td>bonus</td>
      <td>0.095976</td>
    </tr>
    <tr>
      <th>12</th>
      <td>restricted_stock</td>
      <td>0.078907</td>
    </tr>
    <tr>
      <th>9</th>
      <td>exercised_stock_options</td>
      <td>0.075108</td>
    </tr>
    <tr>
      <th>6</th>
      <td>deferred_income</td>
      <td>0.067322</td>
    </tr>
    <tr>
      <th>10</th>
      <td>other</td>
      <td>0.063992</td>
    </tr>
    <tr>
      <th>17</th>
      <td>from_this_person_to_poi</td>
      <td>0.063415</td>
    </tr>
    <tr>
      <th>20</th>
      <td>sent_to_poi_ratio</td>
      <td>0.061489</td>
    </tr>
    <tr>
      <th>2</th>
      <td>total_payments</td>
      <td>0.055093</td>
    </tr>
    <tr>
      <th>0</th>
      <td>salary</td>
      <td>0.049618</td>
    </tr>
    <tr>
      <th>21</th>
      <td>shared_receipt_with_poi_ratio</td>
      <td>0.048545</td>
    </tr>
    <tr>
      <th>8</th>
      <td>expenses</td>
      <td>0.046949</td>
    </tr>
    <tr>
      <th>11</th>
      <td>long_term_incentive</td>
      <td>0.036968</td>
    </tr>
    <tr>
      <th>18</th>
      <td>shared_receipt_with_poi</td>
      <td>0.036005</td>
    </tr>
    <tr>
      <th>15</th>
      <td>from_poi_to_this_person</td>
      <td>0.035730</td>
    </tr>
    <tr>
      <th>16</th>
      <td>from_messages</td>
      <td>0.024428</td>
    </tr>
    <tr>
      <th>14</th>
      <td>to_messages</td>
      <td>0.021906</td>
    </tr>
    <tr>
      <th>19</th>
      <td>received_from_poi_ratio</td>
      <td>0.017253</td>
    </tr>
    <tr>
      <th>1</th>
      <td>deferral_payments</td>
      <td>0.014159</td>
    </tr>
    <tr>
      <th>5</th>
      <td>restricted_stock_deferred</td>
      <td>0.004394</td>
    </tr>
    <tr>
      <th>3</th>
      <td>loan_advances</td>
      <td>0.003810</td>
    </tr>
    <tr>
      <th>13</th>
      <td>director_fees</td>
      <td>0.001639</td>
    </tr>
  </tbody>
</table>

Also, in order to get the better performance, Except manually select the features
based on the model result, I used the pipeline method to incorporate them with different
prediction algorithm, which would be automatically incorporated in the process
and got the best model.


#### Modeling Process

The project is ending up using AdaBoostClassifier model with f1 score 0.44, accuracy
0.44 and recall 0.44, which is pretty balanced in the performance and has the
highest recall in all my models. But if we would prefer higher score in
precision(Out of all the items labeled as positive, how many truly belong to
the positive class), logisticRegression has a better performance with 0.6 precision.

Also, I tried DecisionTreeClassifier, Naive Bayes,
LogisticRegression, Support Vector Machine,K Nearest Neighbors,
RandomForestClassifier, and GradientBoostingClassifier.
The model performance do vary a lot between different models especially
in precision, recall, and f1 score, but in general accuracy and recall is
pretty hard get the higher score than the accuracy.

Parameter Tuning

For the model tuning, if we don't do it, it will impact the model performance. In
the project, I used grid search to tune the model and try to get the best parameter.
For the models I used, below are listed the parameters I tuned,
LogisticRegression- Penalty; C
Support Vector Machine- C; Degree; Kernel
KNN- n_neighbors
Random Forest- n_estimators; max_depth
AdaBoost- n_estimators; algorithm
GradientBoosting - loss; n_estimators; max_depth

#### Model Validation

It's the necessary way after model set up with the training dataset. It's the
process to evaluate the model using test dataset which is not used as training
dataset in the model training.The validation would be a good way to evaluate
the prediction capability of the model, and it is the necessary step to pick up
the best model.

for the validation, I changed to use StratifiedShuffleSplit to evaluate the model
performance as the dataset is pretty imbalanced.(k-fold cross validation might
be better to be used in the balanced dataset. The common
mistake would be only using accuracy to validate the model as the dataset is not balanced.


#### Matrix - Precision & Recall

Precision: How many are classified 'fraud' correctly out of all truely 'fraud' people.
if it is good, it will identify most of people involved in the fraud issue, but might
bring some innocent people in.

Recall: How many are classified 'fraud' correctly out of all labeled 'fraud' people.
if it is good, it will make sure all the labeled fraud people are higher likely involved
in the fraud, but it might miss some people involved in the fraud issue.

#### Final output

Based on the final result, I decide to choose the AdaBoostClassifier with  
precision 0.44974	and recall 0.34450	, and the parameters I chose  are
algorithm - 'SAMME.R', n_estimators 15.
