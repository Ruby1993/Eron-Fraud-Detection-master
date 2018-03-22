Project Summary

1. Summarize for us the goal of this project and how machine learning is useful
in trying to accomplish it. As part of your answer, give some background on the
dataset and how it can be used to answer the project question.
Were there any outliers in the data when you got it, and how did you handle those?

Project Background
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

Problem Simplification
Identify persons who are in the corporate fraud based on the available data
within financial features, email features, and available 'poi' labels.

Outliers Issue
Based on data exploration, there are several outliers I found and will remove,
- TOTAL: This row is the sum of each variable, which need to removed.
- LOCKHART EUGENE E: Except the POI label(false), there is no data in all other
variables.
- THE TRAVEL AGENCY IN THE PARK: Not related to the person based on the name.

2. What features did you end up using in your POI identifier, and what selection
process did you use to pick them? Did you have to do any scaling? Why or why not?
As part of the assignment, you should attempt to engineer your own feature that
does not come ready-made in the dataset -- explain what feature you tried to make,
and the rationale behind it. (You do not necessarily have to use it in the final
analysis, only engineer and test it.) In your feature selection step, if you used
an algorithm like a decision tree, please also give the feature importances of
the features that you use, and if you used an automated feature selection
function like SelectKBest, please report the feature scores and reasons for
your choice of parameter values.  [relevant rubric items: “create new features”,
“intelligently select features”, “properly scale features”]

Feature selection process

Before feature selection, I added three features related to ratio to make the data
more reasonable, which are listed below,

received_from_poi_ratio: Received from poi/all message the person received
(to_messages: the person was included in the to part)
sent_to_poi_ratio: Sent to poi /all message the person sent
(from_messages: the person was included in the from part)
shared_receipt_with_poi_ratio: 'shared_receipt_with_poi'/'to_messages'
[only one record is over 1, which might be a typero, here I just assumed it is 1]

As the number scale in different features are different, I used the MinmaxScaler to
map all the values to 0-1, although tree-based model do not require that. Then
I used two methods one is treebased model and one is SelectKBest. The detaild
score was embeded in the project doc, and what i focused on is the common lowest
scored features which i decide to exclude them in the model, which I listed below,

restricted_stock_deferred
from_message
to_message
deferral_payments
director_fees

3. What algorithm did you end up using? What other one(s) did you try?
How did model performance differ between algorithms?
[relevant rubric item: “pick an algorithm”]

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

4. What does it mean to tune the parameters of an algorithm, and
what can happen if you don’t do this well?  How did you tune the parameters of
your particular algorithm? What parameters did you tune?
(Some algorithms do not have parameters that you need to tune -- if this is the
case for the one you picked, identify and briefly explain how you would have
done it for the model that was not your final choice or a different model that
does utilize parameter tuning, e.g. a decision tree classifier).
[relevant rubric items: “discuss parameter tuning”, “tune the algorithm”]

For the model tuning, if we don't do it, it will impact the model performance. In
the project, I used grid search to tune the model and try to get the best parameter.
For the models I used, below are listed the parameters I tuned,
LogisticRegression- Penalty; C
Support Vector Machine- C; Degree; Kernel
KNN- n_neighbors
Random Forest- n_estimators; max_depth
AdaBoost- n_estimators; algorithm
GradientBoosting - loss; n_estimators; max_depth

5. What is validation, and what’s a classic mistake you can make if you do it
wrong? How did you validate your analysis?
[relevant rubric items: “discuss validation”, “validation strategy”]

for the validation, I used the k-fold cross validation to get the prediction values
and used the precision, recall, f1 to evaluate the model performance. The common
mistake would be only using accuracy to validate the model as the dataset is not balanced.

6. Give at least 2 evaluation metrics and your average performance for each of them.
Explain an interpretation of your metrics that says something
human-understandable about your algorithm’s performance.
[relevant rubric item: “usage of evaluation metrics”]

a. How many are classified 'fraud' correctly out of all truely 'fraud' people.
if it is good, it will identify most of people involved in the fraud issue, but might
bring some innocent people.

b. How many are classified 'fraud' correctly out of all labeled 'fraud' people.
if it is good, it will make sure all the labeled fraud people are higher likely involved
in the fraud, but it might miss some people involved in the fraud issue.
