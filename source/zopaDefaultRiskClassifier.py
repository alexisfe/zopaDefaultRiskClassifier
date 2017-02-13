#https://www.zopa.com/public-loan-book#download
http://nbviewer.jupyter.org/gist/justmarkham/6d5c061ca5aee67c4316471f8c2ae976#Description-of-Variables
#Problem statement
#Data exploration
#Feature engineering
#Algorithm selection and training
#Hyperparameter tuning on validation set
#Algorithm validation
#Algorithm deployment

import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import Imputer
import matplotlib.pyplot as plt

input_dirpath = 'input_data'
input_filepath = 'data_for_loanbook_extract_2017-02-01.csv'

zopaBook = pd.read_csv(os.path.join(os.pardir, input_dirpath, input_filepath))

#Feature engineering
zopaBook['Disbursal date'] = pd.to_datetime(zopaBook['Disbursal date'])
zopaBook['Disbursal year'] = zopaBook['Disbursal date'].dt.year
zopaBook['Disbursal month'] = zopaBook['Disbursal date'].dt.month
zopaBook['Disbursal week day'] = zopaBook['Disbursal date'].dt.weekday
zopaBook['Disbursal month day'] = zopaBook['Disbursal date'].dt.day
zopaBook['Total loans'] = zopaBook.groupby(['Encrypted Member ID'])['Encrypted Loan ID'].transform('count')

zopaUnactiveBook = zopaBook.loc[zopaBook['Latest Status'].isin(['Completed', 'Default'])]
zopaActiveBook = zopaBook.loc[zopaBook['Latest Status'] == 'Active']

label = 'Latest Status'
predictors = ['Disbursal year', 'Disbursal month', 'Disbursal week day', 'Disbursal month day', 'Total loans', 'Total number of payments', 'Original Loan Amount', 'Lending rate', 'Term']

X_train, X_test, y_train, y_test = train_test_split(zopaUnactiveBook[predictors], zopaUnactiveBook[label], test_size=0.2, random_state=1)

lr_params = [{
    'tol':[1e-3, 1e-4, 1e-5],
    'C':[1e-2, 1e-1, 1, 1e1, 1e2, 1e3],
    'max_iter':[100, 200, 300, 400, 500],
    'class_weight': [{'Default': 10}, {'Default': 50}, {'Default': 100}]
}]

rfc_params = [{
    'n_estimators': [5, 10, 15, 20],
    'criterion': ['gini', 'entropy'],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [2, 5, 10],
    'class_weight': [{'Default': 10}, {'Default': 50}, {'Default': 100}]
}]

lr = LogisticRegression(penalty='l2', dual=False, fit_intercept=True, intercept_scaling=1, random_state=1, solver='liblinear', warm_start=True, n_jobs=-1)
rfc = RandomForestClassifier(bootstrap=True, random_state=1, warm_start=True, n_jobs=-1)

clf = GridSearchCV(estimator=rfc, param_grid=rfc_params, n_jobs=-1)
print "Fitting model..."
clf.fit(X_train, y_train)

print "Best estimator found by GridSearchCV: "
print clf.best_estimator_
print "with a score of: "
print clf.best_score_
print "using scorer: "
print clf.scorer_
print "and params: "
print clf.best_params_

print "Predicting on test set..."
y_pred = clf.predict(X_test)

print("Classification report for model %s:\n%s\n"
      % (clf, classification_report(y_test, y_pred)))
print("Confusion matrix:\n%s" % confusion_matrix(y_test, y_pred))

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(false_positive_rate, true_positive_rate)

plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'b',
label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()