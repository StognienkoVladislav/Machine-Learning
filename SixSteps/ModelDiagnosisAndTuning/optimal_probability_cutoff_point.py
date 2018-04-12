import pandas as pd
import numpy as np
import pylab as plt

from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


""" Load data and check the class distribution """

# read the data in
df = pd.read_csv("../Data/data_4/Diabetes.csv")

# target variable % distribution
print(df['class'].value_counts(normalize=True))

""" Build logistic regression model and evaluate performance """
X = df.ix[:, :8]
y = df['class']

# evaluate the model by splitting into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# instantiate a logistic regression model, and fit
model = LogisticRegression()
model = model.fit(X_train, y_train)

# predict class labels for the train set. The predict function converts probability values > .5 to 1 else 0
y_pred = model.predict(X_train)

# generate class probabilities
# Notice that 2 elements will be returned in probs array,
# 1st element is probability fir negative class,
# 2nd element gives probability for positive clas
probs = model.predict_proba(X_train)
y_pred_prob = probs[:, 1]

# generate evaluation metrics
print("Accuracy: ", metrics.accuracy_score(y_train, y_pred))


""" Find optimal cutoff point """
# extract false positive, true positive rate
fpr, tpr, thresholds = metrics.roc_curve(y_train, y_pred_prob)
roc_auc = metrics.auc(fpr, tpr)
print("Area under the ROC curve : {}".format(roc_auc))

i = np.arange(len(tpr)) # index for df
roc = pd.DataFrame({'fpr': pd.Series(fpr, index=i), 'tpr' : pd.Series(tpr,
                    index=i), '1-fpr': pd.Series(1-fpr, index=i), 'tf': pd.Series(tpr -
                    (1-fpr), index=i), 'thresholds': pd.Series(thresholds, index=i)})

roc.ix[(roc.tf-0).abs().argsort()[:1]]

# plot tpr vs 1-fpr
fig, ax = plt.subplots()
plt.plot(roc['tpr'], label='tpr')
plt.plot(roc['1-fpr'], color='red', label='1-fpr')
plt.legend(loc='best')
plt.xlabel('1-False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.show()
