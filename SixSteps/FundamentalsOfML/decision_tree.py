
import pydotplus
import numpy as np
import pandas as pd

from sklearn import tree
from sklearn import datasets, metrics
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.externals.six import StringIO


iris = datasets.load_iris()

# X = iris.data[:, [2, 3]]
X = iris.data
y = iris.target

sc = StandardScaler()
sc.fit(X)

# split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

clf = tree.DecisionTreeClassifier(criterion='entropy', random_state=0)
clf.fit(X_train, y_train)

# generate evaluation metrics
print("Train - Accuracy:", metrics.accuracy_score(y_train, clf.predict(X_train)))
print("Train - Confusion matrix :", metrics.confusion_matrix(y_train, clf.predict(X_train)))
print("Train - classification report: ", metrics.classification_report(y_train, clf.predict(X_train)))
print("Test - Accuracy:", metrics.accuracy_score(y_test, clf.predict(X_test)))
print("Test - Confusion matrix :", metrics.confusion_matrix(y_test, clf.predict(X_test)))
print("Test - classification report: ", metrics.classification_report(y_test, clf.predict(X_test)))

tree.export_graphviz(clf, out_file='tree.dot')

out_data = StringIO()
tree.export_graphviz(clf, out_file=out_data,
                     feature_names=iris.feature_names,
                     class_names=clf.classes_.astype(int).astype(str),
                     filled=True, rounded=True,
                     special_characters=True,
                     node_ids=1,)
graph = pydotplus.graph_from_dot_data(out_data.getvalue())
graph.write_pdf("iris.pdf")    # save to pdf
