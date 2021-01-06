import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sn

data = read_csv('train.csv')

#Extract attribute names from the data frame

feat = data.keys()
print(feat)
#feat_labels = feat.get_values()

#Extract data values from the data frame

dataset = data.values
print('Shape = ',dataset.shape)
X = dataset[:,0:94]
y = dataset[:,94]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
model = RandomForestClassifier(n_estimators=20)
model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print('Score = ', score)

y_predicted = model.predict(X_test)

cm = confusion_matrix(y_test, y_predicted)
print('Confusion Matrix')
print(cm)

plt.figure(figsize=(10,7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()




