Breast Cancer Deep Learning Model
from google.colab import files

uploaded = files.upload()

for fn in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))

import pandas as pd
import io

data = pd.read_csv(io.StringIO(uploaded['data .csv'].decode('utf-8')))

data.head()

del data ['Unnamed: 32']

data.head()

import seaborn as sns

import matplotlib.pyplot as plt

ax = sns.countplot(x='diagnosis', data = data, label='count')

plt.title('Diagnosis Count')

plt.xlabel('Diagnosis')
plt.ylabel('Count')

B, M = data['diagnosis'].value_counts(

)

print('Benign:', B)
print('Malignant:', M)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#settin the dependent and independent columns range
X = data.iloc[:, 2:].values
y = data.iloc[:, 1].values


#Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_X_1 = LabelEncoder()

#encoder deals with missing values in data set
y = labelencoder_X_1.fit_transform(y)


#Splitting the dataset into training and test dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=0)


#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



X_train


X_test


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

# Adding the input and first hidden layer
classifier = Sequential()
classifier.add(Dense(units=16, kernel_initializer='uniform', activation='relu', input_dim=30))
classifier.add(Dropout(rate=0.5))  # Use rate instead of p

# Adding the second hidden layer
classifier.add(Dense(units=16, kernel_initializer='uniform', activation='relu'))
classifier.add(Dropout(rate=0.5))  # Use rate instead of p

# Adding the output layer
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

classifier.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])

# Assuming you have X_train and y_train datasets
classifier.fit(X_train, y_train, batch_size=100, epochs=150)


#predicting the test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)


#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm,annot=True)
plt.savefig('h.png')


#First Accuracy after training
(65+44)/114

(64+44)/114


!pip install keras scikit-learn


!pip install scikeras

from scikeras.wrappers import KerasClassifier

from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense


def built_classifier():
    classifier = Sequential()
    classifier.add(Dense(units=16, kernel_initializer='uniform', activation='relu', input_dim=30))
    classifier.add(Dense(units=16, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer="Adam", loss='binary_crossentropy', metrics=['accuracy'])
    return classifier


classifier = KerasClassifier(build_fn=built_classifier, batch_size=100, epochs=100)
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10, n_jobs=-1)


print(accuracies)


accuracies.mean()


accuracies.std()


import keras
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

def built_classifier(optimizer='adam'):
    classifier = Sequential()
    classifier.add(Dense(units=16, kernel_initializer='uniform', activation='relu', input_dim=30))
    classifier.add(Dense(units=16, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return classifier


classifier = KerasClassifier(build_fn=built_classifier)
parameters = {'batch_size': [10, 32], 'epochs': [100, 500], 'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, scoring='accuracy', cv=10)


grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

