# Music Recommender Demo

## Model
This demo is base on [https://www.youtube.com/watch?v=_uQrJ0TkZlc&t=17963s] and using [coremltools](https://pypi.python.org/pypi/coremltools) python package.
The model have a new user with this profile, what is the kind of music, that this user is interested in, the model will say jazz, or hip hop, or whatever.
Steps in the machine learning projects:
1. Import the Data
2. Clean the Data
3. Split the Data into Training/Test Sets
4. Create a Model
5. Train the Model
6. Make Predictions
7. Evaluate and Improve

## Read the csv file
``` import pandas as pd ```

## Library: side kick learn
```
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
```
## Import the data
```
music_data = pd.read_csv('music.csv')
X = music_data.drop(columns=['genre'])
y = music_data['genre']
```

## Train 4 variables
``` X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) ```

## Create a model: create a new instance of this class
``` model = DecisionTreeClassifier() ```

## Train model
```
model.fit(X, y)

tree.export_graphviz(model, out_file='music-recommender.dot',
                    feature_names=['age', 'gender'],
                    class_names=sorted(y.unique()),
                    label='all',
                    rounded=True,
                    filled=True)

joblib.dump(model, 'music-recommender.joblib')
```
## Predictions: 2 dimensonal array
```
predictions = model.predict(X_test)

score = accuracy_score(y_test, predictions)
score
```
