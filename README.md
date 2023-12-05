
lime-code-in-python


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()

print(fetch_california_housing()['DESCR'][200:1420])

# Separating data into feature variable X and target variable y respectively
from sklearn.model_selection import train_test_split
X = fetch_california_housing()['data']
y = fetch_california_housing()['target']

# Extracting the names of the features from data
features = fetch_california_housing()['feature_names']

# Splitting X & y into training and testing set
X_train, X_test, y_train, y_test = train_test_split(
	X, y, train_size=0.90, random_state=50)
 
 # Creating a dataframe of the data, for a visual check
df = pd.concat([pd.DataFrame(X), pd.DataFrame(y)], axis=1)
df.columns = np.concatenate((features, np.array(['label'])))
print("Shape of data =", df.shape)

# Printing the top 5 rows of the dataframe
df.head()

#INSTANTIATING THE PREDICTION MODEL AND TRAINING IT ON (x_train, y_train) 

# Instantiating the prediction model - an extra-trees regressor
from sklearn.ensemble import ExtraTreesRegressor
reg = ExtraTreesRegressor(random_state=50)

#Fitting the predictino model onto the training set
reg.fit(X_train, y_train)

#Checking the model's performance on the test set
print('R2 score for the model on test set =', reg.score(X_test, y_test))

INSTANTIATING THE EXPLAINER OBJECT
# Importing the module for LimeTabularExplainer
from lime import lime_tabular

# Instantiating the explainer object by passing in the training set,
# and the extracted features
explainer_lime = lime_tabular.LimeTabularExplainer(X_train,
												feature_names=features,
												verbose=True,
												mode='regression')

# Index corresponding to the test vector
i = 10

# Number denoting the top features
k = 5

# Calling the explain_instance method by passing in the:
#    1) ith test vector
#    2) prediction function used by our prediction model('reg' in this case)
#    3) the top features which we want to see, denoted by k
exp_lime = explainer_lime.explain_instance(
    X_test[i], reg.predict, num_features=k)

 # Finally visualizing the explanations
exp_lime.show_in_notebook() 
