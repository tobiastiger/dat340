#!/usr/bin/env python
# coding: utf-8

# In[27]:


import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# In[3]:


# Read the CSV file.
data = pd.read_csv('data/CTG.csv', skiprows=1)

# Select the relevant numerical columns.
selected_cols = ['LB', 'AC', 'FM', 'UC', 'DL', 'DS', 'DP', 'ASTV', 'MSTV', 'ALTV',
                 'MLTV', 'Width', 'Min', 'Max', 'Nmax', 'Nzeros', 'Mode', 'Mean',
                 'Median', 'Variance', 'Tendency', 'NSP']
data = data[selected_cols].dropna()

# Shuffle the dataset.
data_shuffled = data.sample(frac=1.0, random_state=0)

# Split into input part X and output part Y.
X = data_shuffled.drop('NSP', axis=1)

# Map the diagnosis code to a human-readable label.
def to_label(y):
    return [None, 'normal', 'suspect', 'pathologic'][(int(y))]

Y = data_shuffled['NSP'].apply(to_label)

# Partition the data into training and test sets.
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=0)


# In[13]:


clf_dummy = DummyClassifier(strategy='most_frequent')
cross_val_score(clf_dummy, Xtrain, Ytrain)


# In[14]:


clf_tree = DecisionTreeClassifier(random_state=0)
cross_val_score(clf_tree, Xtrain, Ytrain)


# In[16]:


clf_forest = RandomForestClassifier(random_state=0)
cross_val_score(clf_forest, Xtrain, Ytrain)


# In[26]:


clf_SVM = make_pipeline(StandardScaler(),
                    LinearSVC(random_state=0, tol=1e-5, max_iter=10000))
cross_val_score(clf_SVM, Xtrain, Ytrain)


# In[29]:


clf_forest.fit(Xtrain, Ytrain)
Yguess = clf_forest.predict(Xtest)
print(accuracy_score(Ytest, Yguess))


# ## Task 3

# In[ ]:


# Read the CSV file using Pandas.
alldata = pd.read_csv(LOCATION_OF_YOUR_FILE)

# Convert the timestamp string to an integer representing the year.
def get_year(timestamp):
    return int(timestamp[:4])
alldata['year'] = alldata.timestamp.apply(get_year)

# Select the 9 input columns and the output column.
selected_columns = ['price_doc', 'year', 'full_sq', 'life_sq', 'floor', 'num_room', 'kitch_sq', 'full_all']
alldata = alldata[selected_columns]
alldata = alldata.dropna()

# Shuffle.
alldata_shuffled = alldata.sample(frac=1.0, random_state=0)

# Separate the input and output columns.
X = alldata_shuffled.drop('price_doc', axis=1)
# For the output, we'll use the log of the sales price.
Y = alldata_shuffled['price_doc'].apply(np.log)

# Split into training and test sets.
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=0)


# In[ ]:


m1 = DummyRegressor()
cross_validate(m1, Xtrain, Ytrain, scoring='neg_mean_squared_error')


# In[ ]:


regr.fit(Xtrain, Ytrain)
mean_squared_error(Ytest, regr.predict(Xtest))

