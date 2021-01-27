#!/usr/bin/env python
# coding: utf-8

# # Task 1

# ## Step 1

# In[1]:


import pandas as pd


# In[2]:


adult_train = pd.read_csv('data/adult_train.csv')
adult_test = pd.read_csv('data/adult_test.csv')

Xtrain = adult_train.drop('target', axis=1)
Ytrain = adult_train['target']

Xtest = adult_test.drop('target', axis=1)
Ytest = adult_test['target']


# ## Step 2

# In[3]:


from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score


# In[4]:


dicts_training_data = Xtrain.to_dict('records')
dicts_testing_data = Xtest.to_dict('records')

dv = DictVectorizer()
X_train_encoded = dv.fit_transform(dicts_training_data)
X_test_encoded = dv.transform(dicts_testing_data)


# In[5]:


clf_forest = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0)

cv_result = cross_validate(clf_forest, X_train_encoded, Ytrain, cv=5, scoring='accuracy')


# In[6]:


print(cv_result)


# In[7]:


clf_forest.fit(X_train_encoded, Ytrain)
predictions = clf_forest.predict(X_test_encoded)


# In[8]:


accuracy_score(Ytest, predictions)


# ## Step 3

# In[20]:


from sklearn.pipeline import make_pipeline


# In[25]:


Xtrain_dict = Xtrain.to_dict('records')
Xtest_dict = Xtest.to_dict('records')


# In[30]:


pipeline = make_pipeline(
  DictVectorizer(),
  RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0)
)


# In[32]:


pipeline.fit(Xtrain_dict, Ytrain)
accuracy_score(Ytest, pipeline.predict(Xtest_dict))


# # Task 2

# ## DecisionTreeClassifier

# In[45]:


import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


# In[113]:


N = 21
results = np.empty((0, 4), int)
for md in range(1, N):
    clf_tree = DecisionTreeClassifier(max_depth=md, random_state=0)
    clf_tree.fit(X_train_encoded, Ytrain)
    predictions = clf_tree.predict(X_test_encoded)
    accuracy = accuracy_score(Ytest, predictions)
    precision = precision_score(Ytest, predictions, pos_label='<=50K')
    recall = recall_score(Ytest, predictions, pos_label='<=50K')
    results = np.append(results, np.array([[md, accuracy, precision, recall]]), axis=0)


# In[114]:


print(results)


# In[50]:


plt.plot(results[:,0], results[:,1])
plt.show()


# ## RandomForestClassifier

# In[151]:


N_TREES = [1, 50, 100, 150]
MD = 5
results_forest = np.zeros((MD, 4, len(N_TREES)))
i = 0
j = 0
for n_trees in N_TREES:
    for md in range(1, MD + 1):
        clf_tree = DecisionTreeClassifier(max_depth=md, random_state=0)
        clf_tree.fit(X_train_encoded, Ytrain)
        predictions = clf_tree.predict(X_test_encoded)
        accuracy = accuracy_score(Ytest, predictions)
        precision = precision_score(Ytest, predictions, pos_label='<=50K')
        recall = recall_score(Ytest, predictions, pos_label='<=50K')
        results_forest[i, :, j] = [md, accuracy, precision, recall]
        j += 1
    i += 1
    j = 0


# In[149]:


plt.plot(results_forest[0, 0, :], results_forest[0, 1, :])
plt.show()

