#!/usr/bin/env python
# coding: utf-8

# # Task 1

# ## Step 1

# In[2]:


import pandas as pd


# In[8]:


adult_train = pd.read_csv('data/adult_train.csv')
adult_test = pd.read_csv('data/adult_test.csv')

Xtrain = adult_train.drop('target', axis=1)
Ytrain = adult_train['target']

Xtest = adult_test.drop('target', axis=1)
Ytest = adult_test['target']


# ## Step 2

# In[32]:


from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score


# In[45]:


dicts_training_data = Xtrain.to_dict('records')
dicts_testing_data = Xtest.to_dict('records')

dv = DictVectorizer()
X_train_encoded = dv.fit_transform(dicts_training_data)
X_test_encoded = dv.transform(dicts_testing_data)


# In[31]:


clf_forest = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0)

cv_result = cross_validate(clf_forest, X_train_encoded, Ytrain, cv=5, scoring='accuracy')


# In[35]:


print(cv_result)


# In[46]:


clf_forest.fit(X_train_encoded, Ytrain)
predictions = clf_forest.predict(X_test_encoded)


# In[49]:


accuracy_score(Ytest, predictions)


# ## Step 3

# In[50]:


from sklearn.pipeline import make_pipeline


# In[58]:


pipeline = make_pipeline(
  DictVectorizer(),
  RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0)
)


# In[59]:


pipeline


# In[60]:


pipeline.fit(Xtrain, Ytrain)
accuracy_score(Ytest, pipeline.predict(Xtest))

