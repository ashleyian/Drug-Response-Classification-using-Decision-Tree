#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np

print("Libraries imported successfully")


# In[4]:


df = pd.read_csv("C:/Users/Ashley/Downloads/drug200/drug200.csv")

print("data loaded successfully")


# In[6]:


print(df.head())


# In[8]:


print(df.info())


# In[10]:


print(df['Sex'].value_counts())


# In[12]:


from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Libraries imported successflly!")


# In[13]:


the_sex = LabelEncoder()
the_bp = LabelEncoder()
the_chol = LabelEncoder()

the_sex.fit(df['Sex'])
print("Sex Mappings: ", dict(zip(the_sex.classes_, the_sex.transform(the_sex.classes_))))

the_bp.fit(df['BP'])
print("BP Mapping: ", dict(zip(the_bp.classes_, the_bp.transform(the_bp.classes_))))

the_chol.fit(df['Cholesterol'])
print("Cholesterol Mappings: ", dict(zip(the_chol.classes_, the_chol.transform(the_chol.classes_))))

df['Sex'] = the_sex.transform(df['Sex'])
df['BP'] = the_bp.transform(df['BP'])
df['Cholesterol'] = the_chol.transform(df['Cholesterol'])


# In[14]:


x = df.drop('Drug', axis = 1)
y = df['Drug']


# In[17]:


x_train, x_test,y_train, y_test = train_test_split(
x, y, test_size = 0.2, random_state = 42)


# In[19]:


model = DecisionTreeClassifier(criterion = 'entropy', max_depth = 4, random_state = 42)
model.fit (x_train, y_train) 


# In[20]:


y_pred = model.predict(x_test)


# In[21]:


print("Accuracy Score: ", accuracy_score(y_test,y_pred))


# In[23]:


print("Classification report: ", classification_report(y_test, y_pred))


# In[25]:


print("confusion matrix: ", confusion_matrix(y_test, y_pred) )


# In[27]:


plt.figure (figsize = (30,20))
plot_tree(
    model,
    feature_names = x.columns,
    class_names = model.classes_,
    filled = True,
    rounded = True
)
plt.title("Decision Tree Classification")
plt.show()

