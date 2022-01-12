#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


bank= pd.read_csv('F:/Dataset/bank-full.csv',sep=";")


# In[5]:


bank


# In[6]:


bank.info()


# In[9]:


bank1= pd.get_dummies(bank,columns=['job','marital','education','contact','poutcome'])


# In[10]:


bank1


# In[14]:


pd.set_option("display.max.columns",None)


# In[15]:


bank1


# In[17]:


bank1['default']=np.where(bank1['default'].str.contains('yes'),1,0)


# In[18]:


bank1['housing']=np.where(bank1['housing'].str.contains('yes'),1,0)


# In[19]:


bank1['loan']=np.where(bank1['loan'].str.contains('yes'),1,0)


# In[20]:


bank1['y']=np.where(bank1['y'].str.contains('yes'),1,0)


# In[22]:


bank1


# In[24]:


bank1['month'].value_counts()


# In[25]:


order = {'month':{'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}}


# In[26]:


bank1=bank1.replace(order)


# In[27]:


bank


# In[28]:


bank1.info()


# In[36]:


x=pd.concat([bank1.iloc[:,0:11],bank1.iloc[:,12:]],axis=1)


# In[40]:


y= bank1.iloc[:,11]


# In[66]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


# In[67]:


classifier= LogisticRegression()
classifier.fit(x,y)


# In[68]:


y_pred = classifier.predict(x)


# In[70]:


y_predictDataFrame= pd.DataFrame({'actual':y,'predict_prob':y_pred})


# In[71]:


y_predictDataFrame


# In[72]:


confusion_matrix=confusion_matrix(y,y_predict)
print(confusion_matrix)


# In[73]:


(39107+1274)/(39107+4015+815+1274)


# In[75]:


from sklearn.metrics import classification_report
print(classification_report(y,y_pred))


# In[77]:


fpr, tpr, thresholds = roc_curve(y, classifier.predict_proba (x)[:,1])
auc = roc_auc_score(y, y_pred)
plt.plot(fpr, tpr, color='blue', label='logit model ( area  = %0.2f)'%auc)
plt.plot([0, 1], [0, 1], 'k--')


# In[78]:


auc


# In[ ]:




