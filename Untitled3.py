
# coding: utf-8

# In[16]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn


# In[129]:


names=['sepal-length','sepal-width','petal-length','petal-width','class']
Data= pd.read_csv('/home/yanhua/hw3/hw3_dataset/iris.txt',names=names)
#f=open('/home/yanhua/hw3/hw3_dataset/iris.txt', 'r').readlines()
#dataset = list(f)
#print (dataset)
ir1=Data.iloc[:,0].values #  sepal-length
ir2=Data.iloc[:,1].values #  sepal-width
ir3=Data.iloc[:,2].values #  petal-length  
ir4=Data.iloc[:,3].values #  petal-width
sort=Data.iloc[:,4].values # Classsig
X=Data.iloc[:,[0,1,2,3]].values


# In[130]:


U=np.mean(X,axis=0)
x=X-U
sigma=np.mean(np.multiply(x,x),axis=0)
x=x/(sigma**0.5)


# In[131]:


C=1/150*np.dot(x.T,x)
d=np.cov(X.T)


# In[132]:


evalues,evectors=np.linalg.eig(C)


# In[133]:


sorted_indices=np.argsort(-evalues)
sorted_evectors=evectors[:,sorted_indices]


# In[134]:


X_v=X-U
std_x=np.std(X_v)
X_v=X_v/std_x
print (X_v.shape)
print (sorted_evectors.shape)


# In[139]:


from sklearn.metrics import mean_squared_error as mse
#component=n
def nrms(n):
    eigVec =sorted_evectors[:,:n]
    x_proj=np.dot(X_v,eigVec)

    x_rec=np.dot(x_proj,eigVec.T)/std_x

    x_rec=x_rec+np.mean(X,axis=0)

    t=np.max(X,axis=0)-np.min(X,axis=0)
    y=X-x_rec
    d_err=np.sqrt(np.mean(np.power(y,2),axis=0))/t

    nrms=np.mean(d_err)
    nmse=mse(X,x_rec)
    print (nrms)

