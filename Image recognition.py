#!/usr/bin/env python
# coding: utf-8

# In[494]:


import os
import warnings
warnings.simplefilter('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from skimage.io import imread,imshow
from skimage.transform import resize
from skimage.color import rgb2gray
from sklearn.model_selection import train_test_split


# In[495]:


yogesh=os.listdir("C:/Users/Yogesh/Desktop/picture daya/Image-recognition/pewdiepiw/")
ashok=os.listdir("C:/Users/Yogesh/Desktop/picture daya/Image-recognition/jk/")
scarlet=os.listdir("C:/Users/Yogesh/Desktop/picture daya/Image-recognition/jenna/")


# In[496]:


limit=10
yogesh_images = [None]*limit
ashok_images = [None]*limit
scarlet_images = [None]*limit
j=0
for i in yogesh:
    if (j<limit):
        yogesh_images[j]= imread("C:/Users/Yogesh/Desktop/picture daya/Image-recognition/pewdiepiw/"+i)
        j+=1
    else:
        break

j=0
for i in ashok:
    if(j<limit):
        ashok_images[j]=imread("C:/Users/Yogesh/Desktop/picture daya/Image-recognition/jk/"+i)
        j+=1
    else:
        break

j=0
for i in scarlet:
    if(j<limit):
        scarlet_images[j]=imread("C:/Users/Yogesh/Desktop/picture daya/Image-recognition/jenna/"+i)
        j+=1
    else:
        break


# In[497]:


yogesh_gray=[None]*limit
ashok_gray=[None]*limit
scarlet_gray=[None]*limit
j=0

for i in yogesh:
    if(j<limit):
        yogesh_gray[j]=rgb2gray(yogesh_images[j])
        j+=1
    else:
        break
j=0
for i in ashok:
    if(j<limit):
        ashok_gray[j]=rgb2gray(ashok_images[j])
        j+=1
    else:
        break
    

j=0
for i in scarlet:
    if(j<limit):
        scarlet_gray[j]=rgb2gray(scarlet_images[j])
        j+=1
    else:
        break


# In[498]:


imshow(scarlet_gray[0])


# In[499]:


imshow(yogesh_gray[0])


# In[500]:


imshow(ashok_gray[0])


# In[501]:


scarlet_gray[2].shape


# In[502]:


for j in range(10):
    scarlet_temp=scarlet_gray[j]
    scarlet_gray[j]=resize(scarlet_temp,(512,512))


# In[503]:


for j in range(10):
    yogesh_temp=yogesh_gray[j]
    yogesh_gray[j]=resize(yogesh_temp,(512,512))
for j in range(10):
    ashok_temp=ashok_gray[j]
    ashok_gray[j]=resize(ashok_temp,(512,512))


# In[504]:


imshow(ashok_gray[2])


# In[505]:


len_of_scarlet=len(scarlet_gray)
len_of_ashok=len(ashok_gray)
len_of_yogesh=len(yogesh_gray)


# In[506]:


image_size_scarlet=scarlet_gray[1].shape
image_size_yogesh=yogesh_gray[1].shape
image_size_ashok=ashok_gray[1].shape


# In[507]:


image_size_scarlet


# In[508]:


flatten_size_scarlet=image_size_scarlet[0]*image_size_scarlet[1]
flatten_size_yogesh=image_size_yogesh[0]*image_size_yogesh[1]
flatten_size_ashok=image_size_ashok[0]*image_size_ashok[1]


# In[509]:


flatten_size_scarlet


# In[510]:


for i in range(len_of_scarlet):
    scarlet_gray[i]= np.ndarray.flatten(scarlet_gray[i].reshape(flatten_size_scarlet,1))
for i in range(len_of_yogesh):
    yogesh_gray[i]= np.ndarray.flatten(yogesh_gray[i].reshape(flatten_size_yogesh,1))
for i in range(len_of_ashok):
    ashok_gray[i]= np.ndarray.flatten(ashok_gray[i].reshape(flatten_size_ashok,1))


# In[511]:


scarlet_gray=np.dstack(scarlet_gray)
yogesh_gray=np.dstack(yogesh_gray)
ashok_gray=np.dstack(ashok_gray)


# In[512]:


ashok_gray.shape


# In[513]:


scarlet_gray=np.rollaxis(scarlet_gray,axis=2,start=0)
yogesh_gray=np.rollaxis(yogesh_gray,axis=2,start=0)
ashok_gray=np.rollaxis(ashok_gray,axis=2,start=0)
ashok_gray.shape


# In[514]:


yogesh_gray=np.rollaxis(yogesh_gray,axis=2,start=1)
scarlet_gray=np.rollaxis(scarlet_gray,axis=2,start=1)
ashok_gray=np.rollaxis(ashok_gray,axis=2,start=1)


# In[515]:


ashok_gray.shape


# In[516]:


scarlet_gray=scarlet_gray.reshape(len_of_scarlet,flatten_size_scarlet)
yogesh_gray=yogesh_gray.reshape(len_of_yogesh,flatten_size_yogesh)
ashok_gray=ashok_gray.reshape(len_of_ashok,flatten_size_ashok)


# In[517]:


yogesh_gray.shape


# In[518]:


scarlet_data=pd.DataFrame(scarlet_gray)
ashok_data=pd.DataFrame(ashok_gray)
yogesh_data=pd.DataFrame(yogesh_gray)


# In[519]:


yogesh_data


# In[520]:


scarlet_data["label"]="Scarlet"
yogesh_data["label"]="Yogesh"
ashok_data["label"]="Ashok Selvan"


# In[521]:


ashok_data


# In[522]:


img_1=pd.concat([scarlet_data,yogesh_data])


# In[523]:


img=pd.concat([img_1,ashok_data])


# In[524]:


img


# In[525]:


from sklearn.utils import shuffle


# In[526]:


img_shuffle = shuffle(img).reset_index()


# In[527]:


img_shuffle


# In[528]:


img_shuffle=img_shuffle.drop(['index'],axis=1)


# In[529]:


img_shuffle


# In[530]:


img_shuffle.shape


# In[531]:


x = img_shuffle.values[:,:-1]


# In[532]:


y= img_shuffle.values[:,-1]


# In[533]:


x


# In[534]:


y


# In[535]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)


# In[536]:


from sklearn import svm


# In[537]:


clf=svm.SVC()
clf.fit(x_train,y_train)


# In[538]:


y_pred=clf.predict(x_test)


# In[539]:


y_pred


# In[540]:


for i in (np.random.randint(0,6,4)):
    predicted_images = (np.reshape(x_test[i],(512,512)).astype(np.float64))
    plt.title('predicited label:{0}'.format(y_pred[i]))
    plt.imshow(predicted_images,interpolation='nearest',cmap='gray')
    plt.show()


# In[541]:


from sklearn import metrics


# In[542]:


accuracy=metrics.accuracy_score(y_test,y_pred)


# In[543]:


accuracy


# In[392]:


from sklearn.metrics import confusion_matrix


# In[393]:


confusion_matrix(y_test,y_pred)


# In[ ]:




