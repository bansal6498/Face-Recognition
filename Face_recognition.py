#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import face_recognition_models as fr


# In[2]:


images = os.listdir('D:\Data for Machine Learning Projects\Face Rec\Images')


# In[3]:


image_mactched= fr.load_image_file('my name.jpg')


# In[4]:


image_matched_encod= fr.face_encodings(image_matched)[0]


# In[ ]:


for image in images:
    current_image= fr.load_image_file('images/' +image)
    current_image_encod= fr.face_encodings(current_image)[0]
    result= fr.compare_faces([image_matched_encod],current_image_encod)
    if result[0] == True:
        print('Matched: '=image)
    else:
        print('Not Matched: '+image)

