#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf


# In[3]:


get_ipython().system('pip install librosa')


# In[4]:


pip install matplotlib


# In[5]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


filename=r"C:\Users\hp\Downloads\UrbanSound8K\UrbanSound8K\dogbark.wav"


# In[7]:


import IPython.display as ipd
import librosa
import librosa.display


# In[11]:


get_ipython().system('dir')


# In[8]:


### Dog Sound
plt.figure(figsize=(14,5))
data,sample_rate=librosa.load(filename)
librosa.display.waveshow(data,sr=sample_rate)
ipd.Audio(filename)


# In[9]:


sample_rate


# In[15]:


wave_audio


# In[10]:


data


# In[11]:


import pandas as pd


# In[18]:


pip install pandas


# In[12]:


import pandas as pd


# In[13]:


metadata=pd.read_csv(r"C:\Users\hp\Downloads\UrbanSound8K\UrbanSound8K\metadata\UrbanSound8K.csv")


# In[14]:


metadata.head(10)


# In[15]:


metadata['class'].value_counts()


# In[17]:


mfccs = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40)
print(mfccs.shape)


# In[18]:


mfccs


# In[20]:


#### Extracting MFCC's For every audio file
import pandas as pd
import os
import librosa

audio_dataset_path=r"C:\Users\hp\Downloads\UrbanSound8K\UrbanSound8K\audio"
metadata=pd.read_csv(r"C:\Users\hp\Downloads\UrbanSound8K\UrbanSound8K\metadata\UrbanSound8K.csv")
metadata.head()


# In[21]:


def features_extractor(file):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    
    return mfccs_scaled_features


# In[24]:



get_ipython().system('pip install tqdm')
import numpy as np
from tqdm import tqdm
### Now we iterate through every audio file and extract features 
### using Mel-Frequency Cepstral Coefficients
extracted_features=[]
for index_num,row in tqdm(metadata.iterrows()):
    file_name = os.path.join(os.path.abspath(audio_dataset_path),'fold'+str(row["fold"])+'/',str(row["slice_file_name"]))
    final_class_labels=row["class"]
    data=features_extractor(file_name)
    extracted_features.append([data,final_class_labels])


# In[25]:


### converting extracted_features to Pandas dataframe
extracted_features_df=pd.DataFrame(extracted_features,columns=['feature','class'])
extracted_features_df.head()


# In[28]:


### Split the dataset into independent and dependent dataset
X=np.array(extracted_features_df['feature'].tolist())
y=np.array(extracted_features_df['class'].tolist())


# In[29]:


X.shape


# In[30]:


y


# In[31]:


### Label Encoding
###y=np.array(pd.get_dummies(y))
### Label Encoder
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
y=to_categorical(labelencoder.fit_transform(y))


# In[32]:


y


# In[33]:


### Train Test Split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# In[34]:


X_train


# In[35]:


y_train


# In[36]:


X_test


# In[37]:


y_test


# In[38]:


X_train.shape


# In[39]:


y_train.shape


# In[40]:


X_test.shape


# In[41]:


y_train[0]


# In[42]:


X_train[0]


# In[43]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten
from tensorflow.keras.optimizers import Adam
from sklearn import metrics


# In[44]:


### No of classes
num_labels=y.shape[1]


# In[45]:


model=Sequential()
###first layer
model.add(Dense(100,input_shape=(40,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
###second layer
model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dropout(0.5))
###third layer
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.5))

###final layer
model.add(Dense(num_labels))
model.add(Activation('softmax'))


# In[46]:


model.summary()


# In[47]:


model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')


# In[49]:


## Trianing my model
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime 

num_epochs = 100
num_batch_size = 32

checkpointer = ModelCheckpoint(filepath='saved_models/audio_classification.hdf5', 
                               verbose=1, save_best_only=True)
start = datetime.now()

model.fit(X_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(X_test, y_test), callbacks=[checkpointer], verbose=1)


duration = datetime.now() - start
print("Training completed in time: ", duration)


# In[50]:


test_accuracy=model.evaluate(X_test,y_test,verbose=0)
print(test_accuracy[1])


# In[55]:


filename=r"C:\Users\hp\Downloads\UrbanSound8K\UrbanSound8K\drilling_machine.wav"
audio, sample_rate = librosa.load(filename, res_type='kaiser_fast') 
mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)

print(mfccs_scaled_features)
mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)
print(mfccs_scaled_features)
print(mfccs_scaled_features.shape)


# In[57]:


y_predict = np.argmax(model.predict(mfccs_scaled_features), axis=-1)
print(y_predict)


# In[58]:


prediction_class = labelencoder.inverse_transform(y_predict) 
prediction_class


# In[ ]:




