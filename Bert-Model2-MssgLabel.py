import xml.etree.ElementTree as ET
from datetime import time
from xml.dom import minidom
import pandas as pd
import random


MessageDF = pd.read_csv('file2.csv')

labels = MessageDF['Label'].values.tolist()

message = MessageDF['Text'].values.tolist()


train_size = int(len(message)*0.8)
x_train = [str(message[i]) for i in range(train_size)]
y_train = labels[0:train_size]
x_test = [str(message[i]) for i in range(train_size,len(message))]
y_test = labels[train_size:]

from sklearn.model_selection import train_test_split
random_seed = 12342

class_names_value = ['Normal','Predatory']

#Using bert
(x_train, y_train), (x_test, y_test), preproc = ktrain.text.texts_from_array(x_train = x_train, y_train = y_train, 
                                                                             x_test = x_test, y_test = y_test, 
                                                                             class_names = class_names_value, 
                                                                             preprocess_mode = 'bert',
                                                                             maxlen = 500,
                                                                             max_features = 35000) 
model = ktrain.text.text_classifier('bert', train_data = (x_train,y_train) , preproc=preproc)
#Defining the Model
learner = ktrain.get_learner(model,train_data=(x_train,y_train), batch_size=5)

learner.fit(2e-5, 1)

#Testing the previously trained model
learner.validate(val_data = (x_test,y_test))
