import xml.etree.ElementTree as ET
from datetime import time
from xml.dom import minidom
import pandas as pd
import random

#Loading the required data from DataFrame
messages = [((FinalDF['MessageDataFrame'][k])['Text']).tolist() for k in range(len(FinalDF))]

labels = (FinalDF['Label']).values.tolist()

#Creating the Training and Testing sets
train_size = int(len(messages)*0.8)

x_train = messages[0:train_size]
y_train = labels[0:train_size]
x_test = messages[train_size:]
y_test = labels[train_size:]


print("x-train from 0 -2 ",x_train[0:4])
print("x_test at 3 to 5 ",x_test[3:5])
print("y train from 0 to 6",y_train[0:6])
print("y_test from 0 to 2",y_test[0:2])
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
