#Loading the required Packages
import numpy as np
import pandas as pd
import tensorflow_datasets as tfds
import tensorflow as tf

tfds.disable_progress_bar()
MessageDF = pd.read_csv('file2.csv')
MessageDF = MessageDF[~MessageDF['Text'].isin(['hi','hey','hello'])]
MessageDF.append(['5c5b806fbd1826340209616ddb9ed767','23:59:00','hey','Normal'],ignore_index = True)
#Reading the dataset
labels = MessageDF['Label']

labels = labels.replace( to_replace = 'Normal', value = 0)
labels = labels.replace( to_replace = 'Predatory', value = 1)
message = MessageDF['Text'].astype(str)

data = {'Message':message, 'Label':labels}
RequiredDF = pd.concat(data,axis = 1)

#Converting the dataset into the required format.
dataset1 = (
    tf.data.Dataset.from_tensor_slices(
        (
            tf.cast(RequiredDF['Message'].values, tf.string),
            tf.cast(RequiredDF['Label'].values, tf.int64)
        )
    )
)
print(dataset1.element_spec)
train_size = int(0.7 * len(dataset1))

test_size = int(0.3 * len(dataset1))

#Dividing it into test and train datasets.
train_dataset = dataset1.take(train_size)
test_dataset = dataset1.skip(train_size)
test_dataset = dataset1.take(test_size)

train_dataset.element_spec

BUFFER_SIZE = 10000
BATCH_SIZE = 64
train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

#Encoding the data
VOCAB_SIZE = 1000
encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(
    max_tokens=VOCAB_SIZE)
encoder.adapt(train_dataset.map(lambda Message, label: Message))

vocab = np.array(encoder.get_vocabulary())
vocab[:20]

#Building the model.
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping()
model = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(
        input_dim=len(encoder.get_vocabulary()),
        output_dim=64,
        # Use masking to handle the variable sequence lengths
        mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])
history = model.fit(train_dataset, epochs=3,
                    validation_data=test_dataset,
                    validation_steps=30,
                    callbacks=[early_stopping])

#Validating the built model.
test_loss, test_acc = model.evaluate(test_dataset)

#To produce required metrics of the model.
from sklearn import metrics
predicted = model.predict_classes(test_dataset)
print(metrics.classification_report(np.concatenate([y for x, y in test_dataset], axis=0)
, predicted))
