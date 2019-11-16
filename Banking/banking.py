import pandas as pd
import numpy as np
from keras.utils import to_categorical



df = pd.read_csv('train.csv')
target = df['target']
df = df.drop(['ID_code','target'],axis=1)
data = np.array(df)
target = np.array(target)
target = to_categorical(target)

import tensorflow as tf
from tensorflow.contrib.keras import layers
from tensorflow.contrib.keras import models


model = models.Sequential()
num_inputs = df.columns.size
num_hidden = [300, 150, 100, 50]
num_outputs = 2
model.add(layers.Dense(units=num_inputs, activation=tf.nn.relu, input_dim=df.columns.size))
model.add(layers.Dense(units=num_hidden[0], activation=tf.nn.relu))
model.add(layers.Dense(units=num_hidden[1], activation=tf.nn.relu))
model.add(layers.Dense(units=num_hidden[2], activation=tf.nn.relu))
model.add(layers.Dense(units=num_hidden[3], activation=tf.nn.relu))
model.add(layers.Dense(units=num_outputs, activation=tf.nn.softmax))
model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['acc'])

model.fit(data, target, epochs=5)

df2 = pd.read_csv('test.csv')
df2 = df2.drop(['ID_code'],axis=1)
data = np.array(df2)
preds = model.predict(data)
result = np.zeros(len(preds))
for j in range(len(preds)):
	result[j] = np.argmax(preds[j])
Id = ['test_'+str(a) for a in list(range(len(result)))]
results = pd.DataFrame({'ID_code': Id,
						'target': result})
results.to_csv('results.csv', index=False)