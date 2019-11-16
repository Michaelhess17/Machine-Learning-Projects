import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import tensorflow as tf
from tensorflow.contrib.keras import layers
from tensorflow.contrib.keras import models



df = pd.read_csv('train.csv')
target = df['default payment next month']
df = df.drop(['ID', 'default payment next month'], axis=1)

cat_names = ['SEX', 'EDUCATION', 'MARRIAGE']
for i in cat_names:
	df = pd.concat([df, pd.get_dummies(df[i])], axis=1)
df = df.drop(cat_names, axis=1)

scaler = StandardScaler()
for i in df.columns.to_list():
	if i not in cat_names:
		x = df[i].values.astype(float)
		x = x.reshape(-1, 1)
		x_scaled = scaler.fit_transform(x)
		df[i] = x_scaled
df = df.drop(cat_names, axis=1)

data = np.array(df)
target = np.array(target)


model = models.Sequential()
num_inputs = df.columns.size
num_hidden = [100, 150, 100, 50]
num_outputs = target[0].size
model.add(layers.Dense(units=num_inputs, activation=tf.nn.relu, input_dim=df.columns.size))
model.add(layers.Dense(units=num_hidden[0], activation=tf.nn.relu))

model.add(layers.Dense(units=num_hidden[1], activation=tf.nn.relu))

model.add(layers.Dense(units=num_hidden[2], activation=tf.nn.relu))

model.add(layers.Dense(units=num_hidden[3], activation=tf.nn.relu))
model.add(layers.Dense(units=num_outputs, activation=tf.nn.relu))
model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['acc'])

# model.fit(data, target, epochs=500)
