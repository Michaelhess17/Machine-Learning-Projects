import pandas as pd
import numpy as np
from keras.utils import to_categorical
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=42)
df = pd.read_csv('train.csv')
target = df['target']
df = df.drop(['ID_code','target'],axis=1)
data = np.array(df)
target = np.array(target)
target = to_categorical(target)
model.fit(data, target)

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
results.to_csv('results2.csv', index=False)