import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import tensorflow as tf
from tensorflow.contrib.keras import layers
from tensorflow.contrib.keras import models



df = pd.read_csv('hw3_train.csv')