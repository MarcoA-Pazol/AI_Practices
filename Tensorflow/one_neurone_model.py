import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Setting data.
celsius = np.array([-40, -10, 0, 8, 15, 22, 38])
farenheit = np.array([-40, 14, 32, 46, 59, 72, 100])

#Data visualization
sns.scatterplot(celsius['Celsius'], farenheit['Farenheit'])
sns.displot()