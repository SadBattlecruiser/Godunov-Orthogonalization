import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_link = r'out\harmonic0.csv'
data = pd.read_csv(data_link)
print(data)
plt.figure(figsize=(16,10), dpi= 80)
plt.plot(data['t'], data['y0'])
plt.figure(figsize=(16,10), dpi= 80)
plt.plot(data['t'], data['y1'])
plt.figure(figsize=(16,10), dpi= 80)
plt.plot(data['t'], data['y2'])
plt.show()
