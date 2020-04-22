import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data0_link = r'out\harmonic0.csv'
data0 = pd.read_csv(data0_link)
data1_link = r'out\harmonic1.csv'
data1 = pd.read_csv(data1_link)
data2_link = r'out\harmonic2.csv'
data2 = pd.read_csv(data2_link)
#data3_link = r'out\harmonic3.csv'
#data3 = pd.read_csv(data3_link)
data4_link = r'out\harmonic4.csv'
data4 = pd.read_csv(data4_link)

plt.figure(figsize=(16,10), dpi= 80)
plt.plot(data0['t'], data0['y2'])
#plt.plot(data1['t'], data1['y0'])
#plt.plot(data2['t'], data2['y0'])
#plt.plot(data3['t'], data3['y0'])
#plt.plot(data4['t'], data4['y0'])

plt.figure(figsize=(16,10), dpi= 80)
plt.plot(data1['t'], data1['y2'])
#plt.plot(data0['t'], data0['y1'])
#plt.plot(data1['t'], data1['y1'])
#plt.plot(data2['t'], data2['y1'])
#plt.plot(data3['t'], data3['y1'])
#plt.plot(data4['t'], data4['y1'])

plt.figure(figsize=(16,10), dpi= 80)
plt.plot(data2['t'], data2['y2'])
#plt.plot(data0['t'], data0['y2'])
#plt.plot(data1['t'], data1['y2'])
#plt.plot(data2['t'], data2['y2'])
#plt.plot(data3['t'], data3['y2'])
#plt.plot(data4['t'], data4['y2'])
plt.figure(figsize=(16,10), dpi= 80)
plt.plot(data2['t'], data4['y2'])


print(data0)
sum = data0['y2']+data1['y2']+data2['y2']+data4['y2']
plt.figure(figsize=(16,10), dpi= 80)
plt.plot(data2['t'], sum)
plt.show()
