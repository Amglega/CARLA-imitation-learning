#!/usr/bin/python
# -*- coding: utf-8 -*-

# test script to plot the inference times of the different models with and without OpenVINO optimization.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Read data
df = pd.read_csv('OpenVino_times.csv')
df['OpenVINO'] = df['OpenVINO'].astype(bool)
# Plot 
plt.figure(figsize=(12, 6))
plt.grid()
sns.barplot(x='ModelName', y='MeanTime', hue='OpenVINO', data=df)
plt.title('Mean Inference Time of each Model and by use of OpenVino')
plt.ylabel('Mean Inference Time (ms)')
plt.xlabel('Model Name')
plt.tight_layout()
plt.show()