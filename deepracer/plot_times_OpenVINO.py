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
plt.title('Tiempo medio de inferencia por modelo y por uso de OpenVino')
plt.ylabel('Tiempo medio de inferencia (ms)')
plt.xlabel('Nombre de la arquitectura')
plt.tight_layout()
plt.show()