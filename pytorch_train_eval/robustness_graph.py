import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# This script makes violin graphs for the rancom control data collected in 
# robustness_test.py. It reads the CSV file containing the results of the random control tests,
# andd generates violin plots for various metrics 

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
df = pd.read_csv('robustness_results.csv')


print(df.head())


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
fig.suptitle('Evaluación en Simulación (CARLA): Test de Control Aleatorio', fontsize=16, fontweight='bold', y=0.98)


# Vueltas Completadas
sns.violinplot(data=df, x='model', y='laps', ax=axes[0, 0], palette="Set2", inner="box", cut=0)
axes[0, 0].set_title('Vueltas Completadas por Episodio', fontweight='bold')
axes[0, 0].set_ylabel('Vueltas (Fracción)')
axes[0, 0].set_xlabel('')

# Desviación de Trayectoria Media
sns.violinplot(data=df, x='model', y='mean_deviation', ax=axes[0, 1], palette="Set2", inner="box", cut=0)
axes[0, 1].set_title('Desviación de Trayectoria Media', fontweight='bold')
axes[0, 1].set_ylabel('Desviación (metros)')
axes[0, 1].set_xlabel('')

# Velocidad Media
sns.violinplot(data=df, x='model', y='mean_speed', ax=axes[1, 0], palette="Set2", inner="box", cut=0)
axes[1, 0].set_title('Velocidad Media del Vehículo', fontweight='bold')
axes[1, 0].set_ylabel('Velocidad (m/s)')
axes[1, 0].set_xlabel('Arquitectura de Red')

# Invasiones de Carril por Vuelta
sns.violinplot(data=df, x='model', y='mean_lane_invasions', ax=axes[1, 1], palette="Set2", inner="box", cut=0)
axes[1, 1].set_title('Invasiones de Carril por Vuelta', fontweight='bold')
axes[1, 1].set_ylabel('Nº de Invasiones')
axes[1, 1].set_xlabel('Arquitectura de Red')

plt.tight_layout()

#plt.savefig('resultados_modelos_violin.png', dpi=300, bbox_inches='tight')
#plt.savefig('resultados_modelos_violin.pdf', bbox_inches='tight') 

plt.show()
