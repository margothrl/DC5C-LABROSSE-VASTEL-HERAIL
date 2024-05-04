import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./Data_Salary_Market.csv', sep=';')

# Calculer la moyenne des salaires par pays
salaires_moyens_par_pays = df.groupby('Country')['Salary'].mean().reset_index()

# Afficher un bar chart de la moyenne des salaires en fonction du pays
plt.figure(figsize=(10, 6))
bars = plt.bar(salaires_moyens_par_pays['Country'], salaires_moyens_par_pays['Salary'], color='skyblue')
plt.xlabel('Pays')
plt.ylabel('Moyenne des Salaires')
plt.title('Moyenne des Salaires par Pays')
plt.xticks(rotation=45)

# Ajouter les étiquettes au-dessus de chaque barre
plt.bar_label(bars, labels=salaires_moyens_par_pays['Salary'].round(2), padding=3)
plt.tight_layout()  
plt.show()