import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./Data_Salary_Market.csv', sep=';')

df_hommes = df[df['Gender'] == 'Male']

salaires_moyens_par_pays_hommes = df_hommes.groupby('Country')['Salary'].mean().reset_index()

# Afficher les moyennes des salaires par pays
print(salaires_moyens_par_pays_hommes)

# Afficher un bar chart de la moyenne des salaires en fonction du pays pour les hommes uniquement
plt.figure(figsize=(10, 6))
bars = plt.bar(salaires_moyens_par_pays_hommes['Country'], salaires_moyens_par_pays_hommes['Salary'], color='skyblue')
plt.xlabel('Pays')
plt.ylabel('Moyenne des Salaires')
plt.title('Moyenne des Salaires par Pays pour les Hommes')
plt.xticks(rotation=45)
plt.bar_label(bars, labels=salaires_moyens_par_pays_hommes['Salary'].round(2), padding=3)
plt.tight_layout()  
plt.show()