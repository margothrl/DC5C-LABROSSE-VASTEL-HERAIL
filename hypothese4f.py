import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./Data_Salary_Market.csv', sep=';')

# Filtrer pour inclure uniquement les femmes
df_femmes = df[df['Gender'] == 'Female']

# Calculer la moyenne des salaires par pays uniquement pour les femmes
salaires_moyens_par_pays_femmes = df_femmes.groupby('Country')['Salary'].mean().reset_index()

# Afficher les moyennes des salaires par pays
print(salaires_moyens_par_pays_femmes)

# Afficher un bar chart de la moyenne des salaires en fonction du pays pour les femmes uniquement
plt.figure(figsize=(10, 6))
bars = plt.bar(salaires_moyens_par_pays_femmes['Country'], salaires_moyens_par_pays_femmes['Salary'], color='skyblue')
plt.xlabel('Pays')
plt.ylabel('Moyenne des Salaires')
plt.title('Moyenne des Salaires par Pays pour les Femmes')
plt.xticks(rotation=45)
plt.bar_label(bars, labels=salaires_moyens_par_pays_femmes['Salary'].round(2), padding=3)
plt.tight_layout()  
plt.show()