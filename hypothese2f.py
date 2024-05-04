from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd

# Chargement du fichier CSV
df = pd.read_csv('./Data_Salary_Market.csv', sep=';')

df_femme = df.loc[df['Gender'] == 'Female']

# Sélection des variables explicatives (features) et de la variable cible (target)
X = df_femme[['Age']]  # Variable explicative : 'Age'
y = df_femme['Salary']  # Variable cible : 'Salary'

# Division des données en un ensemble d'entraînement et un ensemble de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Création d'une instance du modèle de régression linéaire
model = LinearRegression()

# Entraînement du modèle sur l'ensemble d'entraînement
model.fit(X_train, y_train)

# Prédiction sur l'ensemble de test
y_pred = model.predict(X_test)

# Calcul de l'erreur quadratique moyenne (MSE) et du coefficient de détermination (R^2)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(r2)

# Tracé des points de données
plt.scatter(X_test, y_test, color='black', label='Données réelles')

# Tracé de la ligne de régression
plt.plot(X_test, y_pred, color='blue', linewidth=3, label='Ligne de régression')

plt.xlabel('Âge')
plt.ylabel('Salaires')
plt.title('Régression Linéaire Simple - Salaires des Femmes en fonction de l\'âge')
plt.legend()
plt.show()