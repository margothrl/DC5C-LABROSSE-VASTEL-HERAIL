from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd

# Chargement du fichier CSV pour voir son contenu
df = pd.read_csv('./departretraite_parcsp.csv', sep=";")

#Conserver uniquement les lignes correspondants au type de produit beauté
df_agric = df.loc[df['categorie_socioprofessionnelle'] == '1 - Agriculteurs exploitants']

# Échantillonnage aléatoire pour conserver 50% des données
#df_sampled = df.sample(frac=1)

# Sélection des variables explicatives (features) et de la variable cible (target)
X = df_agric [['annee']]  # Variable explicative : 'Année'
y = df_agric ['age_conjoncturel_de_depart_a_la_retraite']  # Variable cible : 'Age de départ'

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
(mse, r2)
print(r2)

# Tracé des points de données
plt.scatter(X_test, y_test, color='black', label='Données réelles')

# Tracé de la ligne de régression
plt.plot(X_test, y_pred, color='blue', linewidth=3, label='Ligne de régression')

plt.xlabel('Année de départ')
plt.ylabel('Âge de départ')
plt.title("Régression Linéaire Simple - Année de départ en fonction de l'âge pour les Agriculteurs")
plt.legend()
plt.show()
