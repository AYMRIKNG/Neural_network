from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np

def load_data():
    # Chargement du dataset Iris depuis scikit-learn
    iris = load_iris()
    X = iris.data  # Caractéristiques d'entrée (shape: 150 x 4)
    y = iris.target.reshape(-1, 1)  # Étiquettes de classes (0, 1, 2), reshape en colonne (150 x 1)

    # Normalisation des caractéristiques (moyenne = 0, écart-type = 1)
    # Cela améliore l'entraînement du réseau
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Encodage One-Hot des étiquettes de classes :
    # 0 → [1, 0, 0], 1 → [0, 1, 0], 2 → [0, 0, 1]
    encoder = OneHotEncoder(sparse_output=False)
    y_encoded = encoder.fit_transform(y)

    # Découpage en données d'entraînement (80%) et de test (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    # Retourne les données prêtes à être utilisées par le modèle
    return X_train, X_test, y_train, y_test
