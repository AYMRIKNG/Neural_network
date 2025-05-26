from data import load_data
from model import train, predict
import numpy as np

def main():
    X_train, X_test, y_train, y_test = load_data()

    layer_sizes = [4, 8, 6, 3]  # 4 inputs → 2 hidden layers → 3 classes

    print("Entraînement du modèle...")

    # Entraînement du réseau de neurones
    # - epochs = 5000 : nombre d’itérations d’apprentissage
    # - learning_rate = 0.05 : vitesse d’apprentissage
    # modifier les valeur peut affecter le temps de chargement
    parameters = train(X_train, y_train, layer_sizes, epochs=5000, learning_rate=0.05)


    # Prédiction sur le jeu de test
    y_pred = predict(X_test, parameters)
    y_true = np.argmax(y_test, axis=1)

    accuracy = np.mean(y_pred == y_true)
    print(f"\n✅ Accuracy sur test set : {accuracy:.2%}")

if __name__ == "__main__":
    main()
