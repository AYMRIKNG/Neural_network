from data import load_data
from model import train, predict
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

def main():
    X_train, X_test, y_train, y_test = load_data()

    layer_sizes = [4, 8, 6, 3]

    print("Entraînement du modèle...")
    parameters, losses = train(X_train, y_train, layer_sizes, epochs=5000, learning_rate=0.05)


    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Courbe de perte pendant l\'entraînement')
    plt.show()

    y_pred = predict(X_test, parameters)
    y_true = np.argmax(y_test, axis=1)

    accuracy = np.mean(y_pred == y_true)
    print(f"\n✅ Accuracy sur test set : {accuracy:.2%}")

    # 🔍 Afficher les exemples de prédictions (à l’intérieur de main)
    iris = load_iris()
    class_names = iris.target_names

    print("\n🔍 Exemples de prédictions sur les données de test :\n")
    for i in range(5):
        features = X_test[i]
        predicted_class = class_names[y_pred[i]]
        true_class = class_names[y_true[i]]

        print(f"Exemple {i+1}")
        print(f"  Caractéristiques (sépal/pétale) : {features}")
        print(f"  🌼 Classe vraie       : {true_class}")
        print(f"  🤖 Classe prédite    : {predicted_class}")
        print("  ✅ Correct" if y_pred[i] == y_true[i] else "  ❌ Faux")
        print()

if __name__ == "__main__":
    main()
