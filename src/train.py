"""
Module d'apprentissage profond pour la segmentation d'images.
Entraîner le modèle U-Net sur le dataset généré et effectue des prédictions.
"""

import os
import numpy as np
import imageio.v2 as imageio
import tensorflow as tf
from glob import glob
from tqdm import tqdm


def load_dataset(image_dir, label_dir):
    """
    Charge les images et leurs masques (labels) depuis le disque, 
    et les normalise pour l'entraînement du réseau de neurones.

    Args:
        image_dir (str): Chemin vers le dossier des images d'entraînement.
        label_dir (str): Chemin vers le dossier des masques d'entraînement.

    Returns:
        tuple: (X, Y) où X contient les images et Y les masques sous forme de arrays NumPy.
    """
    images = []
    labels = []
    
    image_paths = glob(os.path.join(image_dir, "*.png"))
    
    for img_path in tqdm(image_paths, desc="Chargement du dataset"):
        label_path = img_path.replace("images", "labels").replace(".png", "_label.png")
        
        if not os.path.exists(label_path):
            continue
            
        # Chargement et normalisation (valeurs entre 0 et 1)
        image = imageio.imread(img_path).astype(np.float32) / 255.0
        label = imageio.imread(label_path).astype(np.float32) / 255.0
        
        # Ajout de la dimension du canal (1 canal pour niveaux de gris)
        image = np.expand_dims(image, axis=-1)
        label = np.expand_dims(label, axis=-1)
        
        images.append(image)
        labels.append(label)
        
    return np.array(images), np.array(labels)

def predict_and_save(model, testing_dir, output_dir):
    """
    Effectue des prédictions sur un dossier d'images de test et sauvegarde les masques générés.

    Args:
        model (tf.keras.Model): Le modèle U-Net entraîné.
        testing_dir (str): Chemin vers le dossier contenant les images à tester.
        output_dir (str): Chemin vers le dossier de destination des prédictions.
    """
    testing_images = glob(os.path.join(testing_dir, "*.png"))
    os.makedirs(output_dir, exist_ok=True)
    
    for img_path in tqdm(testing_images, desc="Génération des prédictions"):
        # Chargement et préparation de l'image
        image = imageio.imread(img_path).astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=(0, -1)) # Création de la forme (1, H, W, 1)
        
        # Prédiction
        pred = model.predict(image, verbose=0)[0, :, :, 0]
        
        # Binarisation (Seuillage à 0.5) et conversion en format image
        pred = (pred > 0.5).astype(np.uint8) * 255
        
        # Sauvegarde
        filename = os.path.basename(img_path).replace(".png", "_pred.png")
        save_path = os.path.join(output_dir, filename)
        imageio.imwrite(save_path, pred)
        
    print("✅ Prédiction terminée.")

# === POINT D'ENTRÉE DU SCRIPT ===
if __name__ == "__main__":
    BASE_DIR = r"C:\Users\thblt\OneDrive\Bureau\Centrale\Projet Innov\new_dataset"
    IMAGE_DIR = os.path.join(BASE_DIR, "images")
    LABEL_DIR = os.path.join(BASE_DIR, "labels")
    TESTING_DIR = os.path.join(BASE_DIR, "testing")
    PREDICTIONS_DIR = "predictions"

    # 1. Chargement des données
    print("Étape 1 : Chargement des données...")
    X, Y = load_dataset(IMAGE_DIR, LABEL_DIR)

    # 2. Création et entraînement du modèle
    print("Étape 2 : Entraînement du modèle U-Net...")
    model = unet_model()
    model.fit(X, Y, epochs=5, batch_size=4)

    # 3. Sauvegarde du modèle
    print("Étape 3 : Sauvegarde du modèle...")
    model.save("unet_model.h5")
    
    # 4. Prédiction sur le set de test
    if os.path.exists(TESTING_DIR):
        print("Étape 4 : Inférence sur les données de test...")
        predict_and_save(model, TESTING_DIR, PREDICTIONS_DIR)
    else:
        print(f"⚠️ Dossier de test non trouvé : {TESTING_DIR}")