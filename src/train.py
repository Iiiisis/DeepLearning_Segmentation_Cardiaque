"""
train.py
--------
Script principal d'entraînement et de prédiction
1. Charge les images et les masques PNG en mémoire.
2. Entraîne le modèle U-Net défini dans model.py.
3. Sauvegarde les poids du modèle et les courbes d'apprentissage.
4. Effectue des prédictions sur l'ensemble de test et génère des visualisations.
"""

import os
import imageio
import logging
import warnings
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from model import build_unet
from config import IMAGE_DIR, LABEL_DIR,IMAGE_TEST_DIR, LABEL_TEST_DIR, GRAPH_PATH, PREDICTIONS_DIR, MODEL_PATH, BATCH_SIZE, EPOCHS, THRESHOLD

# === LOGGER ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=DeprecationWarning)

def load_data(image_dir, label_dir):
    """
    Parcourt un dossier, lit les images et leurs labels correspondants,
    et les formate sous forme de tableaux NumPy compatibles avec Keras.
    
    -H (Height - Hauteur) : Le nombre de pixels de l'image de haut en bas.
    -W (Width - Largeur) : Le nombre de pixels de l'image de gauche à droite.
    -1 (Channels - Canaux) : La "profondeur" des couleurs. Le chiffre 1 signifie qu'il y a un seul canal d'information par pixel, ce qui correspond à une image en niveaux de gris.
    """
    
    X, Y = [], []
    img_paths = list(image_dir.glob("*.png"))

    if not img_paths:
        logger.warning(f"Aucune image trouvée dans {image_dir}. Vérifiez le prétraitement.")
        return np.array(X), np.array(Y)

    for img_path in tqdm(img_paths, desc="Chargement des données"):
        lbl_path = label_dir / img_path.name.replace(".png", "_label.png")
        if lbl_path.exists():
            # Normalisation de l'image entre 0 et 1 et ajout de la dimension du canal (H, W, 1)
            X.append(np.expand_dims(imageio.imread(str(img_path)) / 255.0, -1))
            # Ajout de la dimension du canal pour le masque (H, W, 1) sans normalisation (labels entiers)
            Y.append(np.expand_dims(imageio.imread(str(lbl_path)) , -1))
        else:
            logger.warning(f"Label manquant pour {img_path.name}, skip.")

    logger.info(f"{len(X)} paires image/label chargées.")
    return np.array(X), np.array(Y)


if __name__ == "__main__":
    #-- 1. Chargement des données----------------------------
    logger.info("Chargement des données d'entraîmenent...")
    X_train, y_train = load_data(IMAGE_DIR, LABEL_DIR)
    logger.info("Chargement des données de validation...")
    X_val, y_val = load_data(IMAGE_TEST_DIR, LABEL_TEST_DIR)
    
    if len(X_train) == 0:
        logger.error("Aucune donnée d'entraînement chargée. Arrêt.")
        exit(1)

    #--2. Entraînement du modèle----------------------------
    logger.info("Construction et entraînement du modèle...")
    model = build_unet()
    
    # Lancement de la boucle d'apprentissage
    history =model.fit(X_train, y_train, 
                        validation_data=(X_val, y_val) if len(X_val) > 0 else None,
                        epochs=5, 
                        batch_size=BATCH_SIZE)
    
    # Sauvegarde du modèle entraîné
    model.save(str(MODEL_PATH))
    logger.info(f"Modèle sauvegardé : {MODEL_PATH}")

    #--3. Création des graphiques de suivi----------------------------
    logger.info("Génération des graphiques...")
    plt.figure(figsize=(12, 5))

    # Sous-graphique 1 : Évolution de la Loss (Erreur)
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Courbe d\'entraînement (Loss)')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Courbe de validation (Val Loss)')
    plt.title('Évolution de la Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Sous-graphique 2 : Évolution de l'Accuracy (Précision)
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Accuracy Entraînement')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Accuracy Validation')
    plt.title('Évolution de l\'Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Sauvegarde combinée des deux graphiques
    plt.savefig(str(GRAPH_PATH))
    plt.close()
    logger.info(f"Graphiques sauvegardés dans {GRAPH_PATH}")
    
    
    # --- 4. Prédiction sur les images de test ---------------------------------------------
    test_imgs = list(IMAGE_TEST_DIR.glob("*.png"))

    if not test_imgs:
        logger.warning(f"Aucune image de test trouvée dans {IMAGE_TEST_DIR}. Arrêt des prédictions.")
        exit(0)  
        
    else:
        logger.info(f"{len(test_imgs)} images de test trouvées. Lancement des prédictions...")
        
        # 4.1 On définit la palette de couleurs AVANT la boucle (pour ne pas la recréer 100 fois)
        couleurs = [
            (0, 0, 0, 0),       # Classe 0 (Fond) : Transparent
            (0, 1, 1, 1),       # Classe 1 : Cyan (ex: Ventricule Droit)
            (1, 1, 0, 1),       # Classe 2 : Jaune (ex: Myocarde)
            (1, 0, 0, 1)        # Classe 3 : Rouge (ex: Ventricule Gauche)
        ]
        ma_palette = ListedColormap(couleurs)

        # 4.2 On boucle sur chaque image
        for test_img in tqdm(test_imgs, desc="Prédictions"):
            # Formatage de l'image (IRM)
            img = np.expand_dims(imageio.imread(str(test_img)) / 255.0, (0, -1))
            
            # Prédiction Multi-classes
            pred_prob = model.predict(img)[0] # Sortie de taille (512, 512, 4)
            pred_classes = np.argmax(pred_prob, axis=-1) # On garde la classe la plus probable (0, 1, 2 ou 3)
            
            # Création de l'image superposée (Overlay)
            plt.figure(figsize=(5, 5))
            plt.imshow(img[0, :, :, 0], cmap='gray') # L'IRM de fond en niveaux de gris
            plt.imshow(pred_classes, cmap=ma_palette, interpolation='none', alpha=0.5) # Le masque 3 couleurs par-dessus
            plt.axis('off') # On enlève les axes
            
            # Sauvegarde au BON endroit (dans PREDICTIONS_DIR)
            overlay_name = test_img.name.replace(".png", "_overlay.png")
            overlay_path = os.path.join(str(PREDICTIONS_DIR), overlay_name)
            
            plt.savefig(overlay_path, bbox_inches='tight', pad_inches=0)
            plt.close() # IMPORTANT : libère la mémoire de l'ordinateur après chaque image
            
        logger.info(f"Prédictions superposées en couleurs sauvegardées dans : {PREDICTIONS_DIR}")