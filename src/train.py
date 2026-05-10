import os
import numpy as np
import imageio
import logging
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from config import (IMAGE_DIR, LABEL_DIR, IMAGE_TEST_DIR, LABEL_TEST_DIR, 
                    PREDICTIONS_DIR, MODEL_PATH, GRAPH_PATH, BATCH_SIZE, EPOCHS)
from model import unet_model

# --- CONFIGURATION LOGGER ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=DeprecationWarning)

def load_data(image_dir, label_dir):
    """Charge les IRM (normalisées) et les labels (valeurs intactes 0,1,2,3)."""
    X, Y = [], []
    img_paths = list(image_dir.glob("*.png"))

    for img_path in tqdm(img_paths, desc=f"Chargement {image_dir.name}"):
        lbl_path = label_dir / img_path.name.replace(".png", "_label.png")
        if lbl_path.exists():
            X.append(np.expand_dims(imageio.imread(str(img_path)) / 255.0, -1))
            Y.append(np.expand_dims(imageio.imread(str(lbl_path)), -1))

    return np.array(X), np.array(Y)

if __name__ == "__main__":
    os.makedirs(str(PREDICTIONS_DIR), exist_ok=True)

    # === 1. CHARGEMENT DES DONNÉES ===
    logger.info("Chargement des données d'entraînement...")
    X_train, y_train = load_data(IMAGE_DIR, LABEL_DIR)
    
    logger.info("Chargement des données de validation...")
    X_val, y_val = load_data(IMAGE_TEST_DIR, LABEL_TEST_DIR)

    if len(X_train) == 0:
        logger.error("Aucune donnée chargée. Arrêt du script.")
        exit(1)

    # === 2. ENTRAÎNEMENT DU MODÈLE ===
    logger.info("Début de l'entraînement...")
    model = unet_model()
    
    history = model.fit(
        X_train, y_train, 
        validation_data=(X_val, y_val) if len(X_val) > 0 else None,
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE
    )

    model.save(str(MODEL_PATH))
    logger.info(f"Modèle sauvegardé : {MODEL_PATH}")

    # === 3. COURBES D'APPRENTISSAGE ===
    logger.info("Génération des graphiques de Loss/Accuracy...")
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Entraînement (Loss)')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation (Val Loss)')
    plt.title('Évolution de l\'Erreur (Loss)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Entraînement (Accuracy)')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Validation (Val Accuracy)')
    plt.title('Évolution de la Précision (Accuracy)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.savefig(str(GRAPH_PATH))
    plt.close()

    # === 4. PRÉDICTIONS ET PLANCHES VISUELLES ===
    test_imgs = list(IMAGE_TEST_DIR.glob("*.png"))
    
    if not test_imgs:
        logger.warning("Aucune image de test pour la prédiction.")
        exit(0)

    logger.info(f"Génération des planches pour {len(test_imgs)} images de test...")
    
    # Palettes (0=Fond, 1=Cyan VD, 2=Jaune Myocarde, 3=Rouge VG)
    palette_overlay = ListedColormap([(0,0,0,0), (0,0.8,1,1), (1,0.8,0,1), (0.6,0,0,1)])
    palette_masque = ListedColormap([(0,0,0.5,1), (0,0.8,1,1), (1,0.8,0,1), (0.6,0,0,1)])

    for test_img in tqdm(test_imgs, desc="Création des planches"):
        img = np.expand_dims(imageio.imread(str(test_img)) / 255.0, (0, -1))
        
        pred_classes = np.argmax(model.predict(img, verbose=0)[0], axis=-1)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Panneau 1 : IRM
        axes[0].imshow(img[0, :, :, 0], cmap='gray')
        axes[0].set_title('Image IRM', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Panneau 2 : Superposition
        axes[1].imshow(img[0, :, :, 0], cmap='gray')
        axes[1].imshow(pred_classes, cmap=palette_overlay, interpolation='none', alpha=0.7, vmin=0, vmax=3)
        axes[1].set_title('Image + Label', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        # Panneau 3 : Masque seul sur fond bleu nuit
        axes[2].imshow(pred_classes, cmap=palette_masque, interpolation='none', vmin=0, vmax=3)
        axes[2].set_title('Masque prédit', fontsize=14, fontweight='bold')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        result_name = test_img.name.replace(".png", "_planche.png")
        plt.savefig(os.path.join(str(PREDICTIONS_DIR), result_name), facecolor='#FDFDF0')
        plt.close()

    logger.info("✅ Terminé ! Vérifiez le dossier Output/predictions.")