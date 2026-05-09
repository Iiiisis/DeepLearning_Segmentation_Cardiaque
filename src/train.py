import os
import imageio
import logging
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import build_unet
from config import IMAGE_DIR, LABEL_DIR,IMAGE_TEST_DIR, LABEL_TEST_DIR, GRAPH_PATH, PREDICTIONS_DIR, MODEL_PATH, BATCH_SIZE, EPOCHS, THRESHOLD

# === LOGGER ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def load_data(image_dir, label_dir):
    X, Y = [], []
    img_paths = list(image_dir.glob("*.png"))

    if not img_paths:
        logger.warning(f"Aucune image trouvée dans {image_dir}. Vérifiez le prétraitement.")
        return np.array(X), np.array(Y)

    for img_path in tqdm(img_paths, desc="Chargement des données"):
        lbl_path = label_dir / img_path.name.replace(".png", "_label.png")
        if lbl_path.exists():
            X.append(np.expand_dims(imageio.imread(str(img_path)) / 255.0, -1))
            Y.append(np.expand_dims(imageio.imread(str(lbl_path)) / 255.0, -1))
        else:
            logger.warning(f"Label manquant pour {img_path.name}, skip.")

    logger.info(f"{len(X)} paires image/label chargées.")
    return np.array(X), np.array(Y)


if __name__ == "__main__":
    # 1. Chargement
    logger.info("Chargement des données d'entraîmenent...")
    X_train, y_train = load_data(IMAGE_DIR, LABEL_DIR)
    logger.info("Chargement des données de validation...")
    X_val, y_val = load_data(IMAGE_TEST_DIR, LABEL_TEST_DIR)
    
    if len(X_train) == 0:
        logger.error("Aucune donnée d'entraînement chargée. Arrêt.")
        exit(1)

    # ---  Entraînement ---
    logger.info("Construction et entraînement du modèle...")
    model = build_unet()
    
    history =model.fit(X_train, y_train, 
                        validation_data=(X_val, y_val) if len(X_val) > 0 else None,
                        epochs=50, 
                        batch_size=BATCH_SIZE)

    model.save(str(MODEL_PATH))
    logger.info(f"Modèle sauvegardé : {MODEL_PATH}")

    # ---  CRÉATION DES GRAPHIQUES ---
    logger.info("Génération des graphiques...")
    plt.figure(figsize=(12, 5))

    # Graphique de la Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Courbe d\'entraînement (Loss)')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Courbe de validation (Val Loss)')
    plt.title('Évolution de la Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Graphique de l'Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Accuracy Entraînement')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Accuracy Validation')
    plt.title('Évolution de l\'Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Sauvegarde de l'image
    plt.savefig(str(GRAPH_PATH))
    plt.close()
    logger.info(f"Graphiques sauvegardés dans {GRAPH_PATH}")
    
    
    # --- Prédiction sur les images de test ---
    test_imgs = list(IMAGE_TEST_DIR.glob("*.png"))

    if not test_imgs:
        logger.warning("Aucune image de test trouvée dans TESTING_DIR.")
    else:
        logger.info(f"{len(test_imgs)} images de test trouvées. Lancement des prédictions...")
        for test_img in tqdm(test_imgs, desc="Prédictions"):
            img = np.expand_dims(imageio.imread(str(test_img)) / 255.0, (0, -1))
            pred = (model.predict(img)[0, :, :, 0] > THRESHOLD).astype(np.uint8) * 255
            pred_name = test_img.name.replace(".png", "_pred.png")
            pred_path = os.path.join(str(PREDICTIONS_DIR), pred_name)
            imageio.imwrite(pred_path, pred)
        logger.info(f"Prédictions sauvegardées dans : {PREDICTIONS_DIR}")
    
    # Création image superposée
    plt.figure(figsize=(5, 5))
    plt.imshow(img[0, :, :, 0], cmap='gray')
    
    masque_transparent = np.ma.masked_where(pred == 0, pred)
    plt.imshow(masque_transparent, cmap='autumn', alpha=0.5)

    plt.axis('off') # On enlève les axes
    overlay_path = os.path.join(str(PREDICTIONS_DIR), test_img.name.replace(".png", "_overlay.png"))
    plt.savefig(overlay_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    