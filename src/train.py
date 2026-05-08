import os
import numpy as np
import imageio
import logging
from tqdm import tqdm

from config import IMAGE_DIR, LABEL_DIR, TESTING_DIR, PREDICTIONS_DIR, MODEL_PATH, BATCH_SIZE, EPOCHS, THRESHOLD
from model import build_unet

# === LOGGER ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_data():
    X, Y = [], []
    img_paths = list(IMAGE_DIR.glob("*.png"))

    if not img_paths:
        logger.warning("Aucune image trouvée dans IMAGE_DIR. Vérifiez le prétraitement.")
        return np.array(X), np.array(Y)

    for img_path in tqdm(img_paths, desc="Chargement des données"):
        lbl_path = LABEL_DIR / img_path.name.replace(".png", "_label.png")
        if lbl_path.exists():
            X.append(np.expand_dims(imageio.imread(str(img_path)) / 255.0, -1))
            Y.append(np.expand_dims(imageio.imread(str(lbl_path)) / 255.0, -1))
        else:
            logger.warning(f"Label manquant pour {img_path.name}, skip.")

    logger.info(f"{len(X)} paires image/label chargées.")
    return np.array(X), np.array(Y)


if __name__ == "__main__":
    # 1. Chargement
    logger.info("Chargement des données...")
    X, Y = load_data()

    if len(X) == 0:
        logger.error("Aucune donnée chargée. Arrêt.")
        exit(1)

    # 2. Entraînement
    logger.info("Construction et entraînement du modèle...")
    model = build_unet()
    model.fit(X, Y, epochs=EPOCHS, batch_size=BATCH_SIZE)

    model.save(str(MODEL_PATH))
    logger.info(f"Modèle sauvegardé : {MODEL_PATH}")

    # 3. Prédiction sur les images de test
    test_imgs = list(TESTING_DIR.glob("*.png"))

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