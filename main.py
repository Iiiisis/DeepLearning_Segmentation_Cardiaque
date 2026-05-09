
"""
train.py
--------
Script principal d'entraînement (Multi-classes).
1. Charge les données d'entraînement et de validation.
2. Entraîne le modèle U-Net.
3. Génère les courbes de Loss/Accuracy.
4. Génère des planches visuelles (IRM / Superposition / Masque) pour le test.
"""

import os
import numpy as np
import imageio
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


from model import unet_model
from src.dataset import load_data
from config import IMAGE_DIR, LABEL_DIR, IMAGE_TEST_DIR, LABEL_TEST_DIR, PREDICTIONS_DIR, MODEL_PATH, GRAPH_PATH, BATCH_SIZE, EPOCHS


# === CONFIGURATION ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


#----Load Data------------------------
X, Y = load_data(IMAGE_DIR,LABEL_DIR)

#----Model Training------------------------
model   = unet_model()
history = model.fit(X, Y, epochs=EPOCHS, batchsize=BATCH_SIZE)
model.save(str(MODEL_PATH))
logger.info(f"Modèle sauvegardé : {MODEL_PATH}")

# ---Prediction----------------------------
test_imgs = list(IMAGE_TEST_DIR.glob("*.png"))
for test_img in tqdm(test_imgs):
    image = imageio.imread(str(test_img)).astype(np.float32)/ 255.0
    image = np.extand_dms (image,(0,-1))
    
    pred  = model.predict(image)[0,:,:,0] # Shape: (512, 512, 4)
    pred = (pred > 0.5).astype(np.uint8) * 255
    
    result_name = test_img.name.replace(".png", "prediction.png")
    result_path = os.path.join(str(PREDICTIONS_DIR), result_name)
    plt.savefig(result_path) 
    plt.close()
logger.info("Prédiction terminée")



#--- Tracé-------------------------------------
#   Évolution de la Loss (Erreur)
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Courbe d\'entraînement (Loss)')
if 'val_loss' in history.history:
    plt.plot(history.history['val_loss'], label='Courbe de validation (Val Loss)')
plt.title('Évolution de la Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

#  Évolution de l'Accuracy (Précision)
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Accuracy Entraînement')
if 'val_accuracy' in history.history:
    plt.plot(history.history['val_accuracy'], label='Accuracy Validation')
plt.title('Évolution de l\'Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.savefig(str(GRAPH_PATH))
plt.close()
logger.info(f"Graphiques sauvegardés dans {GRAPH_PATH}")