import numpy as np
import imageio
from tqdm import tqdm
from config import IMAGE_DIR, LABEL_DIR, TEST_DIR, PRED_DIR, IMG_SIZE, BATCH_SIZE, EPOCHS


def load_data():
    X, Y = [], []
    for img_path in tqdm(list(IMAGE_DIR.glob("*.png"))):
        lbl_path = LABEL_DIR / img_path.name.replace(".png", "_label.png")
        if lbl_path.exists():
            X.append(np.expand_dims(imageio.imread(img_path) / 255.0, -1))
            Y.append(np.expand_dims(imageio.imread(lbl_path) / 255.0, -1))
    return np.array(X), np.array(Y)

if __name__ == "__main__":
    # 1. Entraînement
    X, Y = load_data()
    model = build_unet()
    model.fit(X, Y, epochs=EPOCHS, batch_size=BATCH_SIZE)
    model.save("unet_model.h5")

    # 2. Prédiction
    PRED_DIR.mkdir(exist_ok=True)
    for test_img in TEST_DIR.glob("*.png"):
        img = np.expand_dims(imageio.imread(test_img) / 255.0, (0, -1))
        pred = (model.predict(img)[0, :, :, 0] > 0.5).astype(np.uint8) * 255
        imageio.imwrite(PRED_DIR / test_img.name.replace(".png", "_pred.png"), pred)