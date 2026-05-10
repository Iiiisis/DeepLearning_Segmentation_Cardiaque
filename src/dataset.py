import imageio
import numpy as np
from tqdm import tqdm

def load_data(image_dir, label_dir):
    """Charge les images (normalisées) et les labels (classes entières)."""
    images, labels = [], []
    img_paths = list(image_dir.glob("*.png"))

    for img_path in tqdm(img_paths, desc=f"Chargement {image_dir.name}"):
        lbl_path = label_dir / img_path.name.replace(".png", "_label.png")
        if lbl_path.exists():
            # L'IRM est divisée par 255 (pour être entre 0 et 1)
            image = imageio.imread(str(img_path)).astype(np.float32) / 255.0
            label = imageio.imread(str(lbl_path)).astype(np.float32)
            
            image = np.expand_dims(image, -1)
            label = np.expand_dims(label, -1)
            
            images.append(image)
            labels.append(label)

    return np.array(images), np.array(labels)