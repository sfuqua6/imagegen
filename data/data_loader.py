import os
import glob
from config import BASE_DATA_DIR, TARGET_IMAGE_SIZE, MAX_PNGS_PER_BATCH
from PIL import Image
import numpy as np
import logging

logger = logging.getLogger("DataLoader")

def get_chapter_paths(base_dir=BASE_DATA_DIR):
    chapters = sorted(
        glob.glob(os.path.join(base_dir, "chapter_*")),
        key=lambda x: int(x.split('_')[-1])
    )
    logger.info(f"Found {len(chapters)} chapters in {base_dir}")
    return chapters

def load_image_batch(image_paths, target_size=TARGET_IMAGE_SIZE):

    batch = []
    batch_paths = []
    for idx, path in enumerate(image_paths):
        try:
            img = Image.open(path).convert("RGB")
            img = img.resize((target_size[1], target_size[0]))  
            img_array = np.array(img, dtype=np.float32) / 255.0
            batch.append(img_array)
            batch_paths.append(path)
        except Exception as e:
            logger.error(f"Error loading image {path}: {e}")
        if len(batch) >= MAX_PNGS_PER_BATCH:
            yield batch, batch_paths
            batch = []
            batch_paths = []
    if batch:
        yield batch, batch_paths
