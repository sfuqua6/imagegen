import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import logging
from config import DEVICE, BATCH_SIZE, TRAINING_EPOCHS, LEARNING_RATE, MODEL_SAVE_PATH, TARGET_IMAGE_SIZE, BASE_DATA_DIR
from models.generation_model import build_generation_model
from data.data_loader import get_chapter_paths, load_image_batch

logger = logging.getLogger("TrainPipeline")

class MangaDataset(Dataset):

    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        from PIL import Image
        try:
            img = Image.open(self.image_paths[idx]).convert("RGB")
            img = img.resize((TARGET_IMAGE_SIZE[1], TARGET_IMAGE_SIZE[0]))  # (width, height)
            img = np.array(img, dtype=np.float32) / 255.0
            # Transpose to (channels, height, width)
            img = np.transpose(img, (2, 0, 1))
            # Dummy condition vector (replace with real data)
            condition = np.zeros(10, dtype=np.float32)
            condition[0] = 1.0  # dummy flag
            return torch.tensor(img), torch.tensor(condition)
        except Exception as e:
            logger.error(f"Error processing {self.image_paths[idx]}: {e}")
            raise e

def collect_all_image_paths():

    from glob import glob
    pattern = os.path.join(BASE_DATA_DIR, "chapter_*", "*.png")
    image_paths = glob(pattern)
    logger.info(f"Collected {len(image_paths)} image paths from {BASE_DATA_DIR}")
    return image_paths

def train_model():

    image_paths = collect_all_image_paths()
    dataset = MangaDataset(image_paths)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)

    # Build model, define loss and optimizer
    condition_dim = 10  # adjust as needed
    model = build_generation_model(condition_dim=condition_dim).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    logger.info("Starting training...")
    model.train()
    for epoch in range(1, TRAINING_EPOCHS+1):
        epoch_loss = 0.0
        for batch_idx, (imgs, conds) in enumerate(dataloader):
            imgs = imgs.to(DEVICE)
            conds = conds.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs, conds)
            loss = criterion(outputs, imgs)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            if batch_idx % 10 == 0:
                logger.info(f"Epoch [{epoch}/{TRAINING_EPOCHS}] Batch [{batch_idx}] Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(dataloader)
        logger.info(f"Epoch [{epoch}/{TRAINING_EPOCHS}] Average Loss: {avg_loss:.4f}")

        # Optionally save a checkpoint
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
    logger.info("Training completed!")
    return model

if __name__ == "__main__":
    train_model()
