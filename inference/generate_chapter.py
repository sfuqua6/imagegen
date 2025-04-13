import os
import torch
import numpy as np
from torchvision.utils import save_image
from models.generation_model import build_generation_model
from config import DEVICE, MODEL_SAVE_PATH, TARGET_IMAGE_SIZE
import logging

logger = logging.getLogger("GenerateChapter")

def load_trained_model(condition_dim=10):
    model = build_generation_model(condition_dim=condition_dim).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    model.eval()
    logger.info("Loaded trained generation model.")
    return model

def generate_panel(model, condition_vector):
    dummy_input = torch.zeros((1, 3, TARGET_IMAGE_SIZE[0]//4*4, TARGET_IMAGE_SIZE[1]//4*4)).to(DEVICE)
    condition_vector = torch.tensor(condition_vector, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        generated = model(dummy_input, condition_vector)
    return generated

def assemble_chapter(panel_tensors, output_path="generated_chapter.png"):
    if panel_tensors:
        chapter_tensor = torch.cat(panel_tensors, dim=0)
        # Save each panel separately, or combine using torchvision.utils
        save_image(chapter_tensor, output_path, nrow=1)
        logger.info(f"Chapter assembled and saved to {output_path}")
    else:
        logger.error("No panels to assemble.")

def generate_chapter(prompt="default condition", num_panels=5):
    condition_vector = np.zeros(10, dtype=np.float32)
    if "red" in prompt.lower():
        condition_vector[0] = 1.0
    panels = []
    model = load_trained_model(condition_dim=10)
    for i in range(num_panels):
        generated_panel = generate_panel(model, condition_vector)
        panels.append(generated_panel)
    assemble_chapter(panels)
    return panels

if __name__ == "__main__":
    generate_chapter(prompt="A dramatic page with red hues", num_panels=5)
