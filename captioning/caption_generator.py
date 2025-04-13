import os
import torch
import webcolors
import numpy as np
from PIL import Image
from colorthief import ColorThief
from transformers import BlipProcessor, BlipForConditionalGeneration
import logging
from config import DEVICE

logger = logging.getLogger("MangaCaptioner")

# Define color mapping for matching purposes
COLOR_NAMES = {
    'red': ['crimson', 'scarlet', 'ruby', 'carmine'],
    'blue': ['azure', 'cobalt', 'sapphire', 'navy'],
    'green': ['emerald', 'jade', 'olive', 'lime'],
    'yellow': ['gold', 'amber', 'mustard', 'lemon'],
    'purple': ['violet', 'lavender', 'mauve', 'lilac'],
    'orange': ['rust', 'pumpkin', 'amber', 'tangerine'],
    'pink': ['magenta', 'rose', 'coral', 'fuchsia'],
    'brown': ['bronze', 'copper', 'umber', 'ochre'],
    'gray': ['slate', 'steel', 'pewter', 'charcoal']
}

def closest_color(requested_color):
    try:
        closest_name = webcolors.rgb_to_name(requested_color)
    except ValueError:
        min_colors = {}
        for name in COLOR_NAMES.keys():
            r, g, b = webcolors.name_to_rgb(name)
            rd = (r - requested_color[0]) ** 2
            gd = (g - requested_color[1]) ** 2
            bd = (b - requested_color[2]) ** 2
            min_colors[rd + gd + bd] = name
        closest_name = min_colors[min(min_colors.keys())]
    return closest_name

logger.info(f"Loading BLIP model on {DEVICE.upper()}")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-large",
    torch_dtype=torch.float16 if DEVICE == 'cuda' else torch.float32
).to(DEVICE)

def analyze_image_colors(image_path: str):
    """Extract dominant colors from an image using ColorThief."""
    try:
        color_thief = ColorThief(image_path)
        palette = color_thief.get_palette(color_count=6, quality=1)
        colors = []
        for color in palette:
            try:
                name = closest_color(color)
                colors.append(name)
            except Exception as e:
                logger.error(f"Error in color matching: {e}")
                continue
        unique_colors = list(set(colors))
        return unique_colors[:3]
    except Exception as e:
        logger.error(f"Error analyzing image colors for {image_path}: {str(e)}")
        return []

def generate_caption(image_path: str) -> str:
    """Generate a caption for a manga panel including dominant colors."""
    try:
        color_terms = analyze_image_colors(image_path)
        raw_image = Image.open(image_path).convert('RGB')
        inputs = processor(
            raw_image,
            "a detailed scan of a manga panel showing",
            return_tensors="pt"
        ).to(DEVICE, torch.float16)
        out = model.generate(
            **inputs,
            max_length=100,
            num_beams=5,
            temperature=0.7,
            repetition_penalty=1.5,
            no_repeat_ngram_size=2
        )
        caption = processor.decode(out[0], skip_special_tokens=True)
        # Extract page number from filename (assumes naming convention like 'chapter_1_page_XX.png')
        page_num = os.path.basename(image_path).split('_')[1].split('.')[0]
        if color_terms:
            if len(color_terms) > 1:
                color_desc = ", ".join(color_terms[:-1]) + f", and {color_terms[-1]}"
            else:
                color_desc = color_terms[0]
            return f"Page {page_num}: {caption} with dominant {color_desc} colors."
        else:
            return f"Page {page_num}: {caption}"
    except Exception as e:
        logger.error(f"Error generating caption for {image_path}: {str(e)}")
        return f"{os.path.basename(image_path)}: Processing failed."
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
