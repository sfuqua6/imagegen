import os
import glob
import logging
from captioning.caption_generator import generate_caption

logger = logging.getLogger("ChapterProcessor")

def process_chapter(chapter_path: str):
    logger.info(f"Processing chapter: {os.path.basename(chapter_path)}")
    try:
        images = sorted(
            glob.glob(os.path.join(chapter_path, "*.png")),
            key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0])
        )
    except Exception as e:
        logger.error(f"Error sorting images in {chapter_path}: {e}")
        return

    captions = []
    for idx, img_path in enumerate(images, 1):
        if idx % 5 == 0:
            # Placeholder for memory clearing if needed
            pass
        caption = generate_caption(img_path)
        captions.append(caption)

    output_file = os.path.join(chapter_path, "detailed_captions.txt")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(captions))
        logger.info(f"Completed processing {os.path.basename(chapter_path)}. Captions written to {output_file}")
    except Exception as e:
        logger.error(f"Error writing captions to {output_file}: {e}")
