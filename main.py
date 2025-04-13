import argparse
import logging
from training.train_pipeline import train_model
from data.chapter_processor import process_chapter
from data.data_loader import get_chapter_paths
from inference.generate_chapter import generate_chapter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Main")

def preprocess_data():
    from config import BASE_DATA_DIR
    chapters = get_chapter_paths(base_dir=BASE_DATA_DIR)
    for chapter in chapters:
        process_chapter(chapter)

def main():
    parser = argparse.ArgumentParser(description="Manga Generation Pipeline")
    parser.add_argument("action", choices=["preprocess", "train", "infer"], help="Action to perform")
    parser.add_argument("--prompt", type=str, default="A dramatic page with red hues", help="Text prompt for inference")
    args = parser.parse_args()

    if args.action == "preprocess":
        logger.info("Starting data preprocessing...")
        preprocess_data()
    elif args.action == "train":
        logger.info("Starting training...")
        train_model()
    elif args.action == "infer":
        logger.info("Starting inference...")
        generate_chapter(prompt=args.prompt, num_panels=5)
    else:
        logger.error("Invalid action specified.")

if __name__ == "__main__":
    main()
