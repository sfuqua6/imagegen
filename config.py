
BASE_DATA_DIR = "manga_chapters"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
MAX_PNGS_PER_BATCH = 50
TRAINING_EPOCHS = 100
MODEL_SAVE_PATH = "models/generation_model.pth"
