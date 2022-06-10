import torch
from dotenv import dotenv_values

environment_variables: dict[str, str] = {**dotenv_values()}

BATCH_SIZE = int(environment_variables.get("BATCH_SIZE", "8"))
EPOCHS = int(environment_variables.get("EPOCHS", "10"))
IMAGE_SIZE = int(environment_variables.get("IMAGE_SIZE", "416"))

# Data Folders
DATA_FOLDER = environment_variables.get("DATA_FOLDER", "./data")

STEPS_FOR_APPLY_GRADIENT = int(
    environment_variables.get("STEPS_FOR_APPLY_GRADIENT", "1")
)
SCORE_THRESHOLD = float(environment_variables.get("SCORE_THRESHOLD", "0.05"))
DEVICE = environment_variables.get("SCORE_THRESHOLD", "cuda" if torch.cuda.is_available() else "cpu")

LOG_DIR = environment_variables.get("LOG_DIR", "./logs")
