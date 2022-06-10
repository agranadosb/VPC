from vpc.conf.env import DATA_FOLDER, BATCH_SIZE, EPOCHS, IMAGE_SIZE
from vpc.constants import ANCHORS
from vpc.model.trainer import VocTrainerYoloV3


def main():
    VocTrainerYoloV3(
        IMAGE_SIZE, BATCH_SIZE, EPOCHS, DATA_FOLDER, ANCHORS, training_ratio=0.8
    ).plot_model().train()
