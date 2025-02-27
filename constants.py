import os

PARENT_DIRNAME = os.path.expanduser("~/Project-Visual-Question-Answering/")
DATA_DIRNAME = os.path.join(PARENT_DIRNAME, "data/")
MODEL_DIRNAME = os.path.join(PARENT_DIRNAME, "model/")
IMAGE_DIRNAME = os.path.join(PARENT_DIRNAME, "image/")

TRAIN_DIRNAME=os.path.join(DATA_DIRNAME,"vaq2.0.TrainImages.txt")
TEST_DIRNAME=os.path.join(DATA_DIRNAME,"vaq2.0.TestImages.txt")
VAL_DIRNAME=os.path.join(DATA_DIRNAME,"vaq2.0.DevImages.txt")

IMAGE_DIR=os.path.join(DATA_DIRNAME,"val2014-resised/")