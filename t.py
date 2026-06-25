import os

TRAIN_DIR = "dataset/real_vs_fake/real-vs-fake/train"
TEST_DIR = "dataset/real_vs_fake/real-vs-fake/test"

print(TRAIN_DIR)
print(TEST_DIR)
print(os.path.exists(TRAIN_DIR))
print(os.path.exists(TEST_DIR))