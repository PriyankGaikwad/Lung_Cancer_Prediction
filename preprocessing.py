import os
import numpy as np
from sklearn.model_selection import train_test_split
import shutil

current_dir = os.path.dirname(os.path.abspath(__file__))

benign_dir = os.path.join(current_dir, 'Bengin cases')
malignant_dir = os.path.join(current_dir, 'Malignant cases')
normal_dir = os.path.join(current_dir, 'Normal cases')

benign_files = [os.path.join(benign_dir, file) for file in os.listdir(benign_dir)]
malignant_files = [os.path.join(malignant_dir, file) for file in os.listdir(malignant_dir)]
normal_files = [os.path.join(normal_dir, file) for file in os.listdir(normal_dir)]

benign_train, benign_val_test = train_test_split(benign_files, test_size=0.2, random_state=42)
malignant_train, malignant_val_test = train_test_split(malignant_files, test_size=0.2, random_state=42)
normal_train, normal_val_test = train_test_split(normal_files, test_size=0.2, random_state=42)

benign_val, benign_test = train_test_split(benign_val_test, test_size=0.5, random_state=42)
malignant_val, malignant_test = train_test_split(malignant_val_test, test_size=0.5, random_state=42)
normal_val, normal_test = train_test_split(normal_val_test, test_size=0.5, random_state=42)

train_dir = os.path.join(current_dir, 'train')
val_dir = os.path.join(current_dir, 'validation')
test_dir = os.path.join(current_dir, 'test')

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

train_benign_dir = os.path.join(train_dir, 'benign')
train_malignant_dir = os.path.join(train_dir, 'malignant')
train_normal_dir = os.path.join(train_dir, 'normal')

val_benign_dir = os.path.join(val_dir, 'benign')
val_malignant_dir = os.path.join(val_dir, 'malignant')
val_normal_dir = os.path.join(val_dir, 'normal')

test_benign_dir = os.path.join(test_dir, 'benign')
test_malignant_dir = os.path.join(test_dir, 'malignant')
test_normal_dir = os.path.join(test_dir, 'normal')

os.makedirs(train_benign_dir, exist_ok=True)
os.makedirs(train_malignant_dir, exist_ok=True)
os.makedirs(train_normal_dir, exist_ok=True)

os.makedirs(val_benign_dir, exist_ok=True)
os.makedirs(val_malignant_dir, exist_ok=True)
os.makedirs(val_normal_dir, exist_ok=True)

os.makedirs(test_benign_dir, exist_ok=True)
os.makedirs(test_malignant_dir, exist_ok=True)
os.makedirs(test_normal_dir, exist_ok=True)

def copy_files(files, dst_dir):
    for file in files:
        destination = os.path.join(dst_dir, os.path.basename(file))
        print("Copying", file, "to", destination)
        try:
            shutil.copy(file, destination)
        except FileNotFoundError as e:
            print("Error copying file:", e)

copy_files(benign_train, train_benign_dir)
copy_files(malignant_train, train_malignant_dir)
copy_files(normal_train, train_normal_dir)

copy_files(benign_val, val_benign_dir)
copy_files(malignant_val, val_malignant_dir)
copy_files(normal_val, val_normal_dir)

copy_files(benign_test, test_benign_dir)
copy_files(malignant_test, test_malignant_dir)
copy_files(normal_test, test_normal_dir)

