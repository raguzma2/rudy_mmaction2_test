import os
import random
import shutil

# Define paths
path_1 = r"C:\Users\rodol\Desktop\basic_salsa_step_data\1_valid_step"       # Replace with the actual path to basic_step files
path_0 = r"C:\Users\rodol\Desktop\basic_salsa_step_data\0_invalid_step"   # Replace with the actual path to non_basic_step files
path_training_folder = r"C:\Users\rodol\Desktop\basic_salsa_step_data\Computer_vision_model\train"       # Replace with the actual training folder path
path_validation_folder = r"C:\Users\rodol\Desktop\basic_salsa_step_data\Computer_vision_model\val"   # Replace with the actual validation folder path

# Output text files
training_text_file = r"C:\Users\rodol\Desktop\basic_salsa_step_data\Computer_vision_model\train_video.txt"            # Replace with the desired training text file path
validation_text_file = r"C:\Users\rodol\Desktop\basic_salsa_step_data\Computer_vision_model\val_video.txt"        # Replace with the desired validation text file path
basic_step_text_file = r"C:\Users\rodol\Desktop\basic_salsa_step_data\Computer_vision_model\basic_step_files.txt"  # Replace with the desired basic_step text file path

# Ensure output directories exist
os.makedirs(path_training_folder, exist_ok=True)
os.makedirs(path_validation_folder, exist_ok=True)

# Get list of files
basic_step_files = [f for f in os.listdir(path_1) if f.endswith('.mp4')]
non_basic_step_files = [f for f in os.listdir(path_0) if f.endswith('.mp4')]

# Create basic_step_files.txt
with open(basic_step_text_file, 'w') as f:
    for file_name in basic_step_files:
        f.write(f"{file_name} 1\n")

# Shuffle files
random.shuffle(basic_step_files)
random.shuffle(non_basic_step_files)

# Calculate split indices
num_basic_total = len(basic_step_files)
num_non_basic_total = len(non_basic_step_files)

num_basic_train = int(0.8 * num_basic_total)
num_basic_val = int(0.1 * num_basic_total)

num_non_basic_train = int(0.8 * num_non_basic_total)
num_non_basic_val = int(0.1 * num_non_basic_total)

# Split files into training and validation
basic_train_files = basic_step_files[:num_basic_train]
basic_val_files = basic_step_files[num_basic_train:num_basic_train+num_basic_val]

non_basic_train_files = non_basic_step_files[:num_non_basic_train]
non_basic_val_files = non_basic_step_files[num_non_basic_train:num_non_basic_train+num_non_basic_val]

# Prepare training data
training_data = [(f, 1) for f in basic_train_files] + [(f, 0) for f in non_basic_train_files]
random.shuffle(training_data)

# Prepare validation data
validation_data = [(f, 1) for f in basic_val_files] + [(f, 0) for f in non_basic_val_files]
random.shuffle(validation_data)

# Write training text file
with open(training_text_file, 'w') as f:
    for file_name, label in training_data:
        f.write(f"{file_name} {label}\n")

# Write validation text file
with open(validation_text_file, 'w') as f:
    for file_name, label in validation_data:
        f.write(f"{file_name} {label}\n")

# Move training files
for file_name in basic_train_files:
    shutil.move(os.path.join(path_1, file_name), os.path.join(path_training_folder, file_name))
for file_name in non_basic_train_files:
    shutil.move(os.path.join(path_0, file_name), os.path.join(path_training_folder, file_name))

# Move validation files
for file_name in basic_val_files:
    shutil.move(os.path.join(path_1, file_name), os.path.join(path_validation_folder, file_name))
for file_name in non_basic_val_files:
    shutil.move(os.path.join(path_0, file_name), os.path.join(path_validation_folder, file_name))
