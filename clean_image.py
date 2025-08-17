'''import os
import pandas as pd
import shutil

# Set your paths
csv_path = r'E:\A-IIIT-B\metadata.csv'
img_folder = r'E:\Cleaned data'
output_folder = r'E:\Dataset_For_Training'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Read CSV
df = pd.read_csv(csv_path)
df = df[df['text'].notnull()]  # Only keep rows with a non-empty description

# Get just the filename (in case CSV includes folders)
df['just_filename'] = df['file_name'].apply(lambda x: os.path.basename(str(x)).strip())
print(f"Images with descriptions: {len(df)}")

# Copy matching images
copied = 0
for fname in df['just_filename']:
    src_file = os.path.join(img_folder, fname)
    dst_file = os.path.join(output_folder, fname)
    if os.path.isfile(src_file):
        shutil.copy2(src_file, dst_file)
        copied += 1
    else:
        print(f"WARNING: File not found for {fname}")

print(f"Done. {copied} images copied to {output_folder}.")'''

import pandas as pd
import os

# Set your file paths
csv_path = r'E:\A-IIIT-B\metadata.csv'
images_folder = r'E:\Cleaned data'
output_csv_path = r'E:\A-IIIT-B\metadata_filtered.csv'

# Load the CSV
csv = pd.read_csv(csv_path)

# Extract just the image filename from the 'file_name' column
csv['just_filename'] = csv['file_name'].apply(lambda x: os.path.basename(str(x)).strip())

# List all images currently in your folder
actual_images = set(f for f in os.listdir(images_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg')))

# Filter: Only keep rows whose file exists in E:\Cleaned data
filtered_csv = csv[csv['just_filename'].isin(actual_images)]

# Save this filtered CSV
filtered_csv.to_csv(output_csv_path, index=False)

print(f"Filtered CSV saved to {output_csv_path}\n"
      f"Total entries before filter: {len(csv)}\n"
      f"Entries after filtering missing images: {len(filtered_csv)}")

