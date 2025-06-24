import os
from PIL import Image
from torchvision.transforms import Resize
from concurrent.futures import ThreadPoolExecutor, as_completed

import tqdm

# === CONFIG ===
depth_input_root = "/mnt/data/kitti/data_depth_annotated/mixed"
depth_output_root = "/mnt/data/kitti/data_depth_annotated/mixed_keep_rgb"

raw_input_root = "/mnt/data/kitti/raw"
raw_output_root = "/mnt/data/kitti/raw_keep_rgb"


raw_files = []
files = []
with open("./data_splits/kitti_eigen_train_files_with_gt.txt") as f:
    for line in f:
        l = line.split()
        if l[1] == 'None': continue

        if os.path.isfile(os.path.join(raw_input_root, l[0])):
            raw_files.append(l[0])

        if os.path.isfile(os.path.join(depth_input_root, l[1])):
            files.append(l[1])

with open("./data_splits/kitti_eigen_test_files_with_gt.txt") as f:
    for line in f:
        l = line.split()
        if l[1] == 'None': continue
        if os.path.isfile(os.path.join(raw_input_root, l[0])):
            raw_files.append(l[0])
        if os.path.isfile(os.path.join(depth_input_root, l[1])):
            files.append(l[1])

resize_to = (636, 192)  # or (w, h)
num_workers = 12  # Adjust to your CPU

# === Create Resize Transform Equivalent ===
if isinstance(resize_to, int):
    resize_transform = Resize(resize_to)
    def resize_image(img): # type: ignore
        return resize_transform(img)
else:
    def resize_image(img):
        return img.resize(resize_to, Image.BILINEAR)

# # === Collect Image Paths ===
depth_paths = []
for file in files:
    in_path = os.path.join(depth_input_root, file)
    out_path = os.path.join(depth_output_root, file)
    depth_paths.append((in_path, out_path))

raw_image_paths = []
for file in raw_files:
    raw_in_path = os.path.join(raw_input_root, file)
    raw_out_path = os.path.join(raw_output_root, file)
    raw_image_paths.append((raw_in_path, raw_out_path))


# === Define Worker Function ===
def process_image(in_out_paths, mode):
    in_path, out_path = in_out_paths
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with Image.open(in_path) as img:
        img = img.convert(mode)  # Grayscale
        resized = resize_image(img)
        resized.save(out_path)
    return out_path

# === Parallel Execution ===
with ThreadPoolExecutor(max_workers=num_workers) as executor:
    futures = [executor.submit(process_image, p, "L") for p in depth_paths] + \
        [executor.submit(process_image, p, "RGB") for p in raw_image_paths]
    for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
        try:
            out = future.result()
        except Exception as e:
            print(f"Error: {e}")
