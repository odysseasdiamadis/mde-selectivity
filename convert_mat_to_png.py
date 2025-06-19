import sys
import os
import h5py
from PIL import Image
import numpy as np
import tqdm
import imageio
import requests

url = 'http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat'
filename = '/tmp/nyu_depth_v2_labeled.mat'

def download_file(url, output_path, chunk_size=1024):
    # Send HTTP GET request with streaming enabled
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        # Create a progress bar
        with tqdm.tqdm(total=total_size, unit='B', unit_scale=True, desc="Download raw dataset") as pbar:
            with open(output_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:  # Filter out keep-alive chunks
                        f.write(chunk)
                        pbar.update(len(chunk))



if len(sys.argv) > 1:
    out_folder = sys.argv[1]
else:
    out_folder = "converted"

if not os.path.isfile(filename):
    print(f"Could not find file {filename}, starting download", file=sys.stderr)
    download_file(url, filename)
    

rgb_folder = os.path.join(out_folder, 'rgb')
depth_folder = os.path.join(out_folder, 'depth')

os.makedirs(out_folder, exist_ok=True)
os.makedirs(rgb_folder, exist_ok=True)
os.makedirs(depth_folder, exist_ok=True)



with h5py.File(filename, 'r') as f:
    images = f['images']
    depths = f['depths']
    names = f['names']
    
    assert images.shape[0] == depths.shape[0], \
        "Dataset is corrupted: rgb images are not the same shape as depth images"

    for idx in tqdm.tqdm(range(images.shape[0]), desc="Converting dataset to images"):
        string_idx = "{:05d}".format(idx)
        string_idx = f[names[idx]][()]
        rgb_path = os.path.join(rgb_folder, f"{string_idx}.png")
        depth_path = os.path.join(depth_folder, f"{string_idx}.tiff")
        
        rgb_image = images[idx]
        depth_image = depths[idx]

        rgb_image = np.transpose(rgb_image, (2, 1, 0))  # Convert to (H, W, 3) if needed
        rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)

        Image.fromarray(rgb_image).save(rgb_path)
        imageio.imwrite(depth_path, depth_image.astype(np.float32))
