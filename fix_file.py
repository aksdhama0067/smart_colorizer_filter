import urllib.request
import numpy as np
import os

# Get the path of current folder

base_dir = os.path.dirname(os.path.abspath(__file__))
target_path = os.path.join(base_dir, "pts_in_hull.npy")

# Direct URL to the raw binary / creator

url = "https://github.com/richzhang/colorization/raw/caffe/resources/pts_in_hull.npy"

print("Downloading fresh, uncorrupted file...")
try:
    urllib.request.urlretrieve(url, target_path)
    print("Download complete!")

    # Automatically verifies the file is healthy by loading it safely
    
    data = np.load(target_path, allow_pickle=True)
    print(f"Verification Success! Shape of array: {data.shape}")
except Exception as e:
    print(f"Error fixing file: {e}")
