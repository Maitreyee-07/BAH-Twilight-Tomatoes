import os
import numpy as np
import rasterio
from skimage.measure import label, regionprops
from PIL import Image

def load_tiff(filepath):
    with rasterio.open(filepath) as src:
        data = src.read(1).astype(np.float32)
    return data

def create_tcc_mask(data, threshold=220, km_per_pixel=4.0, min_radius_km=111, min_area_km2=34800):
    tb_mask = data < threshold
    
    min_area_pixels_radius = np.pi * (min_radius_km / km_per_pixel)**2
    labeled_radius = label(tb_mask)
    radius_filtered_mask = np.zeros_like(tb_mask, dtype=bool)
    
    for region in regionprops(labeled_radius):
        if region.area >= min_area_pixels_radius:
            radius_filtered_mask[labeled_radius == region.label] = True
    
    min_area_pixels_area = min_area_km2 / (km_per_pixel ** 2)
    labeled_area = label(radius_filtered_mask)
    final_mask = np.zeros_like(tb_mask, dtype=bool)
    
    for region in regionprops(labeled_area):
        if region.area >= min_area_pixels_area:
            final_mask[labeled_area == region.label] = True
    
    return final_mask

def save_mask_png(mask, out_path):
    # Convert boolean mask to 0/255 uint8 image
    img = Image.fromarray((mask * 255).astype(np.uint8))
    img.save(out_path)

# === Batch Process ===
input_dir = "mosdac_data"
output_dir = "binary_mask"
os.makedirs(output_dir, exist_ok=True)

for fname in os.listdir(input_dir):
    if fname.endswith('.tif'):
        input_path = os.path.join(input_dir, fname)
        output_path = os.path.join(output_dir, fname.replace('.tif', '.png'))
        
        data = load_tiff(input_path)
        mask = create_tcc_mask(data)
        save_mask_png(mask, output_path)

print("All binary masks saved as PNGs to:", output_dir)
