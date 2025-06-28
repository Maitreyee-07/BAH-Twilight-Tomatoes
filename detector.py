import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import uniform_filter, binary_dilation, label
from skimage.measure import regionprops

# === STEP 1: Load image and convert to Tb ===
file_path = r"E:\BAH\INSAT3D_TIR1_India\3DIMG_07NOV2019_0030_L1C_SGP.tif"
dn = np.array(Image.open(file_path)).astype(np.float32)

print("Image shape (lat, lon):", dn.shape)
tb = 0.05 * dn + 150.0
print("Tb range:", tb.min(), "to", tb.max())

# === STEP 2: Parameters (adjustable) ===
mask_lower = 190.0   # lower bound for DAS spread
mask_upper = 200.0   # upper bound for DAS spread
core_thresh = 190.5  # core of cluster
estimated_km_per_pixel = 6.0  # set this manually after inspection

# === STEP 3: Detect-and-Spread (first stage) ===
def detect_and_spread(tb, core_thresh, spread_low, spread_high, iterations=5):
    core = tb > core_thresh
    spread = (tb >= spread_low) & (tb <= spread_high)
    mask = core.copy()
    for _ in range(iterations):
        mask = binary_dilation(mask, structure=np.ones((3, 3))) & spread | core
    return mask.astype(np.uint8)

das_mask = detect_and_spread(tb, core_thresh, mask_lower, mask_upper)

# === STEP 4: Apply Temperature Density Filter to DAS output ===
density = uniform_filter(das_mask.astype(float), size=9)
density_mask = (density > 0.5).astype(np.uint8)

# === STEP 5: Filter blobs ≥ 111 km radius ===
labeled, _ = label(density_mask)
min_radius_km = 111
min_area_pixels = np.pi * (min_radius_km / estimated_km_per_pixel)**2

valid_mask = np.zeros_like(das_mask)
for region in regionprops(labeled):
    if region.area >= min_area_pixels:
        valid_mask[labeled == region.label] = 1

# === STEP 6: Visualize in correct flow order ===
fig, axs = plt.subplots(1, 4, figsize=(22, 6))

axs[0].imshow(tb, cmap='inferno', vmin=np.percentile(tb, 2), vmax=np.percentile(tb, 98))
axs[0].set_title("Brightness Temperature (K)")
axs[0].axis('off')

axs[1].imshow(das_mask, cmap='gray')
axs[1].set_title("Step 1: DAS Output")
axs[1].axis('off')

axs[2].imshow(density_mask, cmap='gray')
axs[2].set_title("Step 2: After Density Filter")
axs[2].axis('off')

axs[3].imshow(valid_mask, cmap='gray')
axs[3].set_title("Step 3: TCCs ≥111 km Radius")
axs[3].axis('off')

plt.tight_layout()
plt.show()
