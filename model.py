from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import visualize
import numpy as np
import colorsys
import random
import cv2
import os
import matplotlib.pyplot as plt
from collections import defaultdict
import imutils

# Load class names
CLASS_NAMES = open('labels.txt').read().strip().split("\n")

# Assign colors for visualization
hsv = [(i / len(CLASS_NAMES), 1, 1.0) for i in range(len(CLASS_NAMES))]
COLORS = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
random.seed(42)
random.shuffle(COLORS)

# Config class
class SimpleConfig(Config):
    NAME = "tcc_detection"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = len(CLASS_NAMES)  # including background

config = SimpleConfig()

# Initialize model
print("[INFO] Loading Mask R-CNN model...")
model = modellib.MaskRCNN(mode="inference", config=config, model_dir=os.getcwd())

try:
    model.load_weights("tir.h5", by_name=True)
except Exception as e:
    print("[ERROR] Could not load weights. Check path and NUM_CLASSES match. Details:", e)
    exit(1)

# Process images
for s in range(1, 46):
    image_path = f"D:\\SIH\\Test MaskRCNN\\images\\satellite{s}.jpg"
    print(f"[INFO] Processing {image_path}")
    
    # read grayscale image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"[WARN] Could not load {image_path}, skipping.")
        continue

    # convert to 3 channels
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image = imutils.resize(image, width=512)

    # detect
    results = model.detect([image], verbose=1)
    r = results[0]

    if r["masks"] is None or r["masks"].size == 0:
        print("[INFO] No TCCs detected in this image.")
        continue

    # visualize masks
    for i in range(r["rois"].shape[0]):
        classID = r["class_ids"][i]
        if classID >= len(CLASS_NAMES):
            print(f"[WARN] classID {classID} exceeds CLASS_NAMES, skipping")
            continue
        
        mask = r["masks"][:, :, i]
        color = COLORS[classID][::-1]  # BGR
        image = visualize.apply_mask(image, mask, color, alpha=0.5)
        
        # get coordinates of mask
        coords = np.where(mask == True)
        x_coords = coords[1]
        y_coords = coords[0]
        print(f"[INFO] Cluster {i} has {len(x_coords)} pixels.")

        # draw bounding box
        (y1, x1, y2, x2) = r["rois"][i]
        score = r["scores"][i]
        label = CLASS_NAMES[classID]
        text = f"{label}:{i} {score:.3f}"
        cv2.rectangle(image, (x1,y1), (x2,y2), [int(c*255) for c in color], 2)
        cv2.putText(image, text, (x1, y1 - 5 if y1-5>5 else y1+15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, [int(c*255) for c in color], 2)
        
        # optional center of mass
        cx = (x1+x2)//2
        cy = (y1+y2)//2
        print(f"  Center of mass: ({cx},{cy})")

    # save output
    plt.figure(figsize=(8,8))
    plt.imshow(image)
    plt.axis("off")
    output_path = f"D:\\....path....\\{s}.png"
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"[INFO] Saved {output_path}")

