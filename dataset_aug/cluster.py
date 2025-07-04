import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import DBSCAN, KMeans

# === CONFIG ===
input_dir = "binary_mask"
output_dbscan_dir = "dbscan_cluster"
output_kmeans_dir = "kmeans_cluster"
os.makedirs(output_dbscan_dir, exist_ok=True)
os.makedirs(output_kmeans_dir, exist_ok=True)

# === Clustering Function ===
def cluster_tcc_pixels(binary_mask, method='dbscan', eps=10, min_samples=20, k=5):
    coords = np.column_stack(np.where(binary_mask))
    if coords.size == 0:
        return coords, np.array([])

    if method == 'dbscan':
        model = DBSCAN(eps=eps, min_samples=min_samples)
    elif method == 'kmeans':
        model = KMeans(n_clusters=k, random_state=42)
    else:
        raise ValueError("Method must be 'dbscan' or 'kmeans'")
    
    labels = model.fit_predict(coords)
    return coords, labels

# === Plot & Save Clusters ===
def save_cluster_plot(coords, labels, title, out_path):
    if coords.size == 0:
        print(f"[Skipped] No TCC pixels found in: {title}")
        return

    plt.figure(figsize=(7, 7))
    unique_labels = np.unique(labels)
    colors = plt.cm.get_cmap('tab20', len(unique_labels))
    
    for lbl in unique_labels:
        if lbl == -1:
            color = 'k'
            size = 5
            label_name = 'Noise'
        else:
            color = colors(lbl)
            size = 10
            label_name = f'Cluster {lbl}'
        cluster_points = coords[labels == lbl]
        plt.scatter(cluster_points[:, 1], cluster_points[:, 0], s=size, color=color, label=label_name)
    
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.xlabel("Column (X)")
    plt.ylabel("Row (Y)")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")

# === Process All PNG Masks ===
for fname in os.listdir(input_dir):
    if fname.endswith(".png"):
        input_path = os.path.join(input_dir, fname)
        mask = np.array(Image.open(input_path).convert('L')) > 128  # Boolean mask

        # DBSCAN
        coords_db, labels_db = cluster_tcc_pixels(mask, method='dbscan', eps=15, min_samples=30)
        out_db = os.path.join(output_dbscan_dir, fname)
        save_cluster_plot(coords_db, labels_db, f"DBSCAN: {fname}", out_db)

        # KMeans
        coords_km, labels_km = cluster_tcc_pixels(mask, method='kmeans', k=5)
        out_km = os.path.join(output_kmeans_dir, fname)
        save_cluster_plot(coords_km, labels_km, f"KMeans: {fname}", out_km)
