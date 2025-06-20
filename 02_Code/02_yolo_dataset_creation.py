# ---------------------------------------------------------------------------
# SCRIPT: 02_yolo_dataset_creation.py
# PURPOSE: Processes STAC annotation exports to create a final YOLO dataset.
# WORKFLOW:
#   1. Process STAC catalogs to copy source images and GeoJSON labels.
#   2. Clean data by removing empty labels and orphaned images.
#   3. Convert GeoJSON polygon labels to YOLO bounding box format.
#   4. Split the final dataset into train, validation, and test sets.
# ---------------------------------------------------------------------------

import os
import geopandas as gpd
from pystac import Catalog
from shapely.geometry import shape
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import shutil
import glob

def process_stac_catalogs(catalog_paths, tiled_images_dir, staging_dir):
    """Step 1: Process STAC catalogs to copy images and labels to a staging area."""
    print("--- STEP 1: PROCESSING STAC ANNOTATION EXPORTS ---")
    staging_images_dir = os.path.join(staging_dir, "images")
    staging_labels_dir = os.path.join(staging_dir, "labels_geojson")
    os.makedirs(staging_images_dir, exist_ok=True)
    os.makedirs(staging_labels_dir, exist_ok=True)
    for catalog_path in catalog_paths:
        catalog = Catalog.from_file(catalog_path)
        for item in tqdm(catalog.get_all_items(), desc=f"Processing {os.path.basename(os.path.dirname(catalog_path))}"):
            original_filename_base = item.id.replace(".tif", "")
            source_image_path = os.path.join(tiled_images_dir, f"{original_filename_base}.tif")
            if os.path.exists(source_image_path):
                shutil.copy(source_image_path, os.path.join(staging_images_dir, f"{original_filename_base}.tif"))
            if "labels" in item.assets:
                labels_abs_path = os.path.join(os.path.dirname(item.get_self_href()), item.assets["labels"].href)
                if os.path.exists(labels_abs_path):
                    shutil.copy(labels_abs_path, os.path.join(staging_labels_dir, f"{original_filename_base}.geojson"))
    print("STAC processing complete.\n")

def clean_staged_data(staging_dir):
    """Step 2: Clean data by removing empty labels and orphaned images."""
    print("--- STEP 2: CLEANING STAGED DATA ---")
    images_dir = os.path.join(staging_dir, "images")
    labels_dir = os.path.join(staging_dir, "labels_geojson")
    for filename in os.listdir(labels_dir):
        if filename.endswith('.geojson'):
            if os.path.getsize(os.path.join(labels_dir, filename)) < 100: os.remove(os.path.join(labels_dir, filename))
    label_basenames = {os.path.splitext(f)[0] for f in os.listdir(labels_dir)}
    for image_file in os.listdir(images_dir):
        if os.path.splitext(image_file)[0] not in label_basenames: os.remove(os.path.join(images_dir, image_file))
    print("Data cleaning complete.\n")

def convert_geojson_to_yolo(staging_dir, class_map, image_size=(512, 512)):
    """Step 3: Convert GeoJSON annotations to YOLO format .txt files."""
    print("--- STEP 3: CONVERTING LABELS TO YOLO FORMAT ---")
    geojson_dir = os.path.join(staging_dir, "labels_geojson")
    yolo_labels_dir = os.path.join(staging_dir, "labels_yolo")
    os.makedirs(yolo_labels_dir, exist_ok=True)
    img_w, img_h = image_size
    for filename in tqdm(os.listdir(geojson_dir), desc="Converting to YOLO"):
        if not filename.endswith(".geojson"): continue
        gdf = gpd.read_file(os.path.join(geojson_dir, filename))
        yolo_txt_path = os.path.join(yolo_labels_dir, f"{os.path.splitext(filename)[0]}.txt")
        with open(yolo_txt_path, "w") as f:
            for _, row in gdf.iterrows():
                class_name = row.get("class"); class_id = class_map.get(class_name)
                if class_id is None: continue
                minx, miny, maxx, maxy = shape(row.geometry).bounds
                x_center, y_center = (minx + maxx) / 2 / img_w, (miny + maxy) / 2 / img_h
                width, height = (maxx - minx) / img_w, (maxy - miny) / img_h
                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
    print("YOLO format conversion complete.\n")

def split_dataset(staging_dir, final_output_dir, test_size=0.15, val_size=0.15, random_state=42):
    """Step 4: Split files into train, validation, and test sets."""
    print("--- STEP 4: SPLITTING DATASET (70/15/15) ---")
    source_images = os.path.join(staging_dir, "images")
    source_labels = os.path.join(staging_dir, "labels_yolo")
    basenames = [os.path.splitext(f)[0] for f in os.listdir(source_images) if f.endswith('.tif')]
    train_files, val_test_files = train_test_split(basenames, test_size=(test_size + val_size), random_state=random_state)
    val_files, test_files = train_test_split(val_test_files, test_size=(test_size / (test_size + val_size)), random_state=random_state)
    def move_files(filenames, split_name):
        for bn in filenames:
            for folder, ext in [("images", ".tif"), ("labels", ".txt")]:
                dest_dir = os.path.join(final_output_dir, folder, split_name)
                os.makedirs(dest_dir, exist_ok=True)
                source_path = os.path.join(staging_dir, "labels_yolo" if ext == ".txt" else "images", f"{bn}{ext}")
                shutil.copy(source_path, dest_dir)
    move_files(train_files, "train"); move_files(val_files, "valid"); move_files(test_files, "test")
    print("Dataset splitting complete.\n")

def main():
    """Main execution workflow."""
    base_path = os.path.dirname(__file__)
    raw_folder = os.path.abspath(os.path.join(base_path, '../01_Data/01_Raw/'))
    processed_folder = os.path.abspath(os.path.join(base_path, '../01_Data/02_Processed/'))
    
    groundworks_dir = os.path.join(raw_folder, 'groundworks_exports')
    tiled_images_dir = os.path.join(processed_folder, 'zurich_tiled_512x512')
    
    # Automatically find all catalog.json files
    catalog_paths = glob.glob(os.path.join(groundworks_dir, '**', 'catalog.json'), recursive=True)
    if not catalog_paths:
        print(f"Error: No 'catalog.json' files found in {groundworks_dir}. Please check the path."); return
    
    staging_dir = os.path.join(processed_folder, 'temp_yolo_staging')
    final_yolo_dir = os.path.join(processed_folder, 'zurich_yolo_dataset')
    
    class_map = {"Pavement Distress": 0, "Non-Pavement Distress": 1}

    # Run pipeline
    process_stac_catalogs(catalog_paths, tiled_images_dir, staging_dir)
    clean_staged_data(staging_dir)
    convert_geojson_to_yolo(staging_dir, class_map)
    split_dataset(staging_dir, final_yolo_dir)
    shutil.rmtree(staging_dir) # Clean up temporary directory
    print("YOLO dataset creation pipeline finished successfully!")

if __name__ == "__main__":
    main()