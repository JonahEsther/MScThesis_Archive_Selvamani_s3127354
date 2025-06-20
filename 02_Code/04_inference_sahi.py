# ---------------------------------------------------------------------------
# SCRIPT: 04_inference_sahi.py
# PURPOSE: Performs sliced inference on all  road segment images using a
#          trained model and saves georeferenced detections.
# USAGE: > python 04_inference_sahi.py --model_run_name [NAME]
# EXAMPLE: > python 04_inference_sahi.py --model_run_name B_YOLOv11x_Zurich
# ---------------------------------------------------------------------------

import os, argparse, rasterio, torch
import geopandas as gpd
import pandas as pd
from shapely.geometry import box
from tqdm import tqdm
from sahi.predict import get_sliced_prediction, AutoDetectionModel

def main(model_run_name):
    """Main function to run the SAHI inference pipeline."""
    base_path = os.path.dirname(__file__)
    
    # Define paths
    images_folder = os.path.abspath(os.path.join(base_path, '../01_Data/02_Processed/zurich_road_segments_clipped/'))
    model_path = os.path.abspath(os.path.join(base_path, f'../03_Results/training_runs/{model_run_name}/weights/best.pt'))
    output_gpkg_folder = os.path.abspath(os.path.join(base_path, '../03_Results/'))
    output_gpkg_path = os.path.join(output_gpkg_folder, 'mapped_distress.gpkg')
    
    os.makedirs(output_gpkg_folder, exist_ok=True)

    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at '{model_path}'"); return

    # Load model
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='ultralytics', model_path=model_path,
        confidence_threshold=0.2, device="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    print(f"Model '{model_run_name}' loaded on device: {detection_model.device}")

    # Run inference
    all_gdfs = []
    tif_files = [f for f in os.listdir(images_folder) if f.lower().endswith(".tif")]
    for filename in tqdm(tif_files, desc="Running SAHI Inference"):
        tif_path = os.path.join(images_folder, filename)
        result = get_sliced_prediction(tif_path, detection_model, slice_height=512, slice_width=512, overlap_height_ratio=0.2, overlap_width_ratio=0.2)
        if not result.object_prediction_list: continue

        with rasterio.open(tif_path) as raster:
            transform, crs = raster.transform, raster.crs
            geoms, labels, scores = [], [], []
            for pred in result.object_prediction_list:
                x, y, w, h = pred.bbox.to_xywh()
                x1_geo, y1_geo = transform * (x, y)
                x2_geo, y2_geo = transform * (x + w, y + h)
                geoms.append(box(x1_geo, y1_geo, x2_geo, y2_geo))
                labels.append(pred.category.name)
                scores.append(pred.score.value)
            if geoms:
                gdf = gpd.GeoDataFrame({"category":labels, "score":scores, "source_file":os.path.splitext(filename)[0], "geometry":geoms}, crs=crs)
                all_gdfs.append(gdf)

    # Save results
    if all_gdfs:
        combined_gdf = gpd.GeoDataFrame(pd.concat(all_gdfs, ignore_index=True), crs=all_gdfs[0].crs)
        combined_gdf.to_file(output_gpkg_path, driver="GPKG")
        print(f"\n✅ Successfully saved {len(combined_gdf)} detections to '{output_gpkg_path}'")
    else:
        print("\n⚠️ No predictions were found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SAHI inference on road images.")
    parser.add_argument("--model_run_name", type=str, required=True, help="Name of the training run directory (e.g., 'B_YOLOv11x_Zurich').")
    args = parser.parse_args()
    main(args.model_run_name)