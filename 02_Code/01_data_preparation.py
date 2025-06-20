# ---------------------------------------------------------------------------
# SCRIPT: 01_data_preparation.py
# PURPOSE: Prepares all geospatial data and image tiles for the project.
# WORKFLOW:
#   0. Buffer roads and mask out overlapping railway areas hierarchically.
#   1. Download 1km² orthophotos from swisstopo.
#   2. Prune orthophotos that do not contain any road geometries.
#   3. Crop orthophotos to the boundaries of the cleaned road buffers.
#   4. Select a stratified random sample of roads for tiling.
#   5. Tile the cropped images of selected roads into 512x512 squares.
# ---------------------------------------------------------------------------

import os
import rasterio
import requests
import geopandas as gpd
import pandas as pd
import numpy as np
from rasterio.mask import mask
from rasterio.merge import merge
from shapely.geometry import box
from shapely.ops import unary_union
from collections import defaultdict
from tqdm import tqdm

def buffer_and_mask_roads(raw_road_path, railway_path, output_path, hierarchy_col='stufe'):
    """Step 0: Combines buffering and hierarchical masking of road and railway geometries."""
    print("--- STEP 0: HIERARCHICAL BUFFERING AND MASKING ---")
    buffer_map = {'10m Street': 5.0, '8m Street': 4.0, '6m Street': 3.0, '4m Street': 2.0, '3m Street': 1.5, 'Autobahn': 3.5}
    default_buffer = 1.0

    gdf_roads = gpd.read_file(raw_road_path)
    gdf_railways = gpd.read_file(railway_path)

    gdf_roads['geometry'] = gdf_roads.apply(lambda row: row.geometry.buffer(buffer_map.get(row['OBJEKTART'], default_buffer)), axis=1)
    gdf_railways['geometry'] = gdf_railways.buffer(3.5)

    def update_geometry_by_hierarchy(target_gdf, ref_gdf, hierarchy_col):
        for gdf in [target_gdf, ref_gdf]:
            if hierarchy_col not in gdf.columns:
                gdf[hierarchy_col] = 0
            gdf[hierarchy_col] = pd.to_numeric(gdf[hierarchy_col], errors='coerce').fillna(0)
        ref_sindex = ref_gdf.sindex
        target_gdf['updated_geometry'] = target_gdf['geometry'].copy()
        for i, row in tqdm(target_gdf.iterrows(), total=len(target_gdf), desc="Updating geometries"):
            geom = row['updated_geometry']
            possible_matches_index = list(ref_sindex.intersection(geom.bounds))
            possible_matches = ref_gdf.iloc[possible_matches_index]
            geoms_to_subtract = [ref_row['geometry'] for j, ref_row in possible_matches.iterrows()
                               if (target_gdf is not ref_gdf or i != j) and ref_row[hierarchy_col] > row[hierarchy_col] and geom.intersects(ref_row['geometry'])]
            if geoms_to_subtract:
                target_gdf.at[i, 'updated_geometry'] = geom.difference(unary_union(geoms_to_subtract))
        target_gdf['geometry'] = target_gdf['updated_geometry']
        return target_gdf.drop(columns=['updated_geometry'])

    print("Resolving internal road overlaps...")
    gdf_roads = update_geometry_by_hierarchy(gdf_roads, gdf_roads, hierarchy_col)
    print("Masking roads with railways...")
    gdf_roads = update_geometry_by_hierarchy(gdf_roads, gdf_railways, hierarchy_col)
    
    gdf_roads = gdf_roads[~gdf_roads.is_empty]
    gdf_roads.to_file(output_path, driver='GPKG')
    print(f"Saved {len(gdf_roads)} cleaned road buffers to: {output_path}\n")
    return output_path

def download_swiss_images(csv_path, output_folder):
    """Step 1: Downloads images from URLs listed in a CSV file."""
    print(f"--- STEP 1: DOWNLOADING SWISS ORTHOPHOTOS ---")
    os.makedirs(output_folder, exist_ok=True)
    try:
        df = pd.read_csv(csv_path, header=None, names=['url'])
    except FileNotFoundError: return False
    for url in tqdm(df['url'].tolist(), desc="Downloading Orthophotos"):
        try:
            filename = url.split('/')[-1]
            output_path = os.path.join(output_folder, filename)
            if os.path.exists(output_path): continue
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(output_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
        except requests.exceptions.RequestException as e: print(f"Error: {e}")
    print("Download process completed.\n")
    return True

def prune_unnecessary_images(image_folder, road_geopackage):
    """Step 2: Deletes downloaded images that do not intersect with the road network."""
    print(f"--- STEP 2: PRUNING UNNECESSARY IMAGES ---")
    gdf_roads = gpd.read_file(road_geopackage)
    all_roads_union = gdf_roads.unary_union
    image_files = [f for f in os.listdir(image_folder) if f.endswith(".tif")]
    deleted_count = 0
    for image_name in tqdm(image_files, desc="Pruning Images"):
        image_path = os.path.join(image_folder, image_name)
        with rasterio.open(image_path) as src:
            if not box(*src.bounds).intersects(all_roads_union):
                src.close(); os.remove(image_path); deleted_count += 1
    print(f"Pruning complete. Deleted {deleted_count} images.\n")

def crop_roads_from_images(image_folder, road_geopackage, output_folder):
    """Step 3: Crops large images to road segment geometries."""
    print(f"--- STEP 3: CROPPING ROAD SEGMENTS ---")
    os.makedirs(output_folder, exist_ok=True)
    gdf = gpd.read_file(road_geopackage)
    clipped_data = defaultdict(list)
    image_files = [f for f in os.listdir(image_folder) if f.endswith(".tif")]
    for image_name in tqdm(image_files, desc="Clipping Roads from Orthos"):
        with rasterio.open(os.path.join(image_folder, image_name)) as src:
            intersecting = gdf[gdf.intersects(box(*src.bounds))]
            for _, row in intersecting.iterrows():
                out_img, out_trans = mask(src, [row.geometry], crop=True, nodata=0)
                clipped_data[row["unique_id"]].append({"image": out_img, "transform": out_trans, "meta": src.meta.copy()})
    for uuid, clips in tqdm(clipped_data.items(), desc="Saving Cropped Roads"):
        output_path = os.path.join(output_folder, f"{uuid}.tif")
        if os.path.exists(output_path): continue
        if len(clips) == 1: img, trans = clips[0]["image"], clips[0]["transform"]
        else:
            srcs = [rasterio.MemoryFile().open(driver='GTiff', height=c['image'].shape[1], width=c['image'].shape[2], count=c['meta']['count'], dtype=c['image'].dtype, crs=c['meta']['crs'], transform=c['transform']).write(c['image']) for c in clips]
            img, trans = merge(srcs, nodata=0)
        meta = clips[0]["meta"]; meta.update({"driver": "GTiff", "height": img.shape[1], "width": img.shape[2], "transform": trans, "nodata": 0, "compress": "LZW"})
        with rasterio.open(output_path, "w", **meta) as dest: dest.write(img)
    print("Cropping process completed.\n")

def select_roads_for_tiling(road_geopackage, sample_fraction=0.02, stratify_col="OBJEKTART", random_state=42):
    """Step 4: Selects a stratified random sample of roads to be tiled."""
    print(f"--- STEP 4: SELECTING ROADS FOR TILING ---")
    gdf = gpd.read_file(road_geopackage)
    gdf_selected = gdf.groupby(stratify_col, group_keys=False).apply(lambda x: x.sample(frac=sample_fraction, random_state=random_state))
    print(f"Selected {len(gdf_selected)} road segments for tiling.\n")
    return gdf_selected

def tile_road_segments(cropped_roads_folder, gdf_to_tile, output_folder, tile_size_pixels=512, tile_size_meters=51.2):
    """Step 5: Tiles the cropped road images of selected segments."""
    print(f"--- STEP 5: TILING SELECTED ROAD SEGMENTS ---")
    os.makedirs(output_folder, exist_ok=True)
    all_tiles = []
    for _, row in tqdm(gdf_to_tile.iterrows(), desc="Generating Tile Grids"):
        minx, miny, maxx, maxy = row.geometry.bounds
        for i in range(int(np.ceil((maxx-minx)/tile_size_meters))):
            for j in range(int(np.ceil((maxy-miny)/tile_size_meters))):
                tile_geom = box(minx + i*tile_size_meters, miny + j*tile_size_meters, minx + (i+1)*tile_size_meters, miny + (j+1)*tile_size_meters)
                if row.geometry.intersects(tile_geom): all_tiles.append({'tile_name': f"{row['unique_id']}_tile_{len(all_tiles)}", 'unique_id': row['unique_id'], 'geometry': tile_geom})
    gdf_tiles = gpd.GeoDataFrame(all_tiles, crs=gdf_to_tile.crs)
    for _, row in tqdm(gdf_tiles.iterrows(), desc="Creating Tiled Images"):
        image_path = os.path.join(cropped_roads_folder, f"{row['unique_id']}.tif")
        output_path = os.path.join(output_folder, f"{row['tile_name']}.tif")
        if not os.path.exists(image_path) or os.path.exists(output_path): continue
        with rasterio.open(image_path) as src:
            try:
                out_img, out_trans = mask(src, [row.geometry], crop=True, nodata=0)
                if out_img.sum() == 0: continue
                target = np.zeros((src.count, tile_size_pixels, tile_size_pixels), dtype=out_img.dtype)
                h, w = out_img.shape[1], out_img.shape[2]
                target[:, :min(h, tile_size_pixels), :min(w, tile_size_pixels)] = out_img[:, :min(h, tile_size_pixels), :min(w, tile_size_pixels)]
                meta = src.meta.copy(); meta.update({"height": tile_size_pixels, "width": tile_size_pixels, "transform": out_trans, "compress": "LZW", "nodata": 0})
                with rasterio.open(output_path, "w", **meta) as dst: dst.write(target)
            except ValueError: continue
    print("Tiling process completed.")

def main():
    """Main execution workflow."""
    base_path = os.path.dirname(__file__)
    raw_folder = os.path.abspath(os.path.join(base_path, '../01_Data/01_Raw/'))
    processed_folder = os.path.abspath(os.path.join(base_path, '../01_Data/02_Processed/'))
    
    # Input files
    raw_road_gpkg = os.path.join(raw_folder, 'Zurich_road.gpkg')
    railway_gpkg = os.path.join(raw_folder, 'Zurich_rail.gpkg')
    swissimage_csv = os.path.join(raw_folder, 'swissimage_urls.csv') # Assumed name
    
    # Output paths
    orthophotos_folder = os.path.join(raw_folder, 'orthophotos')
    cleaned_road_gpkg = os.path.join(processed_folder, 'zurich_roads_buffered_cleaned.gpkg')
    cropped_roads_folder = os.path.join(processed_folder, 'zurich_road_segments_clipped')
    tiled_images_folder = os.path.join(processed_folder, 'zurich_tiled_512x512')

    # Run pipeline
    cleaned_gpkg = buffer_and_mask_roads(raw_road_gpkg, railway_gpkg, cleaned_road_gpkg)
    if not download_swiss_images(swissimage_csv, orthophotos_folder): return
    prune_unnecessary_images(orthophotos_folder, cleaned_gpkg)
    crop_roads_from_images(orthophotos_folder, cleaned_gpkg, cropped_roads_folder)
    gdf_selected = select_roads_for_tiling(cleaned_gpkg)
    tile_road_segments(cropped_roads_folder, gdf_selected, tiled_images_folder)
    print("\nData preparation pipeline finished successfully!")

if __name__ == "__main__":
    main()