# ---------------------------------------------------------------------------
# SCRIPT: 03_train_models.py
# PURPOSE: Handles training for all four experimental setups via CLI.
# EXPERIMENTS:
#   A: YOLOv8x on Zurich | B: YOLOv11x on Zurich
#   C: Multi-stage      | D: Combined with Offline Augmentation
# USAGE: > python 03_train_yolo.py --experiment [A|B|C|D]
# ---------------------------------------------------------------------------

import os, yaml, argparse, shutil, numpy as np, rasterio, random
from tqdm import tqdm
from ultralytics import YOLO

def create_yaml_file(path, train_img, val_img, test_img, names, nc):
    """Creates a dataset.yaml file required for YOLO training."""
    data = {
        'path': os.path.abspath(os.path.join(os.path.dirname(__file__), '..')),
        'train': os.path.relpath(train_img, os.path.join(os.path.dirname(__file__), '..')),
        'val': os.path.relpath(val_img, os.path.join(os.path.dirname(__file__), '..')),
        'test': os.path.relpath(test_img, os.path.join(os.path.dirname(__file__), '..')),
        'names': names, 'nc': nc
    }
    with open(path, 'w') as f: yaml.dump(data, f, sort_keys=False)
    print(f"Created dataset YAML: {path}")

def transform_yolo_labels(labels, op):
    """Transforms YOLO bounding box coordinates for an augmentation operation."""
    new_labels = []
    for line in labels:
        parts = line.strip().split(); cls, x, y, w, h = parts[0], float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        if op == 'rotate_90': x_n, y_n, w_n, h_n = y, 1-x, h, w
        elif op == 'rotate_180': x_n, y_n, w_n, h_n = 1-x, 1-y, w, h
        elif op == 'rotate_270': x_n, y_n, w_n, h_n = 1-y, x, h, w
        elif op == 'flip_horizontal': x_n, y_n, w_n, h_n = 1-x, y, w, h
        elif op == 'flip_vertical': x_n, y_n, w_n, h_n = x, 1-y, w, h
        else: x_n, y_n, w_n, h_n = x, y, w, h
        new_labels.append(f"{cls} {x_n:.6f} {y_n:.6f} {w_n:.6f} {h_n:.6f}\n")
    return new_labels

def prepare_augmented_combined_dataset(zurich_path, highrpd_path, combined_path):
    """Prepares the augmented and balanced dataset for Experiment D."""
    if os.path.exists(combined_path): shutil.rmtree(combined_path)
    combo_train_img, combo_train_lbl = os.path.join(combined_path, 'images/train'), os.path.join(combined_path, 'labels/train')
    combo_val_img, combo_val_lbl = os.path.join(combined_path, 'images/valid'), os.path.join(combined_path, 'labels/valid')
    for d in [combo_train_img, combo_train_lbl, combo_val_img, combo_val_lbl]: os.makedirs(d)
    
    zurich_train_img, zurich_train_lbl = os.path.join(zurich_path, 'images/train'), os.path.join(zurich_path, 'labels/train')
    aug_count = 0
    for filename in tqdm(os.listdir(zurich_train_img), desc="Augmenting Zurich Train"):
        if not filename.endswith('.tif'): continue
        basename = os.path.splitext(filename)[0]
        with rasterio.open(os.path.join(zurich_train_img, filename)) as src: img, meta = src.read(), src.meta
        with open(os.path.join(zurich_train_lbl, f"{basename}.txt"), 'r') as f: labels = f.readlines()
        augs = {'original':(img,labels),'rotate_90':(np.rot90(img,1,axes=(1,2)),transform_yolo_labels(labels,'rotate_90')),'rotate_180':(np.rot90(img,2,axes=(1,2)),transform_yolo_labels(labels,'rotate_180')),'rotate_270':(np.rot90(img,3,axes=(1,2)),transform_yolo_labels(labels,'rotate_270')),'flip_horizontal':(np.fliplr(img),transform_yolo_labels(labels,'flip_horizontal')),'flip_vertical':(np.flipud(img),transform_yolo_labels(labels,'flip_vertical')),'brightness':(np.clip(img*1.2,0,255).astype(meta['dtype']),labels)}
        for name, (aug_img, aug_lbls) in augs.items():
            meta.update(height=aug_img.shape[1], width=aug_img.shape[2])
            with rasterio.open(os.path.join(combo_train_img, f"{basename}_{name}.tif"), 'w', **meta) as dst: dst.write(aug_img)
            with open(os.path.join(combo_train_lbl, f"{basename}_{name}.txt"), 'w') as f: f.writelines(aug_lbls)
            aug_count += 1
    
    highrpd_train_img, highrpd_train_lbl = os.path.join(highrpd_path, 'images/train'), os.path.join(highrpd_path, 'labels/train')
    highrpd_files = [os.path.splitext(f)[0] for f in os.listdir(highrpd_train_img) if f.endswith('.tif')]
    files_to_copy = random.sample(highrpd_files, min(aug_count, len(highrpd_files)))
    for bn in tqdm(files_to_copy, desc="Copying HighRPD Train"):
        shutil.copy(os.path.join(highrpd_train_img, f"{bn}.tif"), combo_train_img)
        shutil.copy(os.path.join(highrpd_train_lbl, f"{bn}.txt"), combo_train_lbl)
        
    zurich_val_img, zurich_val_lbl = os.path.join(zurich_path, 'images/valid'), os.path.join(zurich_path, 'labels/valid')
    for f in os.listdir(zurich_val_img): shutil.copy(os.path.join(zurich_val_img, f), combo_val_img)
    for f in os.listdir(zurich_val_lbl): shutil.copy(os.path.join(zurich_val_lbl, f), combo_val_lbl)
    highrpd_val_img, highrpd_val_lbl = os.path.join(highrpd_path, 'images/valid'), os.path.join(highrpd_path, 'labels/valid')
    highrpd_val_files = [os.path.splitext(f)[0] for f in os.listdir(highrpd_val_img) if f.endswith('.tif')]
    files_to_copy_val = random.sample(highrpd_val_files, min(len(os.listdir(zurich_val_img)), len(highrpd_val_files)))
    for bn in tqdm(files_to_copy_val, desc="Copying HighRPD Valid"):
        shutil.copy(os.path.join(highrpd_val_img, f"{bn}.tif"), combo_val_img)
        shutil.copy(os.path.join(highrpd_val_lbl, f"{bn}.txt"), combo_val_lbl)
    print("Combined dataset preparation complete.\n")

def train_experiment_A(base_path, models_path, results_path):
    print("\n--- EXPERIMENT A: YOLOv8x on Zurich Dataset ---")
    yaml_path = os.path.join(os.path.dirname(__file__), "zurich_dataset.yaml")
    create_yaml_file(yaml_path, os.path.join(base_path,'02_Processed/zurich_yolo_dataset/images/train'), os.path.join(base_path,'02_Processed/zurich_yolo_dataset/images/valid'), os.path.join(base_path,'02_Processed/zurich_yolo_dataset/images/test'),['Pavement Distress','Non-Pavement Distress'],2)
    model=YOLO(os.path.join(models_path,"yolov8x.pt")); model.train(data=yaml_path,epochs=100,patience=10,batch=4,imgsz=512,optimizer="Adam",lr0=2.5e-4,cos_lr=True,augment=True,name="A_YOLOv8x_Zurich",project=results_path,exist_ok=True)

def train_experiment_B(base_path, models_path, results_path):
    print("\n--- EXPERIMENT B: YOLOv11x on Zurich Dataset ---")
    yaml_path = os.path.join(os.path.dirname(__file__), "zurich_dataset.yaml")
    model=YOLO(os.path.join(models_path,"yolov11x.pt")); model.train(data=yaml_path,epochs=100,patience=10,batch=4,imgsz=512,optimizer="Adam",lr0=2.5e-4,cos_lr=True,augment=True,name="B_YOLOv11x_Zurich",project=results_path,exist_ok=True)

def train_experiment_C(base_path, models_path, results_path):
    print("\n--- EXPERIMENT C: Multi-stage Transfer Learning ---")
    highrpd_yaml_path = os.path.join(os.path.dirname(__file__), "highrpd_dataset.yaml")
    create_yaml_file(highrpd_yaml_path, os.path.join(base_path,'01_Raw/HighRPD/images/train'), os.path.join(base_path,'01_Raw/HighRPD/images/valid'), os.path.join(base_path,'01_Raw/HighRPD/images/test'),['line crack','block crack','pothole'],3)
    model=YOLO(os.path.join(models_path,"yolov11x.pt")); model.train(data=highrpd_yaml_path,epochs=50,lr0=1e-3,name="C1_Pretrain",project=results_path,exist_ok=True)
    
    zurich_yaml_path = os.path.join(os.path.dirname(__file__), "zurich_dataset.yaml")
    stage1_weights = os.path.join(results_path, "C1_Pretrain/weights/last.pt")
    model=YOLO(stage1_weights); model.train(data=zurich_yaml_path,epochs=30,lr0=1e-4,name="C2_Intermediate",project=results_path,exist_ok=True)
    
    stage2_weights = os.path.join(results_path, "C2_Intermediate/weights/last.pt")
    model=YOLO(stage2_weights); model.train(data=zurich_yaml_path,epochs=100,lr0=5e-5,name="C3_Full_Finetune",project=results_path,exist_ok=True)

def train_experiment_D(base_path, models_path, results_path):
    print("\n--- EXPERIMENT D: Combined Dataset (Offline Augmentation) ---")
    combined_path = os.path.join(base_path, '03_Combined_Dataset')
    prepare_augmented_combined_dataset(os.path.join(base_path,'02_Processed/zurich_yolo_dataset'), os.path.join(base_path,'01_Raw/HighRPD'), combined_path)
    
    combined_yaml_path = os.path.join(os.path.dirname(__file__), "combined_dataset.yaml")
    create_yaml_file(combined_yaml_path, os.path.join(combined_path,'images/train'), os.path.join(combined_path,'images/valid'), os.path.join(base_path,'02_Processed/zurich_yolo_dataset/images/test'),['Pavement Distress','Non-Pavement Distress','line crack','block crack','pothole'],5)
    
    model=YOLO(os.path.join(models_path,"yolov11x.pt")); model.train(data=combined_yaml_path,epochs=300,patience=30,batch=16,imgsz=512,augment=False,name="D_Combined_OfflineAug",project=results_path,exist_ok=True)

def main():
    """Main execution block."""
    parser = argparse.ArgumentParser(description="Run YOLO training experiments.")
    parser.add_argument("--experiment", type=str, required=True, choices=['A', 'B', 'C', 'D'], help="Experiment to run (A, B, C, or D).")
    args = parser.parse_args()

    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../01_Data'))
    models_path = os.path.join(base_path, '01_Raw', 'Models')
    results_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../03_Results/training_runs'))

    exp_func = {'A': train_experiment_A, 'B': train_experiment_B, 'C': train_experiment_C, 'D': train_experiment_D}
    exp_func[args.experiment](base_path, models_path, results_path)

if __name__ == "__main__":
    main()