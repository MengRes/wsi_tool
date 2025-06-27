import torch
import torch.nn as nn
import os
import time
import argparse
import h5py
import openslide
from pathlib import Path
import numpy as np
import logging
from datetime import datetime

# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FeatureExtractor:
    """Feature extractor supporting only ResNet50 with configurable patch size"""
    def __init__(self, pretrained: bool = True, patch_size: int = 512):
        self.model = self._load_model(pretrained).to(device)
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        self.model.eval()
        self.patch_size = patch_size
        self.transform = self._get_transform()

    def _load_model(self, pretrained: bool) -> nn.Module:
        import torchvision.models as models
        model = models.resnet50(pretrained=pretrained)
        model = nn.Sequential(*list(model.children())[:-1])
        return model

    def _get_transform(self):
        import torchvision.transforms as transforms
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_features_from_h5(self, h5_file_path: str, wsi_path: str, batch_size: int = 32, verbose: bool = True):
        try:
            wsi = openslide.open_slide(wsi_path)
        except Exception as e:
            print(f"Error opening WSI file {wsi_path}: {e}")
            return None
        try:
            with h5py.File(h5_file_path, "r") as f:
                coords = f["coords"][:]
        except Exception as e:
            print(f"Error reading H5 file {h5_file_path}: {e}")
            return None
        if verbose:
            print(f"Extracting features from {len(coords)} patches...")
        features_list = []
        coords_list = []
        for i in range(0, len(coords), batch_size):
            batch_coords = coords[i:i + batch_size]
            batch_patches = []
            for x, y in batch_coords:
                try:
                    patch = wsi.read_region((x, y), 0, (self.patch_size, self.patch_size)).convert("RGB")
                    patch_tensor = self.transform(patch).unsqueeze(0)
                    batch_patches.append(patch_tensor)
                except Exception as e:
                    print(f"Error extracting patch at ({x}, {y}): {e}")
                    continue
            if not batch_patches:
                continue
            batch_tensor = torch.cat(batch_patches, dim=0).to(device)
            with torch.no_grad():
                batch_features = self.model(batch_tensor)
                batch_features = batch_features.view(batch_features.size(0), -1)
                features_list.append(batch_features.cpu())
                coords_list.append(torch.tensor(batch_coords))
            if verbose and (i // batch_size) % 10 == 0:
                print(f"Processed {i + len(batch_coords)}/{len(coords)} patches")
        if not features_list:
            print("No features extracted!")
            return None
        all_features = torch.cat(features_list, dim=0)
        all_coords = torch.cat(coords_list, dim=0)
        return {"coords": all_coords, "features": all_features}

def setup_logging(save_dir: str):
    """Setup logging configuration"""
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, f"feature_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Feature Extraction from WSI Patches")
    parser.add_argument("--data_h5_dir", type=str, required=True, help="Directory containing H5 patch files")
    parser.add_argument("--data_slide_dir", type=str, required=True, help="Directory containing WSI files")
    parser.add_argument("--slide_ext", type=str, default=".svs", help="Extension of WSI files")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save extracted features and logs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for feature extraction")
    parser.add_argument("--patch_size", type=int, default=512, help="Patch size to extract from WSI (default: 512)")
    parser.add_argument("--no_auto_skip", default=False, action="store_true", help="Do not skip already processed files")
    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.save_dir)
    
    # Create output directories
    pt_files_dir = os.path.join(args.save_dir, "pt_files")
    os.makedirs(pt_files_dir, exist_ok=True)
    
    # Log parameter information
    logger.info("=" * 60)
    logger.info("Feature Extraction Started")
    logger.info("=" * 60)
    logger.info(f"Model: ResNet50 (pretrained)")
    logger.info(f"Patch size: {args.patch_size}x{args.patch_size}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Device: {device}")
    logger.info(f"Data H5 directory: {args.data_h5_dir}")
    logger.info(f"Data slide directory: {args.data_slide_dir}")
    logger.info(f"Save directory: {args.save_dir}")
    logger.info(f"Slide extension: {args.slide_ext}")
    logger.info(f"Auto skip existing files: {not args.no_auto_skip}")
    logger.info("=" * 60)
    
    h5_dir = os.path.join(args.data_h5_dir, "patches")
    h5_files = [f for f in os.listdir(h5_dir) if f.endswith(".h5")]
    dest_files = os.listdir(pt_files_dir)
    extractor = FeatureExtractor(pretrained=True, patch_size=args.patch_size)
    total = len(h5_files)
    
    logger.info(f"Found {total} H5 files to process")
    
    success_count = 0
    error_count = 0
    
    for i, h5_file in enumerate(h5_files):
        # Use WSI filename as feature filename
        wsi_filename = h5_file.replace(".h5", "")
        pt_filename = f"{wsi_filename}.pt"
        
        logger.info(f"\nProgress: {i+1}/{total}")
        logger.info(f"Processing: {wsi_filename}")
        
        if not args.no_auto_skip and pt_filename in dest_files:
            logger.info(f"Skipped {wsi_filename} (already exists)")
            continue
            
        h5_file_path = os.path.join(h5_dir, h5_file)
        slide_file_path = os.path.join(args.data_slide_dir, f"{wsi_filename}{args.slide_ext}")
        
        if not os.path.exists(h5_file_path):
            logger.error(f"H5 file not found: {h5_file_path}")
            error_count += 1
            continue
            
        if not os.path.exists(slide_file_path):
            logger.error(f"WSI file not found: {slide_file_path}")
            error_count += 1
            continue
            
        time_start = time.time()
        
        try:
            result = extractor.extract_features_from_h5(h5_file_path, slide_file_path, batch_size=args.batch_size, verbose=True)
            
            if result is not None:
                pt_path = os.path.join(pt_files_dir, pt_filename)
                torch.save(result, pt_path)
                
                features_shape = result["features"].shape
                coords_shape = result["coords"].shape
                
                logger.info(f"Features shape: {features_shape}")
                logger.info(f"Coordinates shape: {coords_shape}")
                logger.info(f"Saved to: {pt_path}")
                
                success_count += 1
            else:
                logger.error(f"Failed to extract features for {wsi_filename}")
                error_count += 1
                continue
                
            time_elapsed = time.time() - time_start
            logger.info(f"Feature extraction for {wsi_filename} took {time_elapsed:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error processing {wsi_filename}: {e}")
            error_count += 1
            continue
    
    # Log summary information
    logger.info("=" * 60)
    logger.info("Feature Extraction Summary")
    logger.info("=" * 60)
    logger.info(f"Total files processed: {total}")
    logger.info(f"Successfully processed: {success_count}")
    logger.info(f"Errors: {error_count}")
    logger.info(f"Success rate: {success_count/total*100:.1f}%")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()



