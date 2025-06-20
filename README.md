# WSI Tool - Whole Slide Image Processing Tool

This is a Python toolkit for processing Whole Slide Images (WSI), with main features including tissue segmentation, patch extraction, coordinate saving, and visualization.

## Project Structure

```
wsi_tool/
├── wsi_core/                    # Core functional modules
│   ├── __init__.py
│   ├── whole_slide_image.py     # Main WSI processing class
│   ├── wsi_utils.py            # Utility functions
│   ├── util_classes.py         # Helper class definitions
│   └── batch_process_utils.py  # Batch processing tools
├── utils/                       # General utilities
│   ├── __init__.py
│   └── file_utils.py           # File operation utilities
├── create_patches_fp.py        # Main execution script
├── openslide_tutorial/         # OpenSlide tutorial
│   └── openslide_tutorial.ipynb
├── tcga_download/              # TCGA data download tools
│   ├── 01_check.py
│   ├── 02_find_missing.py
│   └── 03_delete_folder.py
├── tcga_oncotree/              # TCGA tumor classification tools
│   └── tcga_oncotree_map.ipynb
└── README.md                   # Project documentation
```

## Core Module Descriptions

### wsi_core/whole_slide_image.py
**Main Function:** Core WSI image processing class
- **WholeSlideImage Class:** Main WSI processing class
- **Tissue Segmentation:** `segmentTissue()` - Automatic detection and segmentation of tissue regions
- **Patch Extraction:** `createPatches_bag_hdf5()` - Extract patch coordinates and save as HDF5 format
- **Visualization:** `visWSI()`, `visHeatmap()` - Generate tissue segmentation and heatmap visualizations
- **Coordinate Processing:** Support saving only patch coordinates without image data

### wsi_core/wsi_utils.py
**Main Function:** Collection of utility functions
- **Patch Quality Detection:** `isBlackPatch()`, `isWhitePatch()` - Detect if patches are blank or too white
- **Coordinate Processing:** `initialize_hdf5_coords_only()`, `savePatchIter_coords_only()` - HDF5 coordinate saving
- **Visualization Tools:** `StitchCoords()`, `DrawMapFromCoords()` - Coordinate-based visualization
- **Sampling Tools:** `sample_rois()`, `top_k()` - Region sampling and sorting

### wsi_core/util_classes.py
**Main Function:** Helper class definitions
- **Contour Checking Classes:** Various contour detection algorithms (`isInContourV1`, `isInContourV2`, etc.)
- **Canvas Class:** `Mosaic_Canvas` - For creating patch mosaic images

### wsi_core/batch_process_utils.py
**Main Function:** Batch processing tools
- **Parameter Management:** Parameter configuration and management for batch processing
- **Progress Tracking:** Processing progress recording and status management

### utils/file_utils.py
**Main Function:** File operation utilities
- **HDF5 Operations:** `save_hdf5()` - General HDF5 file saving
- **Pickle Operations:** `save_pkl()`, `load_pkl()` - Object serialization

## Main Scripts

### create_patches_fp.py
**Main Function:** Main execution script
- **seg_and_patch():** Complete processing pipeline (segmentation + extraction)
- **segment():** Tissue segmentation only
- **patching():** Patch extraction only
- **stitching():** Visualization stitching
- **Batch Processing Support:** Can process entire directories of WSI files
- **Logging:** Detailed operation logs and parameter recording

## Auxiliary Tools

### tcga_download/
**Main Function:** TCGA data download and management
- **01_check.py:** Check download status
- **02_find_missing.py:** Find missing files
- **03_delete_folder.py:** Clean up download directories

### tcga_oncotree/
**Main Function:** TCGA tumor classification
- **tcga_oncotree_map.ipynb:** Tumor type mapping and analysis

### openslide_tutorial/
**Main Function:** OpenSlide learning tutorial
- **openslide_tutorial.ipynb:** OpenSlide usage tutorial

## Key Features

### 1. Efficient Tissue Segmentation
- Automatic tissue region detection
- Support for multiple segmentation parameters
- Filtering of holes and noise

### 2. Flexible Patch Extraction
- Support for multiple extraction strategies
- Option to save only coordinates (saves storage space)
- Quality filtering support (remove blank patches)

### 3. Powerful Visualization
- Tissue segmentation visualization
- Coordinate-based heatmaps
- Support for multiple visualization parameters

### 4. Batch Processing Capability
- Support for batch processing of entire directories
- Detailed progress tracking
- Complete logging

### 5. Storage Optimization
- Option to save only coordinate information
- HDF5 format storage
- Support for incremental saving

## Usage Examples

### Basic Usage
```python
from wsi_core.whole_slide_image import WholeSlideImage

# Load WSI file
wsi = WholeSlideImage("path/to/slide.svs")

# Tissue segmentation
wsi.segmentTissue(seg_level=0, sthresh=8, mthresh=7)

# Extract patch coordinates
result_file = wsi.createPatches_bag_hdf5(
    save_path="output/",
    patch_level=0,
    patch_size=256,
    step_size=256
)
```

### Batch Processing
```python
from create_patches_fp import seg_and_patch

# Batch process entire directory
seg_and_patch(
    source="wsi_files/",
    save_dir="output/",
    patch_save_dir="patches/",
    mask_save_dir="masks/",
    stitch_save_dir="stitches/",
    patch_size=256,
    step_size=256,
    seg=True,
    patch=True,
    stitch=True
)
```

## Dependencies

- **openslide-python:** WSI file reading
- **opencv-python:** Image processing
- **numpy:** Numerical computing
- **h5py:** HDF5 file operations
- **PIL/Pillow:** Image processing
- **matplotlib:** Visualization
- **scipy:** Scientific computing

## Important Notes

1. **Memory Usage:** Pay attention to memory usage when processing large WSI files
2. **Storage Space:** Saving only coordinates can significantly reduce storage space
3. **Parameter Tuning:** Adjust segmentation and extraction parameters based on specific data
4. **Log Management:** Pay attention to log file size during batch processing

## Download Data from TCGA

Downloading TCGA dataset from GDC. First you need install GDC client tool.

