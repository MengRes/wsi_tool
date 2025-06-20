import os
import numpy as np
import time
import argparse
import pdb
import pandas as pd
import logging

# Internal imports
from wsi_core.whole_slide_image import WholeSlideImage
from wsi_core.wsi_utils import StitchCoords
from wsi_core.batch_process_utils import initialize_df


def stitching(file_path, wsi_object, downscale = 64):
    # Start stitching timer
    start = time.time()
    heatmap = StitchCoords(file_path, wsi_object, downscale=downscale, bg_color=(0,0,0), alpha=-1, draw_grid=False)
    total_time = time.time() - start
    
    return heatmap, total_time

def segment(WSI_object, seg_params, filter_params):
    # Start segment timer
    start_time = time.time()
    # Segment
    WSI_object.segmentTissue(**seg_params, filter_params=filter_params)
    # Stop Seg Timers
    seg_time_elapsed = time.time() - start_time   
    return WSI_object, seg_time_elapsed

def patching(WSI_object, **kwargs):
    # Start patch timer
    start_time = time.time()
    # Patch
    file_path = WSI_object.process_contours(**kwargs)

    # Stop Patch Timer
    patch_time_elapsed = time.time() - start_time
    return file_path, patch_time_elapsed


def seg_and_patch(source, save_dir, patch_save_dir, mask_save_dir, stitch_save_dir, 
                  patch_size = 256, step_size = 256, 
                  seg_params = {"seg_level": -1, "sthresh": 8, "mthresh": 7, "close": 4, "use_otsu": False,
                  "keep_ids": "none", "exclude_ids": "none"},
                  filter_params = {"a_t":100, "a_h": 16, "max_n_holes":8}, 
                  vis_params = {"vis_level": -1, "line_thickness": 500},
                  patch_params = {"use_padding": True, "contour_fn": "four_pt"},
                  patch_level = 0,
                  use_default_params = False, 
                  seg = False, save_mask = True, 
                  stitch= False, 
                  patch = False, auto_skip=True, process_list = None):   

    # Setup logging
    log_file = os.path.join(save_dir, "patch_extraction.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ]
    )
    
    # Record start time and parameters
    start_time = time.time()
    logging.info("=" * 80)
    logging.info("Starting batch WSI processing")
    logging.info("=" * 80)
    
    # Record all parameters
    logging.info("Parameter Configuration:")
    logging.info(f"  Source Directory: {source}")
    logging.info(f"  Save Directory: {save_dir}")
    logging.info(f"  Patch Save Directory: {patch_save_dir}")
    logging.info(f"  Mask Save Directory: {mask_save_dir}")
    logging.info(f"  Stitch Save Directory: {stitch_save_dir}")
    logging.info(f"  Patch Size: {patch_size}")
    logging.info(f"  Step Size: {step_size}")
    logging.info(f"  Patch Level: {patch_level}")
    logging.info(f"  Segmentation: {seg}")
    logging.info(f"  Save Mask: {save_mask}")
    logging.info(f"  Stitch: {stitch}")
    logging.info(f"  Extract Patches: {patch}")
    logging.info(f"  Auto Skip: {auto_skip}")
    logging.info(f"  Use Default Params: {use_default_params}")
    
    logging.info("Segmentation Parameters:")
    for key, value in seg_params.items():
        logging.info(f"  {key}: {value}")
    
    logging.info("Filter Parameters:")
    for key, value in filter_params.items():
        logging.info(f"  {key}: {value}")
    
    logging.info("Visualization Parameters:")
    for key, value in vis_params.items():
        logging.info(f"  {key}: {value}")
    
    logging.info("Patch Parameters:")
    for key, value in patch_params.items():
        logging.info(f"  {key}: {value}")

    slides = sorted(os.listdir(source),reverse=True)
    slides = [slide for slide in slides if os.path.isfile(os.path.join(source, slide))]
    logging.info(f"Found {len(slides)} WSI files")
    
    if process_list is None:
        df = initialize_df(slides, seg_params, filter_params, vis_params, patch_params)
        logging.info("Using auto-generated parameter list")
    else:
        df = pd.read_csv(process_list)
        df = initialize_df(df, seg_params, filter_params, vis_params, patch_params)
        logging.info(f"Using specified parameter list: {process_list}")

    mask = df["process"] == 1
    process_stack = df[mask]
    total = len(process_stack)
    logging.info(f"Number of files to process: {total}")

    seg_times = 0.
    patch_times = 0.
    stitch_times = 0.

    for i in range(total):
        df.to_csv(os.path.join(save_dir, "process_list_autogen.csv"), index=False)
        idx = process_stack.index[i]
        slide = process_stack.loc[idx, "slide_id"]
        logging.info(f"\nProcessing Progress: {i+1}/{total} ({((i+1)/total)*100:.1f}%)")
        logging.info(f"Processing: {slide}")
        
        df.loc[idx, "process"] = 0
        slide_id, _ = os.path.splitext(slide)

        if auto_skip and os.path.isfile(os.path.join(patch_save_dir, slide_id + ".h5")):
            logging.info(f"{slide_id} already exists in target location, skipping")
            df.loc[idx, "status"] = "already_exist"
            continue

        # Inialize WSI
        full_path = os.path.join(source, slide)
        try:
            WSI_object = WholeSlideImage(full_path)
            logging.info(f"Successfully loaded WSI file: {full_path}")
        except Exception as e:
            logging.error(f"Failed to load WSI file: {full_path}, error: {e}")
            continue
    
        if use_default_params:
            current_vis_params = vis_params.copy()
            current_filter_params = filter_params.copy()
            current_seg_params = seg_params.copy()
            current_patch_params = patch_params.copy()
            logging.info("Using default parameters")
            
        else:
            current_vis_params = {}
            current_filter_params = {}
            current_seg_params = {}
            current_patch_params = {}

            for key in vis_params.keys():
                current_vis_params.update({key: df.loc[idx, key]})

            for key in filter_params.keys():
                    current_filter_params.update({key: df.loc[idx, key]})

            for key in seg_params.keys():
                current_seg_params.update({key: df.loc[idx, key]})

            for key in patch_params.keys():
                current_patch_params.update({key: df.loc[idx, key]})
            
            logging.info("Using file-specific parameters")

        if current_vis_params["vis_level"] < 0:
            if len(WSI_object.level_dim) == 1:
                current_vis_params["vis_level"] = 0
            else:
                wsi = WSI_object.getOpenSlide()
                best_level = wsi.get_best_level_for_downsample(64)
                current_vis_params["vis_level"] = best_level
            logging.info(f"Setting visualization level: {current_vis_params['vis_level']}")

        if current_seg_params["seg_level"] < 0:
            if len(WSI_object.level_dim) == 1:
                current_seg_params["seg_level"] = 0
            else:
                wsi = WSI_object.getOpenSlide()
                best_level = wsi.get_best_level_for_downsample(64)
                current_seg_params["seg_level"] = best_level
            logging.info(f"Setting segmentation level: {current_seg_params['seg_level']}")

        keep_ids = str(current_seg_params["keep_ids"])
        if keep_ids != "none" and len(keep_ids) > 0:
            str_ids = current_seg_params["keep_ids"]
            current_seg_params["keep_ids"] = np.array(str_ids.split(",")).astype(int)
            logging.info(f"Keeping contour IDs: {current_seg_params['keep_ids']}")
        else:
            current_seg_params["keep_ids"] = []

        exclude_ids = str(current_seg_params["exclude_ids"])
        if exclude_ids != "none" and len(exclude_ids) > 0:
            str_ids = current_seg_params["exclude_ids"]
            current_seg_params["exclude_ids"] = np.array(str_ids.split(",")).astype(int)
            logging.info(f"Excluding contour IDs: {current_seg_params['exclude_ids']}")
        else:
            current_seg_params["exclude_ids"] = []

        # Obtain the width, height of WSI on seg_level image.
        w, h = WSI_object.level_dim[current_seg_params["seg_level"]] 
        if w * h > 1e8:
            logging.error(f"Level size {w} x {h} may be too large to successfully segment, skipping")
            df.loc[idx, "status"] = "failed_seg"
            continue

        df.loc[idx, "vis_level"] = current_vis_params["vis_level"]
        df.loc[idx, "seg_level"] = current_seg_params["seg_level"]

        seg_time_elapsed = -1
        if seg:
            logging.info("Starting segmentation organization...")
            WSI_object, seg_time_elapsed = segment(WSI_object, current_seg_params, current_filter_params)
            logging.info(f"Segmentation completed, time: {seg_time_elapsed:.2f} seconds")

        if save_mask:
            logging.info("Saving segmentation mask...")
            mask = WSI_object.visWSI(**current_vis_params)
            mask_path = os.path.join(mask_save_dir, slide_id+".jpg")
            mask.save(mask_path)
            logging.info(f"Mask saved: {mask_path}")

        patch_time_elapsed = -1
        if patch:
            logging.info("Starting patch extraction...")
            current_patch_params.update({"patch_level": patch_level, "patch_size": patch_size, "step_size": step_size, 
                                         "save_path": patch_save_dir})
            file_path, patch_time_elapsed = patching(WSI_object = WSI_object,  **current_patch_params,)
            logging.info(f"Patch extraction completed, time: {patch_time_elapsed:.2f} seconds")
        
        stitch_time_elapsed = -1
        if stitch:
            logging.info("Starting stitch...")
            file_path = os.path.join(patch_save_dir, slide_id+".h5")
            if os.path.isfile(file_path):
                heatmap, stitch_time_elapsed = stitching(file_path, WSI_object, downscale=64)
                stitch_path = os.path.join(stitch_save_dir, slide_id+".jpg")
                heatmap.save(stitch_path)
                logging.info(f"Stitch completed, time: {stitch_time_elapsed:.2f} seconds")

        logging.info(f"Processing completed - Segmentation: {seg_time_elapsed:.2f}s, Patch: {patch_time_elapsed:.2f}s, Stitch: {stitch_time_elapsed:.2f}s")
        
        df.loc[idx, "status"] = "processed"

        seg_times += seg_time_elapsed
        patch_times += patch_time_elapsed
        stitch_times += stitch_time_elapsed

    average_seg_time = seg_times / total
    average_patch_time = patch_times / total
    average_stitch_time = stitch_times / total

    df.to_csv(os.path.join(save_dir, "process_list_autogen.csv"), index=False)
    
    total_time = time.time() - start_time
    logging.info("=" * 80)
    logging.info("Batch processing completed")
    logging.info("=" * 80)
    logging.info(f"Total processing time: {total_time:.2f} seconds")
    logging.info(f"Average segmentation time: {average_seg_time:.2f} seconds/file")
    logging.info(f"Average patch extraction time: {average_patch_time:.2f} seconds/file")
    logging.info(f"Average stitch time: {average_stitch_time:.2f} seconds/file")
    logging.info(f"Processed file count: {total}")
    logging.info(f"Log file location: {log_file}")
    logging.info("=" * 80)
        
    return average_seg_time, average_patch_time, average_stitch_time


parser = argparse.ArgumentParser(description="WSI Segmentation, Patch, and Stitching.")
parser.add_argument("--source", type = str,
                    help = "path to folder containing raw wsi image files")
parser.add_argument("--step_size", type = int, default = 256,
                    help = "step_size")
parser.add_argument("--patch_size", type = int, default = 256,
                    help = "patch_size")
parser.add_argument("--patch", default = False, action = "store_true")
parser.add_argument("--seg", default = False, action = "store_true")
parser.add_argument("--stitch", default = False, action = "store_true")
parser.add_argument("--no_auto_skip", default = True, action = "store_false")
parser.add_argument("--save_dir", type = str,
                    help="directory to save processed data")
parser.add_argument("--preset", default = None, type = str,
                    help="predefined profile of default segmentation and filter parameters (.csv)")
parser.add_argument("--patch_level", type = int, default = 0, 
                    help = "downsample level at which to patch")
parser.add_argument("--process_list",  type = str, default = None,
                    help = "name of list of images to process with parameters (.csv)")
parser.add_argument("--use_ostu", default = False, action="store_true")


if __name__ == "__main__":
    args = parser.parse_args()

    patch_save_dir = os.path.join(args.save_dir, "patches")
    mask_save_dir = os.path.join(args.save_dir, "masks")
    stitch_save_dir = os.path.join(args.save_dir, "stitches")

    if args.process_list:
        process_list = os.path.join(args.save_dir, args.process_list)
    else:
        process_list = None
    
    print("source: ", args.source)
    print("patch_save_dir: ", patch_save_dir)
    print("mask_save_dir: ", mask_save_dir)
    print("stitch_save_dir: ", stitch_save_dir)

    directories = {"source": args.source, 
                   "save_dir": args.save_dir,
                   "patch_save_dir": patch_save_dir, 
                   "mask_save_dir" : mask_save_dir, 
                   "stitch_save_dir": stitch_save_dir} 

    for key, val in directories.items():
        print("{} : {}".format(key, val))
        if key not in ["source"]:
            os.makedirs(val, exist_ok=True)

    seg_params = {"seg_level": -1, "sthresh": 8, "mthresh": 7, "close": 4, "use_otsu": args.use_ostu,
                  "keep_ids": "none", "exclude_ids": "none"}
    filter_params = {"a_t":100, "a_h": 16, "max_n_holes":8}
    vis_params = {"vis_level": -1, "line_thickness": 250}
    patch_params = {"use_padding": True, "contour_fn": "four_pt"}

    if args.preset:
        preset_df = pd.read_csv(args.preset)
        for key in seg_params.keys():
            seg_params[key] = preset_df.loc[0, key]

        for key in filter_params.keys():
            filter_params[key] = preset_df.loc[0, key]

        for key in vis_params.keys():
            vis_params[key] = preset_df.loc[0, key]

        for key in patch_params.keys():
            patch_params[key] = preset_df.loc[0, key]

    parameters = {"seg_params": seg_params,
                  "filter_params": filter_params,
                  "patch_params": patch_params,
                  "vis_params": vis_params}

    print(parameters)

    average_seg_time, average_patch_time, average_stitch_time = seg_and_patch(**directories, **parameters,
                                            patch_size = args.patch_size, step_size=args.step_size, 
                                            seg = args.seg,  use_default_params=False, save_mask = True, 
                                            stitch= args.stitch,
                                            patch_level=args.patch_level, patch = args.patch,
                                            process_list = process_list, auto_skip=args.no_auto_skip)