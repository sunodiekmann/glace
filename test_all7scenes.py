#!/usr/bin/env python3
import argparse
import logging
import math
import time
from pathlib import Path

import cv2
import json
import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

import dsacstar
from ace_network import Regressor
from dataset import CamLocDataset

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

SCENES = ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"]
DISTORTIONS = ["raw", "blur", "light", "occlusion", "harsh"]


def run_eval_for_split(
    scene_name: str,
    distortion: str,
    base_dir: Path,
    encoder_state_dict,
    head_state_dict,
    image_height: int,
    global_log_file,
    json_results: list,
):
    """
    Evaluate one (scene, distortion) pair and append per-frame results to json_results.
    """
    # Device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    _logger.info(f"Running {scene_name}_{distortion} on device: {device}")

    # Paths
    test_img_dir = base_dir / "test_datasets" / f"{scene_name}_{distortion}"
    test_pose_dir = base_dir / "test_datasets" / f"{scene_name}_pose"
    feat_path = base_dir / "global_features" / scene_name / "features.npy"

    if not test_img_dir.is_dir():
        _logger.warning(f"Skipping {scene_name}_{distortion}: images not found at {test_img_dir}")
        return

    if not test_pose_dir.is_dir():
        _logger.warning(f"Skipping {scene_name}_{distortion}: poses not found at {test_pose_dir}")
        return

    if not feat_path.is_file():
        _logger.warning(f"Skipping {scene_name}_{distortion}: features not found at {feat_path}")
        return

    # Dataset / loader
    testset = CamLocDataset(
        root_dir=test_img_dir,  # dummy; overridden below
        mode=0,
        image_height=image_height,
        override_rgb_dir=test_img_dir,
        override_pose_dir=test_pose_dir,
        override_feat_path=feat_path,
    )
    _logger.info(f"Test images found for {scene_name}_{distortion}: {len(testset)}")

    test_loader = DataLoader(testset, shuffle=False, num_workers=1)

    # Build network from state dicts
    network = Regressor.create_from_split_state_dict(encoder_state_dict, head_state_dict)
    network = network.to(device)
    network.eval()

    # RANSAC params (as before)
    hypotheses = 64
    threshold = 10.0
    inlieralpha = 100.0
    maxpixelerror = 100.0

    total_frames = 0
    start_time = time.time()

    with torch.no_grad():
        for image_B1HW, _, gt_pose_B44, _, intrinsics_B33, _, _, filenames, global_feat, idx in test_loader:
            image_B1HW = image_B1HW.to(device, non_blocking=True)
            global_feat = global_feat.to(device, non_blocking=True)

            # Predict scene coordinates for the batch
            with autocast(enabled=(device.type == "cuda")):
                scene_coordinates_B3HW = network(image_B1HW, global_feat)

            scene_coordinates_B3HW = scene_coordinates_B3HW.float().cpu()

            for scene_coordinates_3HW, gt_pose_44, intrinsics_33, frame_path in zip(
                scene_coordinates_B3HW, gt_pose_B44, intrinsics_B33, filenames
            ):
                total_frames += 1

                # ----------------- Timing start (per-frame latency) -----------------
                frame_t_start = time.time()

                focal_length = intrinsics_33[0, 0].item()
                ppX = intrinsics_33[0, 2].item()
                ppY = intrinsics_33[1, 2].item()
                assert torch.allclose(intrinsics_33[0, 0], intrinsics_33[1, 1])

                frame_name = Path(frame_path).name

                # Infer set_num from parent folder name digits (fallback 1)
                parent_name = Path(frame_path).parent.name
                digits = "".join(ch for ch in parent_name if ch.isdigit())
                set_num = int(digits) if digits else 1

                out_pose = torch.zeros((4, 4))

                inlier_count = dsacstar.forward_rgb(
                    scene_coordinates_3HW.unsqueeze(0),
                    out_pose,
                    hypotheses,
                    threshold,
                    focal_length,
                    ppX,
                    ppY,
                    inlieralpha,
                    maxpixelerror,
                    network.OUTPUT_SUBSAMPLE,
                )

                # ----------------- Timing end -----------------
                latency_ms = (time.time() - frame_t_start) * 1000.0

                # ----------------- JSON per-frame record -----------------
                frame_info = {
                    "dataset": scene_name,              # chess, fire, ...
                    "set_num": set_num,                # inferred as above (or 1)
                    "distortion_type": distortion,     # raw, blur, light, occlusion, harsh
                    "filename": frame_name,            # image filename
                    "pose": out_pose.numpy().tolist(), # 4x4 SE3 matrix as list
                    "latency_ms": float(latency_ms),   # ms as float
                }
                json_results.append(frame_info)
                # ---------------------------------------------------------

    if total_frames == 0:
        _logger.warning(f"No frames evaluated for {scene_name}_{distortion}")
        return

    elapsed = time.time() - start_time
    _logger.info(
        f"Finished {scene_name}_{distortion}: {total_frames} frames, "
        f"elapsed {elapsed:.1f}s (~{elapsed*1000/total_frames:.1f} ms/frame)"
    )

    # You *could* write one summary line to global_log_file here if you like.
    # For now, we'll just log total frames + avg time per frame:
    global_log_file.write(
        f"{scene_name},{distortion},{total_frames},avg_time_ms_per_frame={elapsed*1000/total_frames:.2f}\n"
    )
    global_log_file.flush()


def main():
    parser = argparse.ArgumentParser(
        description="Run ACE/GLACE on all 7-Scenes + distortions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--base_dir",
        type=Path,
        default=Path("/home/suno/7-scenes_modified"),
        help="Base directory containing test_datasets/, global_features/, etc.",
    )
    parser.add_argument(
        "--pretrained_dir",
        type=Path,
        default=Path("pretrained"),
        help="Directory containing 7scenes_<scene>.pt heads.",
    )
    parser.add_argument(
        "--encoder_path",
        type=Path,
        default=Path("ace_encoder_pretrained.pt"),
        help="Path to encoder weights.",
    )
    parser.add_argument(
        "--image_resolution",
        type=int,
        default=480,
        help="Image height for resizing.",
    )
    parser.add_argument(
        "--global_log",
        type=Path,
        default=Path("test_7scenes_all.log"),
        help="Text log file to append per-(scene,distortion) summaries.",
    )
    parser.add_argument(
        "--algorithm_name",
        type=str,
        default="GLACE",
        help="Name of the algorithm (used in JSON filename).",
    )
    parser.add_argument(
        "--json_output",
        type=Path,
        default=None,
        help="Optional explicit path for JSON output; if not set, uses <algorithm_name}_exp_results.json",
    )

    opt = parser.parse_args()

    base_dir = opt.base_dir
    pretrained_dir = opt.pretrained_dir
    encoder_path = opt.encoder_path
    image_height = opt.image_resolution
    global_log_path = opt.global_log

    algorithm_name = opt.algorithm_name
    if opt.json_output is not None:
        json_output_path = opt.json_output
    else:
        json_output_path = Path(f"{algorithm_name}_exp_results.json")

    _logger.info(f"Base dir: {base_dir}")
    _logger.info(f"Pretrained heads dir: {pretrained_dir}")
    _logger.info(f"Encoder path: {encoder_path}")
    _logger.info(f"Global log: {global_log_path}")
    _logger.info(f"JSON output: {json_output_path}")

    # Load encoder once
    encoder_state_dict = torch.load(encoder_path, map_location="cpu")
    _logger.info(f"Loaded encoder from: {encoder_path}")

    # JSON list for all frames across all scenes & distortions
    localization_results = []

    with open(global_log_path, "w", buffering=1) as global_log:
        global_log.write("scene,distortion,frames,avg_time_ms_per_frame\n")

        for scene in SCENES:
            head_path = pretrained_dir / f"7scenes_{scene}.pt"
            if not head_path.is_file():
                _logger.warning(f"Skipping scene {scene}: head not found at {head_path}")
                continue

            head_state_dict = torch.load(head_path, map_location="cpu")
            _logger.info(f"Loaded head weights for {scene} from: {head_path}")

            for distortion in DISTORTIONS:
                run_eval_for_split(
                    scene_name=scene,
                    distortion=distortion,
                    base_dir=base_dir,
                    encoder_state_dict=encoder_state_dict,
                    head_state_dict=head_state_dict,
                    image_height=image_height,
                    global_log_file=global_log,
                    json_results=localization_results,
                )

    # Save JSON once at the end
    try:
        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(localization_results, f, indent=4)
        _logger.info(f"Successfully saved JSON results to {json_output_path}")
    except Exception as e:
        _logger.error(f"Error saving JSON file: {e}")


if __name__ == "__main__":
    main()
