import numpy as np
import os
import cv2
import argparse

from gather_blur import GatherBlur, GatherBlurWithDepth

def main():
    output_folder = "./outputs/"
    img_path = "./examples/001.png"
    img = cv2.imread(img_path)

    h, w, _ = img.shape 
    down_scale = 2.0
    blur_r = 20
    sub_blur_r = int(blur_r / down_scale +0.5)
    sub_h, sub_w = int(h/down_scale), int(w/down_scale)
    sub_img = cv2.resize(img, (sub_w, sub_h))
    
    blured_img = GatherBlur(sub_img, sub_blur_r)
    blured_img = cv2.resize(blured_img, (w, h))
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    save_name = f"{img_name}_gather_blurR_{blur_r}.png"
    save_path = os.path.join(output_folder, save_name)
    cv2.imwrite(save_path, blured_img)

    depth_path = "./examples/001_depth.png"
    depth_map = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
    sub_depth_map = cv2.resize(depth_map, (sub_w, sub_h))
    sub_depth_map_int = sub_depth_map.astype(np.int32)
    focal_depth = 128
    focus_tol = int( 25.0 / (1.0 + (focal_depth/128.0)**2 ) )
    blured_img = GatherBlurWithDepth(sub_img, sub_depth_map_int, focal_depth, focus_tol, sub_blur_r)
    blured_img = cv2.resize(blured_img, (w, h))
    focal_region = np.abs(depth_map - focal_depth) <= focus_tol
    blured_img[focal_region] = img[focal_region]

    img_name = os.path.splitext(os.path.basename(img_path))[0]
    save_name = f"{img_name}_gather_blurR_{blur_r}_focalD{focal_depth}_focalL{focus_tol}.png"
    save_path = os.path.join(output_folder, save_name)
    cv2.imwrite(save_path, blured_img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--blur_r", type=int, default=20, help="Maximun blur radius")
    parser.add_argument("--focal_depth", type=int, default=128, help="Focal depth (0-255, grayscale value)")
    parser.add_argument("--focus_tol", type=float, default=25.0, help='Focal region tolerance)')
    args = parser.parse_args()
    main()