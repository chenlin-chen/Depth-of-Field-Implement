import numpy as np
import os
import cv2
import argparse

from scatter_blur import ScatterBlurOptimWithAlpha

def decomposeBlur(img: np.ndarray, depth: np.ndarray, focal_depth:int, focus_tol:int, max_disk_r:int) -> \
    np.ndarray:

    max_depth_disp = float(max(focal_depth, 255-focal_depth) - focus_tol)
    min_disk_r = 2

    focal_depth_disp = float(min_disk_r)/float(max_disk_r)  * max_depth_disp
    focal_d0 = max(focal_depth - focus_tol - focal_depth_disp, 0)
    focal_d1 = min(focal_depth + focus_tol + focal_depth_disp, 255)

    back_depth = 255 - focal_d1
    front_depth = focal_d0 - 0
    remain_depth = back_depth + front_depth

    band_count = 4
    band_depth_range = remain_depth / float(band_count)
    back_band_count = int(back_depth / band_depth_range + 0.5)
    front_band_count = int(front_depth / band_depth_range + 0.5)
    if back_band_count + front_band_count > band_count:
        if back_band_count > front_band_count:
            back_band_count -= 1
        else:
            front_band_count -= 1
    
    if front_band_count == 0 :
        focal_d0 = 0

    def decompose(img_float:np.ndarray, d0:float, d1:float) -> tuple[np.ndarray, np.ndarray]:
        if d1 < d0 :
            raise ValueError(f"d1 must be greater than d0. Got d0={d0}, d1={d1}")

        d_range = d1 - d0
        d_scale = 4.0 / d_range
        dist_d0 = depth - d0
        dist_d1 = d1 - depth

        min_depth = np.minimum(dist_d0, dist_d1)
        alpha = (1 + (min_depth*d_scale)).clip(0.0, 1.0)[:,:,None]
        premultiplied_img = (img_float * alpha).astype(np.uint8)
        return premultiplied_img, alpha
    
    def doBlur(premultiplied_img:np.ndarray, alpha:np.ndarray, layer_depth) -> np.ndarray:
        depth_r = max(abs(layer_depth - focal_depth)-focus_tol, 0) / max_depth_disp
        disk_r = int(depth_r*max_disk_r + 0.5)
        blur_img = ScatterBlurOptimWithAlpha(premultiplied_img, alpha, disk_r)
        return blur_img

    img_float = img.astype(np.float32)
    h, w, c = img.shape
    output_img = np.zeros((h,w,c), dtype=np.float32)

    pre_d0 = 255
    if back_band_count > 0:
        back_band_range = back_depth / back_band_count
        for i in range(back_band_count):
            d1 = pre_d0
            d0 = pre_d0 - back_band_range  

            premultiplied_img, alpha = decompose(img_float, d0, d1)
            blur_img = doBlur(premultiplied_img, alpha, d1)
            output_img = blur_img * alpha + (1.0 - alpha) * output_img
            pre_d0 = d0

    premultiplied_img, alpha = decompose(img_float, focal_d0, pre_d0)
    blur_img = ScatterBlurOptimWithAlpha(premultiplied_img, alpha, min_disk_r)
    output_img = blur_img * alpha + (1.0 - alpha) * output_img
    pre_d0 = focal_d0

    if front_band_count > 0:
        front_band_range = front_depth / front_band_count

        for i in range(front_band_count):
            d1 = pre_d0
            d0 = max(pre_d0 - front_band_range, 0.0)

            premultiplied_img, alpha = decompose(img_float, d0, d1)
            blur_img = doBlur(premultiplied_img, alpha, d0)
            output_img = blur_img * alpha + (1.0 - alpha) * output_img
            pre_d0 = d0

    return output_img.astype(np.uint8)



def main(args):
    img_path = "./examples/001.png"
    depth_path = "./examples/001_depth.png"
    img = cv2.imread(img_path)
    depth_map = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
    
    h, w, _ = img.shape 
    down_scale = 2.0
    blur_r = args.blur_r
    
    sub_blur_r = int(blur_r / down_scale +0.5)
    sub_h, sub_w = int(h/down_scale), int(w/down_scale)
    sub_img = cv2.resize(img, (sub_w, sub_h))
    sub_depth_map = cv2.resize(depth_map, (sub_w, sub_h))
    sub_depth_map_int = sub_depth_map.astype(np.int32)

    focal_depth = args.focal_depth
    focus_tol = int( args.focus_tol / (1.0 + (focal_depth/128.0)**2 ) )
    blured_img = decomposeBlur(sub_img, sub_depth_map_int, focal_depth, focus_tol, sub_blur_r)
    blured_img = cv2.resize(blured_img, (w, h))

    focal_region = np.abs(depth_map - focal_depth) <= focus_tol
    blured_img[focal_region] = img[focal_region]

    output_folder = "./outputs"
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    save_name = f"{img_name}_scatter_LayerBlurR_{blur_r}_focalD{focal_depth}_focalTol{focus_tol}.png"
    save_path = os.path.join(output_folder, save_name)
    cv2.imwrite(save_path, blured_img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--blur_r", type=int, default=20, help="Maximun blur radius")
    parser.add_argument("--focal_depth", type=int, default=128, help="Focal depth (0-255, grayscale value)")
    parser.add_argument("--focus_tol", type=float, default=25.0, help='Focal region tolerance')
    args = parser.parse_args()
    main(args)