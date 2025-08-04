import numpy as np

def GetIntergal(img: np.ndarray, blur_r: int) -> np.ndarray:
    """
    Compute the integral image (summed-area table) of an input image with symmetric padding.
    Args:
        img (np.ndarray): Input image of shape (H, W, C), where H is height, W is width, and C is the number of channels.
        blur_r (int): Blur radius. Determines how much padding is added around the image.

    Returns:
        np.ndarray: Integral image with shape (H + 2 * blur_r + 1, W + 2 * blur_r + 1, C).
                    The extra +1 on height and width allows easy use of the integral image formula 
                    without negative indexing or special boundary handling.

    Notes:
        - Symmetric padding is applied before computing the integral image to preserve edge information.
        - An extra row and column of zeros are added at the top and left for boundary alignment.
        - The result is typically used to compute fast box filters or average pooling over regions.
    """

    pad_img = np.pad(img, ((blur_r, blur_r), (blur_r, blur_r), (0, 0)), 'symmetric').astype(np.uint32)
    integral_img = pad_img.cumsum(axis=0).cumsum(axis=1)
    integral_img = np.pad(integral_img, ((1, 0), (1, 0), (0, 0)), 'constant', constant_values=0)

    return integral_img

def GatherBlur(img: np.ndarray, blur_r: int) -> np.ndarray:
    # Ensure input is 3D
    if img.ndim != 3:
        raise ValueError("Input image must be 3D (H x W x C)")

    h, w, c = img.shape
    integral_img = GetIntergal(img, blur_r)
    
    blur_d = (2*blur_r + 1)
    # Extract summed regions using the integral image trick
    block_rb = integral_img[blur_d:, blur_d:]
    block_rt = integral_img[:h, blur_d:]
    block_lb = integral_img[blur_d:, :w]
    block_lt = integral_img[:h, :w]

    blur_size = blur_d ** 2
    blured_img = (block_rb - block_rt - block_lb + block_lt) / blur_size
    
    return blured_img.astype(np.uint8)

def GatherBlurWithDepth(img:np.ndarray, depth:np.ndarray, focal_depth:int, focus_tol:int, max_blur_r:int) \
    -> np.ndarray:

    # Ensure input is 3D
    if img.ndim != 3:
        raise ValueError("Input image must be 3D (H x W x C)")

    integral_img = GetIntergal(img, max_blur_r)

    h, w, c = img.shape
    max_depth_disp = float(max(focal_depth, 255-focal_depth) - focus_tol)
    depth_r_map = np.maximum(np.abs(depth - focal_depth)-focus_tol, 0) / max_depth_disp
    blur_r_map = (depth_r_map*max_blur_r + 0.5).astype(np.int32)
    blured_img = np.zeros_like(img)

    for y in range(h):
        intergal_y = y+max_blur_r+1
        intergal_x = max_blur_r + 1
        for x in range(w):
            blur_r = blur_r_map[y, x]
            blur_d = 2 * blur_r + 1

            integral_rb = integral_img[intergal_y+blur_r, intergal_x+blur_r]
            integral_rt = integral_img[intergal_y-blur_r-1, intergal_x+blur_r]
            integral_lb = integral_img[intergal_y+blur_r, intergal_x-blur_r-1]
            integral_lt = integral_img[intergal_y-blur_r-1, intergal_x-blur_r-1]
            blur_size = blur_d ** 2
            blured_img[y, x, :] = (integral_rb - integral_rt - integral_lb + integral_lt) / blur_size

            intergal_x += 1

    return blured_img

if __name__ == "__main__":
    import cv2
    import os
    img_path = "./examples/001.png"
    img = cv2.imread(img_path)

    h, w, _ = img.shape 
    down_scale = 2.0
    sub_h, sub_w = int(h/down_scale), int(w/down_scale)
    sub_img = cv2.resize(img, (sub_w, sub_h))

    blur_r = 10
    sub_blur_r = int(blur_r / down_scale +0.5)
    blur_img = GatherBlur(sub_img, sub_blur_r)
    blur_img = cv2.resize(blur_img, (w, h))
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    save_path = f"./{img_name}_gather_blurR_{blur_r}.png"
    cv2.imwrite(save_path, blur_img)
    

    depth_path = "./examples/001_depth.png"
    depth_map = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
    sub_depth_map = cv2.resize(depth_map, (sub_w, sub_h))
    sub_depth_map_int = sub_depth_map.astype(np.int32)
    focal_depth = 128
    focus_tol = 0
    blur_img = GatherBlurWithDepth(sub_img, sub_depth_map_int, focal_depth, focus_tol, sub_blur_r)
    blur_img = cv2.resize(blur_img, (w, h))
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    save_path = f"./{img_name}_gather_blurR_{blur_r}_focalD{focal_depth}_focalL{focus_tol}.png"
    cv2.imwrite(save_path, blur_img)
    
