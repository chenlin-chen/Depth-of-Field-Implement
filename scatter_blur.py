import numpy as np
import os

def sumArea(img:np.ndarray, disk_r: int) -> tuple[np.ndarray, np.ndarray]:
    h, w, c = img.shape
    sum_table = np.zeros((h,w,c), np.int32)
    count_table = np.zeros((h,w), np.int32)
    disk_table = np.zeros((2*disk_r+1), np.int32)

    for i, dy in enumerate(range(-disk_r, disk_r+1)):
        disk_table[i] = int((disk_r*disk_r - dy*dy)**0.5);  

    for y in range(h):
        for x in range(w):
            for i, dy in enumerate(range(-disk_r, disk_r+1)):
                target_y = y + dy
                if(target_y < 0 or target_y >= h):
                    continue

                dx = disk_table[i]
                x_l = max(x - dx, 0)
                x_r = min(x + dx, w-1)
        
                sum_table[target_y, x_l, :] += img[y, x, :] 
                count_table[target_y, x_l] += 1

                if(x_r < w-1):
                    sum_table[target_y, x_r+1, :] -= img[y, x, :]
                    count_table[target_y, x_r+1] -= 1

    return sum_table, count_table

def ScatterBlurOptim(img:np.ndarray, disk_r:int) -> np.ndarray:
    sum_area, sum_alpha_area = sumArea(img, disk_r)

    sum_area = sum_area.cumsum(axis=1).astype(np.float32)
    sum_alpha_area = sum_alpha_area.cumsum(axis=1).astype(np.float32)
    sum_alpha_area[sum_alpha_area == 0.0] = 1.0

    blur_img = sum_area / sum_alpha_area[:,:, None]
    blur_img = (blur_img + 0.5).astype(np.uint8)
    return blur_img

def sumAreaWithAlpha(img:np.ndarray, alpha:np.ndarray, disk_r: int) -> tuple[np.ndarray, np.ndarray]:
    h, w, c = img.shape
    sum_table = np.zeros((h,w,c), np.int32)
    sum_alpha_table = np.zeros((h,w), np.float32)
    disk_table = np.zeros((2*disk_r+1), np.int32)

    for i, dy in enumerate(range(-disk_r, disk_r+1)):
        disk_table[i] = int((disk_r*disk_r - dy*dy)**0.5);  

    for y in range(h):
        for x in range(w):
            if alpha[y][x] == 0.0:
                continue 

            for i, dy in enumerate(range(-disk_r, disk_r+1)):
                target_y = y + dy
                if(target_y < 0 or target_y >= h):
                    continue

                dx = disk_table[i]
                x_l = max(x - dx, 0)
                x_r = min(x + dx, w-1)
        
                sum_table[target_y, x_l, :] += img[y, x, :] 
                sum_alpha_table[target_y, x_l] += alpha[y][x]

                if(x_r < w-1):
                    sum_table[target_y, x_r+1, :] -= img[y, x, :]
                    sum_alpha_table[target_y, x_r+1] -= alpha[y][x]

    return sum_table, sum_alpha_table

def ScatterBlurOptimWithAlpha(img:np.ndarray, alpha:np.ndarray, disk_r:int) -> np.ndarray:
    sum_area, count_area = sumAreaWithAlpha(img, alpha, disk_r)

    sum_area = sum_area.cumsum(axis=1).astype(np.float32)
    count_area = count_area.cumsum(axis=1).astype(np.float32)
    count_area[count_area == 0] = 1

    blur_img = sum_area / count_area[:,:, None]
    blur_img = (blur_img + 0.5).astype(np.uint8)
    return blur_img

def sumAreaWithDepth(img:np.ndarray, depth:np.ndarray, focal_depth:int, focus_tol:int, max_disk_r: int) \
    -> tuple[np.ndarray, np.ndarray]:

    def createDiskTable():
        radius2disktable = {}
        radius2disktable[0] = np.zeros(1, np.int32)

        for disk_r in range(1, max_disk_r+1):
            disk_table = np.zeros((2*disk_r+1), np.int32)
            for i, dy in enumerate(range(-disk_r, disk_r+1)):
                disk_table[i] = int((disk_r*disk_r - dy*dy)**0.5);  
    
            radius2disktable[disk_r] = disk_table
        return radius2disktable

    h, w, c = img.shape
    sum_table = np.zeros((h,w,c), np.int32)
    count_table = np.zeros((h,w), np.int32)
    radius2disktable = createDiskTable()

    max_depth_disp = float(max(focal_depth, 255-focal_depth) - focus_tol)
    for y in range(h):
        for x in range(w):
            depth_r = max(abs(depth[y,x] - focal_depth)-focus_tol, 0) / max_depth_disp
            disk_r = int(depth_r*max_disk_r + 0.5)

            disk_table = radius2disktable[disk_r]
            for i, dy in enumerate(range(-disk_r, disk_r+1)):
                target_y = y + dy
                if(target_y < 0 or target_y >= h):
                    continue

                dx = disk_table[i]
                x_l = max(x - dx, 0)
                x_r = min(x + dx, w-1)
        
                sum_table[target_y, x_l, :] += img[y, x, :] 
                count_table[target_y, x_l] += 1

                if(x_r < w-1):
                    sum_table[target_y, x_r+1, :] -= img[y, x, :]
                    count_table[target_y, x_r+1] -= 1

    return sum_table, count_table

def ScatterBlurWithDepth(img:np.ndarray, depth:np.ndarray, focal_depth:int, focus_tol:int, max_disk_r:int) \
    -> np.ndarray:
    sum_area, count_area = sumAreaWithDepth(img, depth, focal_depth, focus_tol, max_disk_r)

    sum_area = sum_area.cumsum(axis=1).astype(np.float32)
    count_area = count_area.cumsum(axis=1).astype(np.float32)
    count_area[count_area == 0] = 1

    blur_img = sum_area / count_area[:,:, None]
    blur_img = (blur_img + 0.5).astype(np.uint8)
    return blur_img

if __name__ == "__main__":
    import cv2
    img_path = "./examples/001.png"
    img = cv2.imread(img_path)

    h, w, _ = img.shape 
    down_scale = 2.0
    sub_h, sub_w = int(h/down_scale), int(w/down_scale)
    sub_img = cv2.resize(img, (sub_w, sub_h))

    blur_r = 10
    sub_blur_r = int(blur_r / down_scale +0.5)
    '''
    blur_img = ScatterBlurOptim(sub_img, blur_r)
    blur_img = cv2.resize(blur_img, (w, h))
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    save_path = f"./{img_name}_blurR_{blur_r}.png"
    cv2.imwrite(save_path, blur_img)
    '''

    depth_path = "./examples/001_depth.png"
    depth_map = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
    sub_depth_map = cv2.resize(depth_map, (sub_w, sub_h))
    sub_depth_map_int = sub_depth_map.astype(np.int32)
    focal_depth = 128
    focus_tol = 0
    blur_img = ScatterBlurWithDepth(sub_img, sub_depth_map_int, focal_depth, focus_tol, blur_r)
    blur_img = cv2.resize(blur_img, (w, h))
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    save_path = f"./{img_name}_blurR_{blur_r}_focal_{focal_depth}.png"
    cv2.imwrite(save_path, blur_img)