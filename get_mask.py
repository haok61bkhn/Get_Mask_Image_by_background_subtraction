import cv2
import numpy as np
import os
import time 

id=6
fg="mask/"+str(id)+"_fg.jpg"
bg="mask/"+str(id)+"_bg.jpg"

FOREGROUND_IMG = fg
BACKGROUND_IMG = bg

def blur_color_img(img, kernel_width=5, kernel_height=5, sigma_x=2, sigma_y=2):
    img = np.copy(img) # we don't modify the original image
    img[:,:,0] = cv2.GaussianBlur(img[:,:,0], ksize=(kernel_width, kernel_height), sigmaX=sigma_x, sigmaY=sigma_y)
    img[:,:,1] = cv2.GaussianBlur(img[:,:,1], ksize=(kernel_width, kernel_height), sigmaX=sigma_x, sigmaY=sigma_y)
    img[:,:,2] = cv2.GaussianBlur(img[:,:,2], ksize=(kernel_width, kernel_height), sigmaX=sigma_x, sigmaY=sigma_y)
    return img   

def background_subtraction(fg_img, bg_img, diff_threshold=30):
    # fg_img = blur_color_img(fg_img,3 , 3, 4, 4)
    # bg_img = blur_color_img(bg_img,3, 3, 4, 4)
    mask = fg_img - bg_img
    mask = np.abs(mask)
    mask = np.mean(mask, axis=2, keepdims=False)
    mask[mask<diff_threshold] = 0
    mask[mask>=diff_threshold] = 255
    # mask = mask.astype(np.uint8)
    # mask = cv2.medianBlur(mask, 7)
    return mask
    
def main(foreground_img, background_img):
    fg_img = cv2.imread(foreground_img) # [h, w, 3]
    bg_img = cv2.imread(background_img) # [h, w, 3]
    t1=time.time()
    mask = background_subtraction(fg_img, bg_img)
    print("time ",time.time()-t1)
    new_fg = np.zeros([fg_img.shape[0], fg_img.shape[1], 4]) # png image --> has 4-dims instead of 3-dims like color image
    new_fg[:,:,:3] = fg_img
    new_fg[:,:,3] = mask

    cv2.imwrite("mask_test/output.png",new_fg)
    cv2.imwrite("mask_test/fg.jpg",fg_img)
    cv2.imwrite("mask_test/bg.jpg",bg_img)
 
if __name__ == "__main__":
    print('Running Background Subtraction for: %s and %s' % (FOREGROUND_IMG, BACKGROUND_IMG))
    main(foreground_img=FOREGROUND_IMG, background_img=BACKGROUND_IMG)
   
