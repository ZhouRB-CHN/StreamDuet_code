import os

import cv2


def sort_frames(frames,start_frame,end_frame):
    batch_fnames = sorted(
        [f"{str(idx).zfill(10)}.{frames[idx].split('.')[-1]}" for idx in range(start_frame, end_frame)])


def list_frames(raw_images):
    return [f for f in os.listdir(raw_images) if f.endswith('.png') or f.endswith('.jpg')]

def get_images_length(raw_images_path):
    return len([f for f in os.listdir(raw_images_path) if f.endswith('.png') or f.endswith('.jpg')])

def get_image_extension(raw_images_path):
    for f in os.listdir(raw_images_path):
        if f.endswith(".png"):
            return "png"
        elif f.endswith(".jpg") or f.endswith(".jpeg"):
            return "jpg"
    return None



def draw_bboxes_on_image(results, image, output_path):

    image_copy = image.copy()


    for region in results.regions:
        height, width = image_copy.shape[:2]
        x_min = int(region.x * width)
        y_min = int(region.y * height)
        x_max = int((region.x + region.w) * width)
        y_max = int((region.y + region.h) * height)


        cv2.rectangle(image_copy, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)


    cv2.imwrite(output_path, image_copy)

