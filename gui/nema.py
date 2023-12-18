#!/pkg/mamba/envs/yolo8/bin/python
from ultralytics import YOLO
import cv2
import csv
# import torch
import numpy as np
import argparse
import os

rkn_name = "RKN"
fln_name = "FLN"
egg_name = "EGG"
deg_name = "DEG"

seg_classes = {0.0: rkn_name, 1.0: fln_name}
det_classes = {0.0: egg_name, 1.0: deg_name}

def divide_image_into_grid(image, grid_size):
    height, width = image.shape[:2]
    grid_height = height // grid_size[0]
    grid_width = width // grid_size[1]
    subimages = []
    coordinates = []
    for y in range(grid_size[0]):
        for x in range(grid_size[1]):
            y1 = y * grid_height
            y2 = (y + 1) * grid_height
            x1 = x * grid_width
            x2 = (x + 1) * grid_width
            subimage = image[y1:y2, x1:x2]
            subimages.append(subimage)
            coordinates.append((x1, y1))
    return subimages, coordinates


def divide_image_into_crop(image, size):
    height, width, _ = image.shape
    if height < size * 2 or width < size * 2:
        return [image], [(0, 0)]
    grid_width = (width + size - 1) // size
    grid_height = (height + size - 1) // size
    images = []
    coordinates = []
    for i in range(grid_width):
        for j in range(grid_height):
            left = i * size
            upper = j * size
            right = min((i + 1) * size, width)
            lower = min((j + 1) * size, height)
            grid_image = image[upper:lower, left:right]
            images.append(grid_image)
            coordinates.append((left, upper))
    return images, coordinates


def segment(seg_model, img, original_image=None, cordinates=None, confidence=0.5):
    all_results = []
    seg_results = seg_model.predict(source=img.copy(), show=False, save=False, save_txt=False, stream=False, conf=confidence)
    if original_image is None: original_image = img
    for result in seg_results:
        boxes = result.boxes.cpu().numpy()
        # masks = result.masks.xy
        for i, box in enumerate(boxes):
            score = float(box.conf)
            cls = seg_classes[box.cls[0]]
            color = (0, 255, 0) if cls == rkn_name else (0, 0, 255)
            x1, y1, x2, y2 = box.xyxy[0]
            if cordinates is not None:
                x1 += cordinates[0]
                y1 += cordinates[1]
                x2 += cordinates[0]
                y2 += cordinates[1]
            all_results.append([int(x1), int(y1), int(x2), int(y2), score, cls])
            # segments = []
            # for pixel in masks[i]:
            #     x, y = pixel
            #     if cordinates is not None:
            #         x += cordinates[0]
            #         y += cordinates[1]
            #     segments.append((int(x), int(y)))
            # cv2.polylines(original_image, [np.array(segments)], True, color, 2)
            # cv2.fillPoly(original_image, [np.array(segments)], color)
    return all_results

def detect(det_model, img, original_image=None, cordinates=None, confidence=0.5):
    all_results = []
    det_results = det_model.predict(source=img.copy(), show=False, save=False, save_txt=False, stream=False, conf=confidence)
    if original_image is None: original_image = img
    for result in det_results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            score = float(box.conf)
            cls = det_classes[box.cls[0]]
            color = (150, 150, 0) if cls == egg_name else (150, 0, 150)
            x1, y1, x2, y2 = box.xyxy[0]
            if cordinates is not None:
                x1 += cordinates[0]
                y1 += cordinates[1]
                x2 += cordinates[0]
                y2 += cordinates[1]
            all_results.append([int(x1), int(y1), int(x2), int(y2), score, cls])
    return all_results

def calculate_iou(box1, box2):
    x1, y1, x2, y2, *_ = box1
    x1_, y1_, x2_, y2_, *_ = box2
    inter_area = max(0, min(x2, x2_) - max(x1, x1_)) * max(0, min(y2, y2_) - max(y1, y1_))
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_ - x1_) * (y2_ - y1_)
    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area
    return iou


def iou_correct(det_results, seg_results, img):
    numbers = {rkn_name: 0, fln_name: 0, egg_name: 0, deg_name: 0}
    for det in det_results:
        x1, y1, x2, y2, score, cls = det
        numbers[cls] += 1
        for seg in seg_results:
            if calculate_iou(det, seg) > 0.5:
                seg_results.remove(seg)
                cv2.circle(img, (int(x1), int(y1)), 5, (0, 255, 0), 10)
                break
    for seg in seg_results:
        x1, y1, x2, y2, score, cls = seg
        numbers[cls] += 1
    results = det_results + seg_results
    return results, numbers

def annotate_img(results, img):
    colors = {rkn_name: (0, 255, 0), fln_name: (0, 0, 255), egg_name: (150, 150, 0), deg_name: (150, 0, 150)}
    for u in results:
        x1, y1, x2, y2, score, cls = u
        color = colors[cls]
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
        cv2.putText(img, f"{cls} {score:.2f}", (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 3)
    return results


def write_to_csv(image_path, results, output_csv):
    header = ["IMAGE", "TOTAL JUVENILES", f"{rkn_name}", f"{fln_name}", "TOTAL EGGS", f"{egg_name}", f"{deg_name}"]
    if output_csv:
        file_exists = os.path.isfile(output_csv)
        with open(output_csv, mode='a', newline='') as file:  # 'a' for append mode
            writer = csv.writer(file)
            # Write header if the file is newly created
            if not file_exists:
                writer.writerow(header)
            writer.writerow([
                image_path,
                results[rkn_name] + results[fln_name],
                results[rkn_name], 
                results[fln_name],
                results[egg_name] + results[deg_name],
                results[egg_name], 
                results[deg_name], 
            ])

def run(image_path, grid_size=None, crop_size=None, save_folder=None, confidence=0.5, confidence_egg=0.5):
    image = cv2.imread(image_path)
    seg_model = YOLO("/doc/code/DEG/nematodes/other/seg/runs/segment/big2/weights/best.pt")
    det_model = YOLO("/doc/code/DEG/nematodes/other/det/runs/detect/big/weights/best.pt")

    if grid_size is None and crop_size is None:
        det_results = detect(det_model, image, confidence=confidence_egg)
        seg_results = segment(seg_model, image, confidence=confidence)
        results, numbers = iou_correct(det_results, seg_results, image)
        annotate_img(results, image)
    else:
        if crop_size is not None:
            subimages, cordinates = divide_image_into_crop(image, crop_size)
        else:
            subimages, cordinates = divide_image_into_grid(image, grid_size)
        all_detected = []
        all_segmented = []
        for i, img in enumerate(subimages):
            all_detected.extend(detect(det_model, img, image, cordinates[i], confidence=confidence_egg))
            all_segmented.extend(segment(seg_model, img, image, cordinates[i], confidence=confidence))
        results, numbers = iou_correct(all_detected, all_segmented, image)
        annotate_img(results, image)

    if save_folder is not None:
        annotated_image_path = os.path.join(save_folder, os.path.basename(image_path))
        cv2.imwrite(annotated_image_path, image)

    print("\n\n--------------------------------------------")
    print(f"{rkn_name}     : {numbers[rkn_name]}\
          \n{fln_name}     : {numbers[fln_name]}\
          \n{egg_name}     : {numbers[egg_name]}\
          \n{deg_name}     : {numbers[deg_name]}\
          \n\nTOTAL    : {numbers[rkn_name] + numbers[fln_name] + numbers[egg_name] + numbers[deg_name]}")
    print("--------------------------------------------")
    return numbers

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images using YOLO models.")
    parser.add_argument("image_path", help="Path to the image or folder of images")
    parser.add_argument("--thresh_juv", help="Threshold value for juvenile segmentation")
    parser.add_argument("--thresh_egg", help="Threshold value for egg detection")
    parser.add_argument("--grid_size", help="Grid size in format '2x2' or '512'")
    parser.add_argument("--output_csv", help="Path to the output CSV file")
    parser.add_argument("--save_folder", help="Path to the folder to save annotated images")

    args = parser.parse_args()
    image_path = args.image_path
    output_csv = args.output_csv
    if args.thresh_juv:
        confidence = float(args.thresh_juv)
    else:
        confidence = 0.5
    if args.thresh_egg:
        confidence_egg = float(args.thresh_egg)
    else:
        confidence_egg = 0.5
    if args.grid_size:
        try:
            grid_size = tuple(map(int, args.grid_size.split('x')))
            crop_size = None
        except ValueError:
            try:
                crop_size = int(args.grid_size)
                grid_size = None
            except ValueError:
                print("Invalid grid size format. Please use format like '2x2' or '512'")
                exit(1)
    else:
        grid_size = None
        crop_size = None
    if args.save_folder:
        save_folder = args.save_folder
        os.makedirs(save_folder, exist_ok=True)
    else:
        save_folder = None

    if os.path.isdir(image_path):
        for filename in os.listdir(image_path):
            if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".tif"):
                current_image_path = os.path.join(image_path, filename)
                results = run(current_image_path, grid_size, crop_size, save_folder, confidence=confidence, confidence_egg=confidence_egg)
                write_to_csv(current_image_path, results, output_csv)
    else:
        results = run(image_path, grid_size, crop_size, save_folder, confidence=confidence, confidence_egg=confidence_egg)
        write_to_csv(image_path, results, output_csv)