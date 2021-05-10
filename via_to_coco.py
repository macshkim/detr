import os
import cv2
import math
from itertools import chain
import numpy as np
import json
from os import listdir
from os.path import isfile, join


def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))


if __name__ == '__main__':

    image_dir = '/Users/mackim/datasets/garment_images/images'
    anno_dir = '/Users/mackim/datasets/garment_images/via_annotation_json'
    coco_file = '/Users/mackim/datasets/garment_images/via_project_coco.json'

    # image_files = [file for file in listdir(image_dir) if file.endswith('.png') or file.endswith('.jpeg')]

    class_keyword = 'label'

    anno_files = []
    for file in listdir(anno_dir):
        anno_file = join(anno_dir, file)
        if isfile(anno_file):
            anno_files.append(anno_file)

    images_ids_dict = {}
    images_info = []
    classes = set()
    image_id = 0
    data_length = 0
    for anno_file in anno_files:
        with open(anno_file) as in_file:
            label_data = json.load(in_file)
            data_length += len(label_data)

            for v in label_data.values():
                file_name = v["filename"].replace(' ', '-')
                # image_files.remove(file_name)
                images_ids_dict[file_name] = image_id
                img = cv2.imread(os.path.join(image_dir, file_name), 0)
                # print(file_name)
                height, width = img.shape[:2]
                assert (height > 0) and (width > 0)

                images_info.append({"file_name": file_name, "id": image_id, "width": width, "height": height})

                regions_list = v['regions']
                if len(regions_list) > 0:
                    for region in regions_list:
                        if class_keyword in region['region_attributes']:
                            class_label = region['region_attributes'][class_keyword]
                            classes.add(class_label)
                image_id += 1

    # print(image_files)

    category_ids_dict = {c: i for i, c in enumerate(classes)}
    categories = [{"supercategory": class_keyword, "id": v, "name": k} for k, v in category_ids_dict.items()]

    suffix_zeros = math.ceil(math.log10(data_length))
    annotations = []
    for anno_file in anno_files:
        with open(anno_file) as in_file:
            label_data = json.load(in_file)

            # suffix_zeros = math.ceil(math.log10(len(label_data)))
            for i, v in enumerate(label_data.values()):
                file_name = v["filename"].replace(' ', '-')
                for j, r in enumerate(v["regions"]):
                    if class_keyword in r["region_attributes"]:
                        x, y = r["shape_attributes"]["all_points_x"], r["shape_attributes"]["all_points_y"]
                        annotations.append({
                            "segmentation": [list(chain.from_iterable(zip(x, y)))],
                            "area": PolyArea(x, y),
                            "bbox": [min(x), min(y), max(x) - min(x), max(y) - min(y)],
                            "image_id": images_ids_dict[file_name],
                            "category_id": category_ids_dict[r["region_attributes"][class_keyword]],
                            "id": int(f"{i:0>{suffix_zeros}}{j:0>{suffix_zeros}}"),
                            "iscrowd": 0
                        })

    coco = {
        "images": images_info,
        "categories": categories,
        "annotations": annotations
    }

    with open(coco_file, "w") as f:
        json.dump(coco, f)

    print("COCO annotation file has been generated for {} images".format(data_length))

'''
COCO annotation file has been generated for 419 images
'''