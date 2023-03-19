"""
This script creates a new dataset with the related annotations for a certain category from the TACO dataset. 
As the annotations follow the COCO format dataset, we can parse easily the json file. 
It will split the data in training and evaluation set, and also copy the images in new directoy
"""
import sys
import json
from unicodedata import category
import os
import glob 
from shutil import copy

######################################
## User parameter's
######################################

category="Cigarette"
TACO_dataset_dir = "../Deep-Recycling/TACO/data/"
TACO_annotations = "../Deep-Recycling/TACO/data/annotations.json"
split_ratio=0.8                         # percentage for the training
dir_dst_train="./SimpleTrash/data/train/"
dir_dst_eval="./SimpleTrash/data/eval/"

# First we access the category id 
annotations=open(TACO_annotations)
data=json.load(annotations)

cat_id=[]
for cat in data["categories"]:
    if cat["name"]==category:
        cat_id.append(cat["id"])

# Then all images with this category

image_id=[]
for cat in cat_id:
    for annotations in data["annotations"]:
        if annotations["category_id"] in cat_id:
            image_id.append(annotations["image_id"])

image_id=list(set(image_id))    #remove duplicates id

# You can check if the number seem correct with the tacodataset.org 
print(f"Number of images qualified as '{category}': ", len(image_id))

# Then we split the dataset in training and evaluation sets according to splitting ratio
import random
random.seed(000)

train_image_id=[]
eval_image_id=[]

idx=range(len(image_id))
nb_train_image=int(split_ratio*len(image_id))

train_idx=random.sample(idx, k=nb_train_image)

for id in train_idx:
    train_image_id.append(image_id[id])
for img_id in image_id:
    if img_id not in train_image_id:
        eval_image_id.append(img_id)

# Now get the split at the file_name level
file_name_train=[]
file_name_eval=[]

for image in data["images"]:
    if image["id"] in train_image_id:
        file_name_train.append(image["file_name"])
for image in data["images"]:
    if image["id"] in eval_image_id:
        file_name_eval.append(image["file_name"])

print("Double check")
print("Total number of pictures", len(image_id))
print("Number of train images", len(train_image_id))
print("Number of train images (file_name)", len(file_name_train))
print("Number of eval images", len(eval_image_id))
print("Number of eval images (file_name)", len(file_name_eval))

# Now we can create a replica of the annotations (GT) only for the desired category
# And aslo copy the data in the train and eval directory
for dir in ['train', 'eval']:
    if dir == 'train':
        directory=dir_dst_train
        list_image_id=train_image_id
    else:
        directory=dir_dst_eval
        list_image_id=eval_image_id
    new_annotations = {}

    new_annotations["info"]=data["info"]

    images_coco=[]
    for idx, image in enumerate(data["images"]):
        image_dict={}
        if image["id"] in list_image_id:    
            image_dict["id"]=image["id"]
            image_dict["width"]=image["width"]
            image_dict["height"]=image["height"]
            file=TACO_dataset_dir+image["file_name"]
            file_name=f"{idx}.jpg"
            file_dst=directory+file_name
            copy(file, file_dst)
            image_dict["file_name"]=file_name
            image_dict["license"]=image["license"]
            image_dict["flickr_url"]=image["flickr_url"]
            image_dict["coco_url"]=image["coco_url"]
            image_dict["date_captured"]=image["date_captured"]
            image_dict["flickr_640_url"]=image["flickr_640_url"]
            images_coco.append(image_dict)
    new_annotations["images"]=images_coco

    annotations_coco=[]
    for annotations in data["annotations"]:
        image_dict={}
        if annotations["image_id"] in list_image_id and annotations["category_id"] in cat_id:   
            image_dict["id"]=annotations["id"]
            image_dict["image_id"]=annotations["image_id"]
            image_dict["category_id"]=0     #only 1 categroy
            image_dict["segmentation"]=annotations["segmentation"]
            image_dict["area"]=annotations["area"]
            image_dict["bbox"]=annotations["bbox"]
            image_dict["iscrowd"]=annotations["iscrowd"]
            annotations_coco.append(image_dict)
    new_annotations["annotations"]=annotations_coco

    new_annotations["licenses"]=data["licenses"]

    new_annotations["categories"]={'supercategory': category, 'id': 0, 'name':category}

    output_dir=directory+"annotations.json"
    with open(output_dir, 'w') as f:
        json.dump(new_annotations, f)

print("Done!")


