import torch
import clip
from torchvision.datasets import VOCSegmentation
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import cv2

from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC



device = "cuda" if torch.cuda.is_available() else "cpu"
model, _ = clip.load("CS-RN101", device=device)
preprocess_img =  Compose([Resize((224, 224), interpolation=BICUBIC), ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])


preprocess_target =  Compose([Resize((224, 224), interpolation=BICUBIC), ToTensor()])

class VOCSegmentationCustom(VOCSegmentation):
    def __getitem__(self, index):
        img = self.images[index]
        target = self.masks[index]

        img = Image.open(img) #.convert('RGB')
        
        cv2_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        pre_target = Image.open(target)
        target = Image.open(target) #.convert('L')
        
        
        if self.transforms is not None:
            img, target, pre_target = self.transforms(img, target, pre_target)

        return img, cv2_img, target, pre_target

# Load the dataset with the preprocess transformation
test_dataset = VOCSegmentationCustom(
    root='path/to/VOC2012', year='2012', image_set='val', download=False, 
    transforms=lambda img,target,pre_target: (preprocess_img(img), preprocess_target(target), preprocess_target(pre_target))
)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)


voc_classes = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", 
    "bus", "car", "cat", "chair", "cow", 
    "dining table", "dog", "horse", "motorbike", "person", 
    "potted plant", "sheep", "sofa", "train", "tv/monitor"
]

all_texts = voc_classes 

##########################################################################################

import torch.nn as nn
sizes = [512, 384, 256]

layers_text = []
for i in range(len(sizes) - 2):
    layers_text.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
    layers_text.append(nn.BatchNorm1d(sizes[i + 1]))
    layers_text.append(nn.ReLU(inplace=True))
layers_text.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
text_projector = nn.Sequential(*layers_text)

size_img = [512, 256]
layers_img = []
# for i in range(len(sizes) - 2):
#     layers_img.append(nn.Linear(size_img[i], size_img[i + 1], bias=False))
#     layers_img.append(nn.BatchNorm1d(197))
#     layers_img.append(nn.ReLU(inplace=True))
layers_img.append(nn.Linear(size_img[-2], size_img[-1], bias=False))
text_projector = nn.Sequential(*layers_text).to(device)

image_projector = nn.Sequential(*layers_img).to(device)


# model_path = '/home/samyakr2/Redundancy/DualCoOp/output/coco_with_SSL_90_0.003R/model_best.pth.tar'
model_path = '/home/samyakr2/Redundancy/DualCoOp/output/coco_with_SSL_90_0.002R/model_best.pth.tar'
state_dict = torch.load(model_path)

projector_weights_text = {}
projector_weights_img = {}

for kays in state_dict['state_dict'].keys():
    if 'text_projector' in kays:
        projector_weights_text[kays[15:]] = state_dict['state_dict'][kays]
    if 'image_projector' in kays:
        projector_weights_img[kays[16:]] = state_dict['state_dict'][kays]


text_projector.load_state_dict(projector_weights_text)
image_projector.load_state_dict(projector_weights_img)


##########################################################################################

import numpy as np
from sklearn.metrics import jaccard_score

def min_max_normalize(segmentation_map):
    """
    Normalize the segmentation map to the range [0, 1] using Min-Max normalization.
    """
    min_val = np.min(segmentation_map)
    max_val = np.max(segmentation_map)
    return (segmentation_map - min_val) / (max_val - min_val)

def compute_iou(ground_truth, binarized_map):
    """
    Compute the Intersection over Union (IoU) between the ground truth and the binarized segmentation map.
    """
    # Flatten the arrays to compute IoU
    return jaccard_score(ground_truth.flatten(), binarized_map.flatten(), average='binary')

def find_best_threshold(normalized_map, ground_truth, class_idx, step_size=0.01):
    """
    Perform a grid search to find the optimal threshold for a single class segmentation map.
    """
    best_threshold = 0
    best_iou = 0

    thresholds = np.arange(0, 1 + step_size, step_size)

    for threshold in thresholds:
        binarized_map = (normalized_map >= threshold).astype(np.uint8) #* class_idx

        ground_truth_clip = np.clip(ground_truth, 0,1)
        
        iou = compute_iou(ground_truth_clip, binarized_map)

        if iou > best_iou:
            best_iou = iou
            best_threshold = threshold

    return best_threshold, best_iou

def process_segmentation_map(segmentation_map, ground_truth, num_classes, step_size=0.01):
    """
    Apply the grid search strategy to find the best threshold for each class in the segmentation map.
    """
    best_thresholds = []
    best_ious = []

    for class_idx in num_classes:
        # Extract the segmentation map and ground truth for the current class
        class_segmentation_map = segmentation_map[:,:,:,int(class_idx)-1]
        class_ground_truth = (ground_truth==int(class_idx))* class_idx

        # Normalize the segmentation map
        normalized_map = min_max_normalize(class_segmentation_map)

        # Find the best threshold using grid search
        best_threshold, best_iou = find_best_threshold(normalized_map, class_ground_truth, class_idx, step_size)

        best_thresholds.append(best_threshold)
        best_ious.append(best_iou)

    return best_thresholds, best_ious



############################# CLIP SURGERY #############################

import torch
from tqdm import tqdm
import time
from IPython.display import display, clear_output

with torch.no_grad():
    i = 0
    iou_scores = []
    text_feats = clip.encode_text_with_prompt_ensemble(model, all_texts, device)
        
    for images, cv2_img, targets, pre_target in tqdm(test_loader):

        images = images.to(device)
        targets = targets.to(device)

        image_features = model.encode_image(images)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        
        text_features = text_feats
    

        features = image_features @ text_features.t()
        similarity = clip.clip_feature_surgery(image_features, text_features)
        similarity_map = clip.get_similarity_map(similarity[:, 1:, :], 224)

        targets = 255.*targets
        targets[targets == 255] = 0 ## only keep 0 for background, 255 was border, made it background        
        num_classes = torch.unique(targets)[1:].tolist()
        
        best_thresholds, best_ious = process_segmentation_map(similarity_map.cpu().numpy(), targets[0].cpu().numpy(), num_classes)
        iou_scores.append(np.mean(best_ious))
        running_mean = np.mean(iou_scores)
        # clear_output(wait=True)
        # print(f"Progress: {running_mean}%")
        # time.sleep(0.1)
        

print(np.mean(iou_scores))

############################# CLIP SURGERY + Ours #############################

# import torch
# from tqdm import tqdm
# import time
# from IPython.display import display, clear_output

# with torch.no_grad():
#     i = 0
#     iou_scores = []
#     text_feats = clip.encode_text_with_prompt_ensemble(model, all_texts, device)
        
#     for images, cv2_img, targets, pre_target in tqdm(test_loader):

#         images = images.to(device)
#         targets = targets.to(device)

        

#         image_features = model.encode_image(images)
#         image_features = image_features / image_features.norm(dim=1, keepdim=True)
#         img_feat = image_projector(image_features)
#         image_features = img_feat

#         text_features = text_projector(text_feats)
    
#         features = image_features @ text_features.t()
#         similarity = clip.clip_feature_surgery(image_features, text_features)
#         similarity_map = clip.get_similarity_map(similarity[:, 1:, :], 224)

#         targets = 255.*targets
#         targets[targets == 255] = 0 ## only keep 0 for background, 255 was border, made it background        
#         num_classes = torch.unique(targets)[1:].tolist()
        
#         best_thresholds, best_ious = process_segmentation_map(similarity_map.cpu().numpy(), targets[0].cpu().numpy(), num_classes)
#         iou_scores.append(np.mean(best_ious))
#         running_mean = np.mean(iou_scores)
        
# print(np.mean(iou_scores))



############################# Ours #############################

# import torch
# from tqdm import tqdm
# import time
# from IPython.display import display, clear_output

# with torch.no_grad():
#     i = 0
#     iou_scores = []
#     text_feats = clip.encode_text_with_prompt_ensemble(model, all_texts, device)
        
#     for images, cv2_img, targets, pre_target in tqdm(test_loader):

#         images = images.to(device)
#         targets = targets.to(device)

        

#         image_features = model.encode_image(images)
#         image_features = image_features / image_features.norm(dim=1, keepdim=True)
#         img_feat = image_projector(image_features)
#         image_features = img_feat

#         text_features = text_projector(text_feats)
    
#         features = image_features @ text_features.t()
#         # similarity = clip.clip_feature_surgery(image_features, text_features)
#         # similarity_map = clip.get_similarity_map(similarity[:, 1:, :], 224)
#         similarity_map = clip.get_similarity_map(features[:, 1:, :], 224)
        

#         targets = 255.*targets
#         targets[targets == 255] = 0 ## only keep 0 for background, 255 was border, made it background        
#         num_classes = torch.unique(targets)[1:].tolist()
        
#         best_thresholds, best_ious = process_segmentation_map(similarity_map.cpu().numpy(), targets[0].cpu().numpy(), num_classes)
#         iou_scores.append(np.mean(best_ious))
#         running_mean = np.mean(iou_scores)

# print(np.mean(iou_scores))
