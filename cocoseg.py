import torch
from torchvision.transforms import functional as F
from pycocotools.coco import COCO

from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import clip
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC


# Load the COCO 2017 test set
coco = COCO('/home/samyakr2/multilabel/data/coco/annotations/instances_val2017.json')
image_dir = '/home/samyakr2/multilabel/data/coco/val2017/'

device = "cuda" if torch.cuda.is_available() else "cpu"




model, _ = clip.load("CS-RN101", device=device)
preprocess_img =  Compose([Resize((224, 224), interpolation=BICUBIC), ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])

preprocess_target =  Compose([Resize((224, 224), interpolation=BICUBIC), ToTensor()])



image_ids = coco.getImgIds()

cat_ids = coco.getCatIds()
cats = coco.loadCats(cat_ids)
cat_id_to_name = {cat['id']: cat['name'] for cat in cats}




import numpy as np
from sklearn.metrics import jaccard_score
import time


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

def process_segmentation_map(segmentation_map, complete_mask, num_classes, step_size=0.01):
    """
    Apply the grid search strategy to find the best threshold for each class in the segmentation map.
    """
    best_thresholds = []
    best_ious = []

    for classes in num_classes:


        class_ground_truth = (complete_mask==int(classes))* classes

        pred_mask = segmentation_map[:,:,:,classes-1]
        class_ground_truth =  np.expand_dims(class_ground_truth, axis=0)
        
        normalized_map = min_max_normalize(pred_mask)

        
        # Find the best threshold using grid search
        best_threshold, best_iou = find_best_threshold(normalized_map, class_ground_truth, step_size)

        best_thresholds.append(best_threshold)
        best_ious.append(best_iou)

    return best_thresholds, best_ious



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



classnames = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                           "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                           "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
                           "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
                           "kite",
                           "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
                           "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
                           "orange",
                           "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
                           "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
                           "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                           "teddy bear", "hair drier", "toothbrush"]

all_texts = classnames 

with torch.no_grad():
    text_feats = clip.encode_text_with_prompt_ensemble(model, all_texts, device)

#################################################################### CLIP SURGERY ####################################################
# with torch.no_grad():
#     iou_scores = []
#     for img_id in tqdm(image_ids):
#         img_info = coco.loadImgs(img_id)[0]
#         img_path = image_dir + img_info['file_name']
        
#         img = Image.open(img_path).convert('RGB')
#         img = preprocess_img(img).to(device)
#         # plt.imshow(img.permute(1,2,0).cpu())
#         # plt.show()
#         img = img.unsqueeze(0)
        
#         image_features = model.encode_image(img)
#         text_features = text_feats
#         features = image_features @ text_features.t()
#         similarity = clip.clip_feature_surgery(image_features, text_features)

#         similarity_map = clip.get_similarity_map(similarity[:, 1:, :], 224)
        
#         true_ann_ids = coco.getAnnIds(imgIds=img_id)
#         true_anns = coco.loadAnns(true_ann_ids)
#         iou_per_img = []

#         if len(true_anns)>0:

#             complete_mask = np.zeros((224, 224), dtype=np.uint8)
#             for annotation in true_anns:
#                 mask = cv2.resize(coco.annToMask(annotation), (224,224))
#                 category_id = annotation['category_id']
#                 our_id = classnames.index(cat_id_to_name[category_id])
#                 complete_mask[mask == 1] = our_id +1 
#             num_classes = np.unique(complete_mask)[1:]
#             best_thresholds, best_ious = process_segmentation_map(similarity_map.cpu().numpy(), complete_mask, num_classes)
#             iou_scores.append(np.mean(best_ious))
#             running_mean = np.nanmean(iou_scores)
#             # clear_output(wait=True)
#             print('running iou: ', running_mean)
#         # time.sleep(0.1)
# print(np.mean(iou_scores))


#################################################################### Ours ####################################################

# with torch.no_grad():
#     iou_scores = []
#     for img_id in tqdm(image_ids):
#         img_info = coco.loadImgs(img_id)[0]
#         img_path = image_dir + img_info['file_name']
        
#         img = Image.open(img_path).convert('RGB')
#         img = preprocess_img(img).to(device)
#         # plt.imshow(img.permute(1,2,0).cpu())
#         # plt.show()
#         img = img.unsqueeze(0)
        
#         image_features = model.encode_image(img)
        
#         image_features = image_features / image_features.norm(dim=1, keepdim=True)
#         img_feat = image_projector(image_features)
#         image_features = img_feat

#         text_features = text_projector(text_feats)
    
#         features = image_features @ text_features.t()

#         # similarity = clip.clip_feature_surgery(image_features, text_features)
#         # similarity_map = clip.get_similarity_map(similarity[:, 1:, :], 224)
#         similarity_map = clip.get_similarity_map(features[:, 1:, :], 224)
        
        
#         true_ann_ids = coco.getAnnIds(imgIds=img_id)
#         true_anns = coco.loadAnns(true_ann_ids)
#         iou_per_img = []

#         if len(true_anns)>0:
#             complete_mask = np.zeros((224, 224), dtype=np.uint8)
#             for annotation in true_anns:
#                 mask = cv2.resize(coco.annToMask(annotation), (224,224))
#                 category_id = annotation['category_id']
#                 our_id = classnames.index(cat_id_to_name[category_id])
#                 complete_mask[mask == 1] = our_id +1 
#             num_classes = np.unique(complete_mask)[1:]
#             best_thresholds, best_ious = process_segmentation_map(similarity_map.cpu().numpy(), complete_mask, num_classes)
#             iou_scores.append(np.mean(best_ious))
#             running_mean = np.nanmean(iou_scores)
#             print("running_mean: ", running_mean)

# print(running_mean)


#################################################################### Ours + CS ####################################################

with torch.no_grad():
    iou_scores = []
    for img_id in tqdm(image_ids):
        img_info = coco.loadImgs(img_id)[0]
        img_path = image_dir + img_info['file_name']
        
        img = Image.open(img_path).convert('RGB')
        img = preprocess_img(img).to(device)
        # plt.imshow(img.permute(1,2,0).cpu())
        # plt.show()
        img = img.unsqueeze(0)
        
        image_features = model.encode_image(img)
        
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        img_feat = image_projector(image_features)
        image_features = img_feat

        text_features = text_projector(text_feats)
    
        features = image_features @ text_features.t()

        similarity = clip.clip_feature_surgery(image_features, text_features)
        similarity_map = clip.get_similarity_map(similarity[:, 1:, :], 224)
        # similarity_map = clip.get_similarity_map(features[:, 1:, :], 224)
        
        
        true_ann_ids = coco.getAnnIds(imgIds=img_id)
        true_anns = coco.loadAnns(true_ann_ids)
        iou_per_img = []

        if len(true_anns)>0:
            complete_mask = np.zeros((224, 224), dtype=np.uint8)
            for annotation in true_anns:
                mask = cv2.resize(coco.annToMask(annotation), (224,224))
                category_id = annotation['category_id']
                our_id = classnames.index(cat_id_to_name[category_id])
                complete_mask[mask == 1] = our_id +1 
            num_classes = np.unique(complete_mask)[1:]
            best_thresholds, best_ious = process_segmentation_map(similarity_map.cpu().numpy(), complete_mask, num_classes)
            iou_scores.append(np.mean(best_ious))
            running_mean = np.nanmean(iou_scores)
            print("running_mean: ", running_mean)

print(running_mean)
