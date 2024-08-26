import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import scipy.io
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"
parser = argparse.ArgumentParser(description="A script to demonstrate command-line arguments.")
    
parser.add_argument('--arch', type=str, required=True, help="The architecture to use (e.g., 'rn101').")
parser.add_argument('--thres', type=int, required=True, help="The architecture to use (e.g., 'rn101').")

args = parser.parse_args()

class PascalVOCContextDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (string): Directory with all the images and annotations.
            split (string): 'train', 'val', or 'test'.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # Load the file paths
        self.image_dir = os.path.join(self.root_dir, 'JPEGImages')
        self.mask_dir = os.path.join(self.root_dir, 'context/trainval')
        
        split_file = os.path.join(self.root_dir, 'ImageSets', 'Segmentation', f'{split}.txt')
        
        with open(split_file, 'r') as file:
            self.image_names = file.read().splitlines()

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        # print('samyak', self.image_names[idx])
        
        img_name = os.path.join(self.image_dir, self.image_names[idx] + '.jpg')
        mask_name = os.path.join(self.mask_dir, self.image_names[idx] + '.mat')

        file_path = '/home/samyakr2/CLIP_Surgery/path/to/VOC2012/VOCdevkit/VOC2012/context/labels_map_dict.json'
        with open(file_path, 'r') as file:
            data_dict = json.load(file)

        image = Image.open(img_name).convert('RGB')
        mask = scipy.io.loadmat(mask_name)['LabelMap']
        unique_out = np.unique(mask)
        for cal in unique_out:
            mask[mask==cal] = data_dict.get(str(cal), 0)

        mask = Image.fromarray(mask.astype(np.uint8))
        
        # if self.transform:
        image = self.transform(image)
        to_tensor = transforms.ToTensor()
        mask = to_tensor(mask) 

        return image , mask

# Example of data transformation
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    
])
# transform = transforms.Compose([
#     # transforms.Resize((512, 512)),
#     transforms.ToTensor()
# ])
# Example of creating the dataset
dataset = PascalVOCContextDataset(root_dir='/home/samyakr2/CLIP_Surgery/path/to/VOC2012/VOCdevkit/VOC2012', split='context_gt', transform = transform)

# Example of creating the dataloader
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)



import torch.nn as nn
# sizes = [512, 384, 256] ## RN101
sizes = [1024, 768, 512] ## RN50

layers_text = []
for i in range(len(sizes) - 2):
    layers_text.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
    layers_text.append(nn.BatchNorm1d(sizes[i + 1]))
    layers_text.append(nn.ReLU(inplace=True))
layers_text.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
text_projector = nn.Sequential(*layers_text)

# size_img = [512, 256] ## RN101
size_img = [1024, 512]  ## RN50
layers_img = []
# for i in range(len(sizes) - 2):
#     layers_img.append(nn.Linear(size_img[i], size_img[i + 1], bias=False))
#     layers_img.append(nn.BatchNorm1d(197))
#     layers_img.append(nn.ReLU(inplace=True))
layers_img.append(nn.Linear(size_img[-2], size_img[-1], bias=False))
text_projector = nn.Sequential(*layers_text).to(device)

image_projector = nn.Sequential(*layers_img).to(device)


# model_path = '/home/samyakr2/Redundancy/DualCoOp/output/coco_with_SSL_90_0.003R/model_best.pth.tar'
# model_path = '/home/samyakr2/Redundancy/DualCoOp/output/coco_with_SSL_90_0.002R/model_best.pth.tar'
# model_path = '/home/samyakr2/Redundancy/DualCoOp/output/voc_with_SSL_90%/model_best.pth.tar'
model_path = '/home/samyakr2/Redundancy/DualCoOp/output/coco_RN50_SSL_90%_0.002R/model_best.pth.tar'


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




import clip 
from torchmetrics.classification import MulticlassJaccardIndex
import torch.nn.functional as F
from tqdm import tqdm


voc_context_classes = ['background','aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'table', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor', 'bag', 'bed', 'bench', 'book', 'building', 'cabinet', 'ceiling', 'cloth', 'computer', 'cup', 'door', 'fence', 'floor', 'flower', 'food', 'grass', 'ground', 'keyboard', 'light', 'mountain', 'mouse', 'curtain', 'platform', 'sign', 'plate', 'road', 'rock', 'shelves', 'sidewalk', 'sky', 'snow', 'bedclothes', 'track', 'tree', 'truck', 'wall', 'water', 'window', 'wood']

threshold = 0.9
# model, _ = clip.load("CS-RN101", device=device)

model, _ = clip.load("CS-RN50", device=device)

if args.arch == 'CS':
    with torch.no_grad():
        text_feats = clip.encode_text_with_prompt_ensemble(model, voc_context_classes[1:], device)
        text_features = text_feats

        metric_iou = MulticlassJaccardIndex(num_classes=len(voc_context_classes), average=None, ignore_index=0).to('cpu')
        postive_pred_iou = []
        postive_only_iou = []

        for images, targets in tqdm(dataloader):
            images = images.to(device)
            targets = targets#.to(device)
            mask_shape = targets.shape[-2:]

            image_features = model.encode_image(images)
            image_features = F.normalize(image_features, dim=-1)
            # text_features = F.normalize(text_feats, dim=-1)
            
            similarity = clip.clip_feature_surgery( image_features, text_features)
            similarity_map = clip.get_similarity_map(similarity[:, 1:, :], mask_shape)
            simialrity_map_argmax = similarity_map.argmax(dim = -1) + 1

            if args.thres == 1:
                logits_soft_max = (100*similarity_map).softmax(dim=-1).max(dim=-1)[0]
                # print(logits_soft_max.min(), logits_soft_max.mean(), logits_soft_max.max())
                simialrity_map_argmax[logits_soft_max < threshold] = 0 ## threshold to ignore background

            iou_scores = metric_iou(simialrity_map_argmax.cpu(), (targets[0]*255).to(int))
            positive_iou = iou_scores[torch.unique(simialrity_map_argmax).cpu()] ## Keep only postive classes for IoU, note postive means from our prediction, not GT
            postive_pred_iou.append(torch.nanmean(positive_iou).item())
            
            positive_iou_v2 = iou_scores[iou_scores>0]
            postive_only_iou.append(torch.nanmean(positive_iou_v2).item())


    print('positive_pred', np.nanmean(postive_pred_iou) * 100)
    print('postive_only_iou', np.nanmean(postive_only_iou)*100)

elif args.arch == 'Ours':
    with torch.no_grad():
        text_feats = clip.encode_text_with_prompt_ensemble(model, voc_context_classes[1:], device)
        # text_features = text_feats

        metric_iou = MulticlassJaccardIndex(num_classes=len(voc_context_classes), average=None, ignore_index=0).to('cpu')
        postive_pred_iou = []
        postive_only_iou = []
        
        for images, targets in tqdm(dataloader):
            images = images.to(device)
            targets = targets#.to(device)
            mask_shape = targets.shape[-2:]

            image_features = model.encode_image(images)
            image_features = F.normalize(image_features, dim=-1)
            img_feat = image_projector(image_features)
            image_features = img_feat
            
            text_feat = text_projector(text_feats)
            text_features = F.normalize(text_feat, dim=-1) 

            features = image_features @ text_features.t()
            # similarity = clip.clip_feature_surgery( image_features, text_features)
            # similarity_map = clip.get_similarity_map(similarity[:, 1:, :], mask_shape)
            similarity_map = clip.get_similarity_map(features[:, 1:, :], mask_shape)
            simialrity_map_argmax = similarity_map.argmax(dim = -1) +  1


            # similarity = clip.clip_feature_surgery( image_features, text_features)
            # similarity_map = clip.get_similarity_map(similarity[:, 1:, :], mask_shape)
            # simialrity_map_argmax = similarity_map.argmax(dim = -1) + 1

            if args.thres == 1:
                logits_soft_max = (100*similarity_map).softmax(dim=-1).max(dim=-1)[0]
                # print(logits_soft_max.min(), logits_soft_max.mean(), logits_soft_max.max())
                simialrity_map_argmax[logits_soft_max < threshold] = 0 ## threshold to ignore background

            iou_scores = metric_iou(simialrity_map_argmax.cpu(), (targets[0]*255).to(int))
            positive_iou = iou_scores[torch.unique(simialrity_map_argmax).cpu()] ## Keep only postive classes for IoU, note postive means from our prediction, not GT
            postive_pred_iou.append(torch.nanmean(positive_iou).item())
            
            positive_iou_v2 = iou_scores[iou_scores>0]
            postive_only_iou.append(torch.nanmean(positive_iou_v2).item())

    print('positive_pred', np.nanmean(postive_pred_iou) * 100)
    print('postive_only_iou', np.nanmean(postive_only_iou)*100)

elif args.arch == 'CS_Ours':
    with torch.no_grad():
        text_feats = clip.encode_text_with_prompt_ensemble(model, voc_context_classes[1:], device)
        # text_features = text_feats

        metric_iou = MulticlassJaccardIndex(num_classes=len(voc_context_classes), average=None, ignore_index=0).to('cpu')
        postive_pred_iou = []
        postive_only_iou = []
        
        for images, targets in tqdm(dataloader):
            images = images.to(device)
            targets = targets#.to(device)
            mask_shape = targets.shape[-2:]

            image_features = model.encode_image(images)
            image_features = F.normalize(image_features, dim=-1)
            img_feat = image_projector(image_features)
            image_features = img_feat
            
            text_feat = text_projector(text_feats)
            text_features = F.normalize(text_feat, dim=-1) 

            # features = image_features @ text_features.t()
            similarity = clip.clip_feature_surgery( image_features, text_features)
            similarity_map = clip.get_similarity_map(similarity[:, 1:, :], mask_shape)
            # similarity_map = clip.get_similarity_map(features[:, 1:, :], mask_shape)
            simialrity_map_argmax = similarity_map.argmax(dim = -1) +  1


            # similarity = clip.clip_feature_surgery( image_features, text_features)
            # similarity_map = clip.get_similarity_map(similarity[:, 1:, :], mask_shape)
            # simialrity_map_argmax = similarity_map.argmax(dim = -1) + 1

            if args.thres == 1:
                logits_soft_max = (100*similarity_map).softmax(dim=-1).max(dim=-1)[0]
                # print(logits_soft_max.min(), logits_soft_max.mean(), logits_soft_max.max())
                simialrity_map_argmax[logits_soft_max < threshold] = 0 ## threshold to ignore background

            iou_scores = metric_iou(simialrity_map_argmax.cpu(), (targets[0]*255).to(int))
            positive_iou = iou_scores[torch.unique(simialrity_map_argmax).cpu()] ## Keep only postive classes for IoU, note postive means from our prediction, not GT
            postive_pred_iou.append(torch.nanmean(positive_iou).item())
            
            positive_iou_v2 = iou_scores[iou_scores>0]
            postive_only_iou.append(torch.nanmean(positive_iou_v2).item())

    print('positive_pred', np.nanmean(postive_pred_iou) * 100)
    print('postive_only_iou', np.nanmean(postive_only_iou)*100)
