import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from torchmetrics.classification import MulticlassJaccardIndex
import torch
import clip
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
import argparse
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser(description="A script to demonstrate command-line arguments.")
    
# Add the --arch argument
parser.add_argument('--arch', type=str, required=True, help="The architecture to use (e.g., 'rn101').")
# parser.add_argument('--thres', type=int, required=True, help="The architecture to use (e.g., 'rn101').")

# Parse the arguments
args = parser.parse_args()
BICUBIC = InterpolationMode.BICUBIC



class CustomImageDataset(Dataset):
    def __init__(self, root_dir, image_folder="images/val2017", label_folder="annotations/val2017", transform=None, target_transform=None):
        self.image_dir = os.path.join(root_dir, image_folder)
        self.label_dir = os.path.join(root_dir, label_folder)
        self.image_filenames = sorted(os.listdir(self.image_dir))
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        label_path = os.path.join(self.label_dir, self.image_filenames[idx].replace('.jpg', '.png'))

        image = Image.open(img_path).convert("RGB")
        label = Image.open(label_path).convert("L")  # Assume labels are grayscale images

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label


root_dir = '/home/samyakr2/CLIP_Surgery/coco_stuff/dataset/'
transform = transforms.Compose([
    transforms.Resize((448, 448), interpolation=BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
])
target_transform = transforms.Compose([
#     transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

dataset = CustomImageDataset(root_dir=root_dir, transform=transform, target_transform=target_transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)



file_path = '/home/samyakr2/CLIP_Surgery/coco_stuff/labels.txt'
with open(file_path, 'r') as file:
    line = file.readlines()
coco_stuff_labels = []
for idx in range (len(line)):
    coco_stuff_labels.append(line[idx].strip().split(':')[1].strip())



import torch.nn as nn
sizes = [512, 384, 256]
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _ = clip.load("CS-RN101", device=device)

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


if args.arch == 'CS':
    with torch.no_grad():
        text_feats = clip.encode_text_with_prompt_ensemble(model, coco_stuff_labels[1:], device)
        metric_iou = MulticlassJaccardIndex(num_classes=len(coco_stuff_labels), average=None, ignore_index=0).to('cpu')

        postive_pred_iou = []
        postive_only_iou = []
        
        for img, lab in tqdm(dataloader):
            img = img.to(device)
    #         print(lab.shape)
    #         break
            
            image_features = model.encode_image(img)
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_feats, dim=-1)
            
            similarity = clip.clip_feature_surgery(image_features, text_features)
            similarity_map = clip.get_similarity_map(similarity[:, 1:, :], lab.shape[-2:])
            similarity_map_argmax = similarity_map.argmax(-1)+1
            
            lab = lab*255 + 1
            lab[lab==256] = 0
            
            iou_scores = metric_iou(similarity_map_argmax.cpu(), lab[0])#.to(int))
            positive_iou = iou_scores[torch.unique(similarity_map_argmax).cpu()] ## Keep only postive classes for IoU, note postive means from our prediction, not GT
            postive_pred_iou.append(torch.nanmean(positive_iou).item())

            positive_iou_v2 = iou_scores[iou_scores>0]
            postive_only_iou.append(torch.nanmean(positive_iou_v2).item())

        print('positive_pred', np.nanmean(postive_pred_iou) * 100)
        print('postive_only_iou', np.nanmean(postive_only_iou)*100)

elif args.arch == 'Ours':
    with torch.no_grad():
        text_feats = clip.encode_text_with_prompt_ensemble(model, coco_stuff_labels[1:], device)
        metric_iou = MulticlassJaccardIndex(num_classes=len(coco_stuff_labels), average=None, ignore_index=0).to('cpu')

        postive_pred_iou = []
        postive_only_iou = []
        
        for img, lab in tqdm(dataloader):
            img = img.to(device)

            image_features = model.encode_image(img)
            image_features = F.normalize(image_features, dim=-1)
            img_feat = image_projector(image_features)
            image_features = img_feat

            text_feat = text_projector(text_feats)
            text_features = F.normalize(text_feat, dim=-1)
            

            features = image_features @ text_features.t()
            similarity_map = clip.get_similarity_map(features[:, 1:, :], lab.shape[-2:])
            similarity_map_argmax = similarity_map.argmax(-1)+1
            
            lab = lab*255 + 1
            lab[lab==256] = 0
            
            iou_scores = metric_iou(similarity_map_argmax.cpu(), lab[0])#.to(int))
            positive_iou = iou_scores[torch.unique(similarity_map_argmax).cpu()] ## Keep only postive classes for IoU, note postive means from our prediction, not GT
            postive_pred_iou.append(torch.nanmean(positive_iou).item())

            positive_iou_v2 = iou_scores[iou_scores>0]
            postive_only_iou.append(torch.nanmean(positive_iou_v2).item())

        print('positive_pred', np.nanmean(postive_pred_iou) * 100)
        print('postive_only_iou', np.nanmean(postive_only_iou)*100)


elif args.arch == 'CS_Ours':
    with torch.no_grad():
        text_feats = clip.encode_text_with_prompt_ensemble(model, coco_stuff_labels[1:], device)
        metric_iou = MulticlassJaccardIndex(num_classes=len(coco_stuff_labels), average=None, ignore_index=0).to('cpu')

        postive_pred_iou = []
        postive_only_iou = []
        
        for img, lab in tqdm(dataloader):
            img = img.to(device)

            image_features = model.encode_image(img)
            image_features = F.normalize(image_features, dim=-1)
            img_feat = image_projector(image_features)
            image_features = img_feat

            text_feat = text_projector(text_feats)
            text_features = F.normalize(text_feat, dim=-1)
            

            # features = image_features @ text_features.t()
            similarity = clip.clip_feature_surgery(image_features, text_features)
            similarity_map = clip.get_similarity_map(similarity[:, 1:, :], lab.shape[-2:])
            similarity_map_argmax = similarity_map.argmax(-1)+1
            
            lab = lab*255 + 1
            lab[lab==256] = 0
            
            iou_scores = metric_iou(similarity_map_argmax.cpu(), lab[0])#.to(int))
            positive_iou = iou_scores[torch.unique(similarity_map_argmax).cpu()] ## Keep only postive classes for IoU, note postive means from our prediction, not GT
            postive_pred_iou.append(torch.nanmean(positive_iou).item())

            positive_iou_v2 = iou_scores[iou_scores>0]
            postive_only_iou.append(torch.nanmean(positive_iou_v2).item())

        print('positive_pred', np.nanmean(postive_pred_iou) * 100)
        print('postive_only_iou', np.nanmean(postive_only_iou)*100)