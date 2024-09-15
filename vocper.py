from os.path import join
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.transforms import transforms


import clip
from torchvision.datasets import VOCSegmentation
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC


from torchmetrics.classification import MulticlassJaccardIndex
import argparse
parser = argparse.ArgumentParser(description="A script to demonstrate command-line arguments.")
    
# Add the --arch argument
parser.add_argument('--arch', type=str, required=True, help="The architecture to use (e.g., 'rn101').")
parser.add_argument('--thres', type=int, required=True, help="The architecture to use (e.g., 'rn101').")
parser.add_argument('--model_arch', type=str, required=True, help="The architecture to use (e.g., 'rn101').")

# Parse the arguments
args = parser.parse_args()



OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)


device = "cuda" if torch.cuda.is_available() else "cpu"
if args.model_arch == 'CS-RN101':
    sizes = [512, 384, 256] ## RN101
    model, _ = clip.load("CS-RN101", device=device)
    size_img = [512, 256] ## RN101

elif args.model_arch == 'CS-RN50':
    sizes = [1024, 768, 512] ## RN101
    model, _ = clip.load("CS-RN50", device=device)
    size_img = [1024, 512]  ## RN50

elif args.model_arch == 'RN50':
    sizes = [1024, 768, 512] ## RN101
    model, _ = clip.load("RN50", device=device)
    size_img = [1024, 512]  ## RN50

elif args.model_arch == 'RN101':
    sizes = [512, 384, 256] ## RN101
    model, _ = clip.load("RN101", device=device)
    size_img = [512, 256]  ## RN50

preprocess_img =  Compose([Resize((224, 224), interpolation=BICUBIC), ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])


preprocess_target =  Compose([Resize((224, 224), interpolation=BICUBIC), ToTensor()])

class PascalVOC(Dataset):
    CLASSES = ('background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'table', 'dog', 'horse', 'motorbike', 'person', 'plant', 'sheep', 'sofa', 'train', 'monitor')

    PALETTE = torch.tensor([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
                           [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
                           [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
                           [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0],
                           [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]], dtype=torch.uint8)

    def __init__(self,
                 root,
                 split='train',
                 transform=None,
                 only_image=False,
                 aug=True,
                 nclass=None,
                 only_mask=False,
                 split_file=None,
                 ignore_index=-1,
                 return_path=False):
        super(PascalVOC, self).__init__()
        self.nclass = nclass if nclass is not None else self.PALETTE.shape[0]
        self.only_image = only_image
        self.only_mask = only_mask
        self.split = split
        self.return_path = return_path
        self.ignore_index = ignore_index
        assert self.split in ['train', 'trainval', 'val'], f'{self.split} must be in ["train", "trainval", "val"]'
        self.split = 'trainaug' if aug and (self.split == 'train') else self.split
        self.root = join(root, 'VOCdevkit/VOC2012/') if split_file is None else root
        self.transform = transform



        self.anno_type = 'SegmentationClassAug' if aug else 'SegmentationClass'
        txt_file = join(self.root, split_file) if split_file is not None \
            else join(self.root, 'ImageSets', 'Segmentation', self.split + '.txt')

        self.samples = []
        with open(txt_file) as f:
            samples_tmp = f.readlines()
        samples_tmp = list(map(lambda elem: elem.strip(), samples_tmp))
        self.samples.extend(samples_tmp)

        samples_list = []
        self.image_files = []
        self.label_files = []
        for sample in self.samples:
            if split_file is not None:
                img = f'{str(sample)}.jpg'
                label = f'{str(sample)}.png'
            else:
                img = f'JPEGImages/{str(sample)}.jpg'
                label = f'{self.anno_type}/{str(sample)}.png'
            self.image_files.append(join(self.root, img))
            self.label_files.append(join(self.root, label))

    def __len__(self):
        return len(self.image_files)


    def __getitem__(self, idx):

        image_path = self.image_files[idx]
        label_path = self.label_files[idx]

        img, msk = Image.open(image_path).convert("RGB"), Image.open(label_path).convert("RGB")

        # if self.img_transform is not None:
        images, rgb_target = self.transform(img, msk)

        h, w = rgb_target.shape[1:]
        one_hot_seg_mask = self.ignore_index * torch.ones((h, w), dtype=torch.long)
        for color_idx in range(self.nclass):
            idx = (rgb_target == self.PALETTE[color_idx].unsqueeze(-1).unsqueeze(-1))
            valid_idx = (idx.sum(0) == 3)#.unsqueeze(0)
            one_hot_seg_mask[valid_idx] = color_idx

        if self.return_path:
            path_to_img_msk = {}
            path_to_img_msk["img_path"] = image_path
            path_to_img_msk["label_path"] = label_path
            return images, one_hot_seg_mask, path_to_img_msk

        return images, one_hot_seg_mask


class ToTensorMask(nn.Module):
    def __init__(self):
        super(ToTensorMask, self).__init__()

    def forward(self, mask):
        return torch.as_tensor(np.array(mask), dtype=torch.int64).permute(2, 0, 1)


class SegmentationTransforms(object):
    def __init__(self, size, img_transforms=None, resize_mask=False):
        self.img_transforms = img_transforms if img_transforms is not None else transforms.Compose([
            transforms.Resize(size=size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(OPENAI_DATASET_MEAN, OPENAI_DATASET_STD),
        ])
        self.mask_transforms = transforms.Compose([
            transforms.Resize(size=size) if resize_mask else nn.Identity(),
            ToTensorMask(),
        ])

    def __call__(self, image, mask):
        return self.img_transforms(image), self.mask_transforms(mask)


root_path_voc = 'path/to/VOC2012'
dataset = PascalVOC(root=root_path_voc, split='val',
                        transform=SegmentationTransforms((448, 448), resize_mask=False),
                        aug=False, only_image=False, only_mask=False, ignore_index=-1)

test_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=8)

import torch.nn as nn


layers_text = []
for i in range(len(sizes) - 2):
    layers_text.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
    layers_text.append(nn.BatchNorm1d(sizes[i + 1]))
    layers_text.append(nn.ReLU(inplace=True))
layers_text.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
text_projector = nn.Sequential(*layers_text)

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
# model_path = '/home/samyakr2/Redundancy/DualCoOp/output/coco_RN50_SSL_90%_0.002R/model_best.pth.tar'
# model_path = '/home/samyakr2/Redundancy/DualCoOp/output/voc_RN50_SSL_90_0.002R/model_best.pth.tar'
# model_path = '/home/samyakr2/Redundancy/DualCoOp/output/voc2007-DualCoop-RN101e51-0.02R/model_best.pth.tar'
# model_path = '/home/samyakr2/Redundancy/DualCoOp/output/voc2007-DualCoop-RN101e51-1e-05R/model_best.pth.tar'

# model_path = '/home/samyakr2/Redundancy/DualCoOp/output/coco_RN101_SSL_90_0.004R/model_best.pth.tar'
# model_path = '/home/samyakr2/Redundancy/DualCoOp/output/voc2007-DualCoop-RN50e51-0.01R/model_best.pth.tar'
# model_path = '/home/samyakr2/Redundancy/DualCoOp/output/coco_RN50_SSL_90_0.004R/model_best.pth.tar'
# model_path = '/home/samyakr2/Redundancy/DualCoOp/output/coco-DualCoop-RN50-cosine-bs32-e51/model_best.pth.tar'
# model_path = '/home/samyakr2/Redundancy/DualCoOp/output/coco-DualCoop-RN50e51-0.0008R/model_best.pth.tar'
# model_path= '/home/samyakr2/Redundancy/DualCoOp/output/voc2007-DualCoop-RN101SSL_p0.9-0.03R/model_best.pth.tar'
# model_path = '/home/samyakr2/Redundancy/DualCoOp/output/voc2007-DualCoop-RN101SSL_p1.0-0.005R/model_best.pth.tar'
# model_path = '/home/samyakr2/Redundancy/DualCoOp/output/voc2007-DualCoop-RN101SSL_p0.9-0.005R/model_best.pth.tar'
# model_path = '/home/samyakr2/Redundancy/DualCoOp/output/voc2007-DualCoop-RN101SSL_p0.9-0.01R/model_best.pth.tar'
# model_path = '/home/samyakr2/Redundancy/DualCoOp/output/voc2007-DualCoop-RN101SSL_p1.0-0.0005R/model_best.pth.tar'
# model_path = '/home/samyakr2/Redundancy/DualCoOp/output/voc2007-DualCoop-RN101SSL_posneg_p0.9-0.01R/model_best.pth.tar'
# model_path = '/home/samyakr2/Redundancy/DualCoOp/output/voc2007-DualCoop-RN101SSL_posneg_p1.0-0.01R/model_best.pth.tar'
# model_path = '/home/samyakr2/Redundancy/DualCoOp/output/voc2007-DualCoop-RN101SSL_posneg_p1.0-0.05R/model_best.pth.tar'
model_path = '/home/samyakr2/Redundancy/DualCoOp/output/voc2007-DualCoop-RN101SSL_posneg_p1.0-0.15R/model_best.pth.tar'


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


import torch
from tqdm import tqdm
import time 
from IPython.display import display, clear_output
import torch.nn.functional as F


voc_classes_background = [ "background",
    "aeroplane", "bicycle", "bird", "boat", "bottle", 
    "bus", "car", "cat", "chair", "cow", 
    "dining table", "dog", "horse", "motorbike", "person", 
    "potted plant", "sheep", "sofa", "train", "tv/monitor"
]

threshold = 0.2
if args.arch == 'CS':
    with torch.no_grad():
        i = 0
        iou_scores = []
        
        text_feats = clip.encode_text_with_prompt_ensemble(model, voc_classes_background[1:], device)
        metric_iou = MulticlassJaccardIndex(num_classes=len(voc_classes_background), average=None, ignore_index=0).to('cpu')
        postive_pred_iou = []
        postive_only_iou = []

        for images, targets in tqdm(test_loader):

            images = images.to(device)
            targets = targets#.to(device)
            mask_shape = targets.shape[-2:]

            image_features = model.encode_image(images)
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_feats, dim=-1)
            
            

            # features = 100.0 * image_features @ text_features.t()
            similarity = clip.clip_feature_surgery( image_features, text_features)
            similarity_map = clip.get_similarity_map(similarity[:, 1:, :], mask_shape)
            simialrity_map_argmax = similarity_map.argmax(dim = -1) +  1
            targets[targets == -1] = 0
            
            if args.thres == 1:
                logits_soft_max = (100*similarity_map).softmax(dim=-1).max(dim=-1)[0]
                simialrity_map_argmax[logits_soft_max < threshold] = 0 ## threshold to ignore background
            
            iou_scores = metric_iou(simialrity_map_argmax.cpu(), targets)#.to(int))
            positive_iou = iou_scores[torch.unique(simialrity_map_argmax).cpu()] ## Keep only postive classes for IoU, note postive means from our prediction, not GT
            postive_pred_iou.append(torch.nanmean(positive_iou).item())
            
            positive_iou_v2 = iou_scores[iou_scores>0]
            postive_only_iou.append(torch.nanmean(positive_iou_v2).item())

    print('positive_pred', np.nanmean(postive_pred_iou) * 100)
    print('postive_only_iou', np.nanmean(postive_only_iou)*100)

elif args.arch == 'CLIP_VV':
    with torch.no_grad():
        iou_scores = []
        text_feats = clip.encode_text_with_prompt_ensemble(model, voc_classes_background[1:], device)
        metric_iou = MulticlassJaccardIndex(num_classes=len(voc_classes_background), average=None, ignore_index=0).to('cpu')
        postive_pred_iou = []
        postive_only_iou = []
        
        for images, targets in tqdm(test_loader):

            images = images.to(device)
            targets = targets#.to(device)
            mask_shape = targets.shape[-2:]


            image_features = model.encode_image(images)
            image_features = F.normalize(image_features, dim=-1)
            
            text_features = F.normalize(text_feats, dim=-1)
            
            
            features = image_features @ text_features.t()
            similarity_map = clip.get_similarity_map(features[:, 1:, :], mask_shape)
            simialrity_map_argmax = similarity_map.argmax(dim = -1) +  1

            targets[targets == -1] = 0 ## making edges background
            if args.thres == 1:
                logits_soft_max = (100*similarity_map).softmax(dim=-1).max(dim=-1)[0]
                simialrity_map_argmax[logits_soft_max < threshold] = 0 ## threshold to ignore background

            iou_scores = metric_iou(simialrity_map_argmax.cpu(), targets)#.to(int))
            positive_iou = iou_scores[torch.unique(simialrity_map_argmax).cpu()] ## Keep only postive classes for IoU, note postive means from our prediction, not GT
            postive_pred_iou.append(torch.nanmean(positive_iou).item())
            
            positive_iou_v2 = iou_scores[iou_scores>0]
            postive_only_iou.append(torch.nanmean(positive_iou_v2).item())

    print('positive_pred', np.nanmean(postive_pred_iou) * 100)
    print('postive_only_iou', np.nanmean(postive_only_iou)*100)


elif args.arch == 'Ours':
    with torch.no_grad():
        iou_scores = []
        text_feats = clip.encode_text_with_prompt_ensemble(model, voc_classes_background[1:], device)
        metric_iou = MulticlassJaccardIndex(num_classes=len(voc_classes_background), average=None, ignore_index=0).to('cpu')
        postive_pred_iou = []
        postive_only_iou = []
        
        for images, targets in tqdm(test_loader):

            images = images.to(device)
            targets = targets#.to(device)
            mask_shape = targets.shape[-2:]


            image_features = model.encode_image(images)
            image_features = F.normalize(image_features, dim=-1)
            img_feat = image_projector(image_features)
            image_features = img_feat
            # print('img_feat', img_feat.shape)
            # break


            text_feat = text_projector(text_feats)

            text_features = F.normalize(text_feat, dim=-1)
            
            
            features = image_features @ text_features.t()
            similarity_map = clip.get_similarity_map(features[:, 1:, :], mask_shape)
            simialrity_map_argmax = similarity_map.argmax(dim = -1) +  1

            targets[targets == -1] = 0 ## making edges background
            if args.thres == 1:
                logits_soft_max = (100*similarity_map).softmax(dim=-1).max(dim=-1)[0]
                simialrity_map_argmax[logits_soft_max < threshold] = 0 ## threshold to ignore background

            iou_scores = metric_iou(simialrity_map_argmax.cpu(), targets)#.to(int))
            positive_iou = iou_scores[torch.unique(simialrity_map_argmax).cpu()] ## Keep only postive classes for IoU, note postive means from our prediction, not GT
            postive_pred_iou.append(torch.nanmean(positive_iou).item())
            
            positive_iou_v2 = iou_scores[iou_scores>0]
            postive_only_iou.append(torch.nanmean(positive_iou_v2).item())

    print('positive_pred', np.nanmean(postive_pred_iou) * 100)
    print('postive_only_iou', np.nanmean(postive_only_iou)*100)


elif args.arch == 'CS_Ours':
    with torch.no_grad():
        iou_scores = []
        text_feats = clip.encode_text_with_prompt_ensemble(model, voc_classes_background[1:], device)
        metric_iou = MulticlassJaccardIndex(num_classes=len(voc_classes_background), average=None, ignore_index=0).to('cpu')
        postive_pred_iou = []
        postive_only_iou = []
        
        for images, targets in tqdm(test_loader):

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


            targets[targets == -1] = 0 ## making edges background
            if args.thres == 1:
                logits_soft_max = (100*similarity_map).softmax(dim=-1).max(dim=-1)[0]
                simialrity_map_argmax[logits_soft_max < threshold] = 0 ## threshold to ignore background

            iou_scores = metric_iou(simialrity_map_argmax.cpu(), targets)#.to(int))
            positive_iou = iou_scores[torch.unique(simialrity_map_argmax).cpu()] ## Keep only postive classes for IoU, note postive means from our prediction, not GT
            postive_pred_iou.append(torch.nanmean(positive_iou).item())
            
            positive_iou_v2 = iou_scores[iou_scores>0]
            postive_only_iou.append(torch.nanmean(positive_iou_v2).item())

    print('positive_pred', np.nanmean(postive_pred_iou) * 100)
    print('postive_only_iou', np.nanmean(postive_only_iou)*100)


            
            