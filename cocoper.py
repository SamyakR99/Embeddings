from os.path import join
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import torch.nn.functional as F
import clip
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
from IPython.display import display, clear_output
from pycocotools.coco import COCO
import argparse


from torchmetrics.classification import MulticlassJaccardIndex
import time



parser = argparse.ArgumentParser(description="A script to demonstrate command-line arguments.")
    
# Add the --arch argument
parser.add_argument('--arch', type=str, required=True, help="The architecture to use (e.g., 'rn101').")
parser.add_argument('--thres', type=int, required=True, help="The architecture to use (e.g., 'rn101').")

# Parse the arguments
args = parser.parse_args()


# Load the COCO 2017 test set
coco = COCO('/home/samyakr2/multilabel/data/coco/annotations/instances_val2017.json')
image_dir = '/home/samyakr2/multilabel/data/coco/val2017/'

device = "cuda" if torch.cuda.is_available() else "cpu"


model, _ = clip.load("CS-RN101", device=device)
preprocess_img =  Compose([Resize((448, 448), interpolation=BICUBIC), ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])

preprocess_target =  Compose([ToTensor()])



image_ids = coco.getImgIds()

cat_ids = coco.getCatIds()
cats = coco.loadCats(cat_ids)
cat_id_to_name = {cat['id']: cat['name'] for cat in cats}



classnames = ['background',"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
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

all_texts = classnames[1:]

with torch.no_grad():
    text_feats = clip.encode_text_with_prompt_ensemble(model, all_texts, device)



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

if args.arch == 'CS':

    with torch.no_grad():
        iou_scores = []
        threshold = 0.9
        metric_iou = MulticlassJaccardIndex(num_classes=len(classnames), ignore_index=0).to('cpu')
        for img_id in tqdm(image_ids):
        # for img_id in [image_ids[129]]:
            img_info = coco.loadImgs(img_id)[0]
            img_path = image_dir + img_info['file_name']
            
            img = Image.open(img_path).convert('RGB')
            img = preprocess_img(img).to(device)
            # plt.imshow(img.permute(1,2,0).cpu())
            # plt.show()
            img = img.unsqueeze(0)
            

            true_ann_ids = coco.getAnnIds(imgIds=img_id)
            true_anns = coco.loadAnns(true_ann_ids)
            
            if len(true_anns)>0:
                
                complete_mask = np.zeros(coco.annToMask(true_anns[0]).shape, dtype=np.uint8)
                for annotation in true_anns:
                    mask = coco.annToMask(annotation)
                    category_id = annotation['category_id']
                    our_id = classnames.index(cat_id_to_name[category_id])
                    complete_mask[mask == 1] = our_id
            
            # print('UNIQUE', np.unique(complete_mask))
            # print('complete mask shape', complete_mask.shape)
            
            image_features = model.encode_image(img)
            image_features = F.normalize(image_features, dim=-1)
            
            text_features = F.normalize(text_feats, dim=-1) 
    #         features = image_features @ text_features.t()
            similarity = clip.clip_feature_surgery(image_features, text_features)

            similarity_map = clip.get_similarity_map(similarity[:, 1:, :], complete_mask.shape)
            similarity_map_argmax = similarity_map.argmax(-1)+1
            if args.thres == 1:
                logits_soft_max = (100*similarity_map).softmax(dim=-1).max(dim=-1)[0]
                similarity_map_argmax[logits_soft_max < threshold] = 0 ## threshold to ignore background
            # print('similarity_map',similarity_map_argmax.shape)
            # print(torch.unique(similarity_map_argmax))
            metric_iou.update(similarity_map_argmax.cpu(), torch.tensor(np.expand_dims(complete_mask,0)))
            
        print(metric_iou.compute().item())

elif args.arch == 'Ours':
    with torch.no_grad():
        threshold = 0.9
        metric_iou = MulticlassJaccardIndex(num_classes=len(classnames), ignore_index=0).to('cpu')
        for img_id in tqdm(image_ids):
        # for img_id in [image_ids[129]]:
            img_info = coco.loadImgs(img_id)[0]
            img_path = image_dir + img_info['file_name']
            
            img = Image.open(img_path).convert('RGB')
            img = preprocess_img(img).to(device)
            # plt.imshow(img.permute(1,2,0).cpu())
            # plt.show()
            img = img.unsqueeze(0)
            

            true_ann_ids = coco.getAnnIds(imgIds=img_id)
            true_anns = coco.loadAnns(true_ann_ids)
            
            if len(true_anns)>0:
                
                complete_mask = np.zeros(coco.annToMask(true_anns[0]).shape, dtype=np.uint8)
                for annotation in true_anns:
                    mask = coco.annToMask(annotation)
                    category_id = annotation['category_id']
                    our_id = classnames.index(cat_id_to_name[category_id])
                    complete_mask[mask == 1] = our_id
            
            # print('UNIQUE', np.unique(complete_mask))
            # print('complete mask shape', complete_mask.shape)
            
            image_features = model.encode_image(img)
            image_features = F.normalize(image_features, dim=-1)
            
            img_feat = image_projector(image_features)
            image_features = img_feat
            
            text_feat = text_projector(text_feats)
            text_features = F.normalize(text_feat, dim=-1) 


            features = image_features @ text_features.t()
            # similarity = clip.clip_feature_surgery(image_features, text_features)

            similarity_map = clip.get_similarity_map(features[:, 1:, :], complete_mask.shape)
            similarity_map_argmax = similarity_map.argmax(-1)+1
            if args.thres == 1:
                logits_soft_max = (100*similarity_map).softmax(dim=-1).max(dim=-1)[0]
                similarity_map_argmax[logits_soft_max < threshold] = 0 ## threshold to ignore background
            # print('similarity_map',similarity_map_argmax.shape)
            # print(torch.unique(similarity_map_argmax))
            metric_iou.update(similarity_map_argmax.cpu(), torch.tensor(np.expand_dims(complete_mask,0)))
        print(metric_iou.compute().item())

elif args.arch == 'CS_OUR':
    with torch.no_grad():
        threshold = 0.9
        metric_iou = MulticlassJaccardIndex(num_classes=len(classnames), ignore_index=0).to('cpu')
        for img_id in tqdm(image_ids):
        # for img_id in [image_ids[129]]:
            img_info = coco.loadImgs(img_id)[0]
            img_path = image_dir + img_info['file_name']
            
            img = Image.open(img_path).convert('RGB')
            img = preprocess_img(img).to(device)
            # plt.imshow(img.permute(1,2,0).cpu())
            # plt.show()
            img = img.unsqueeze(0)
            

            true_ann_ids = coco.getAnnIds(imgIds=img_id)
            true_anns = coco.loadAnns(true_ann_ids)
            
            if len(true_anns)>0:
                
                complete_mask = np.zeros(coco.annToMask(true_anns[0]).shape, dtype=np.uint8)
                for annotation in true_anns:
                    mask = coco.annToMask(annotation)
                    category_id = annotation['category_id']
                    our_id = classnames.index(cat_id_to_name[category_id])
                    complete_mask[mask == 1] = our_id
            
            # print('UNIQUE', np.unique(complete_mask))
            # print('complete mask shape', complete_mask.shape)
            
            image_features = model.encode_image(img)
            image_features = F.normalize(image_features, dim=-1)
            
            img_feat = image_projector(image_features)
            image_features = img_feat
            
            text_feat = text_projector(text_feats)
            text_features = F.normalize(text_feat, dim=-1) 


            # features = image_features @ text_features.t()
            similarity = clip.clip_feature_surgery(image_features, text_features)

            similarity_map = clip.get_similarity_map(similarity[:, 1:, :], complete_mask.shape)
            similarity_map_argmax = similarity_map.argmax(-1)+1
            if args.thres == 1:
                logits_soft_max = (100*similarity_map).softmax(dim=-1).max(dim=-1)[0]
                similarity_map_argmax[logits_soft_max < threshold] = 0 ## threshold to ignore background
            # print('similarity_map',similarity_map_argmax.shape)
            # print(torch.unique(similarity_map_argmax))
            metric_iou.update(similarity_map_argmax.cpu(), torch.tensor(np.expand_dims(complete_mask,0)))
        print(metric_iou.compute().item())
        