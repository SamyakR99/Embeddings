import torch
import torch.nn as nn

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from copy import deepcopy
import torch.nn.functional as F

_tokenizer = _Tokenizer()

__all__ = ['dualcoop', 'DualCoop']


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    model = clip.build_model_conv_proj(state_dict or model.state_dict(), cfg)

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class DualCoop(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.visual_encoder_type = cfg.MODEL.BACKBONE.NAME
        # self.prompt_learner = MLCPromptLearner(cfg, classnames, clip_model)

        # self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = cfg.TRAINER.COOP_MLC.LS
        self.dtype = clip_model.dtype
        self.cfg = cfg
        self.clip_model = clip_model        
        self.classnames = classnames


        pos_template = 'A photo of a {}'
        neg_template = 'Not a photo of a {}'
        self.neg_texts = [neg_template.format(label) for label in self.classnames]
        self.pos_texts = [pos_template.format(label) for label in self.classnames]

        self.tokenized_prompts_pos = []
        self.tokenized_prompts_neg = []
        for p_pos, p_neg in zip(self.pos_texts, self.neg_texts):
            self.tokenized_prompts_pos.append(clip.tokenize(p_pos))
            self.tokenized_prompts_neg.append(clip.tokenize(p_neg))
        
        sizes = [512, 384, 256] ## RN101
        # sizes = [1024, 768, 512] ## RN50

        layers_text = []
        for i in range(len(sizes) - 2):
            layers_text.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers_text.append(nn.BatchNorm1d(sizes[i + 1]))
            layers_text.append(nn.ReLU(inplace=True))
        layers_text.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.text_projector = nn.Sequential(*layers_text)

        size_img = [512, 256] ## RN101
        # size_img = [1024, 512]  ## RN50
        layers_img = []
        # for i in range(len(sizes) - 2):
        #     layers_img.append(nn.Linear(size_img[i], size_img[i + 1], bias=False))
        #     layers_img.append(nn.BatchNorm1d(197))
        #     layers_img.append(nn.ReLU(inplace=True))
        layers_img.append(nn.Linear(size_img[-2], size_img[-1], bias=False))
        self.text_projector = nn.Sequential(*layers_text)
        
        self.image_projector = nn.Sequential(*layers_img)
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)
        self.lambd = 1#0.051 ## change here: originally 0.0051

        
        

    def off_diagonal(self, x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, image, cls_id=None):
        # get image and text features
        device = torch.device("cuda")
        image_feature, attn_weights = self.image_encoder(image.type(self.dtype))
        
        img_feat_interm = image_feature.permute(0,2,1)
        image_features = self.image_projector(img_feat_interm).permute(0,2,1)
        
        
        tokenized_prompts_pos = torch.cat(self.tokenized_prompts_pos).to(device)
        tokenized_prompts_neg = torch.cat(self.tokenized_prompts_neg).to(device)
        
        text_features_neg = self.clip_model.encode_text(tokenized_prompts_neg)
        text_features_pos = self.clip_model.encode_text(tokenized_prompts_pos)

        text_features = torch.cat((text_features_neg, text_features_pos), dim = 0)

        
        # text_features_final_pos =  self.text_projector(text_features_pos)
        # text_features_final_neg =  self.text_projector(text_features_neg)
        text_features_final = self.text_projector(text_features)
        

        # normalize features
        # text_features_final_pos = text_features_final_pos / text_features_final_pos.norm(dim=-1, keepdim=True)
        # text_features_final_neg = text_features_final_neg / text_features_final_neg.norm(dim=-1, keepdim=True)
        # text_features_final = torch.cat((text_features_final_neg, text_features_final_pos), dim = 0)

        text_features = text_features_final / text_features_final.norm(dim=-1, keepdim=True)

        # text_features.
        # breakpoint()
        

        ### Interm Test
        # c = self.bn(text_features).T @ self.bn(text_features)

        c_neg = self.bn(text_features[:len(self.classnames),:]) @ self.bn(text_features[:len(self.classnames),:]).T
        c_pos = self.bn(text_features[len(self.classnames):,:]) @ self.bn(text_features[len(self.classnames):,:]).T
        
        # c = self.bn(text_features) @ self.bn(text_features).T

        # breakpoint()
        on_diag_pos = torch.diagonal(c_pos).add_(-1).pow_(2).sum()
        off_diag_pos = self.off_diagonal(c_pos).pow_(2).sum()
        on_diag_neg = torch.diagonal(c_neg).add_(-1).pow_(2).sum()
        off_diag_neg = self.off_diagonal(c_neg).pow_(2).sum()
        
        loss_ssl_pos = on_diag_pos + self.lambd * off_diag_pos
        loss_ssl_neg = on_diag_neg + self.lambd * off_diag_neg
        loss_ssl = loss_ssl_pos+ loss_ssl_neg
        # print(loss_ssl_pos, loss_ssl_neg)
        
        image_features_norm = image_features / image_features.norm(dim=1, keepdim=True)

        # Class-Specific Region Feature Aggregation
        output = 20 * F.conv1d(image_features_norm, text_features[:, :, None])


        b, c, _ = output.shape
        # output_half = output[:,  c // 2:]
        w = F.softmax(output, dim=-1)
        output = 5 * (output * w).sum(-1)

        b, c = output.shape

        # convert the shape of logits to [b, 2, num_class]
        logits = output.resize(b, 2, c//2)

        return logits, loss_ssl

    @property
    def network_name(self):
        name = ''
        name += 'DualCoop-{}'.format(self.visual_encoder_type)
        return name

    def backbone_params(self):
        params = []
        for name, param in self.named_parameters():
            if "image_encoder" in name and "prompt_learner" not in name and 'attnpool' not in name:
                params.append(param)
        return params

    def attn_params(self):
        params = []
        for name, param in self.named_parameters():
            if 'attnpool' in name and 'image_encoder' in name:
                params.append(param)
                print(name)
        return params

    def prompt_params(self):
        params = []
        for name, param in self.named_parameters():
            if "prompt_learner" in name:
                params.append(param)
        return params

    def text_proj(self):
        params = []
        for name, param in self.named_parameters():
            if "text_projector" in name:
                params.append(param)
        return params

    def img_proj(self):
        params = []
        for name, param in self.named_parameters():
            if "image_projector" in name:
                params.append(param)
        return params

def dualcoop(cfg, classnames, **kwargs):
    print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
    clip_model = load_clip_to_cpu(cfg)

    clip_model.float()

    print("Building dualcoop")
    model = DualCoop(cfg, classnames, clip_model)

    if not cfg.TRAINER.FINETUNE_BACKBONE:
        print('Freeze the backbone weights')
        backbone_params = model.backbone_params()
        for param in backbone_params:
            param.requires_grad_(False)

    if not cfg.TRAINER.FINETUNE_ATTN:
        print('Freeze the attn weights')
        attn_params = model.attn_params()
        for param in attn_params:
            param.requires_grad_(False)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)

    # Note that multi-gpu training could be slow because CLIP's size is
    # big, which slows down the copy operation in DataParallel
    device_count = torch.cuda.device_count()
    if device_count > 1:
        print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
        model = nn.DataParallel(model)
    return model
