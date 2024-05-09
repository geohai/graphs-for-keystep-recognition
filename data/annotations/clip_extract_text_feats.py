import os
import numpy as np
import torch
import open_clip

path_annts = 'data/annotations/egoexo-segmentwise'
path_save = os.path.join(path_annts, 'descriptions_features')
os.makedirs(path_save, exist_ok=True)
model_name = 'ViT-H-14'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained='laion2b_s32b_b79k', device=device)
model.eval()

tokenizer = open_clip.get_tokenizer(model_name)

video_names = sorted(os.listdir(os.path.join(path_annts, 'descriptions')))
for vn in video_names:
    with open(os.path.join(path_annts, 'descriptions', vn), 'r') as f:
        texts = f.readlines()
    #TODO Use the raw descriptions. Currently all the spaces have been replaced by _
    #texts = [t.strip() for t in texts]
    texts = [' '.join(t.strip().split('_')) for t in texts]

    text_tokens = tokenizer(texts).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_tokens)

    with open(os.path.join(path_save, os.path.splitext(vn)[0] + '_0.npy'), 'wb') as f:
        np.save(f, text_features.numpy(force=True))

    print (f'{vn} processed')
