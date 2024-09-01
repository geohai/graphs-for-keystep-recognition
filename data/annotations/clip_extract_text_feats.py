import os
import numpy as np
import torch
import open_clip



model_name = 'ViT-H-14'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained='laion2b_s32b_b79k', device=device)
model.eval()

tokenizer = open_clip.get_tokenizer(model_name)


def get_clip_text_features(texts):
    #TODO Use the raw descriptions. Currently all the spaces have been replaced by _
    #texts = [t.strip() for t in texts]
    texts = [' '.join(t.strip().split('_')) for t in texts]
    # print(texts)

    text_tokens = tokenizer(texts).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_tokens)

    return text_features


if __name__ == '__main__':
    #  output_dir = '/home/juro4948/gravit/GraVi-T/data/features/clip-videorecap-feats'
    # text_annotation_dir = '/home/juro4948/gravit/GraVi-T/data/annotations/egoexo-segmentwise/videorecap-clip'
    print('Running main')
    path_annts = 'data/annotations/egoexo-segmentwise/'
    annts_type = 'detic-object-list-anchors_hands_sorted' #'detic-object-list-3objects_sorted' #detic-object-list-3times' #'videorecap-clip' #'descriptions'
    path_save = os.path.join(path_annts, f'{annts_type}_features')
    os.makedirs(path_save, exist_ok=True)

    video_names = sorted(os.listdir(os.path.join(path_annts, annts_type)))
    for vn in video_names:
        with open(os.path.join(path_annts, annts_type, vn), 'r') as f:
            texts = f.readlines()

        text_features = get_clip_text_features(texts)

        with open(os.path.join(path_save, os.path.splitext(vn)[0] + '_0.npy'), 'wb') as f:
            np.save(f, text_features.numpy(force=True))

        print (f'Saved to {os.path.join(path_save, os.path.splitext(vn)[0] + "_0.npy")}')