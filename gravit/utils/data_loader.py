import os
import scipy
import numpy as np
import re
from statistics import mode

def crop_to_start_and_end(feature, label):
    """
    Crop videos and labels. 
    Note that the labels and features must be at same sampling rate (aligned) when input to this function.
    """
    # Crop beginning: to index of first label that is not "action_start" 
    begin_label_index = next((i for i, x in enumerate(label) if x != 'action_start'), None)
    label = label[begin_label_index:]

    # Crop end: to index of first "action_end" label
    end_label_index = label.index(next(x for x in label if x == "action_end"))
    label = label[:end_label_index]

    # Crop features the same way
    if feature is not None:
        feature = feature[begin_label_index:]
        feature = feature[:end_label_index]
        
    return feature, label


def get_segments_and_batch_idxs(label):
    """ Returns the segmentwise labels and the corresponding batch indices for the given labels"""
    batch = 0
    batch_idx_designation = []
    segment_labels = []

    # Find segment starts
    segment_labels.append(label[0])
    batch_idx_designation.append(batch)
    for i in range(1, len(label)):
        if label[i] != label[i-1]:
            segment_labels.append(label[i])
            batch += 1
        batch_idx_designation.append(batch)

    if batch+1 != len(segment_labels):
        print(f'Number of segments: {batch+1} | Number of labels: {len(segment_labels)}')
        print(segment_labels)
        raise ValueError('Number of segments and labels do not match')
    
    return segment_labels, batch_idx_designation
    
def load_and_fuse_modalities(data_file, combine_method='concat', dataset=None, sample_rate=1, is_multiview=None):
    feature = None
    # if dataset == '50salads':
    #     feature = np.transpose(np.load(data_file))


    # EgoExo Swin OmnivoreL Features
    # elif 'egoexo-omnivore' in dataset:
    # Multiview approach for EgoExo-OmnivoreL -> Load all view features and concatenate
    if is_multiview is not None and is_multiview == True:
        take_name = data_file.split('/')[-1].split('.npy')[0].rsplit('_', 1)[0]        

        root = '/'.join(data_file.split('/')[:-1])
        
        # gopro_ext = f'{take_name}*.npy'

        # pattern = re.compile(gopro_ext)

        import glob
        filenames = glob.glob(os.path.join(root,f'{take_name}*.npy'))
        if len(filenames) < 3:
            print(f'{take_name} has less than 3 views. ')

        feat_length = None
        
        for fp in filenames:
            this_view_feature = np.load(fp)
            if feature is None:
                feature = this_view_feature
                feat_length = this_view_feature.shape[0]
            elif combine_method == 'concat':
                assert this_view_feature.shape[0] == feat_length, 'Feature lengths of different views do not match'
                feature = np.concatenate([feature, this_view_feature], axis=1)

            if 'fair_cooking_05_2' in fp:
                print(f'Loaded {fp} with shape: {this_view_feature.shape}')


        if feature is None:
            raise ValueError('No features loaded')

        # load and concatenate all views' features
        # Check all filenames against the base filename "take_name" and concatenate if they match
        # for fn in os.listdir(root):
        #     # print(fn)
        #     if pattern.match(fn):
        #         print(fn)
        #         this_view_feature = np.load(os.path.join(root, fn))
        #         if feature is None:
        #             feature = this_view_feature
        #             if combine_method == 'concat':
        #                 feature = np.concatenate([feature, this_view_feature], axis=1)
    
    else:
        feature = np.load(data_file)
        # if 'fair_cooking_05_2' in data_file:
        #         print(f'Loaded {data_file} with shape: {feature.shape}')
 
        # print(f'Loaded EgoExo-OmnivoreL. shape: {feature.shape}, {data_file}')
     
    ######## 50 Salads: Load and Concatenate Bridge-Prompt Features ###########
    # id = data_file.split('rgb-')[1].split('.npy')[0]
    # brp_file = os.path.join(root_data, f'features/Bridge-Prompt/downsampled/rgb-{id}.npy')
    # brp_features = np.load(brp_file)
    # # print(brp_features.shape)

    # ## There is an issue with BrP features for a few sessions are slightly shorter in legnth. For these just drop
    # if brp_features.shape[0] < num_frame:
    #     num_frame = brp_features.shape[0] # UPDATE FRAME LENGTH
        
    #     if rgb_features.any() != None:
    #         rgb_features = rgb_features[0: num_frame, :]
        
    #     if accel_features.any() != None:
    #         accel_features = accel_features[0: num_frame, :]

    # ## ------------------------------------------- ##
    # if combine_method == 'concat':
    #     if accel_features.any() != None:
    #         feature = np.concatenate([rgb_features, accel_features], axis=1)
    #     else:
    #         feature = rgb_features
    #     if brp_features.any() != None:
    #         feature = np.concatenate([feature, brp_features], axis=1)

    #     num_frame = feature.shape[0]
    #     print('Concatenated. New shape: ')
    #     print(feature.shape)

    # ##### For just Br-P feats
    # # feature = brp_features
    
    return feature


def load_labels(actions, root_data, annotation_dataset, video_id, feature=None, sample_rate=1, load_descriptions=False, verbose=0):
    if annotation_dataset == '50salads':
        return load_and_trim_labels(video_id, actions, root_data=root_data, dataset=annotation_dataset, sample_rate=sample_rate, feature=feature)

    elif load_descriptions:
        with open(os.path.join(root_data, f'annotations/{annotation_dataset}/descriptions/{video_id}.txt')) as f:
            return [line.strip() for line in f]

    else:
        if not os.path.exists(os.path.join(root_data, f'annotations/{annotation_dataset}/groundTruth/{video_id}.txt')):
            print(f'File not found: {video_id}. Likely need to split video_id. See data_loader.py')

        with open(os.path.join(root_data, f'annotations/{annotation_dataset}/groundTruth/{video_id}.txt')) as f:
            return [actions[line.strip()] for line in f]

       

#####
def load_accelerometer(id, root_data):
    ###### Load Accel Features ######
    accel_file = os.path.join(root_data, f'features/accelerometer/{id}.npy')
    accel_features = np.load(accel_file)

    for i in range(10):
        triaxial = accel_features[:, i*3:i*3+3]

        # Normalize each axis independently
        for j in range(triaxial.shape[1]):  # Assuming the second dimension is the axis
            axis_data = triaxial[:, j]
            if (axis_data != 0).sum() == 0: # if all recordings are 0 on the axis then pass
                continue
            normalized_axis = (axis_data - np.min(axis_data)) / (np.max(axis_data) - np.min(axis_data))
            triaxial[:, j] = normalized_axis

        accel_features[:, i*3:i*3+3] = triaxial

    return accel_features


def load_and_trim_labels(video_id, actions, root_data='../data', dataset='50salads', sample_rate=1, feature=None):
    """
    feature: must input np array or pd dataframe of features to trim the labels to
    This function loads the ground truth annotations for a video, and trims them to match the length of the features.
    For BrP 50Salads feature/label pairs, since labels tend to be longer than features, and they appear to be aligned at the start.

    """
    # Get a list of ground-truth action labels
    with open(os.path.join(root_data, f'annotations/{dataset}/groundTruth/{video_id}.txt')) as f:
        label = [actions[line.strip()] for line in f]
     
    ##### Shorten Labels to match #####
    if feature is not None:
        if feature.shape[0] != len(label):
            print(f'Trimming labels to match feature length: {video_id}')

        new_length = feature.shape[0]*sample_rate
        label = label[0: new_length]

    return label