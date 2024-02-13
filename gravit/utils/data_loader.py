import os
import scipy
import numpy as np
import re

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
    

def load_and_fuse_modalities(data_file, combine_method, dataset='50salads', sample_rate=1, is_multiview=None):
    feature = None

    if dataset == '50salads':
        feature = np.transpose(np.load(data_file))
 
    # EgoExo Bridge Prompt Features
    elif dataset == 'egoexo':
        feature = np.load(data_file)

    # EgoExo Swin OmnivoreL Features
    elif 'egoexo-omnivore' in dataset:
        # Multiview approach for EgoExo-OmnivoreL -> Load all view features and concatenate
        # TODO: Add argument for multiview graph (rather than concatenate)
        if is_multiview is not None and is_multiview == True:
            take_name = data_file.split('/')[-1].split('0.npy')[0]
            root = '/'.join(data_file.split('/')[:-1])
            gopro_ext = f'{take_name}..npy'
            pattern = re.compile(gopro_ext)

            # load and concatenate all views' features
            # Check all filenames against the base filename "take_name" and concatenate if they match
            for fn in os.listdir(root):
                if pattern.match(fn):
                    this_view_feature = np.load(os.path.join(root, fn))
                    if feature is None:
                        feature = this_view_feature
                        if combine_method == 'concat':
                            feature = np.concatenate([feature, this_view_feature], axis=1)
     
        else:
            feature = np.load(data_file)

        # print(f'Loaded EgoExo-OmnivoreL. New shape: {feature.shape}')
     
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


def load_labels(actions, root_data, dataset, video_id, sample_rate=1, feature=None, load_raw=False):
    if dataset == '50salads' or dataset == 'egoexo':
        label = load_and_trim_labels(video_id, actions, root_data=root_data, dataset=dataset, sample_rate=sample_rate, feature=feature)
    elif 'egoexo-omnivore' in dataset:
        label = load_egoexo_omnivideo_integer_labels(video_id, actions, root_data=root_data, dataset=dataset, feature=feature, load_raw=load_raw)

    return label


def load_egoexo_omnivideo_integer_labels(video_id, actions, root_data='../data', dataset='egoexo',feature=None, load_raw=False):
    """
    Loads EgoExo Labels that correspond with Omnivore Sampling/Windowing strategy. Assuming 30Hz sampling rate, the window size 
    is 32 frames, and the stride is 16 frames. Aggregate the labels for these windows by taking the mode of the labels in each window.
    """
    # Get a list of ground-truth action labels
    with open(os.path.join(root_data, f'annotations/{dataset}/groundTruth/{video_id}.txt')) as f:
        label = [actions[line.strip()] for line in f]
        
    if load_raw == False:
        new_labels = [scipy.stats.mode(label[i:i+32])[0] for i in range(0, len(label), 16) if i+32 < len(label)]
        # add last label to the end
        if len(new_labels) < feature.shape[0]:
            last_window = label[-32:]
            new_labels.append(scipy.stats.mode(last_window)[0])

    label = new_labels

    return label

       
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
    For BrP 50Salads feature/label pairs, since labels tend to be shorter than features, and they appear to be aligned at the start.

    """
    # Get a list of ground-truth action labels
    with open(os.path.join(root_data, f'annotations/{dataset}/groundTruth/{video_id}.txt')) as f:
        label = [actions[line.strip()] for line in f]
     
    # print(f'Original Length of labels: {len(label)}')

    ##### Shorten Labels to match #####
    if feature is not None:
        if feature.shape[0] != len(label):
            print(f'Trimming labels to match feature length: {video_id}')

        new_length = feature.shape[0]*sample_rate
        label = label[0: new_length]

    # print(f'Final Length of labels: {len(label)}')
    # print('-----------')

    return label