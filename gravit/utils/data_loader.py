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


def concatenate_features(features1, features2):
    assert features1.shape[0] == features2.shape, f'Feature lengths do not match, {features1.shape[0]}, {features2.shape[0]}'
    return np.concatenate([features1, features2], axis=1)


def load_features(data_file):
    return np.load(data_file)


def load_labels(actions, root_data, annotation_dataset, video_id,  load_descriptions=False, trimmed=True, verbose=0):
    if annotation_dataset == '50salads':
        return load_and_trim_labels(video_id, actions, root_data=root_data, dataset=annotation_dataset, sample_rate=sample_rate, feature=feature)

    elif load_descriptions:
        if trimmed:
            with open(os.path.join(root_data, f'annotations/{annotation_dataset}/descriptions/{video_id}.txt')) as f:
            
                return [line.strip() for line in f]
        else:
            with open(os.path.join(root_data, f'annotations/{annotation_dataset}/descriptions_untrimmed/{video_id}.txt')) as f:
                return ['' if line.strip() in ['action_start', 'action_end'] else line.strip() for line in f]

    else:
        if trimmed:
            with open(os.path.join(root_data, f'annotations/{annotation_dataset}/groundTruth/{video_id}.txt')) as f:
                return [actions[line.strip()] for line in f]
        else:
            with open(os.path.join(root_data, f'annotations/{annotation_dataset}/groundTruth_untrimmed/{video_id}.txt')) as f:
                return ['' if line.strip() in ['action_start', 'action_end'] else line.strip() for line in f]



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