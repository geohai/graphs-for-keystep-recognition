import os
import scipy
import numpy as np
import re
from statistics import mode
import json

CLI_OUTPUT_DIR = "/local/juro4948/data/egoexo4d/egoexo"
ANNOTATIONS_PATH = os.path.join(CLI_OUTPUT_DIR, "annotations")


def get_segment_labels_by_batch_idxs(label, batch_idx_designation):
    """ Returns the segmentwise labels and the corresponding batch indices for the given labels"""
    assert len(label) == len(batch_idx_designation), "Length of label and batch_idx_designation must be the same"
    segment_labels = []

    for idx in np.unique(batch_idx_designation):
        if idx == -1:
            continue
        # select indices where batch_idx_designation is idx
        labels_per_segment = [label[i] for i in range(len(label)) if batch_idx_designation[i] == idx]

        segment_labels.append(labels_per_segment)

    return segment_labels


def load_batch_indices(batch_idx_path, take_name):
    batch_idx_file = os.path.join(batch_idx_path, f'{take_name}.txt')
    with open(batch_idx_file, 'r') as f:
        loaded = [int(line.strip()) for line in f]
    return loaded


def load_features(filepath):
    return np.load(filepath, allow_pickle=True)


def load_spatial_features(filepath, verbose=False):
    spatial_feature = load_features(filepath)
    # spatial_feature = spatial_feature.reshape(spatial_feature.shape[0], spatial_feature.shape[1]*spatial_feature.shape[2])
    if verbose:
        print(f'Shape of spatial features: {spatial_feature.shape}')
    return spatial_feature


def load_labels(actions, root_data, annotation_dataset, video_id,  load_descriptions=False):
    if load_descriptions:
        with open(os.path.join(root_data, f'annotations/{annotation_dataset}/descriptions/{video_id}.txt')) as f:
            return [line.strip() for line in f]

    else:
        with open(os.path.join(root_data, f'annotations/{annotation_dataset}/groundTruth/{video_id}.txt')) as f:
            return [actions[line.strip()] for line in f]


def load_labels_raw(root_data, annotation_dataset, video_id):
    with open(os.path.join(root_data, f'annotations/{annotation_dataset}/groundTruth/{video_id}.txt')) as f:
        return [line.strip() for line in f]


def load_atomic_action_descriptions(root_data, annotation_dataset, video_id):
    with open(os.path.join(root_data, f'annotations/{annotation_dataset}/aads/{video_id}.txt')) as f:
        return [line.strip() for line in f]