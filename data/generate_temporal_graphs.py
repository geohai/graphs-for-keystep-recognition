import os
import glob
import torch
import argparse
import numpy as np
from functools import partial
from multiprocessing import Pool
from torch_geometric.data import Data
import scipy
from gravit.models.fusion.multihead_attention import *
from gravit.utils.data_loader import load_and_fuse_modalities, load_labels
import re




def generate_temporal_graph(data_file, args, path_graphs, actions, train_ids, all_ids):
    """
    Generate temporal graphs of a single video
    """
    
    combine_method = 'concat'
    skip = args.skip_factor

    video_id = os.path.splitext(os.path.basename(data_file))[0]
    if args.is_multiview is not None and args.is_multiview == True:
        video_id = video_id[0:-2] 

    # Load the features and labels
    feature = load_and_fuse_modalities(data_file, combine_method, video_id, actions, root_data=args.root_data, dataset=args.dataset, sample_rate=args.sample_rate, is_multiview=args.is_multiview)
    label = load_labels(video_id, actions, root_data=args.root_data, dataset=args.dataset, sample_rate=args.sample_rate, feature=feature)
    num_frame = feature.shape[0]


    # crop to start and end
    if args.crop == True:
        begin_label_index = label.index(next(x for x in label if x != "action_start"))
        feature = feature[begin_label_index:]
        label = label[begin_label_index:]

        end_label_index = label.index(next(x for x in label if x != "action_end"))
        feature = feature[:end_label_index]
        label = label[:end_label_index]
        num_frame = feature.shape[0]
        


    # Get a list of the edge information: these are for edge_index and edge_attr
    node_source = []
    node_target = []
    edge_attr = []
    for i in range(num_frame):
        for j in range(num_frame):
            # Frame difference between the i-th and j-th nodes
            frame_diff = i - j

            # The edge ij connects the i-th node and j-th node
            # Positive edge_attr indicates that the edge ij is backward (negative: forward)
            if abs(frame_diff) <= args.tauf:
                node_source.append(i)
                node_target.append(j)
                edge_attr.append(np.sign(frame_diff))

            # Make additional connections between non-adjacent nodes
            # This can help reduce over-segmentation of predictions in some cases
            elif skip:
                if (frame_diff % skip == 0) and (abs(frame_diff) <= skip*args.tauf):
                    node_source.append(i)
                    node_target.append(j)
                    edge_attr.append(np.sign(frame_diff))

    # x: features
    # g: global_id
    # edge_index: information on how the graph nodes are connected
    # edge_attr: information about whether the edge is spatial (0) or temporal (positive: backward, negative: forward)
    # y: labels

    graphs = Data(x = torch.tensor(np.array(feature, dtype=np.float32), dtype=torch.float32),
                  g = all_ids.index(video_id),
                  edge_index = torch.tensor(np.array([node_source, node_target], dtype=np.int64), dtype=torch.long),
                  edge_attr = torch.tensor(edge_attr, dtype=torch.float32),
                  y = torch.tensor(np.array(label, dtype=np.int64)[::args.sample_rate], dtype=torch.long))

    if video_id in train_ids:
        torch.save(graphs, os.path.join(path_graphs, 'train', f'{video_id}.pt'))
    else:
        torch.save(graphs, os.path.join(path_graphs, 'val', f'{video_id}.pt'))


if __name__ == "__main__":
    """
    Generate temporal graphs from the extracted features
    """

    parser = argparse.ArgumentParser()
    # Default paths for the training process
    parser.add_argument('--root_data',     type=str,   help='Root directory to the data', default='./data')
    parser.add_argument('--dataset',       type=str,   help='Name of the dataset', default='50salads')
    parser.add_argument('--features',      type=str,   help='Name of the features', required=True)

    # Hyperparameters for the graph generation
    parser.add_argument('--tauf',          type=int,   help='Maximum frame difference between neighboring nodes', required=True)
    parser.add_argument('--skip_factor',   type=int,   help='Make additional connections between non-adjacent nodes', default=10)
    parser.add_argument('--sample_rate',   type=int,   help='Downsampling rate for the input', default=1) #downsample rate for labels (Julia-my labels are at 30Hz)
    parser.add_argument('--is_multiview',   type=bool,   help='Using Multiview Features?', default=False)
    parser.add_argument('--crop',   type=bool,   help='Crop action_start and action_end', default=False)
    

    args = parser.parse_args()

    # Build a mapping from action classes to action ids
    actions = {}
    with open(os.path.join(args.root_data, f'annotations/{args.dataset}/mapping.txt')) as f:
        for line in f:
            aid, cls = line.strip().split(' ')
            actions[cls] = int(aid)
    # print (actions)

    # Get a list of all video ids
    all_ids = sorted([os.path.splitext(v)[0] for v in os.listdir(os.path.join(args.root_data, f'annotations/{args.dataset}/groundTruth'))])
 
    # Iterate over different splits
    print ('This process might take a few minutes')

    list_splits = sorted(os.listdir(os.path.join(args.root_data, f'features/{args.features}')))
    # print(list_splits)
    for split in list_splits:
        # Get a list of training video ids
        with open(os.path.join(args.root_data, f'annotations/{args.dataset}/splits/train.{split}.bundle')) as f:
            train_ids = [os.path.splitext(line.strip())[0] for line in f]

        path_graphs = os.path.join(args.root_data, f'graphs/{args.features}_{args.tauf}_{args.skip_factor}/{split}')
        os.makedirs(os.path.join(path_graphs, 'train'), exist_ok=True)
        os.makedirs(os.path.join(path_graphs, 'val'), exist_ok=True)

        list_data_files = sorted(glob.glob(os.path.join(args.root_data, f'features/{args.features}/{split}/*.npy')))

        if args.is_multiview:
            list_data_files = sorted(glob.glob(os.path.join(args.root_data, f'features/{args.features}/{split}/*_0.npy')))

        
        with Pool(processes=35) as pool:
            pool.map(partial(generate_temporal_graph, args=args, path_graphs=path_graphs, actions=actions, train_ids=train_ids, all_ids=all_ids), list_data_files)

        print (f'Graph generation for {split} is finished')
