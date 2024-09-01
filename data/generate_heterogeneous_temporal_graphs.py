import os
import glob
import torch
import argparse
import numpy as np
from functools import partial
from multiprocessing import Pool
from torch_geometric.data import Data
from gravit.utils.data_loader import *
from gravit.utils.parser import get_args, get_cfg
from torch_geometric.data import HeteroData

def compute_similarity_metric(node_i, node_j, metric):
    if metric == 'cosine':
        return np.dot(node_i, node_j) / (np.linalg.norm(node_i) * np.linalg.norm(node_j))
    elif metric == 'gaussian':
        sigma = 2
        return np.exp(-np.linalg.norm(node_i - node_j) ** 2 / (2 * (sigma ** 2)))
    elif metric == 'inner_product':
        return np.dot(node_i, node_j)


def generate_heterogeneous_temporal_graph(data_file, args, path_graphs, actions, train_ids, all_ids, list_multiview_data_files=[]):
    """
    Generate heterogeneous temporal graphs of a single video
    """

    skip = args.skip_factor
    batch_idx_designation = 0
    take_name = os.path.splitext(os.path.basename(data_file))[0]

    # # Load the features and labels
    feature = load_features(data_file)
    list_feature_multiview = []
    for multiview_data_file in list_multiview_data_files:
        feature_multiview = np.load(multiview_data_file)
        assert feature.shape == feature_multiview.shape, f'feature.shape: {feature.shape}, feature_multiview.shape: {feature_multiview.shape}'
        list_feature_multiview.append(feature_multiview)


    if args.add_text:
        if not os.path.exists(os.path.join(args.text_dir, take_name + '.npy')):
            raise ValueError(f'Text feature not found for {os.path.join(args.text_dir, take_name + ".npy")}')
            # print(f'Text feature not found for {os.path.join(args.text_dir, take_name + ".npy")}')
            # return
        text_feature = load_features(os.path.join(args.text_dir, take_name + '.npy'))
    else:
        text_feature = [[]]

 
    if not os.path.exists(os.path.join(args.root_data, f'annotations/{args.dataset}/groundTruth/{take_name}.txt')):
        take_name = take_name.rsplit('_', 1)[0]


    ############ REMOVE ############
    # load features
    text_feature2 = load_features(os.path.join(f'/home/juro4948/gravit/GraVi-T/data/features/{args.object_position_nodes}', take_name + '.npy'))
    text_feature2 = text_feature2.reshape(text_feature2.shape[0], text_feature2.shape[1]*text_feature2.shape[2])
    # print(f'text_feature.shape: {text_feature2.shape}')

    # concatenate 
    print(f'Concatenating text and heatmap')
    text_feature = np.hstack((text_feature, text_feature2))

    print(f'text_feature.shape: {text_feature.shape}')

    ################################

    #  load pre-averaged segmentwise features
    if cfg['load_segmentwise']:
        label = load_labels(trimmed=True, video_id=take_name, actions=actions, root_data=args.root_data, annotation_dataset=args.dataset) 
        
        if len(feature) != len(label):
            print(take_name)
            print(f'Length of feature: {len(feature)} | Length of label: {len(label)}')
            raise ValueError('Length of feature and label does not match')
        
    else:
        label = load_labels(trimmed=True, video_id=take_name, actions=actions, root_data=args.root_data, annotation_dataset=args.dataset) 
        # print(f'Length of label: {len(label)} | Length of feature: {len(feature)}')
        batch_idx_path = os.path.join(args.root_data, 'annotations', args.dataset, 'batch_idx')
        untrimmed_batch_idxs = load_batch_indices(batch_idx_path, take_name)
        batch_idx_designation = [i for i in untrimmed_batch_idxs if i != -1]
        label = get_segment_labels_by_batch_idxs(label, batch_idx_designation=batch_idx_designation)
        label = [mode(label[i]) for i in range(len(label))]



    num_frame = feature.shape[0]
    # print(f'take_name: {take_name} | Num Frames: {num_frame} | Num Labels: {len(label)}')

    # # Get a list of the edge information: these are for edge_index and edge_attr
    counter_similarity_edges_added = 0
    node_source = []
    node_target = []
    edge_attr = []

    hetero_node_source = []
    hetero_node_target = []
    hetero_edge_attr = []

    num_view = len(list_feature_multiview)+1
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

                # add edges between the same view in different frames
                for k in range(1, num_view):
                    node_source.append(i+num_frame*k)
                    node_target.append(j+num_frame*k)
                    edge_attr.append(np.sign(frame_diff))

                # add edges between different views in the same frame
                if frame_diff == 0:
                    for k in range(1, num_view):
                        node_source.append(i)
                        node_target.append(j+num_frame*k)
                        edge_attr.append(-1)

                # add edges between heterogenous nodes (text to ego) in the same frame
                if frame_diff == 0:
                    if args.add_text:
                        hetero_node_source.append(i)
                        hetero_node_target.append(j)
                        hetero_edge_attr.append(-1)
                        

            # Make additional connections between non-adjacent nodes
            # This can help reduce over-segmentation of predictions in some cases
            elif skip:
                if (frame_diff % skip == 0) and (abs(frame_diff) <= skip*args.tauf):
                    node_source.append(i)
                    node_target.append(j)
                    edge_attr.append(np.sign(frame_diff))

                    for k in range(num_view):
                        node_source.append(i+num_frame)
                        node_target.append(j+num_frame)
                        edge_attr.append(np.sign(frame_diff))

            # Add similarity-based connections
            # print(args.similarity_metric)
            if args.similarity_metric is not None and i != j:
                similarity = compute_similarity_metric(feature[i], feature[j], metric=args.similarity_metric)
                if similarity > args.similarity_threshold:
                    # print(f'Adding similarity edge between {i} and {j} with similarity {similarity}')
                    node_source.append(i)
                    node_target.append(j)
                    edge_attr.append(np.sign(frame_diff))  # try 0
                    counter_similarity_edges_added += 1
    
    if args.similarity_metric is not None:
        print(f'{counter_similarity_edges_added} similarity edges | {len(node_source) - counter_similarity_edges_added} | ' + "{:.1f}%".format(counter_similarity_edges_added / len(node_source) * 100) + " % of Total edges")

    # x: features
    # g: global_id
    # edge_index: information on how the graph nodes are connected
    # edge_attr: information about whether the edge is spatial (0) or temporal (positive: backward, negative: forward)
    # y: labels

    if num_view > 1:
        feature_multiview = np.concatenate(list_feature_multiview)
        feature = np.concatenate((np.array(feature, dtype=np.float32), feature_multiview))
        label = label*num_view
     
    
    graphs = HeteroData()

    # define node types and their feature matrix [num_nodes, num_features]
    graphs['omnivore'].x = torch.tensor(np.array(feature, dtype=np.float32), dtype=torch.float32)
    graphs['omnivore', 'to', 'omnivore'].edge_index = torch.tensor(np.array([node_source, node_target], dtype=np.int32), dtype=torch.long)
    graphs['omnivore', 'to', 'omnivore'].edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
    g = all_ids.index(take_name)
    graphs['omnivore'].g = torch.tensor([g], dtype=torch.long)

    graphs['text'].x = torch.tensor(np.array(text_feature, dtype=np.float32), dtype=torch.float32)
    graphs['omnivore', 'to', 'text'].edge_index = torch.tensor(np.array([hetero_node_source, hetero_node_target], dtype=np.int32), dtype=torch.long)
    graphs['omnivore', 'to', 'text'].edge_attr = torch.tensor(hetero_edge_attr, dtype=torch.float32)
    

    # labels for omnivore nodes 
    graphs['omnivore'].y = torch.tensor(np.array(label, dtype=np.int16)[::args.sample_rate], dtype=torch.long)

    ## for framewise features need to save the corresponding framewise segment idx
    # graphs['omnivore'].batch_idxs = batch_idx_designation


    if take_name in train_ids:
        torch.save(graphs, os.path.join(path_graphs, 'train', f'{take_name}.pt'))
    else:
        torch.save(graphs, os.path.join(path_graphs, 'val', f'{take_name}.pt'))


if __name__ == "__main__":
    """
    Generate temporal graphs from the extracted features
    """

    parser = argparse.ArgumentParser()
    # Default paths for the training process
    parser.add_argument('--root_data',     type=str,   help='Root directory to the data', default='./data')
    parser.add_argument('--dataset',       type=str,   help='Name of the dataset (annotation dir)')
    parser.add_argument('--features',      type=str,   help='Name of the features', required=False)
    parser.add_argument('--cfg',      type=str,   help='Path to config file containing parameters. ', required=False)

    # Hyperparameters for the graph generation
    parser.add_argument('--tauf',          type=int,   help='Maximum frame difference between neighboring nodes', required=False)
    parser.add_argument('--skip_factor',   type=int,   help='Make additional connections between non-adjacent nodes', default=1000)
    parser.add_argument('--sample_rate',   type=int,   help='Downsampling rate for the input', default=1) #downsample rate for labels (Julia-my labels are at 30Hz)
    # parser.add_argument('--is_multiview',   type=bool,   help='Using Multiview Features?', default=False)
    parser.add_argument('--add_multiview',   help='Whether to add multiview features', action="store_true")
    parser.add_argument('--add_text',   help='Whether to add text features', action="store_true")
    parser.add_argument('--crop',   type=bool,   help='Crop action_start and action_end', default=False)
    
    args = parser.parse_args()

    # load config file
    if args.cfg is not None:
        cfg = get_cfg(args)
        print(cfg)

    if cfg['object_position_nodes'] is not None:
        args.object_position_nodes = cfg['object_position_nodes']
    if args.features is None:
        args.features = cfg['features_dataset']
    if args.dataset is None:
        args.dataset = cfg['annotations_dataset']

    if args.tauf is None:
        args.tauf= cfg['tauf']
    if args.skip_factor is None:
        args.skip_factor = cfg['skip_factor']
    args.similarity_metric = cfg['similarity_metric']
    if args.similarity_metric == 'None':
        args.similarity_metric = None
    if cfg['similarity_threshold'] is not None:
        args.similarity_threshold = cfg['similarity_threshold']

    print(f'Tauf: {args.tauf} | Skip Factor: {args.skip_factor} | Similarity Metric: {args.similarity_metric} | Similarity Threshold: {args.similarity_threshold}')
    print(f'Features: {args.features} | Dataset: {args.dataset}')
    # Build a mapping from action classes to action ids
    actions = {}
    with open(os.path.join(args.root_data, f'annotations/{args.dataset}/mapping.txt')) as f:
        for line in f:
            aid, cls = line.strip().split(' ')
            actions[cls] = int(aid)

    # Get a list of all video ids
    all_ids = sorted([os.path.splitext(v)[0] for v in os.listdir(os.path.join(args.root_data, f'annotations/{args.dataset}/groundTruth'))])
 
    # Iterate over different splits
    print ('This process might take a few minutes')

    list_splits = sorted(os.listdir(os.path.join(args.root_data, f'features/{args.features}')))

    for split in list_splits:
        # Get a list of training video ids
        print(f'Reading splits at {os.path.join(args.root_data, f"annotations/{args.dataset}/splits/train.{split}.bundle")}')
        with open(os.path.join(args.root_data, f'annotations/{args.dataset}/splits/train.{split}.bundle')) as f:
            train_ids = [os.path.splitext(line.strip())[0] for line in f]
            print(f'Number of training videos: {len(train_ids)}')

        # path_graphs = os.path.join(args.root_data, f'graphs/{args.features}_{args.tauf}_{args.skip_factor}/{split}')
        path_graphs = os.path.join(args.root_data, f'graphs/{cfg["graph_name"]}/{split}')
        
        os.makedirs(os.path.join(path_graphs, 'train'), exist_ok=True)
        os.makedirs(os.path.join(path_graphs, 'val'), exist_ok=True)

        list_data_files = sorted(glob.glob(os.path.join(args.root_data, f'features/{args.features}/{split}/*/*.npy')))
        multiview_data_files = {}
        if args.add_multiview:
            for multiview_data in sorted(glob.glob(os.path.join(args.root_data, f'features/{args.features}-exo/{split}/*/*.npy'))):
                vid = '_'.join(os.path.basename(multiview_data).split('_')[:-1])
                data_sp = 'train'
                if vid not in train_ids:
                    data_sp = 'val'
                matching_data_file = os.path.join(args.root_data, f'features/{args.features}/{split}/{data_sp}/{vid}_0.npy')
                assert matching_data_file in list_data_files, f'check {matching_data_file}'
                if matching_data_file not in multiview_data_files:
                    multiview_data_files[matching_data_file] = []
                multiview_data_files[matching_data_file].append(multiview_data)

        if args.add_text:
            text_dir = os.path.join(args.root_data, f'annotations/{cfg["annotations_dataset"]}/{cfg["text_dataset"]}/')
            args.text_dir = text_dir
        else:
            args.text_dir = None


        #with Pool(processes=35) as pool:
        #    pool.map(partial(generate_temporal_graph, args=args, path_graphs=path_graphs, actions=actions, train_ids=train_ids, all_ids=all_ids), list_data_files)
        for data_file in list_data_files:
            generate_heterogeneous_temporal_graph(data_file, args=args, path_graphs=path_graphs, actions=actions, train_ids=train_ids, all_ids=all_ids, list_multiview_data_files=multiview_data_files[data_file] if data_file in multiview_data_files else '')


        print (f'Graph generation for {split} is finished')
