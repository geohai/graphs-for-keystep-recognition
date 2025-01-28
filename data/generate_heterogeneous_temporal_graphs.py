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


from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt

takes_json_path = '/home/juro4948/gravit/data/egoexo4d/egoexo4d/egoexo/takes.json'
with open(takes_json_path) as f:
    takes = json.load(f)


def find_best_exo_view(take_name):
    take_name = take_name.rsplit('_', 1)[0]
    for take_dict in takes:
        if take_dict['take_name'] == take_name:
            try:
                return take_dict['best_exo'][-1]
            except:
                print('Best exo view not found')
                return 1
    if 'sfu_cooking015_6' in take_name or 'georgiatech_cooking_08_01_6' in take_name:
        return 1
    raise ValueError(f'Best exo view not found for {take_name}')

def compute_similarity_metric(node_i, node_j, metric):
    if metric == 'cosine':
        return np.dot(node_i, node_j) / (np.linalg.norm(node_i) * np.linalg.norm(node_j))
    elif metric == 'gaussian':
        sigma = 2
        return np.exp(-np.linalg.norm(node_i - node_j) ** 2 / (2 * (sigma ** 2)))
    elif metric == 'inner_product':
        return np.dot(node_i, node_j)


def generate_heterogeneous_temporal_graph(data_file, args, path_graphs, actions, train_ids, all_ids, list_multiview_data_files=[], list_multidepth_data_files=[]):
    """
    Generate heterogeneous temporal graphs of a single video
    """

    skip = args.skip_factor
    batch_idx_designation = 0
    take_name = os.path.splitext(os.path.basename(data_file))[0]

    # # Load the features and label
    feature = load_features(data_file)
    list_feature_multiview = []
    
    # code to find best exo
    best_exo_num = find_best_exo_view(take_name)
    num_view = 1
    # print(f'Best exo view for {take_name}: {best_exo_num}')
    for multiview_data_file in list_multiview_data_files:
        take_num = multiview_data_file[-1]
        # if take_num == best_exo_num:
        feature_multiview = np.load(multiview_data_file)
        assert feature.shape == feature_multiview.shape, f'feature.shape: {feature.shape}, feature_multiview.shape: {feature_multiview.shape}'
        list_feature_multiview.append(feature_multiview)


    if args.add_text:
        if not os.path.exists(os.path.join(args.text_dir, take_name + '.npy')):
            # print(f'Text feature not found for {os.path.join(args.text_dir, take_name + ".npy")}')
            # return 
            raise ValueError(f'Text feature not found for {os.path.join(args.text_dir, take_name + ".npy")}')
        text_feature = load_features(os.path.join(args.text_dir, take_name + '.npy'))

 
    # # load depth featres (spatial--> depth)
    # if args.add_spatial:
    #     if not os.path.exists(os.path.join(args.root_data, 'features', args.spatial_dir, take_name + '.npy')):
    #         print(f'Spatial feature not found for {os.path.join(args.root_data, "features", args.spatial_dir, take_name + ".npy")}')
    #         return

    #     spatial_feature = load_spatial_features(filepath=os.path.join(args.root_data, 'features', args.spatial_dir, take_name + '.npy'),
    #                                         verbose=False)    

        # print(list_multidepth_data_files)    
        # list_feature_multidepth = []
        # for multidepth_data_file in list_multidepth_data_files:
        #     feature_multidepth = load_spatial_features(multidepth_data_file)
        #     assert spatial_feature.shape == feature_multidepth.shape, f'feature.shape: {spatial_feature.shape}, feature_multidepth.shape: {feature_multidepth.shape}'
        #     list_feature_multidepth.append(feature_multidepth)

        # node_feature = spatial_feature
    else:
        node_feature = [[]]

    if not os.path.exists(os.path.join(args.root_data, f'annotations/{args.dataset}/groundTruth/{take_name}.txt')):
        take_name = take_name.rsplit('_', 1)[0]
  
    # feature = spatial_feature
        # create a random vector that is the same size as a single object feature
        # spatial_feature = np.random.rand(spatial_feature.shape[0], spatial_feature.shape[1])
    # concat 
    # if args.add_text and args.add_spatial:
    #     node_feature = np.hstack((text_feature, spatial_feature))
    # elif args.add_text:
    #     node_feature = text_feature
    # elif args.add_spatial:
    #     node_feature = spatial_feature
    

    #  load pre-averaged segmentwise features
    if cfg['load_segmentwise']:
       if split == 'test':
           label = load_labels_raw( root_data=args.root_data, annotation_dataset=args.dataset, video_id=take_name)
       else:
           label = load_labels(video_id=take_name, actions=actions, root_data=args.root_data, annotation_dataset=args.dataset)
    else:
        label = load_labels(trimmed=True, video_id=take_name, actions=actions, root_data=args.root_data, annotation_dataset=args.dataset) 
        batch_idx_path = os.path.join(args.root_data, 'annotations', args.dataset, 'batch_idx')
        untrimmed_batch_idxs = load_batch_indices(batch_idx_path, take_name)
        batch_idx_designation = [i for i in untrimmed_batch_idxs if i != -1]
        label = get_segment_labels_by_batch_idxs(label, batch_idx_designation=batch_idx_designation)
        label = [mode(label[i]) for i in range(len(label))]


    num_frame = feature.shape[0]
    # if args.add_spatial:
    #     assert spatial_feature.shape[0] == num_frame, f'feature.shape: {feature.shape}, spatial_feature.shape: {spatial_feature.shape}'
    if args.add_text:
        assert text_feature.shape[0] == num_frame, f'feature.shape: {feature.shape}, text_feature.shape: {text_feature.shape}'
        if text_feature.shape[0] != num_frame:
            print(f'{take_name}--omnivore: {feature.shape}, text_feature: {text_feature.shape}')
            return


    # # Get a list of the edge information: these are for edge_index and edge_attr
    counter_similarity_edges_added = 0
    # edges between omnivore
    node_source = []
    node_target = []
    edge_attr = []

    # edges between mode 2 (depth)
    # if args.add_spatial:
    #     # edges between omnivore and text
    #     spatial_hetero_node_source = []
    #     spatial_hetero_node_target = []
    #     spatial_hetero_edge_attr = []

        # spatial_node_source = []
        # spatial_node_target = []
        # spatial_edge_attr = []


    if args.add_text:
        # edges between omnivore and text
        text_hetero_node_source = []
        text_hetero_node_target = []
        text_hetero_edge_attr = []

        # edges between text nodes (temporal)
        text_node_source = []
        text_node_target = []
        text_edge_attr = []


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

                # # connect ego spatial nodes across frames
                # if args.add_spatial:
                #     spatial_node_source.append(i)
                #     spatial_node_target.append(j)
                #     spatial_edge_attr.append(np.sign(frame_diff))

                # # connect text nodes across frames
                if args.add_text:
                    # if i != j:
                    text_node_source.append(i)
                    text_node_target.append(j)
                    text_edge_attr.append(np.sign(frame_diff))

                
                # add edges between same view across frames
                for k in range(1, num_view):
                    node_source.append(i+num_frame*k) # num_frame*k is the offset for the next view
                    node_target.append(j+num_frame*k)
                    edge_attr.append(np.sign(frame_diff))

                    # # # connect multiview-spatial nodes across frames
                    # if args.add_spatial:
                    #     spatial_node_source.append(i+num_frame*k)
                    #     spatial_node_target.append(j+num_frame*k)
                    #     spatial_edge_attr.append(np.sign(frame_diff))

                # add edges between exo views to ego view in the same frame
                if frame_diff == 0:
                    for k in range(1, num_view):
                        node_source.append(i)   # ego view
                        node_target.append(j+num_frame*k) # each exo view (i==j)
                        edge_attr.append(-2)


                    # if args.add_spatial:
                    #     # # # add edges between heterogenous nodes (spatial to  corresponding omnivore) in the same frame
                    #     # ego spatial to ego omnivore
                    #     spatial_hetero_node_source.append(i)
                    #     spatial_hetero_node_target.append(j)
                    #     spatial_hetero_edge_attr.append(-2)
                        
                        # for k in range(1, num_view):
                        #     # exo to ego edges for spatial nodes
                        #     spatial_node_source.append(i)
                        #     spatial_node_target.append(j+num_frame*k)
                        #     spatial_edge_attr.append(-1)

                            # # multiview-spatial to omnivore corresponding view
                            # spatial_hetero_node_source.append(i+num_frame*k)
                            # spatial_hetero_node_target.append(j+num_frame*k)
                            # spatial_hetero_edge_attr.append(-1)


                    # # # add edges between text to  ego omnivore in the same frame
                    if args.add_text:
                        text_hetero_node_source.append(i)
                        text_hetero_node_target.append(j)
                        text_hetero_edge_attr.append(-2)
                         

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

            # # Add similarity-based connections
            # # print(args.similarity_metric)
            # if args.similarity_metric is not None and i != j:
            #     similarity = compute_similarity_metric(feature[i], feature[j], metric=args.similarity_metric)
            #     if similarity > args.similarity_threshold:
            #         # print(f'Adding similarity edge between {i} and {j} with similarity {similarity}')
            #         node_source.append(i)
            #         node_target.append(j)
            #         edge_attr.append(np.sign(frame_diff))  # try 0
            #         counter_similarity_edges_added += 1
    
   
    # x: features
    # g: global_id
    # edge_index: information on how the graph nodes are connected
    # edge_attr: information about whether the edge is spatial (0) or temporal (positive: backward, negative: forward)
    # y: labels

    if num_view > 1:
        feature_multiview = np.concatenate(list_feature_multiview)
        feature = np.concatenate((np.array(feature, dtype=np.float32), feature_multiview))
        label = label*num_view
        # if args.add_spatial:
        #     feature_multidepth = np.concatenate(list_feature_multidepth)
        #     node_feature = np.concatenate((node_feature, feature_multidepth))

    
    graphs = HeteroData()

    # define node types and their feature matrix [num_nodes, num_features]
    graphs['omnivore'].x = torch.tensor(np.array(feature, dtype=np.float32), dtype=torch.float32)
    graphs['omnivore', 'to', 'omnivore'].edge_index = torch.tensor(np.array([node_source, node_target], dtype=np.int32), dtype=torch.long)
    graphs['omnivore', 'to', 'omnivore'].edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
    g = all_ids.index(take_name)
    graphs['omnivore'].g = torch.tensor([g], dtype=torch.long)

    # if args.add_spatial:
    #     graphs['spatial'].x = torch.tensor(np.array(node_feature, dtype=np.float32), dtype=torch.float32)
    #     graphs['omnivore', 'to', 'spatial'].edge_index = torch.tensor(np.array([spatial_hetero_node_source, spatial_hetero_node_target], dtype=np.int32), dtype=torch.long)
    #     graphs['omnivore', 'to', 'spatial'].edge_attr = torch.tensor(spatial_hetero_edge_attr, dtype=torch.float32)

        # ## addd edge types for multiview
        # graphs['spatial', 'to', 'spatial'].edge_index = torch.tensor(np.array([spatial_node_source, spatial_node_target], dtype=np.int32), dtype=torch.long)
        # graphs['spatial', 'to', 'spatial'].edge_attr = torch.tensor(spatial_edge_attr, dtype=torch.float32)

    if args.add_text:
        graphs['text'].x = torch.tensor(np.array(text_feature, dtype=np.float32), dtype=torch.float32)
        graphs['omnivore', 'to', 'text'].edge_index = torch.tensor(np.array([text_hetero_node_source, text_hetero_node_target], dtype=np.int32), dtype=torch.long)
        graphs['omnivore', 'to', 'text'].edge_attr = torch.tensor(text_hetero_edge_attr, dtype=torch.float32)


        graphs['text', 'to', 'text'].edge_index = torch.tensor(np.array([text_node_source, text_node_target], dtype=np.int32), dtype=torch.long)
        graphs['text', 'to', 'text'].edge_attr = torch.tensor(text_edge_attr, dtype=torch.float32)
        assert graphs["text"].x.shape[0] == graphs["omnivore"].x.shape[0], f'Number of nodes for text: {graphs["text"].x.shape[0]}, Number of nodes for omnivore: {graphs["omnivore"].x.shape[0]}'


    # labels for omnivore nodes 
    graphs['omnivore'].y = torch.tensor(np.array(label, dtype=np.int16)[::args.sample_rate], dtype=torch.long)


    # if split == 'test':
    #    torch.save(graphs, os.path.join(path_graphs, 'test', f'{take_name}.pt'))
    #    print(f'Saved graph for {take_name} to {os.path.join(path_graphs, "test", take_name + ".pt")}')
    #    return
    if take_name in train_ids:
        torch.save(graphs, os.path.join(path_graphs, 'train', f'{take_name}.pt'))
        print(f'Saved graph for {take_name} to {os.path.join(path_graphs, "train", take_name + ".pt")}')
    else:
        torch.save(graphs, os.path.join(path_graphs, 'val', f'{take_name}.pt'))
        print(f'Saved graph for {take_name} to {os.path.join(path_graphs, "val", take_name + ".pt")}')



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
    parser.add_argument('--add_multiview',   help='Whether to add multiview features', action="store_true")
    parser.add_argument('--add_text',   help='Whether to add text features', action="store_true")
    parser.add_argument('--add_spatial',   help='Whether to add spatial features', action="store_true")
    parser.add_argument('--crop',   type=bool,   help='Crop action_start and action_end', default=False)
    
    args = parser.parse_args()

    # load config file
    if args.cfg is not None:
        cfg = get_cfg(args)
        print(cfg)


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
        if split == 'test':
            continue
        # Get a list of training video ids
        if split != 'test':
            print(f'Reading splits at {os.path.join(args.root_data, f"annotations/{args.dataset}/splits/train.{split}.bundle")}')
            with open(os.path.join(args.root_data, f'annotations/{args.dataset}/splits/train.{split}.bundle')) as f:
                train_ids = [os.path.splitext(line.strip())[0] for line in f]
                print(f'Number of training videos: {len(train_ids)}')
        else:
            train_ids = []


        # path_graphs = os.path.join(args.root_data, f'graphs/{args.features}_{args.tauf}_{args.skip_factor}/{split}')
        path_graphs = os.path.join(args.root_data, f'graphs/{cfg["graph_name"]}/{split}')
        
        if split == 'test':
            os.makedirs(os.path.join(path_graphs, 'test'), exist_ok=True)
        else:
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
            args.text_dir = os.path.join(args.root_data, f'annotations/{cfg["annotations_dataset"]}/{cfg["text_dataset"]}/')

        multidepth_data_files = {}
        if args.add_spatial:
            args.spatial_dir = cfg['spatial_dataset']
            if args.add_multiview:

                # get number of views
                print(os.path.join(args.root_data, f'features/{args.spatial_dir}-exo/*/*.npy'))
                for multiview_data in sorted(glob.glob(os.path.join(args.root_data, f'features/{args.spatial_dir}-exo/*/*.npy'))):
                    vid = os.path.basename(multiview_data).split('.npy')[0]
                    data_sp = 'train'
                    if vid not in train_ids:
                        data_sp = 'val'
                    matching_data_file = os.path.join(args.root_data, f'features/{args.features}/{split}/{data_sp}/{vid}_0.npy')
                    assert matching_data_file in list_data_files, f'check {matching_data_file}'
                    # if matching_data_file not in multidepth_data_files:
                    #     print(f'{matching_data_file} not in omnivore ego files')
                    if matching_data_file not in multidepth_data_files:
                        multidepth_data_files[matching_data_file] = []
                    multidepth_data_files[matching_data_file].append(multiview_data)


        # print(f'Multiview depth files {multidepth_data_files}')


        #with Pool(processes=35) as pool:
        #    pool.map(partial(generate_temporal_graph, args=args, path_graphs=path_graphs, actions=actions, train_ids=train_ids, all_ids=all_ids), list_data_files)
        for data_file in list_data_files:
            generate_heterogeneous_temporal_graph(data_file, args=args, path_graphs=path_graphs, actions=actions, train_ids=train_ids, all_ids=all_ids, list_multiview_data_files=multiview_data_files[data_file] if data_file in multiview_data_files else '', list_multidepth_data_files=multidepth_data_files[data_file] if data_file in multidepth_data_files else [])


        print (f'Graph generation for {split} is finished')
