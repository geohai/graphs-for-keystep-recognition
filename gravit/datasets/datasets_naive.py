import os
import glob
import torch
from torch.utils.data import Dataset
from gravit.utils.data_loader import load_features, load_labels
import numpy as np


# class EgoExoOmnivoreDataset(Dataset):
#     """
#     General class for graph dataset
#     """

#     def __init__(self, path_graphs):
#         super(EgoExoOmnivoreDataset, self).__init__()
#         self.all_graphs = sorted(glob.glob(os.path.join(path_graphs, '*.pt')))
#         print('Length of dataset: ', len(self.all_graphs))

#     def __len__(self):
#         return len(self.all_graphs)

#     def __getitem__(self, idx):
#         data = torch.load(self.all_graphs[idx])
#         print(data.shape)
#         return data



# Simple dataset for non-graph structured data
class EgoExoOmnivoreDataset(Dataset):
    def __init__(self, split, features_dataset, annotations_dataset, validation=False, eval_mode=False):
        self.root_data = './data'
        self.is_multiview = None
        self.dataset = features_dataset
        self.annotations = annotations_dataset
        self.data_files = []
        self.split = split
        self.total_dimensions = 0
        self.validation = validation
        self.eval_mode = eval_mode
        
        # one hot encoding
        self.actions = self.__load_action_classes_mapping__()
        self.num_classes = len(self.actions)  # Assuming self.actions is a dictionary mapping class names to indices
        print(f'Number of classes: {self.num_classes}')

        # list of all feature files
        if validation == True:
            # if self.is_multiview:
            #     self.data_files = sorted(glob.glob(os.path.join(self.root_data, f'features/{self.dataset}/split{self.split}/val/*_0.npy')))
            # else:
            self.data_files = sorted(glob.glob(os.path.join(self.root_data, f'features/{self.dataset}/split{self.split}/val/*.npy')))
        
        else:
             # if self.is_multiview:
            #     self.data_files = sorted(glob.glob(os.path.join(self.root_data, f'features/{self.dataset}/split{self.split}/train/*_0.npy')))
            # else:
            self.data_files = sorted(glob.glob(os.path.join(self.root_data, f'features/{self.dataset}/split{self.split}/train/*.npy')))
            print(f'Loading data from {self.root_data}/features/{self.dataset}/split{self.split}/train/*.npy')
        

        # # Load and sum the dimensions of all data files
        # for data_file in self.data_files:
        #     data = np.load(data_file)
        #     self.total_dimensions += data.shape[0]

        # # build a mapping from val in total dimensions to a file+frame
        # self.val_to_file_frame = {}
        # start = 0
        # for data_file in self.data_files:
        #     data = np.load(data_file)
        #     end = start + data.shape[0]
        #     for i in range(start, end):
        #         self.val_to_file_frame[i] = (data_file, i-start)
        #     start = end
        # print('Number of samples: ', self.total_dimensions)

        # build a mapping from val in total dimensions to a file+frame
        self.val_to_file_frame = {}
        start = 0
        for i, data_file in enumerate(self.data_files):
            self.val_to_file_frame[i] = data_file
        self.total_dimensions = len(self.val_to_file_frame)

        print('Number of videos: ', self.total_dimensions)
      
    def __len__(self):
        return self.total_dimensions

    def __getitem__(self, idx):
        data_file = self.val_to_file_frame[idx] # frame_num == segment_num in the video

        video_id = os.path.splitext(os.path.basename(data_file))[0]

        # if self.is_multiview is not None and self.is_multiview == True:
        #     video_id = video_id[0:-2] 

        # Load the features and labels
        feature = load_features(data_file)
        take_name = os.path.splitext(os.path.basename(data_file))[0]
        take_name = take_name.rsplit('_', 1)[0]
        actions = self.actions
        
        label = load_labels(video_id=take_name, actions=actions, root_data='./data', annotation_dataset=self.annotations) 
        
        if len(feature) != len(label):
            print(take_name)
            print(f'Length of feature: {len(feature)} | Length of label: {len(label)}')
            raise ValueError('Length of feature and label does not match')


        # # now get the specific frame (segment)
        # feature = feature[frame_num]
        # label = label[frame_num]

        # # One-hot encode the label
        # label_one_hot = torch.zeros(self.num_classes)
        # label_one_hot[label] = 1
        # label = label_one_hot

        
        # one hot encode label
        label_one_hot = torch.zeros(len(label), self.num_classes) # (num_segments, num_classes)

        for i in range(len(label)):
            sample_label = label[i]
            label_one_hot[i][sample_label] = 1

        label = label_one_hot


        # feature = torch.tensor(feature).unsqueeze(0)  # Add batch dimension
        # label = label.clone().detach().to(dtype=torch.float)  # Add batch dimension

        # print(feature.shape)
        # print(label.shape)


        if self.eval_mode:
            samples = [(feature[i], label[i], video_id) for i in range(len(feature))]
            return samples

        samples = [(feature[i], label[i]) for i in range(len(feature))]
        return samples
    
    def __load_action_classes_mapping__(self):
        # Build a mapping from action classes to action ids
        actions = {}
        with open(os.path.join(self.root_data, f'annotations/{self.annotations}/mapping.txt')) as f:
            for line in f:
                aid, cls = line.strip().split(' ')
                actions[cls] = int(aid)
        return actions



# # Simple dataset for non-graph structured data
# class EgoExoOmnivoreDataset(Dataset):
#     def __init__(self, split, features_dataset, annotations_dataset, validation=False, eval_mode=False):
#         self.root_data = './data'
#         self.is_multiview = None
#         self.dataset = features_dataset
#         self.annotations = annotations_dataset
#         self.data_files = []
#         self.split = split
#         self.total_dimensions = 0
#         self.validation = validation
#         self.eval_mode = eval_mode
        
#         # one hot encoding
#         self.actions = self.__load_action_classes_mapping__()
#         self.num_classes = len(self.actions)  # Assuming self.actions is a dictionary mapping class names to indices
#         print(f'Number of classes: {self.num_classes}')

#         # list of all feature files
#         if validation == True:
#             # if self.is_multiview:
#             #     self.data_files = sorted(glob.glob(os.path.join(self.root_data, f'features/{self.dataset}/split{self.split}/val/*_0.npy')))
#             # else:
#             self.data_files = sorted(glob.glob(os.path.join(self.root_data, f'features/{self.dataset}/split{self.split}/val/*.npy')))
        
#         else:
#              # if self.is_multiview:
#             #     self.data_files = sorted(glob.glob(os.path.join(self.root_data, f'features/{self.dataset}/split{self.split}/train/*_0.npy')))
#             # else:
#             self.data_files = sorted(glob.glob(os.path.join(self.root_data, f'features/{self.dataset}/split{self.split}/train/*.npy')))
#             print(f'Loading data from {self.root_data}/features/{self.dataset}/split{self.split}/train/*.npy')
        

#         # Load and sum the dimensions of all data files
#         for data_file in self.data_files:
#             data = np.load(data_file)
#             self.total_dimensions += data.shape[0]

#         # build a mapping from val in total dimensions to a file+frame
#         self.val_to_file_frame = {}
#         start = 0
#         for data_file in self.data_files:
#             data = np.load(data_file)
#             end = start + data.shape[0]
#             for i in range(start, end):
#                 self.val_to_file_frame[i] = (data_file, i-start)
#             start = end

#         print('Number of samples: ', self.total_dimensions)
      
#     def __len__(self):
#         return self.total_dimensions

#     def __getitem__(self, idx):
#         data_file, frame_num = self.val_to_file_frame[idx] # frame_num == segment_num in the video

#         video_id = os.path.splitext(os.path.basename(data_file))[0]
#         # if self.is_multiview is not None and self.is_multiview == True:
#         #     video_id = video_id[0:-2] 

#         # Load the features and labels
#         feature = load_features(data_file)
#         take_name = os.path.splitext(os.path.basename(data_file))[0]
#         take_name = take_name.rsplit('_', 1)[0]
#         actions = self.actions
        
#         label = load_labels(video_id=take_name, actions=actions, root_data='./data', annotation_dataset=self.annotations) 
        
#         if len(feature) != len(label):
#             print(take_name)
#             print(f'Length of feature: {len(feature)} | Length of label: {len(label)}')
#             raise ValueError('Length of feature and label does not match')

#         # print(f'Length of feature: {len(feature)} | Length of label: {len(label)} | Frame number: {frame_num}')
        
   
#         # now get the specific frame (segment)
#         feature = feature[frame_num]
#         label = label[frame_num]

#         # One-hot encode the label
#         label_one_hot = torch.zeros(self.num_classes)
#         label_one_hot[label] = 1
#         label = label_one_hot


#         feature = torch.tensor(feature).unsqueeze(0)  # Add batch dimension
#         label = label.clone().detach().to(dtype=torch.float)  # Add batch dimension

#         if self.eval_mode:
#             return feature, label, video_id, frame_num

#         return feature, label
    
#     def __load_action_classes_mapping__(self):
#         # Build a mapping from action classes to action ids
#         actions = {}
#         with open(os.path.join(self.root_data, f'annotations/{self.annotations}/mapping.txt')) as f:
#             for line in f:
#                 aid, cls = line.strip().split(' ')
#                 actions[cls] = int(aid)
#         return actions

