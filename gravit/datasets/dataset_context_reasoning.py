import os
import glob
import torch
from torch_geometric.data import Data, Dataset
from gravit.utils.data_loader import load_and_fuse_modalities, load_labels
import numpy as np

class GraphDataset(Dataset):
    """
    General class for graph dataset
    """

    def __init__(self, path_graphs):
        super(GraphDataset, self).__init__()
        self.all_graphs = sorted(glob.glob(os.path.join(path_graphs, '*.pt')))

    def len(self):
        return len(self.all_graphs)

    def get(self, idx):
        data = torch.load(self.all_graphs[idx])
        return data



# Simple dataset for non-graph structured data
class EgoExoOmnivoreTrainDataset(torch.utils.data.Dataset):
    def __init__(self, split):
        self.root_data = './data'
        self.is_multiview = None
        self.crop = False
        self.dataset = 'egoexo-omnivore-aria'
        self.tauf = 10
        self.skip_factor = 10
        self.data_files = []
        self.split = split
        self.sample_rate = 1
        self.total_dimensions = 0
        
        # one hot encoding
        self.actions = self.__load_action_classes_mapping__()
        self.num_classes = len(self.actions)  # Assuming self.actions is a dictionary mapping class names to indices

        # list of all feature files
        self.data_files = sorted(glob.glob(os.path.join(self.root_data, f'features/{self.dataset}/split{self.split}/*.npy')))

        if self.is_multiview:
            self.data_files = sorted(glob.glob(os.path.join(self.root_data, f'features/{self.dataset}/split{self.split}/*_0.npy')))

        # Load and sum the dimensions of all data files
        for data_file in self.data_files:
            data = np.load(data_file)
            self.total_dimensions += data.shape[0]

        # build a mapping from val in total dimensions to a file+frame
        self.val_to_file_frame = {}
        start = 0
        for data_file in self.data_files:
            data = np.load(data_file)
            end = start + data.shape[0]
            for i in range(start, end):
                self.val_to_file_frame[i] = (data_file, i-start)
            start = end
      
    def __len__(self):
        # return the total number of frames in the dataa -> each frame is a sample
        return self.total_dimensions

    def __getitem__(self, idx):
        data_file, frame_num = self.val_to_file_frame[idx]

        video_id = os.path.splitext(os.path.basename(data_file))[0]
        if self.is_multiview is not None and self.is_multiview == True:
            video_id = video_id[0:-2] 

        # Load the features and labels
        feature = load_and_fuse_modalities(data_file, 'concat', video_id, root_data=self.root_data, 
                                           dataset=self.dataset, sample_rate=self.sample_rate, is_multiview=self.is_multiview)
       
        label = load_labels(video_id=video_id, actions=self.actions, root_data=self.root_data, dataset=self.dataset, sample_rate=self.sample_rate, feature=feature)
   
        # now get the specific frame
        feature = feature[frame_num]
        label = label[frame_num]

        if self.crop == True:
            feature, label = self.__crop_to_start_and_end__(feature, label)

        
        # One-hot encode the label
          # Assuming self.actions is a dictionary mapping class names to indices
        label_one_hot = torch.zeros(self.num_classes)
        label_one_hot[label] = 1
        label = label_one_hot

        feature = torch.tensor(feature).unsqueeze(0)  # Add batch dimension
        label = torch.tensor(label, dtype=torch.float)  # Add batch dimension

        # print(feature.shape, label.shape)
        # print('***************')

        return feature, label
    

    def __load_action_classes_mapping__(self):
        # Build a mapping from action classes to action ids
        actions = {}
        with open(os.path.join(self.root_data, f'annotations/{self.dataset}/mapping.txt')) as f:
            for line in f:
                aid, cls = line.strip().split(' ')
                actions[cls] = int(aid)
        return actions

    def __crop_to_start_and_end__(self, feature, label):
        # crop to start and end
        begin_label_index = label.index(next(x for x in label if x != "action_start"))
        feature = feature[begin_label_index:]
        label = label[begin_label_index:]

        end_label_index = label.index(next(x for x in label if x != "action_end"))
        feature = feature[:end_label_index]
        label = label[:end_label_index]
        return feature, label

                    
# Simple dataset for non-graph structured data
class EgoExoOmnivoreValDataset(torch.utils.data.Dataset):
    def __init__(self, split):
        self.root_data = './data'
        self.is_multiview = None
        self.crop = False
        self.dataset = 'egoexo-omnivore-aria'
        self.tauf = 10
        self.skip_factor = 10
        self.data_files = []
        self.split = split
        self.sample_rate = 1
        self.total_dimensions = 0
        
        # one hot encoding
        self.actions = self.__load_action_classes_mapping__()
        self.num_classes = len(self.actions)  # Assuming self.actions is a dictionary mapping class names to indices

        # list of all feature files
        print('Assuming 5 splits for now')
        for i in range(1, 6):
            if i == split:
                continue
                    # list of all feature files
            self.data_files = sorted(glob.glob(os.path.join(self.root_data, f'features/{self.dataset}/split{self.split}/*.npy')))

            if self.is_multiview:
                self.data_files = sorted(glob.glob(os.path.join(self.root_data, f'features/{self.dataset}/split{self.split}/*_0.npy')))

        self.data_files.sort()

        # Load and sum the dimensions of all data files
        for data_file in self.data_files:
            data = np.load(data_file)
            self.total_dimensions += data.shape[0]

        # build a mapping from val in total dimensions to a file+frame
        self.val_to_file_frame = {}
        start = 0
        for data_file in self.data_files:
            data = np.load(data_file)
            end = start + data.shape[0]
            for i in range(start, end):
                self.val_to_file_frame[i] = (data_file, i-start)
            start = end
      
    def __len__(self):
        # return the total number of frames in the dataa -> each frame is a sample
        return self.total_dimensions

    def __getitem__(self, idx):
        data_file, frame_num = self.val_to_file_frame[idx]

        video_id = os.path.splitext(os.path.basename(data_file))[0]
        if self.is_multiview is not None and self.is_multiview == True:
            video_id = video_id[0:-2] 

        # Load the features and labels
        feature = load_and_fuse_modalities(data_file, 'concat', video_id, root_data=self.root_data, 
                                           dataset=self.dataset, sample_rate=self.sample_rate, is_multiview=self.is_multiview)
       
        label = load_labels(video_id=video_id, actions=self.actions, root_data=self.root_data, dataset=self.dataset, sample_rate=self.sample_rate, feature=feature)
   
        # now get the specific frame
        feature = feature[frame_num]
        label = label[frame_num]

        if self.crop == True:
            feature, label = self.__crop_to_start_and_end__(feature, label)

        
        # One-hot encode the label
          # Assuming self.actions is a dictionary mapping class names to indices
        label_one_hot = torch.zeros(self.num_classes)
        label_one_hot[label] = 1
        label = label_one_hot

        feature = torch.tensor(feature).unsqueeze(0)  # Add batch dimension
        label = torch.tensor(label, dtype=torch.float)  # Add batch dimension

        # print(feature.shape, label.shape)
        # print('***************')

        return feature, label
    

    def __load_action_classes_mapping__(self):
        # Build a mapping from action classes to action ids
        actions = {}
        with open(os.path.join(self.root_data, f'annotations/{self.dataset}/mapping.txt')) as f:
            for line in f:
                aid, cls = line.strip().split(' ')
                actions[cls] = int(aid)
        return actions

    def __crop_to_start_and_end__(self, feature, label):
        # crop to start and end
        begin_label_index = label.index(next(x for x in label if x != "action_start"))
        feature = feature[begin_label_index:]
        label = label[begin_label_index:]

        end_label_index = label.index(next(x for x in label if x != "action_end"))
        feature = feature[:end_label_index]
        label = label[:end_label_index]
        return feature, label
    
