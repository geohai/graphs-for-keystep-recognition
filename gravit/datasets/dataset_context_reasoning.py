import os
import glob
import torch
from torch_geometric.data import Dataset

class GraphDataset(Dataset):
    """
    General class for graph dataset
    """

    def __init__(self, path_graphs):
        super(GraphDataset, self).__init__()
        self.all_graphs = sorted(glob.glob(os.path.join(path_graphs, '*.pt')))
        print('Length of dataset: ', len(self.all_graphs))

    def len(self):
        return len(self.all_graphs)

    def get(self, idx):
        data = torch.load(self.all_graphs[idx])
        return data

class TestGraphDataset(Dataset):
    """
    General class for graph dataset
    """

    def __init__(self, path_graphs):
        super(TestGraphDataset, self).__init__()
        self.all_graphs = sorted(glob.glob(os.path.join(path_graphs, '*.pt')))
        print('Length of dataset: ', len(self.all_graphs))

    def len(self):
        return len(self.all_graphs)

    def get(self, idx):
        sample = self.all_graphs[idx]
        fn = os.path.basename(sample)
        take_name = fn.split('.pt')[0]
        data = torch.load(sample)
        return (take_name, data)

