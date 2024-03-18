import os
import sys
sys.path.append('/home/blde8334/research/GraVi-T-custom') # Added for Blake's sys path
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from gravit.utils.parser import get_args, get_cfg
from gravit.utils.logger import get_logger
from gravit.models import build_model, get_loss_func
# from gravit.datasets import EgoExoOmnivoreDataset, GraphDataset

import os
import glob
import torch
from torch.utils.data import Dataset
from gravit.utils.data_loader import load_and_fuse_modalities, load_labels
import numpy as np
import os
import scipy
import numpy as np
import re
# Simple dataset for non-graph structured data
class EgoExoOmnivoreDataset(Dataset):
    def __init__(self, split, features_dataset, annotations_dataset, load_raw_labels=False, validation=False, eval_mode=False):
        print('Initializing EgoExoOmnivoreDataset')
        self.root_data = './data'
        self.is_multiview = None
        self.crop = False
        self.dataset = features_dataset
        self.annotations = annotations_dataset
        self.tauf = 10
        self.skip_factor = 10
        self.data_files = []
        self.split = split
        self.sample_rate = 1
        self.total_dimensions = 0
        self.validation = validation
        self.eval_mode = eval_mode
        self.load_raw_labels = load_raw_labels
        
        # one hot encoding
        self.actions = self.__load_action_classes_mapping__()
        self.num_classes = len(self.actions)  # Assuming self.actions is a dictionary mapping class names to indices
        print('EgoExoOmnivoreDataset - Number of Classes: ', self.num_classes)

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

        print('EgoExoOmnivoreDataset - Number of samples: ', self.total_dimensions)
      
    def __len__(self):
        # return the total number of frames in the data -> each frame is a sample
        return 500 #self.total_dimensions
        #return 1 # test with smaller dimension

    def __getitem__(self, idx):
        data_file, frame_num = self.val_to_file_frame[idx]

        video_id = os.path.splitext(os.path.basename(data_file))[0]
        if self.is_multiview is not None and self.is_multiview == True:
            video_id = video_id[0:-2]
        # Load the features and labels
        feature = load_and_fuse_modalities(data_file, 'concat', dataset=self.dataset, sample_rate=self.sample_rate, is_multiview=self.is_multiview)
        print('Calling load_labels: ', video_id, self.actions, self.root_data, self.annotations, self.sample_rate, feature, self.load_raw_labels)
        label = load_labels(video_id=video_id, actions=self.actions, root_data=self.root_data, annotation_dataset=self.annotations, 
                            sample_rate=self.sample_rate, feature=feature, load_raw=self.load_raw_labels)
        
        # checkFrameNum = frame_num
        # checkLabelLength = len(label)
        # if checkFrameNum > checkLabelLength:
        #     print("Warning, frame_num greater than label length")
        #     print(f'frame_num: {frame_num}, len(label): {len(label)}')

        # now get the specific frame
        feature = feature[frame_num]
        label = label[frame_num]
        # Print the shape of the feature
        print("Feature shape:", feature.shape)
        if self.crop == True:
            feature, label = self.__remove_start_and_end__(feature, label)

        # One-hot encode the label
          # Assuming self.actions is a dictionary mapping class names to indices
        label_one_hot = torch.zeros(self.num_classes)
        label_one_hot[label] = 1
        label = label_one_hot

        feature = torch.tensor(feature).unsqueeze(0)  # Add batch dimension
        label = label.clone().detach().to(dtype=torch.float)  # Add batch dimension

        if self.eval_mode:
            return feature, label, video_id, frame_num

        return feature, label
    
    def __load_action_classes_mapping__(self):
        # Build a mapping from action classes to action ids
        actions = {}
        with open(os.path.join(self.root_data, f'annotations/{self.dataset}/mapping.txt')) as f:
            for line in f:
                aid, cls = line.strip().split(' ')
                actions[cls] = int(aid)
        return actions

    def __remove_start_and_end__(self, feature, label):
        # remove all samples with labels "action_start" and "action_end"
        keep_indices = [i for i, x in enumerate(label) if x != "action_start"]
        feature = [feature[i] for i in keep_indices]
        label = [label[i] for i in keep_indices]

        keep_indices = [i for i, x in enumerate(label) if x != "action_end"]
        feature = [feature[i] for i in keep_indices]
        label = [label[i] for i in keep_indices]

        return feature, label



def load_labels(actions, root_data, annotation_dataset, video_id, sample_rate=1, feature=None, load_raw=False):
    print('-------------------------------')
    print(annotation_dataset)
    if 'egoexo-omnivore' in annotation_dataset:
        print('Loading egoexo-omnivore labels...')
        label = load_egoexo_omnivideo_integer_labels(video_id, actions, root_data=root_data, dataset=annotation_dataset, feature=feature, load_raw=load_raw)
    else:
        print('Loading regular labels...')
        label = load_and_trim_labels(video_id, actions, root_data=root_data, dataset=annotation_dataset, sample_rate=sample_rate, feature=feature)
    
    return label


def load_egoexo_omnivideo_integer_labels(video_id, actions, root_data='../data', dataset='egoexo',feature=None, load_raw=False):
    """
    Loads EgoExo Labels that correspond with Omnivore Sampling/Windowing strategy. Assuming 30Hz sampling rate, the window size 
    is 32 frames, and the stride is 16 frames. Aggregate the labels for these windows by taking the mode of the labels in each window.
    """
    # Get a list of ground-truth action labels
    try:
        with open(os.path.join(root_data, f'annotations/{dataset}/groundTruth/{video_id}.txt')) as f:
            label = [actions[line.strip()] for line in f]
            
    except:
        video_id = video_id.rsplit('_', 1)[0]
        # print(video_id)
        with open(os.path.join(root_data, f'annotations/{dataset}/groundTruth/{video_id}.txt')) as f:
            label = [actions[line.strip()] for line in f]
        # print('Success')


    if load_raw == False:
        new_labels = [scipy.stats.mode(label[i:i+32])[0] for i in range(0, len(label), 16) if i+32 < len(label)]
        # add last label to the end
        if len(new_labels) < feature.shape[0]:
            last_window = label[-32:]
            new_labels.append(scipy.stats.mode(last_window)[0])

        label = new_labels

    return label


def load_and_trim_labels(video_id, actions, root_data='../data', dataset='50salads', sample_rate=1, feature=None):
    """
    feature: must input np array or pd dataframe of features to trim the labels to
    This function loads the ground truth annotations for a video, and trims them to match the length of the features.
    For BrP 50Salads feature/label pairs, since labels tend to be shorter than features, and they appear to be aligned at the start.

    """
    # Get a list of ground-truth action labels
    print(os.path.join(root_data, f'annotations/{dataset}/groundTruth/{video_id}.txt'))
    try:
        with open(os.path.join(root_data, f'annotations/{dataset}/groundTruth/{video_id}.txt')) as f:
            label = [actions[line.strip()] for line in f]
    except:
        video_id = video_id.rsplit('_', 1)[0]
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




def train(cfg):
    """
    Run the training process given the configuration
    """

    # Input and output paths
    path_graphs = os.path.join(cfg['root_data'], f'graphs/{cfg["graph_name"]}')
    if cfg['split'] is not None:
        path_graphs = os.path.join(path_graphs, f'split{cfg["split"]}')
    path_result = os.path.join(cfg['root_result'], f'{cfg["exp_name"]}')
    os.makedirs(path_result, exist_ok=True)
    print("cfg:",cfg)

    # Prepare the logger and save the current configuration for future reference
    logger = get_logger(path_result, file_name='train')
    logger.info(cfg['exp_name'])
    logger.info('Saving the configuration file')
    with open(os.path.join(path_result, 'cfg.yaml'), 'w') as f:
        yaml.dump({k: v for k, v in cfg.items() if v is not None}, f, default_flow_style=False, sort_keys=False)

    # Build a model and prepare the data loaders
    logger.info('Preparing a model and data loaders')
    device = 'cpu' #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device:",device)
    model = build_model(cfg, device)
    print("model:",device)


    

    ## Use the EgoExoOmnivore dataset
    print('Calling the Dataloader: split:',cfg['split'],'features_dataset:',cfg['features_dataset'],'annotations_dataset:',cfg['annotations_dataset'],'validation:',False,'eval_mode:',False,'load_raw_labels:',True)
    h = EgoExoOmnivoreDataset(cfg['split'], features_dataset=cfg['features_dataset'], annotations_dataset=cfg['annotations_dataset'], validation=False, eval_mode=False, load_raw_labels=True)
    # print('Done')
    # quit()

    train_loader = DataLoader(EgoExoOmnivoreDataset(cfg['split'], features_dataset=cfg['features_dataset'], annotations_dataset=cfg['annotations_dataset'], validation=False, eval_mode=False, load_raw_labels=True),
                               batch_size=cfg['batch_size'], shuffle=True, num_workers=128)
    val_loader = DataLoader(EgoExoOmnivoreDataset(cfg['split'], features_dataset=cfg['features_dataset'], annotations_dataset=cfg['annotations_dataset'], validation=True, eval_mode=False, load_raw_labels=True), 
                            batch_size=cfg['batch_size'], shuffle=False, num_workers=128)



    # Prepare the experiment
    loss_func = get_loss_func(cfg)
    loss_func_val = get_loss_func(cfg, 'val')
    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['sch_param'])

    # Run the training process
    logger.info('Training process started')
    print(f'Length of train_loader:', len(train_loader))

    min_loss_val = float('inf')
    for epoch in range(1, cfg['num_epoch']+1):
        print(f'------- Epoch: {epoch} --------')
        model.train()

        # Train for a single epoch
        loss_sum = 0.
        for data in train_loader:
            optimizer.zero_grad()
         
            x, y = data
            # x = x.to(device)
            # y = y.to(device)
            # Added for egoexo-brp-aria
            x = x.float().to(device)
            y = y.float().to(device)
            print("x dtype :",x.dtype)
            print("y dtype :",y.dtype)
            print("x shape :",x.shape)
            print("y shape :",y.shape)
            logits = model(x)
            logits = logits.squeeze(1)
                
            # print(logits.dtype, y.dtype)
            # print(logits.shape, y.shape)

            loss = loss_func(logits, y)
            # print(loss)
            loss.backward()
            loss_sum += loss.item()
            optimizer.step()

        # Adjust the learning rate
        scheduler.step()

        loss_train = loss_sum / len(train_loader)
        print(loss_train)

        # Get the validation loss
        loss_val = val(val_loader, cfg['use_spf'], model, device, loss_func_val)
        print(loss_val)

        # Save the best-performing checkpoint
        if loss_val < min_loss_val:
            min_loss_val = loss_val
            epoch_best = epoch
            torch.save(model.state_dict(), os.path.join(path_result, 'ckpt_best.pt'))

        # Log the losses for every epoch
        logger.info(f'Epoch [{epoch:03d}|{cfg["num_epoch"]:03d}] loss_train: {loss_train:.4f}, loss_val: {loss_val:.4f}, best: epoch {epoch_best:03d}')

    logger.info('Training finished')


def val(val_loader, use_spf, model, device, loss_func):
    """
    Run a single validation process
    """

    model.eval()
    loss_sum = 0
    with torch.no_grad():
        for data in val_loader:  
           
            x, y = data
            #x = x.to(device)
            # y = y.to(device)
            # Added for egoexo-brp-aria
            x = x.float().to(device)
            y = y.float().to(device)
            print("x dtype :",x.dtype)
            print("y dtype :",y.dtype)
            print("x shape :",x.shape)
            print("y shape :",y.shape)
            logits = model(x)
            logits = logits.squeeze(1)
               
            loss = loss_func(logits, y)
            loss_sum += loss.item()

    return loss_sum / len(val_loader)


if __name__ == "__main__":
    args = get_args()
    cfg = get_cfg(args)

    train(cfg)