import torch
from torch.nn import Module, ModuleList, Conv1d, Sequential, ReLU, Dropout, functional as F
from torch_geometric.nn import to_hetero, Linear, EdgeConv, GATv2Conv, SAGEConv, BatchNorm, RGCNConv
import numpy as np
from torch_geometric.data import HeteroData



class DilatedResidualLayer(Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = Conv1d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv_1x1 = Conv1d(out_channels, out_channels, kernel_size=1)
        self.relu = ReLU()
        self.dropout = Dropout()

    def forward(self, x):
        out = self.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + out


# This is for the iterative refinement (we refer to MSTCN++: https://github.com/sj-li/MS-TCN2)
class Refinement(Module):
    def __init__(self, final_dim, num_layers=10, interm_dim=64):
        super(Refinement, self).__init__()
        self.conv_1x1 = Conv1d(final_dim, interm_dim, kernel_size=1)
        self.layers = ModuleList([DilatedResidualLayer(2**i, interm_dim, interm_dim) for i in range(num_layers)])
        self.conv_out = Conv1d(interm_dim, final_dim, kernel_size=1)

    def forward(self, x):
        f = self.conv_1x1(x)
        for layer in self.layers:
            f = layer(f)
        out = self.conv_out(f)
        return out
    


class SPELL_HETEROGENEOUS(Module):
    def __init__(self, cfg):
        super(SPELL_HETEROGENEOUS, self).__init__()
        self.add_text = cfg['add_text']
        self.add_spatial = cfg['add_spatial']
        print('Add text:', self.add_text)
        print('Add spatial:', self.add_spatial)
        self.use_spf = cfg['use_spf'] # whether to use the spatial features
        self.num_modality = cfg['num_modality']
        if self.use_spf:
            self.layer_spf = Linear(-1, cfg['proj_dim']) # projection layer for spatial features

        # manually add the first layer (dependent on node input dim)
        channels = [cfg['channel1'], cfg['channel2']]
        input_dim = cfg['input_dim'] 
        
        self.layer011_omnivore = Linear(input_dim, channels[0]) 
         
        print(f'Add text: {self.add_text}, Add spatial: {self.add_spatial}')

        # hetero gnn
        if self.add_spatial and not self.add_text:
            spatial_input_dim = cfg['spatial_input_dim']
            self.layer011_spatial = Linear(spatial_input_dim, channels[0])
            node_types = ['spatial', 'omnivore']
            edge_types = [ ('omnivore', 'to', 'omnivore'), ('spatial', 'to', 'spatial'), ('omnivore', 'to', 'spatial')] # 

        elif self.add_text and not self.add_spatial:
            
            text_input_dim = cfg['text_input_dim']
            self.layer011_text = Linear(text_input_dim, channels[0])
            node_types = ['text', 'omnivore']
            edge_types = [ ('omnivore', 'to', 'omnivore'), ('omnivore', 'to', 'text') ] # , , ('text', 'to', 'text')


        elif self.add_text and self.add_spatial:
            spatial_input_dim = cfg['spatial_input_dim']
            text_input_dim = cfg['text_input_dim']
            self.layer011_text = Linear(text_input_dim, channels[0])
            self.layer011_spatial = Linear(spatial_input_dim, channels[0])
            node_types = ['text', 'spatial', 'omnivore']
            edge_types = [ ('omnivore', 'to', 'omnivore'), ('omnivore', 'to', 'text'), ('omnivore', 'to', 'spatial') ] # ('spatial', 'to', 'spatial')
        
        else:
            node_types = ['omnivore']
            edge_types = [ ('omnivore', 'to', 'omnivore')]
        
        metadata = (node_types, edge_types)
        base_spell = SPELL(cfg)
        self.model = to_hetero(base_spell, metadata, aggr='sum')
        
        # print('Model:', self.model)

    def forward(self, data: HeteroData, c=None):
        if self.add_text == False and self.add_spatial == False:
            data.x_dict = {'omnivore': data.x_dict['omnivore']}
            data.edge_index_dict = {('omnivore', 'to', 'omnivore'): data.edge_index_dict['omnivore', 'to', 'omnivore']}
            data.edge_attr_dict = {('omnivore', 'to', 'omnivore'): data.edge_attr_dict['omnivore', 'to', 'omnivore']}
        

        if self.use_spf:
            feature_dim = data.x_dict['omnivore'].shape[1]
            x = data.x_dict['omnivore']
            x_omnivore = self.layer011_omnivore(torch.cat((x[:, :feature_dim//self.num_modality], self.layer_spf(c)), dim=1)) 
            if 'text' in data.x_dict.keys():
                feature_dim = data.x_dict['text'].shape[1]
                x = data.x_dict['text']
                x_text = self.layer011_text(torch.cat((x[:, :feature_dim//self.num_modality], self.layer_spf(c)), dim=1)) 
            if 'spatial' in data.x_dict.keys():
                feature_dim = data.x_dict['spatial'].shape[1]
                x = data.x_dict['spatial']
                x_spatial = self.layer011_spatial(torch.cat((x[:, :feature_dim//self.num_modality], self.layer_spf(c)), dim=1))

        else:
            feature_dim = data.x_dict['omnivore'].shape[1]
            x = data.x_dict['omnivore']
            x_omnivore = self.layer011_omnivore(x[:, :feature_dim//self.num_modality])
            if 'text' in data.x_dict.keys():
                feature_dim = data.x_dict['text'].shape[1]
                x = data.x_dict['text']
                x_text = self.layer011_text(x[:, :feature_dim//self.num_modality]) 
            if 'spatial' in data.x_dict.keys():
                feature_dim = data.x_dict['spatial'].shape[1]
                x = data.x_dict['spatial']
                x_spatial = self.layer011_spatial(x[:, :feature_dim//self.num_modality])


        data['omnivore'].x = x_omnivore
        if 'text' in data.x_dict.keys():
            data['text'].x = x_text
        if 'spatial' in data.x_dict.keys():
            data['spatial'].x = x_spatial

        
        # print('---------')
        # for key in data.x_dict.keys():
        #     print(f'Key: {key}, Shape: {data.x_dict[key].shape}')
        # for key in data.edge_index_dict.keys():
        #     print(f'Key: {key}, Shape: {data.edge_index_dict[key].shape}')
        # for key in data.edge_attr_dict.keys():
        #     print(f'Key: {key}, Shape: {data.edge_attr_dict[key].shape}')

        x_dict = self.model(data.x_dict, data.edge_index_dict, data.edge_attr_dict, c)
        
        return x_dict['omnivore']
        

class SPELL(Module):
    def __init__(self, cfg):
        super(SPELL, self).__init__()
        
        self.use_ref = cfg['use_ref']
        self.num_modality = cfg['num_modality']

        channels = [cfg['channel1'], cfg['channel2']]
        final_dim = cfg['final_dim']
        
        num_att_heads = cfg['num_att_heads']
        dropout = cfg['dropout']

        ######
        self.layer11 = EdgeConv(Sequential(Linear(2*channels[0], channels[0]), ReLU(), Linear(channels[0], channels[1])))
        self.batch11 = BatchNorm(channels[1])
        self.layer12 = EdgeConv(Sequential(Linear(2*channels[0], channels[0]), ReLU(), Linear(channels[0], channels[1])))
        self.batch12 = BatchNorm(channels[1])
        self.layer13 = EdgeConv(Sequential(Linear(2*channels[0], channels[0]), ReLU(), Linear(channels[0], channels[1])))
        self.batch13 = BatchNorm(channels[1])

        # self.layer31 = SAGEConv(channels[1], final_dim)
        # self.layer32 = SAGEConv(channels[1], final_dim)
        # self.layer33 = SAGEConv(channels[1], final_dim)
        self.layer31 = RGCNConv(channels[1], final_dim, num_relations=2)
        self.layer32 = RGCNConv(channels[1], final_dim, num_relations=2)
        self.layer33 = RGCNConv(channels[1], final_dim, num_relations=2)

        #####

        
        if self.num_modality == 2:
            self.layer012 = Linear(-1, channels[0])

        self.batch01 = BatchNorm(channels[0])
        self.relu = ReLU()
        self.dropout = Dropout(dropout)

        # self.layer11 = EdgeConv(Sequential(Linear(2*channels[0], channels[0]), ReLU(), Linear(channels[0], channels[0])))
        # self.batch11 = BatchNorm(channels[0])
        # self.layer12 = EdgeConv(Sequential(Linear(2*channels[0], channels[0]), ReLU(), Linear(channels[0], channels[0])))
        # self.batch12 = BatchNorm(channels[0])
        # self.layer13 = EdgeConv(Sequential(Linear(2*channels[0], channels[0]), ReLU(), Linear(channels[0], channels[0])))
        # self.batch13 = BatchNorm(channels[0])

        # if num_att_heads > 0:
        #     self.layer21 = GATv2Conv(channels[0], channels[1], heads=num_att_heads, add_self_loops=False) #DEBUGGING, set add_self_loops to False
        # else:
        #     self.layer21 = SAGEConv(channels[0], channels[1])
        #     num_att_heads = 1
        # self.batch21 = BatchNorm(channels[1]*num_att_heads)

        # self.layer31 = SAGEConv(channels[1]*num_att_heads, final_dim)
        # self.layer32 = SAGEConv(channels[1]*num_att_heads, final_dim)
        # self.layer33 = SAGEConv(channels[1]*num_att_heads, final_dim)

        if self.use_ref:
            self.layer_ref1 = Refinement(final_dim)
            self.layer_ref2 = Refinement(final_dim)
            self.layer_ref3 = Refinement(final_dim)


    def forward(self, x, edge_index, edge_attr, c=None):
        x = self.batch01(x)
        x = self.relu(x)

        edge_index_f = edge_index[:, edge_attr<=0]
        edge_index_b = edge_index[:, edge_attr>=0]
        edge_type = (edge_attr != -2).type(torch.int64)
        edge_type_f = (edge_attr[edge_attr<=0] != -2).type(torch.int64)
        edge_type_b = (edge_attr[edge_attr>=0] != -2).type(torch.int64)


        ######## Forward-graph stream
        x1 = self.layer11(x, edge_index_f)

        x1 = self.batch11(x1)
        x1 = self.relu(x1)
        x1 = self.dropout(x1)
        # x1 = self.layer21(x1, edge_index_f)
        # x1 = self.batch21(x1)
        # x1 = self.relu(x1)
        # x1 = self.dropout(x1)

        ######## Backward-graph stream
        x2 = self.layer12(x, edge_index_b)
        x2 = self.batch12(x2)
        x2 = self.relu(x2)
        x2 = self.dropout(x2)
        # x2 = self.layer21(x2, edge_index_b)
        # x2 = self.batch21(x2)
        # x2 = self.relu(x2)
        # x2 = self.dropout(x2)

        ######## Undirected-graph stream
        x3 = self.layer13(x, edge_index)
        x3 = self.batch13(x3)
        x3 = self.relu(x3)
        x3 = self.dropout(x3)
        # x3 = self.layer21(x3, edge_index)
        # x3 = self.batch21(x3)
        # x3 = self.relu(x3)
        # x3 = self.dropout(x3)

        x1 = self.layer31(x1, edge_index_f, edge_type_f)
        x2 = self.layer32(x2, edge_index_b, edge_type_b)
        x3 = self.layer33(x3, edge_index, edge_type)

        out = x1+x2+x3
            
        
        if self.use_ref:
            xr0 = torch.permute(out, (1, 0)).unsqueeze(0)
            xr1 = self.layer_ref1(torch.softmax(xr0, dim=1))
            xr2 = self.layer_ref2(torch.softmax(xr1, dim=1))
            xr3 = self.layer_ref3(torch.softmax(xr2, dim=1))
            out = torch.stack((xr0, xr1, xr2, xr3), dim=0).squeeze(1).transpose(2, 1).contiguous()

        return out
