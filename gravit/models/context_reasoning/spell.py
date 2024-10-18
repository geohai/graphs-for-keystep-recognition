import torch
from torch.nn import Module, ModuleList, Conv1d, Sequential, ReLU, Dropout, functional as F
from torch_geometric.nn import Linear, EdgeConv, GATv2Conv, SAGEConv, BatchNorm
import torch_geometric
import numpy as np
# from gravit.utils.batch_process import graph_to_nn_batch, nn_batch_to_graph, multiview_graph_to_nn_batch


# def multiview_graph_to_nn_batch(x, batch_idxs, view_idxs, max_seq_len=25): 
#     batch_numbers = torch.unique(batch_idxs).sort()[0]
#     view_numbers = torch.unique(view_idxs).sort()[0]

#     batch_tensors = []
#     for batch_num in batch_numbers:
#         for view_num in view_numbers:
#             indices = torch.where((batch_idxs == batch_num) & (view_idxs == view_num))[0]
#             batch_data = torch.index_select(x, 0, indices)
#             padded_data = F.pad(batch_data, (0, 0, 0, max_seq_len - batch_data.size(0)))
#             padded_data = padded_data[:max_seq_len, :]
#             padded_data = padded_data.unsqueeze(0).transpose(1, 2)
#             batch_tensors.append(padded_data)

#     return torch.cat(batch_tensors, dim=0)


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
    

# class SubgraphConv1D(Module):
#     def __init__(self, in_channels, out_channels, max_seq_len=40, kernel_size=3, stride=1, padding=1):
#         super(SubgraphConv1D, self).__init__()
#         self.conv1D = Conv1d(in_channels, out_channels, kernel_size, stride, padding)
#         self.batch_norm = BatchNorm(out_channels)
#         self.relu = ReLU()
#         self.dropout = Dropout()
#         self.max_seq_len = max_seq_len
#         self.output_dim = int((self.max_seq_len - kernel_size + 2*padding) / stride + 1)

#     def forward(self, x, batch):
#         x = graph_to_nn_batch(x, batch, max_seq_len=self.max_seq_len)
#         out = self.conv1D(x)
#         out = self.batch_norm(out)
#         out = self.relu(out)
#         out = self.dropout(out)
#         out,batch = nn_batch_to_graph(out, batch, self.output_dim, self.max_seq_len)
#         return out, batch


class SPELL(Module):
    def __init__(self, cfg, save_feats=False):
        super(SPELL, self).__init__()
        self.use_spf = cfg['use_spf'] # whether to use the spatial features
        self.use_ref = cfg['use_ref']
        self.num_modality = cfg['num_modality']
        self.save_feats = save_feats

        channels = [cfg['channel1'], cfg['channel2']]
        final_dim = cfg['final_dim']
        input_dim = cfg['input_dim'] 
        
        num_att_heads = cfg['num_att_heads']
        dropout = cfg['dropout']

        # ######## SUBGRAPH AGGREGATION ########
        # self.max_seq_len = 25
        # mamba_config= MambaConfig(vocab_size=497, hidden_size=1536, num_hidden_layers=4) # ~30M params with this config
        # model = MambaModel(mamba_config)
        # model = MambaSeqEmbedding(model) # wrapper to access the mamba features

        # # self.subgraph_agg = model
        # self.subgraph_agg = torch.nn.DataParallel(model) # Distribute Mamba across available GPUs
        # # self.average_subgraph = torch_geometric.nn.pool.global_mean_pool # for calculating segment average features
        # ######################################

        if self.use_spf:
            self.layer_spf = Linear(-1, cfg['proj_dim']) # projection layer for spatial features

        self.layer011 = Linear(input_dim, channels[0]) 
        if self.num_modality == 2:
            self.layer012 = Linear(-1, channels[0])

        self.batch01 = BatchNorm(channels[0])
        self.relu = ReLU()
        self.dropout = Dropout(dropout)

        self.layer11 = EdgeConv(Sequential(Linear(2*channels[0], channels[0]), ReLU(), Linear(channels[0], channels[0])))
        self.batch11 = BatchNorm(channels[0])
        self.layer12 = EdgeConv(Sequential(Linear(2*channels[0], channels[0]), ReLU(), Linear(channels[0], channels[0])))
        self.batch12 = BatchNorm(channels[0])
        self.layer13 = EdgeConv(Sequential(Linear(2*channels[0], channels[0]), ReLU(), Linear(channels[0], channels[0])))
        self.batch13 = BatchNorm(channels[0])

        if num_att_heads > 0:
            self.layer21 = GATv2Conv(channels[0], channels[1], heads=num_att_heads)
        else:
            self.layer21 = SAGEConv(channels[0], channels[1])
            num_att_heads = 1
        self.batch21 = BatchNorm(channels[1]*num_att_heads)

        self.layer31 = SAGEConv(channels[1]*num_att_heads, final_dim)
        self.layer32 = SAGEConv(channels[1]*num_att_heads, final_dim)
        self.layer33 = SAGEConv(channels[1]*num_att_heads, final_dim)

        if self.use_ref:
            self.layer_ref1 = Refinement(final_dim)
            self.layer_ref2 = Refinement(final_dim)
            self.layer_ref3 = Refinement(final_dim)


    def forward(self, x, edge_index, edge_attr, c=None, batch=None, view_idx=None):

        # ################################################################################
        ## Mamba

        # # when there are multiple views in the graph
        # if view_idx is not None and torch.unique(view_idx).shape[0] > 1:
        #     # print(f'x: {x.shape} | batch: {torch.unique(batch).shape}, {batch.shape} | view_idx: {torch.unique(view_idx).shape}')
            

        #     ####### Uncomment to calculate subgraph average features #######
        #     # # create new batch index for multiview using batch and view_idx
        #     # batch_view = batch * torch.unique(view_idx).shape[0] + view_idx
        #     # avg = self.average_subgraph(x, batch=batch_view.to(x.device))
        #     # # sort avg by the batch_view indices
        #     # avg = avg[torch.unique(batch_view).argsort()]
        #     # # avg is in order of b0v0, b0v1, b0v2,
        #     ################################################################


        #     # print(f'x: {x.shape} | batch: {torch.unique(batch).shape} | view_idx: {torch.unique(view_idx).shape}')
        #     x = multiview_graph_to_nn_batch(x, batch, view_idx, max_seq_len=self.max_seq_len) # parse graph to tensor to input to mamba [num_views*batch_size, feature_dim, max_seq_len]
        #     x = x.reshape(x.shape[0], x.shape[2], x.shape[1]) # rearrange to [num_views*batch_size, max_seq_len, feature_dim]
        #     x = self.subgraph_agg(x) # run through mamba
        #     edge_index, edge_attr = self.rebuild_edge_matrix(x, view_idx)  # reconstruct edge matrix based on the new segmentwise features, view_idx only needed to get num_views
        
        # # for single view
        # else:

        #     ####### Uncomment to calculate subgraph average features #######
        #     # # when there is a single view in the graph
        #     # avg = self.average_subgraph(x, batch=batch)
        #     ################################################################

        #     x = graph_to_nn_batch(x, batch, max_seq_len=self.max_seq_len) # output: first dimension is batch size [batch_size, feature_dim, max_seq_len)]
        #     x = x.reshape(x.shape[0], x.shape[2], x.shape[1])   # rearrange the dimensions to [batch_size, max_seq_len, feature_dim]
        #     x = self.subgraph_agg(x)
        #     edge_index, edge_attr = self.rebuild_edge_matrix(x, view_idx)  # reconstruct edge matrix based on the new segmentwise features

        ####### Uncomment to calculate subgraph average features #######
        # # append the average subgraph to the feature vector
        # # print(f'x: {x.shape} | avg: {avg.shape}')
        # x = torch.cat((x, avg), dim=1)
        ################################################################
 

        feature_dim = x.shape[1]

        if self.use_spf:
            x_visual = self.layer011(torch.cat((x[:, :feature_dim//self.num_modality], self.layer_spf(c)), dim=1))
        else:
            x_visual = self.layer011(x[:, :feature_dim//self.num_modality])

        if self.num_modality == 1:
            x = x_visual
        elif self.num_modality == 2:
            x_audio = self.layer012(x[:, feature_dim//self.num_modality:])
            x = x_visual + x_audio

        x = self.batch01(x)
        x = self.relu(x)

        edge_index_f = edge_index[:, edge_attr<=0]
        edge_index_b = edge_index[:, edge_attr>=0]

  

        ######## Forward-graph stream
        x1 = self.layer11(x, edge_index_f)
        x1 = self.batch11(x1)
        x1 = self.relu(x1)
        x1 = self.dropout(x1)
        x1 = self.layer21(x1, edge_index_f)
        x1 = self.batch21(x1)
        x1 = self.relu(x1)
        x1 = self.dropout(x1)

        ######## Backward-graph stream
        x2 = self.layer12(x, edge_index_b)
        x2 = self.batch12(x2)
        x2 = self.relu(x2)
        x2 = self.dropout(x2)
        x2 = self.layer21(x2, edge_index_b)
        x2 = self.batch21(x2)
        x2 = self.relu(x2)
        x2 = self.dropout(x2)

        ######## Undirected-graph stream
        x3 = self.layer13(x, edge_index)
        x3 = self.batch13(x3)
        x3 = self.relu(x3)
        x3 = self.dropout(x3)
        x3 = self.layer21(x3, edge_index)
        x3 = self.batch21(x3)
        x3 = self.relu(x3)
        x3 = self.dropout(x3)

        x1 = self.layer31(x1, edge_index_f)
        x2 = self.layer32(x2, edge_index_b)
        x3 = self.layer33(x3, edge_index)

        out = x1+x2+x3
            
        
        if self.use_ref:
            xr0 = torch.permute(out, (1, 0)).unsqueeze(0)
            xr1 = self.layer_ref1(torch.softmax(xr0, dim=1))
            xr2 = self.layer_ref2(torch.softmax(xr1, dim=1))
            xr3 = self.layer_ref3(torch.softmax(xr2, dim=1))
            out = torch.stack((xr0, xr1, xr2, xr3), dim=0).squeeze(1).transpose(2, 1).contiguous()

        return out

    # def rebuild_edge_matrix(self, x, view_idx=None):
    #     if view_idx is not None and torch.unique(view_idx).shape[0] > 1:
    #         return self.rebuild_edge_matrix_multiview(x, view_idx)
    #     node_source = []
    #     node_target = []
    #     edge_attr = []
    #     # print(f'x: {x.shape}')
    #     # print(f'view_idx: {view_idx.shape}')
    #     for i in range(x.shape[0]):
    #         for j in range(x.shape[0]):
    #             # Frame difference between the i-th and j-th nodes
    #             frame_diff = i - j

    #             # The edge ij connects the i-th node and j-th node
    #             # Positive edge_attr indicates that the edge ij is backward (negative: forward)
    #             if abs(frame_diff) <= 1: # tauf
    #                 node_source.append(i)
    #                 node_target.append(j)
    #                 edge_attr.append(np.sign(frame_diff))

    #     edge_index = torch.tensor(np.array([node_source, node_target], dtype=np.int64), dtype=torch.long).to(x.device)
    #     edge_attr = torch.tensor(edge_attr, dtype=torch.float32).to(x.device)

    #     return edge_index, edge_attr
    
    # def rebuild_edge_matrix_multiview(self, x, view_idx):              
    #     num_view = torch.unique(view_idx).shape[0]
    #     num_frame = x.shape[0] // num_view  # num frames is the number of batches
    #     node_source = []
    #     node_target = []
    #     edge_attr = []

    #     for i in range(num_frame):
    #         for j in range(num_frame):
    #             # Frame difference between the i-th and j-th nodes
    #             frame_diff = i - j

    #             # The edge ij connects the i-th node and j-th node
    #             # Positive edge_attr indicates that the edge ij is backward (negative: forward)
    #             if abs(frame_diff) <= 1:
    #                 node_source.append(i)
    #                 node_target.append(j)
    #                 edge_attr.append(np.sign(frame_diff))

    #                 for k in range(1, num_view):
    #                     node_source.append(i+num_frame*k)
    #                     node_target.append(j+num_frame*k)
    #                     edge_attr.append(np.sign(frame_diff))

    #                 if frame_diff == 0:
    #                     for k in range(1, num_view):
    #                         node_source.append(i)
    #                         node_target.append(j+num_frame*k)
    #                         edge_attr.append(1)

    #     edge_index = torch.tensor(np.array([node_source, node_target], dtype=np.int64), dtype=torch.long).to(x.device)
    #     edge_attr = torch.tensor(edge_attr, dtype=torch.float32).to(x.device)

    #     return edge_index, edge_attr
        
