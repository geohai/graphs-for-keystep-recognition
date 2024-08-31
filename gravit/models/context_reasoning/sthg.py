import torch
from torch.nn import Module, ModuleList, Conv1d, Sequential, ReLU, Dropout, functional as F
from torch_geometric.nn import Linear, HeteroConv, EdgeConv, GATv2Conv, SAGEConv, BatchNorm
import torch_geometric
import numpy as np
# from gravit.utils.batch_process import graph_to_nn_batch, nn_batch_to_graph, multiview_graph_to_nn_batch

from transformers import MambaConfig, MambaModel
from gravit.mamba.models import MambaSeqEmbedding


def graph_to_nn_batch(x, batch, max_seq_len=25):
    batch_numbers = torch.unique(batch).sort()[0]

    batch_tensors = []
    for batch_num in batch_numbers:
        indices = torch.where(batch == batch_num)[0]
        batch_data = torch.index_select(x, 0, indices)
        padded_data = F.pad(batch_data, (0, 0, 0, max_seq_len - batch_data.size(0)))
        padded_data = padded_data[:max_seq_len, :]
        padded_data = padded_data.unsqueeze(0).transpose(1, 2)
        batch_tensors.append(padded_data)

    return torch.cat(batch_tensors, dim=0)


def multiview_graph_to_nn_batch(x, batch_idxs, view_idxs, max_seq_len=25): 
    batch_numbers = torch.unique(batch_idxs).sort()[0]
    view_numbers = torch.unique(view_idxs).sort()[0]

    batch_tensors = []
    for batch_num in batch_numbers:
        for view_num in view_numbers:
            indices = torch.where((batch_idxs == batch_num) & (view_idxs == view_num))[0]
            batch_data = torch.index_select(x, 0, indices)
            padded_data = F.pad(batch_data, (0, 0, 0, max_seq_len - batch_data.size(0)))
            padded_data = padded_data[:max_seq_len, :]
            padded_data = padded_data.unsqueeze(0).transpose(1, 2)
            batch_tensors.append(padded_data)

    return torch.cat(batch_tensors, dim=0)


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


class STHG(Module):
    def __init__(self, cfg, save_feats=False):
        super(STHG, self).__init__()
        self.use_spf = cfg['use_spf'] # whether to use the spatial features
        self.use_ref = cfg['use_ref']
        self.num_modality = cfg['num_modality']
        self.save_feats = save_feats

        channels = [cfg['channel1'], cfg['channel2']]
        final_dim = cfg['final_dim']
        input_dim = cfg['input_dim'] 
        
        num_att_heads = cfg['num_att_heads']
        dropout = cfg['dropout']

        ######## SUBGRAPH AGGREGATION ########
        self.max_seq_len = 25
        mamba_config= MambaConfig(vocab_size=497, hidden_size=1536, num_hidden_layers=4) # ~30M params with this config
        model = MambaModel(mamba_config)
        model = MambaSeqEmbedding(model) # wrapper to access the mamba features

        # self.subgraph_agg = model
        self.subgraph_agg = torch.nn.DataParallel(model) # Distribute Mamba across available GPUs
        # self.average_subgraph = torch_geometric.nn.pool.global_mean_pool # for calculating segment average features
        ######################################

        if self.use_spf:
            self.layer_spf = Linear(-1, cfg['proj_dim']) # projection layer for spatial features

        print(f'Input dim: {input_dim}  Final dim: {final_dim}')
        self.layer011 = Linear(input_dim, channels[0])
        text_dim = 1024
        self.layer01t = Linear(text_dim, text_dim)
        if self.num_modality == 2:
            self.layer012 = Linear(-1, channels[0])

        self.batch01 = BatchNorm(channels[0])
        self.batch01t = BatchNorm(text_dim)
        self.relu = ReLU()
        self.dropout = Dropout(dropout)

        self.layer11 = HeteroConv({
            ('o', 'oo', 'o'): EdgeConv(Sequential(Linear(2*channels[0], channels[0]), ReLU(), Linear(channels[0], channels[0]))),
            ('t', 'to', 'o'): SAGEConv((text_dim, channels[0]), channels[0]),
            ('t', 'tt', 't'): EdgeConv(Sequential(Linear(2*text_dim, text_dim), ReLU(), Linear(text_dim, text_dim))),
            })
        self.batch11 = BatchNorm(channels[0])
        self.batch11t = BatchNorm(text_dim)
        self.layer12 = HeteroConv({
            ('o', 'oo', 'o'): EdgeConv(Sequential(Linear(2*channels[0], channels[0]), ReLU(), Linear(channels[0], channels[0]))),
            ('t', 'to', 'o'): SAGEConv((text_dim, channels[0]), channels[0]),
            ('t', 'tt', 't'): EdgeConv(Sequential(Linear(2*text_dim, text_dim), ReLU(), Linear(text_dim, text_dim))),
            })
        self.batch12 = BatchNorm(channels[0])
        self.batch12t = BatchNorm(text_dim)
        self.layer13 = HeteroConv({
            ('o', 'oo', 'o'): EdgeConv(Sequential(Linear(2*channels[0], channels[0]), ReLU(), Linear(channels[0], channels[0]))),
            ('t', 'to', 'o'): SAGEConv((text_dim, channels[0]), channels[0]),
            ('t', 'tt', 't'): EdgeConv(Sequential(Linear(2*text_dim, text_dim), ReLU(), Linear(text_dim, text_dim))),
            })
        self.batch13 = BatchNorm(channels[0])
        self.batch13t = BatchNorm(text_dim)

        if num_att_heads > 0:
            self.layer21 = HeteroConv({
                ('o', 'oo', 'o'): GATv2Conv(channels[0], channels[1], heads=num_att_heads),
                ('t', 'to', 'o'): GATv2Conv((text_dim, channels[0]), channels[1], heads=num_att_heads),
                ('t', 'tt', 't'): GATv2Conv(text_dim, channels[1], heads=num_att_heads),
                })
        else:
            self.layer21 = HeteroConv({
                ('o', 'oo', 'o'): SAGEConv(channels[0], channels[1]),
                ('t', 'to', 'o'): SAGEConv((text_dim, channels[0]), channels[1]),
                ('t', 'tt', 't'): SAGEConv(text_dim, channels[1]),
                })
            num_att_heads = 1
        self.batch21 = BatchNorm(channels[1]*num_att_heads)
        self.batch21t = BatchNorm(channels[1]*num_att_heads)

        self.layer31 = HeteroConv({
            ('o', 'oo', 'o'): SAGEConv(channels[1]*num_att_heads, final_dim),
            ('t', 'to', 'o'): SAGEConv((text_dim, channels[1]), final_dim),
            ('t', 'tt', 't'): SAGEConv(text_dim*num_att_heads, final_dim),
            })
        self.layer32 = HeteroConv({
            ('o', 'oo', 'o'): SAGEConv(channels[1]*num_att_heads, final_dim),
            ('t', 'to', 'o'): SAGEConv((text_dim, channels[1]), final_dim),
            ('t', 'tt', 't'): SAGEConv(text_dim*num_att_heads, final_dim),
            })
        self.layer33 = HeteroConv({
            ('o', 'oo', 'o'): SAGEConv(channels[1]*num_att_heads, final_dim),
            ('t', 'to', 'o'): SAGEConv((text_dim, channels[1]), final_dim),
            ('t', 'tt', 't'): SAGEConv(text_dim*num_att_heads, final_dim),
            })

        if self.use_ref:
            self.layer_ref1 = Refinement(final_dim)
            self.layer_ref2 = Refinement(final_dim)
            self.layer_ref3 = Refinement(final_dim)


    def forward(self, x_dict, edge_index_dict, edge_attr_dict, c=None, batch=None, view_idx=None):

        """
        # ################################################################################
        ## Mamba

        # when there are multiple views in the graph
        if view_idx is not None and torch.unique(view_idx).shape[0] > 1:
            # print(f'x: {x.shape} | batch: {torch.unique(batch).shape}, {batch.shape} | view_idx: {torch.unique(view_idx).shape}')
            

            ####### Uncomment to calculate subgraph average features #######
            # # create new batch index for multiview using batch and view_idx
            # batch_view = batch * torch.unique(view_idx).shape[0] + view_idx
            # avg = self.average_subgraph(x, batch=batch_view.to(x.device))
            # # sort avg by the batch_view indices
            # avg = avg[torch.unique(batch_view).argsort()]
            # # avg is in order of b0v0, b0v1, b0v2,
            ################################################################


            # print(f'x: {x.shape} | batch: {torch.unique(batch).shape} | view_idx: {torch.unique(view_idx).shape}')
            x = multiview_graph_to_nn_batch(x, batch, view_idx, max_seq_len=self.max_seq_len) # parse graph to tensor to input to mamba [num_views*batch_size, feature_dim, max_seq_len]
            x = x.reshape(x.shape[0], x.shape[2], x.shape[1]) # rearrange to [num_views*batch_size, max_seq_len, feature_dim]
            x = self.subgraph_agg(x) # run through mamba
            edge_index, edge_attr = self.rebuild_edge_matrix(x, view_idx)  # reconstruct edge matrix based on the new segmentwise features, view_idx only needed to get num_views
        
        # for single view
        else:

            ####### Uncomment to calculate subgraph average features #######
            # # when there is a single view in the graph
            # avg = self.average_subgraph(x, batch=batch)
            ################################################################

            x = graph_to_nn_batch(x, batch, max_seq_len=self.max_seq_len) # output: first dimension is batch size [batch_size, feature_dim, max_seq_len)]
            x = x.reshape(x.shape[0], x.shape[2], x.shape[1])   # rearrange the dimensions to [batch_size, max_seq_len, feature_dim]
            x = self.subgraph_agg(x)
            edge_index, edge_attr = self.rebuild_edge_matrix(x, view_idx)  # reconstruct edge matrix based on the new segmentwise features

        ####### Uncomment to calculate subgraph average features #######
        # # append the average subgraph to the feature vector
        # # print(f'x: {x.shape} | avg: {avg.shape}')
        # x = torch.cat((x, avg), dim=1)
        ################################################################
        """

        x = self.layer011(x_dict['o'])
        x = self.batch01(x)
        x = self.relu(x)

        xt = self.layer01t(x_dict['t'])
        xt = self.batch01t(xt)
        xt = self.relu(xt)

        x_dict = {'o': x, 't': xt}

        edge_index_dict_f = {
                ('o', 'oo', 'o'): edge_index_dict[('o', 'oo', 'o')][:, edge_attr_dict[('o', 'oo', 'o')]<=0],
                ('t', 'to', 'o'): edge_index_dict[('t', 'to', 'o')],
                ('t', 'tt', 't'): edge_index_dict[('t', 'tt', 't')][:, edge_attr_dict[('t', 'tt', 't')]<=0],
                }
        edge_index_dict_b = {
                ('o', 'oo', 'o'): edge_index_dict[('o', 'oo', 'o')][:, edge_attr_dict[('o', 'oo', 'o')]>=0],
                ('t', 'to', 'o'): edge_index_dict[('t', 'to', 'o')],
                ('t', 'tt', 't'): edge_index_dict[('t', 'tt', 't')][:, edge_attr_dict[('t', 'tt', 't')]>=0],
                }
  

        ######## Forward-graph stream
        x1 = self.layer11(x_dict, edge_index_dict_f)
        x1['o'] = self.batch11(x1['o'])
        x1['o'] = self.relu(x1['o'])
        x1['o'] = self.dropout(x1['o'])
        x1['t'] = self.batch11t(x1['t'])
        x1['t'] = self.relu(x1['t'])
        x1['t'] = self.dropout(x1['t'])
        x1 = self.layer21(x1, edge_index_dict_f)
        x1['o'] = self.batch21(x1['o'])
        x1['o'] = self.relu(x1['o'])
        x1['o'] = self.dropout(x1['o'])
        x1['t'] = self.batch21t(x1['t'])
        x1['t'] = self.relu(x1['t'])
        x1['t'] = self.dropout(x1['t'])

        ######## Backward-graph stream
        x2 = self.layer12(x_dict, edge_index_dict_b)
        x2['o'] = self.batch12(x2['o'])
        x2['o'] = self.relu(x2['o'])
        x2['o'] = self.dropout(x2['o'])
        x2['t'] = self.batch12t(x2['t'])
        x2['t'] = self.relu(x2['t'])
        x2['t'] = self.dropout(x2['t'])
        x2 = self.layer21(x2, edge_index_dict_b)
        x2['o'] = self.batch21(x2['o'])
        x2['o'] = self.relu(x2['o'])
        x2['o'] = self.dropout(x2['o'])
        x2['t'] = self.batch21t(x2['t'])
        x2['t'] = self.relu(x2['t'])
        x2['t'] = self.dropout(x2['t'])

        ######## Undirected-graph stream
        x3 = self.layer13(x_dict, edge_index_dict)
        x3['o'] = self.batch13(x3['o'])
        x3['o'] = self.relu(x3['o'])
        x3['o'] = self.dropout(x3['o'])
        x3['t'] = self.batch13t(x3['t'])
        x3['t'] = self.relu(x3['t'])
        x3['t'] = self.dropout(x3['t'])
        x3 = self.layer21(x3, edge_index_dict)
        x3['o'] = self.batch21(x3['o'])
        x3['o'] = self.relu(x3['o'])
        x3['o'] = self.dropout(x3['o'])
        x3['t'] = self.batch21t(x3['t'])
        x3['t'] = self.relu(x3['t'])
        x3['t'] = self.dropout(x3['t'])

        x1 = self.layer31(x1, edge_index_dict_f)
        x2 = self.layer32(x2, edge_index_dict_b)
        x3 = self.layer33(x3, edge_index_dict)

        out = x1['o']+x2['o']+x3['o']
        return out

    def rebuild_edge_matrix(self, x, view_idx=None):
        if view_idx is not None and torch.unique(view_idx).shape[0] > 1:
            return self.rebuild_edge_matrix_multiview(x, view_idx)
        node_source = []
        node_target = []
        edge_attr = []
        # print(f'x: {x.shape}')
        # print(f'view_idx: {view_idx.shape}')
        for i in range(x.shape[0]):
            for j in range(x.shape[0]):
                # Frame difference between the i-th and j-th nodes
                frame_diff = i - j

                # The edge ij connects the i-th node and j-th node
                # Positive edge_attr indicates that the edge ij is backward (negative: forward)
                if abs(frame_diff) <= 1: # tauf
                    node_source.append(i)
                    node_target.append(j)
                    edge_attr.append(np.sign(frame_diff))

        edge_index = torch.tensor(np.array([node_source, node_target], dtype=np.int64), dtype=torch.long).to(x.device)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32).to(x.device)

        return edge_index, edge_attr
    
    def rebuild_edge_matrix_multiview(self, x, view_idx):              
        num_view = torch.unique(view_idx).shape[0]
        # print(f'Num views: {num_view}')
        num_frame = x.shape[0] // num_view  # num frames is the number of batches
        # print(f'Num frames: {num_frame}')
        node_source = []
        node_target = []
        edge_attr = []

        for i in range(num_frame):
            for j in range(num_frame):
                # Frame difference between the i-th and j-th nodes
                frame_diff = i - j

                # The edge ij connects the i-th node and j-th node
                # Positive edge_attr indicates that the edge ij is backward (negative: forward)
                if abs(frame_diff) <= 1:
                    node_source.append(i)
                    node_target.append(j)
                    edge_attr.append(np.sign(frame_diff))

                    for k in range(1, num_view):
                        node_source.append(i+num_frame*k)
                        node_target.append(j+num_frame*k)
                        edge_attr.append(np.sign(frame_diff))

                    if frame_diff == 0:
                        for k in range(1, num_view):
                            node_source.append(i)
                            node_target.append(j+num_frame*k)
                            edge_attr.append(1)

        edge_index = torch.tensor(np.array([node_source, node_target], dtype=np.int64), dtype=torch.long).to(x.device)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32).to(x.device)

        return edge_index, edge_attr
        

    # def concatenate_frames_in_subgraph(self, x, batch):
    #     ##########
    #     # Concatenate within batch
    #     for count, b in enumerate(batch.unique(sorted=True)):
    #         this_batch_features = x[batch == b]
    #         # Reduce subgraph to concatenated feature vector
    #         for i, this_feature in enumerate(this_batch_features):
    #             this_feature = this_feature.unsqueeze(0)
    #             if i == 0:
    #                 concated = this_feature
    #             else:
    #                 concated = torch.cat((concated, this_feature), dim=1)

    #         # If subgraph num nodes is not equal to the max num nodes in the batch, pad with zeros
    #         if concated.shape[1] > 10*2048:
    #             concated = concated[:, :10*2048]
    #         elif concated.shape[1] < 10*2048:
    #             padding = torch.zeros(concated.shape[0], (10*2048-concated.shape[1])).to(concated.device)
    #             concated = torch.cat((concated, padding), dim=1)

    #         # Append each subgraph's feature vector to the new out tensor
    #         if count == 0:
    #             new_out = concated
    #         else:
    #             new_out = torch.cat((new_out, concated), dim=0)

    #         new_out = new_out.to(x.device)
    #     return new_out
    

    # def pool(self, pooling_layer, x, edge_index, edge_attr, batch):
    #     # print(f'Before pooling - x: {x.shape}')
    #     # print(f'Batch: {batch.shape}')
    #     out = pooling_layer.forward(x, edge_index, edge_attr, batch)
    #     x = out[0]
    #     batch = out[3]
    #     edge_index, edge_attr = self.rebuild_edge_matrix(x)
    #     edge_index_f = edge_index[:, edge_attr<=0]
    #     edge_index_b = edge_index[:, edge_attr>=0]
    #     return x, edge_index, edge_attr, batch, edge_index_f, edge_index_b

    # def pooling_layer(self, x, edge_index, edge_attr, batch):
            
    #         if self.pooling_method == 'topk' or self.pooling_method == 'adaptive':
    #             out = self.pooling1.forward(x, edge_index, edge_attr, batch)
    #             x = out[0]
    #             batch = out[3]

    #             x = self.pooling2(x, batch)
    #             edge_index, edge_attr = self.rebuild_edge_matrix(x)

    #         elif self.pooling_method == 'concatenate':
    #             print(f'Before pooling - x: {x.shape}')
    #             for count, b in enumerate(batch.unique(sorted=True)):
    #                 this_batch_features = x[batch == b]

    #                 for i, this_feature in enumerate(this_batch_features):
    #                     this_feature = this_feature.unsqueeze(0)
    #                     if i == 0:
    #                         concated = this_feature
    #                     else:
    #                         concated = torch.cat((concated, this_feature), dim=1)

    #                 x = torch_geometric.nn.pool.global_mean_pool(concated)

    #                 if count == 0:
    #                     new_x = x
    #                 else:
    #                     new_x = torch.cat((new_x, x), dim=1)

    #                 new_x = new_x.to(x.device)
    #                 new_batch = torch.arange(0, new_x.shape[0], 1).to(x.device)
                
  
    #             batch = new_batch

    #             edge_index, edge_attr = self.rebuild_edge_matrix(x)

    #         # print(f'Pooling operation Final: x: {x.shape} | edge_index: {edge_index.shape} | edge_attr: {edge_attr.shape} | batch: {batch.shape}') # | perm: {perm.shape} | score: {score}
    #         return x, edge_index, edge_attr, batch
    


