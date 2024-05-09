from torch.nn import Module, Linear, ModuleList, Conv1d, Sequential, ReLU, Dropout, functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer, Linear, Transformer,  AdaptiveAvgPool2d
from transformers import MambaConfig, MambaModel


class TimeSeriesTransformer(Module):
    def __init__(self, input_size, num_layers, num_heads, hidden_size, output_size):
        super(TimeSeriesTransformer, self).__init__()
        self.encoder_layer = TransformerEncoderLayer(
            d_model=input_size,
            nhead=num_heads,
            dim_feedforward=hidden_size
        )
        self.transformer_encoder = TransformerEncoder(
            self.encoder_layer,
            num_layers=num_layers
        )
        self.linear = Linear(input_size, output_size)

    def forward(self, x):
        output = self.transformer_encoder(x)
        output = self.linear(output[:, -1, :])  # Use only the last timestep for prediction
        return output



class MLP(Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = Linear(input_size, hidden_size)
        self.fc2 = Linear(hidden_size, hidden_size)
        self.fc3 = Linear(hidden_size, output_size)
        self.relu = ReLU()
        self.dropout = Dropout(0.2)
        self.pool = AdaptiveAvgPool2d((1, None))

    def forward(self, x):
        # print(f'x: {x.shape}')
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.pool(x)
        # now remove the second dimension which is 1
        x = x.squeeze(1)
        return x
        


class SimpleMLP(Module):
    def __init__(self, cfg):
        input_dim = cfg['input_dim']
        final_dim = cfg['final_dim']
        hidden_size = 1056
        super(SimpleMLP, self).__init__()
        self.fc1 = Linear(input_dim, hidden_size) 
        self.relu = ReLU()                          
        self.fc2 = Linear(hidden_size, final_dim)

    def forward(self, x):
        # average  across the second dimension
        x = x.mean(dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
        
    

# For 2 stage training with Mamba -> GraVi-T
class MambaEndtoEnd(Module):
    def __init__(self, mamba_model, mamba_output_dim, embedding_output_dim):
        super().__init__()
        self.mamba = mamba_model
        self.cls_head = Linear(mamba_output_dim, embedding_output_dim)

    def forward(self, x):
        x = self.mamba(inputs_embeds=x)
        # print(x)
        x = x.last_hidden_state
        x = x[:, -1, :]

        # print(x.shape)
        # x = x.mean(dim=1)
        # print(x.shape)
        return self.cls_head(x)

    

# For 1 stage joint training of Mamba+GraVi-T
class MambaSeqEmbedding(Module):
    def __init__(self, mamba_model):
        super().__init__()
        self.mamba = mamba_model

    def forward(self, x):
        x = self.mamba(inputs_embeds=x)
        x = x.last_hidden_state # last hidden state has length of sequence_length 
        x = x[:, -1, :] # get the last element of the last hidden state corresponding to last sequence element

        # print(x.shape)
        # x = x.mean(dim=1)
        # print(x.shape)
        return x

