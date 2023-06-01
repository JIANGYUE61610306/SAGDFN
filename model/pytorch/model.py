from sqlalchemy import true
import torch
import torch.nn as nn
from torch.nn import functional as F
from model.pytorch.cell import DCGRUCell
import numpy as np
from entmax import sparsemax, entmax15, entmax_bisect
from lib import utils
import time
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def cosine_similarity_torch(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)


def softmax_adj_unity(x):
    x = F.softmax(x/0.5, dim=-1)
    shape = x.size()
    _, k = x.data.max(-1)
    y_hard = torch.zeros(*shape).to(device)
    y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
    y = torch.autograd.Variable(y_hard - x.data) + x
    # y = y_hard - x.data + x
    return y


def entmax_adj(x):
    alpha = torch.tensor(1.5, requires_grad=True).to(device)
    x = entmax_bisect(x, alpha)
    shape = x.size()
    _, k = x.data.max(-1)
    y_hard = torch.zeros(*shape).to(device)
    y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
    y = torch.autograd.Variable(y_hard - x.data) + x
    return y


class Seq2SeqAttrs:
    def __init__(self, **model_kwargs):
        #self.adj_mx = adj_mx
        self.max_diffusion_step = int(model_kwargs.get('max_diffusion_step', 2))
        self.cl_decay_steps = int(model_kwargs.get('cl_decay_steps', 1000))
        self.filter_type = model_kwargs.get('filter_type', 'laplacian')
        self.num_nodes = int(model_kwargs.get('num_nodes', 1))
        self.num_rnn_layers = int(model_kwargs.get('num_rnn_layers', 1))
        self.rnn_units = int(model_kwargs.get('rnn_units'))
        self.hidden_state_size = self.num_nodes * self.rnn_units
        self.threshold = model_kwargs.get('threshold', 0.5)


class EncoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, **model_kwargs):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, **model_kwargs)
        self.input_dim = int(model_kwargs.get('input_dim', 1))
        self.seq_len = int(model_kwargs.get('seq_len'))  # for the encoder
        self.dcgru_layers = nn.ModuleList(
            [DCGRUCell(self.rnn_units, self.max_diffusion_step, self.num_nodes,
                       filter_type=self.filter_type) for _ in range(self.num_rnn_layers)])

    def forward(self, inputs, adj,node_index, hidden_state=None):
        """
        Encoder forward pass.
        :param inputs: shape (batch_size, self.num_nodes * self.input_dim)
        :param hidden_state: (num_layers, batch_size, self.hidden_state_size)
               optional, zeros if not provided
        :return: output: # shape (batch_size, self.hidden_state_size)
                 hidden_state # shape (num_layers, batch_size, self.hidden_state_size)
                 (lower indices mean lower layers)
        """
        batch_size, _ = inputs.size()
        if hidden_state is None:
            hidden_state = torch.zeros((self.num_rnn_layers, batch_size, self.hidden_state_size),
                                       device=device)
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num], adj,node_index)
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        return output, torch.stack(hidden_states)  # runs in O(num_layers) so not too slow


class DecoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, **model_kwargs):
        # super().__init__(is_training, adj_mx, **model_kwargs)
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, **model_kwargs)
        self.output_dim = int(model_kwargs.get('output_dim', 1))
        self.horizon = int(model_kwargs.get('horizon', 1))  # for the decoder
        self.decoder_type = model_kwargs.get('decoder')
        if self.decoder_type == 'GRU':
            self.projection_layer = nn.Linear(self.rnn_units, self.output_dim)
        else:
            self.projection_layer = nn.Linear(self.rnn_units, 12) # 12 is the forecasting length here
        self.dcgru_layers = nn.ModuleList(
            [DCGRUCell(self.rnn_units, self.max_diffusion_step, self.num_nodes,
                       filter_type=self.filter_type) for _ in range(self.num_rnn_layers)])

    def forward(self, inputs, adj,node_index, hidden_state=None):
        """
        :param inputs: shape (batch_size, self.num_nodes * self.output_dim)
        :param hidden_state: (num_layers, batch_size, self.hidden_state_size)
               optional, zeros if not provided
        :return: output: # shape (batch_size, self.num_nodes * self.output_dim)
                 hidden_state # shape (num_layers, batch_size, self.hidden_state_size)
                 (lower indices mean lower layers)
        """
        batch_size, _ = inputs.size()
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num], adj, node_index)
            hidden_states.append(next_hidden_state)
            output = next_hidden_state
        # print('output shape from DecoderModel before projection', output.shape)
        projected = self.projection_layer(output.view(-1, self.rnn_units))
        if self.decoder_type == 'GRU':
            output = projected.view(-1, self.num_nodes * self.output_dim)
        else:
        # print('the output size is: ', output.shape)
            output = projected.view(batch_size, self.num_nodes * self.output_dim, 12) # 12 is the forecasting length here
            output = torch.transpose(output,0,2)
            output = torch.transpose(output,1,2)

        return output, torch.stack(hidden_states)

class ATTenModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, **model_kwargs):
        # super().__init__(is_training, adj_mx, **model_kwargs)
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, **model_kwargs)
        self.embedding_dim = 100
        self.fc_out = nn.Linear(self.embedding_dim * 2, self.embedding_dim)
        self.fc_cat = nn.Linear(self.embedding_dim, 2)
        self.fc2 = torch.nn.Linear(2, 1)
    def forward(self, x):
        x = torch.relu(self.fc_out(x))
        x = self.fc_cat(x)
        adj = entmax_adj(x)
        

        # np.savez('adj_aft', adj = adj.cpu().detach().numpy())
        # print('adj shape is: ', adj.shape)
        adj = self.fc2(adj)
        return adj

class SAGDFNModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, logger, **model_kwargs):
        super().__init__()
        Seq2SeqAttrs.__init__(self, **model_kwargs)
        self.encoder_model = EncoderModel(**model_kwargs)
        self.decoder_model = DecoderModel(**model_kwargs)
        self.decoder_type = model_kwargs.get('decoder')
        self.cl_decay_steps = int(model_kwargs.get('cl_decay_steps', 1000))
        self.use_curriculum_learning = bool(model_kwargs.get('use_curriculum_learning', False))
        self._logger = logger
        self.num_heads = 1
        self.att_block = [ATTenModel(**model_kwargs) for _ in range(self.num_heads)]
        self.att_fc = torch.nn.Linear(self.num_heads, 1)
        self.att_block = nn.ModuleList(self.att_block)
        self.embedding_dim = 100 # precise embedding dimension used in graph learning process
        self.emb_dim = model_kwargs.get('emb_dim') # randomly initial embedding dimension
        self.hidden_drop = torch.nn.Dropout(0.2)
        if self.emb_dim == 200:
            self.fc3 = torch.nn.Linear(self.emb_dim, self.embedding_dim)
            self.fc4 = torch.nn.Linear(2, 1)
        else:
            self.fc3 = torch.nn.Linear(self.emb_dim, 1000)
            self.fc4 = torch.nn.Linear(1000, 500)
            self.fc5 = torch.nn.Linear(500, self.embedding_dim)

        self.bn3 = torch.nn.BatchNorm1d(self.embedding_dim)
        self.fc_out = nn.Linear(self.embedding_dim * 2, self.embedding_dim)
        self.fc_cat = nn.Linear(self.embedding_dim, 2)
        self.sigmoid = nn.Sigmoid()

        ### For sparse implementation
        self.node_lib = torch.ones(self.num_nodes)
        self.neigb = 100
        self.sub = 20
        self.node_index = torch.multinomial(self.node_lib , self.num_nodes*self.neigb, replacement = True)
        self.node_index = torch.reshape(self.node_index, (self.num_nodes, self.neigb))
        


    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (
                self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def encoder(self, inputs, adj, node_index):
        """
        Encoder forward pass
        :param inputs: shape (seq_len, batch_size, num_sensor * input_dim)
        :return: encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
        """
        encoder_hidden_state = None
        for t in range(self.encoder_model.seq_len):
            _, encoder_hidden_state = self.encoder_model(inputs[t], adj, node_index, encoder_hidden_state) 
        # print('encoder model shape from cell, ', encoder_hidden_state.shape)
        return encoder_hidden_state

    def decoder(self, encoder_hidden_state, adj, node_index, labels=None, batches_seen=None):
        """
        Decoder forward pass
        :param encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
        :param labels: (self.horizon, batch_size, self.num_nodes * self.output_dim) [optional, not exist for inference]
        :param batches_seen: global step [optional, not exist for inference]
        :return: output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        """
        batch_size = encoder_hidden_state.size(1)
        go_symbol = torch.zeros((batch_size, self.num_nodes * self.decoder_model.output_dim),
                                device=device)
        decoder_hidden_state = encoder_hidden_state
        decoder_input = go_symbol

        outputs = []
        if self.decoder_type == 'GRU':
            for t in range(self.decoder_model.horizon):
                decoder_output, decoder_hidden_state = self.decoder_model(decoder_input, adj,node_index,
                                                                        decoder_hidden_state)
                decoder_input = decoder_output
                outputs.append(decoder_output)
                if self.training and self.use_curriculum_learning:
                    c = np.random.uniform(0, 1)
                    if c < self._compute_sampling_threshold(batches_seen):
                        decoder_input = labels[t]
            outputs = torch.stack(outputs)
            return outputs
        else:
            decoder_output, decoder_hidden_state = self.decoder_model(decoder_input, adj,node_index,
                                                                        decoder_hidden_state)
            return decoder_output


    def filter_neigb(self, x, node_index, sub):
        senders = []
        # start_time = time.time()
        for t in range(x.size(0)):
            senders.append(x[node_index[t,:],:])
        # end_time1 = time.time()
        # print('Time to build sender for comparison is: ', end_time1 - start_time)
        senders = torch.stack(senders) # senders shape:(N, k, d)
        x_copy = x.clone().detach()
        x_copy = torch.unsqueeze(x_copy, 1)
        difference = (senders - x_copy)*(senders-x_copy)
        difference = torch.sum(difference, 2)
        sorted_node, indices = torch.sort(difference)
        del senders, x_copy, sorted_node, difference
        new_node_index = []
        # start_time = time.time()
        for t in range(x.size(0)):
            new_node_index.append(node_index[t, indices[t,:-sub]])
        new_node_index = torch.stack(new_node_index)

        ### count the top K elements in the new node index
        unique, counts = torch.unique(new_node_index, return_counts = True)
        sorted, indices = torch.sort(counts, descending=True)
        new_node_index = torch.gather(unique, 0, indices)[:self.neigb-sub]
        new_node_index = torch.unsqueeze(new_node_index, 0)
        sub_index = torch.randint(self.num_nodes, (1,sub))
        new_node_index = torch.squeeze(torch.cat((new_node_index, sub_index), 1), 0) 

        return new_node_index

    def forward(self, label, inputs, node_feas, labels=None, batches_seen=None, batch_idx=None):
        """
        :param inputs: shape (seq_len, batch_size, num_sensor * input_dim)
        :param labels: shape (horizon, batch_size, num_sensor * output)
        :param batches_seen: batches seen till now
        :return: output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        """
        # print(node_feas.shape)
        x = node_feas.view(self.num_nodes, -1)
        if self.emb_dim == 200:
            x = self.fc3(x)
            x = F.relu(x)
            x = x.view(self.num_nodes, -1)
        else: # initial embedding dimension 2000, if change, need modify here
            x = self.fc3(x)
            x = F.relu(x)
            x = self.fc4(x)
            x = F.relu(x)
            x = self.fc5(x)
            x = F.relu(x)

        x = self.bn3(x)        ### node embedding: (N, d)      
        
        ### filtering function applied here
        if self.num_nodes < 1600: # no need filtering for graph less than 1600 nodes as GPU should be able to handle this, also can DIY upon your device
            receivers = torch.repeat_interleave(x, x.shape[0], axis=0)
            senders = torch.tile(x, (x.shape[0], 1))
        else:
            if batch_idx < 100 and batches_seen <50:
                # self._logger.info("Current num_batches:{}".format(batch_idx))
                self.node_index = self.filter_neigb(x, self.node_index, self.sub)
            else:
                self.node_index = self.node_index[0,:]
            senders = []
            # start_time = time.time()
            for t in range(x.size(0)):
                senders.append(x[self.node_index,:])
            senders = torch.stack(senders)
            # end_time1 = time.time()
            # print('Time to build sender is: ', end_time1 - start_time)
            senders = torch.reshape(senders, (self.num_nodes*self.neigb, -1))
            receivers = torch.repeat_interleave(x, self.neigb, axis=0)
        
        x = torch.cat([senders, receivers], dim=1)
        # print('X shape is: ', x.shape)
        adj_list = []
        if self.num_heads ==1:
            x = torch.relu(self.fc_out(x))
            x = self.fc_cat(x)
            adj = entmax_adj(x)
            adj = adj[:, 0].clone().reshape(self.num_nodes, -1)
            adj_save = adj.clone().detach()
        else:
            for i, attention in enumerate(self.att_block):
                adj= attention(x)
                adj_list.append(adj)
                # print('adj shape is: ', adj.shape)
            adj = torch.stack(adj_list, dim =1)
            adj = torch.squeeze(adj, dim =2)
            # print('adj shape after stack and squeeze is: ', adj.shape)
            adj = self.att_fc(adj)
            adj_save = adj.clone().detach()
            # adj = unity(adj)
            adj = adj.clone().reshape(self.num_nodes, -1)
        if self.num_nodes < 1600:
            mask = torch.eye(self.num_nodes, self.num_nodes).bool().to(device)
            adj.masked_fill_(mask, 0)
        # print(inputs.shape)
        # print('start encoding')
        encoder_hidden_state = self.encoder(inputs, adj, self.node_index)
        self._logger.debug("Encoder complete, starting decoder")
        # print('start decoding')
        outputs = self.decoder(encoder_hidden_state, adj,self.node_index, labels, batches_seen=batches_seen)
        self.node_index = torch.unsqueeze(self.node_index, 0)
        self.node_index = torch.repeat_interleave(self.node_index, self.num_nodes, 0)
        # self._logger.debug("Decoder complete")
        if batches_seen == 0:
            self._logger.info(
                "Total trainable parameters {}".format(count_parameters(self))
            )

        return outputs, x.softmax(-1)[:, 0].clone().reshape(self.num_nodes, -1), adj_save
