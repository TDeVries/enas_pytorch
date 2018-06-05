import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions.categorical import Categorical


class Controller(nn.Module):
    '''
    https://github.com/melodyguan/enas/blob/master/src/cifar10/general_controller.py
    '''
    def __init__(self,
                 search_for="macro",
                 search_whole_channels=True,
                 num_layers=12,
                 num_branches=6,
                 out_filters=36,
                 lstm_size=32,
                 lstm_num_layers=2,
                 tanh_constant=1.5,
                 temperature=None,
                 skip_target=0.4,
                 skip_weight=0.8):
        super(Controller, self).__init__()

        self.search_for = search_for
        self.search_whole_channels = search_whole_channels
        self.num_layers = num_layers
        self.num_branches = num_branches
        self.out_filters = out_filters

        self.lstm_size = lstm_size
        self.lstm_num_layers = lstm_num_layers
        self.tanh_constant = tanh_constant
        self.temperature = temperature

        self.skip_target = skip_target
        self.skip_weight = skip_weight

        self._create_params()

    def _create_params(self):
        '''
        https://github.com/melodyguan/enas/blob/master/src/cifar10/general_controller.py#L83
        '''
        self.w_lstm = nn.LSTM(input_size=self.lstm_size,
                              hidden_size=self.lstm_size,
                              num_layers=self.lstm_num_layers)

        self.g_emb = nn.Embedding(1, self.lstm_size)  # Learn the starting input

        if self.search_whole_channels:
            self.w_emb = nn.Embedding(self.num_branches, self.lstm_size)
            self.w_soft = nn.Linear(self.lstm_size, self.num_branches, bias=False)
        else:
            assert False, "Not implemented error: search_whole_channels = False"

        self.w_attn_1 = nn.Linear(self.lstm_size, self.lstm_size, bias=False)
        self.w_attn_2 = nn.Linear(self.lstm_size, self.lstm_size, bias=False)
        self.v_attn = nn.Linear(self.lstm_size, 1, bias=False)

        self._reset_params()

    def _reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight, -0.1, 0.1)

        nn.init.uniform_(self.w_lstm.weight_hh_l0, -0.1, 0.1)
        nn.init.uniform_(self.w_lstm.weight_ih_l0, -0.1, 0.1)

    def forward(self):
        '''
        https://github.com/melodyguan/enas/blob/master/src/cifar10/general_controller.py#L126
        '''
        h0 = None  # setting h0 to None will initialize LSTM state with 0s

        anchors = []
        anchors_w_1 = []

        arc_seq = {}
        entropys = []
        log_probs = []
        skip_count = []
        skip_penaltys = []

        inputs = self.g_emb.weight
        skip_targets = torch.tensor([1.0 - self.skip_target, self.skip_target]).cuda()

        for layer_id in range(self.num_layers):
            if self.search_whole_channels:
                inputs = inputs.unsqueeze(0)
                output, hn = self.w_lstm(inputs, h0)
                output = output.squeeze(0)
                h0 = hn

                logit = self.w_soft(output)
                if self.temperature is not None:
                    logit /= self.temperature
                if self.tanh_constant is not None:
                    logit = self.tanh_constant * torch.tanh(logit)

                branch_id_dist = Categorical(logits=logit)
                branch_id = branch_id_dist.sample()

                arc_seq[str(layer_id)] = [branch_id]

                log_prob = branch_id_dist.log_prob(branch_id)
                log_probs.append(log_prob.view(-1))
                entropy = branch_id_dist.entropy()
                entropys.append(entropy.view(-1))

                inputs = self.w_emb(branch_id)
                inputs = inputs.unsqueeze(0)
            else:
                # https://github.com/melodyguan/enas/blob/master/src/cifar10/general_controller.py#L171
                assert False, "Not implemented error: search_whole_channels = False"

            output, hn = self.w_lstm(inputs, h0)
            output = output.squeeze(0)

            if layer_id > 0:
                query = torch.cat(anchors_w_1, dim=0)
                query = torch.tanh(query + self.w_attn_2(output))
                query = self.v_attn(query)
                logit = torch.cat([-query, query], dim=1)
                if self.temperature is not None:
                    logit /= self.temperature
                if self.tanh_constant is not None:
                    logit = self.tanh_constant * torch.tanh(logit)

                skip_dist = Categorical(logits=logit)
                skip = skip_dist.sample()
                skip = skip.view(layer_id)

                arc_seq[str(layer_id)].append(skip)

                skip_prob = torch.sigmoid(logit)
                kl = skip_prob * torch.log(skip_prob / skip_targets)
                kl = torch.sum(kl)
                skip_penaltys.append(kl)

                log_prob = skip_dist.log_prob(skip)
                log_prob = torch.sum(log_prob)
                log_probs.append(log_prob.view(-1))

                entropy = skip_dist.entropy()
                entropy = torch.sum(entropy)
                entropys.append(entropy.view(-1))

                # Calculate average hidden state of all nodes that got skips
                # and use it as input for next step
                skip = skip.type(torch.float)
                skip = skip.view(1, layer_id)
                skip_count.append(torch.sum(skip))
                inputs = torch.matmul(skip, torch.cat(anchors, dim=0))
                inputs /= (1.0 + torch.sum(skip))

            else:
                inputs = self.g_emb.weight

            anchors.append(output)
            anchors_w_1.append(self.w_attn_1(output))

        self.sample_arc = arc_seq

        entropys = torch.cat(entropys)
        self.sample_entropy = torch.sum(entropys)

        log_probs = torch.cat(log_probs)
        self.sample_log_prob = torch.sum(log_probs)

        skip_count = torch.stack(skip_count)
        self.skip_count = torch.sum(skip_count)

        skip_penaltys = torch.stack(skip_penaltys)
        self.skip_penaltys = torch.mean(skip_penaltys)
