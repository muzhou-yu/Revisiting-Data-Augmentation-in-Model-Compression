import torch
import torch.nn as nn
import torch.nn.functional as F

class S2S_LSTM(nn.Module):
    def __init__(self, embed_size=512, feat_num=4, rnn_layers_num=2):
        super(S2S_LSTM, self).__init__()
        self.embed_size = embed_size
        self.feat_num = 4
        self.init_flag = False
        self.downsample = True # global pooling the features first
        self.teacher_embed = None
        self.student_embed = None
        self.encoder = None
        self.decoder = None
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def init_layers(self, stu_feat, tea_feat):
        self.teacher_embed = []
        self.student_embed = []

        for i in range(self.feat_num):
            self.teacher_embed.append(nn.Linear(tea_feat[i].size(-1), self.embed_size))
            self.student_embed.append(nn.Linear(stu_feat[i].size(-1), self.embed_size))

        self.teacher_embed = nn.ModuleList(self.teacher_embed).cuda()
        self.student_embed = nn.ModuleList(self.student_embed).cuda()

        self.encoder = nn.LSTM(
            input_size=self.embed_size,
            hidden_size=self.embed_size,
            num_layers=2,
            bias=True,
            batch_first=True,
            dropout=0.5,
            bidirectional=True
        ).cuda()
        self.decoder = nn.LSTM(
            input_size=self.embed_size,
            hidden_size=self.embed_size,
            num_layers=2,
            bias=True,
            batch_first=True,
            dropout=0.5,
            bidirectional=True
        ).cuda()

    def forward(self, stu_feat, tea_feat):
        batchsize = stu_feat[0].size(0)
        for i in range(len(stu_feat)):
            stu_feat[i] = self.global_pool(stu_feat[i]).view(batchsize, -1)
            tea_feat[i] = self.global_pool(tea_feat[i]).view(batchsize, -1)
        if self.init_flag is False:
            self.init_layers(stu_feat, tea_feat)
            self.init_flag = True
        for i in range(len(stu_feat)):
            stu_feat[i] = self.student_embed[i](stu_feat[i])
            tea_feat[i] = self.teacher_embed[i](tea_feat[i])
        stu_feat = torch.stack(stu_feat, dim=1)
        tea_feat = torch.stack(tea_feat, dim=1)
        encoding_feat, _ = self.encoder(stu_feat)
        decoding_feat, _ = self.decoder(tea_feat)
        loss = torch.dist(encoding_feat, decoding_feat) * 0.05

        return loss
