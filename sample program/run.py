from torch import nn, optim
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from functools import partial
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding

ACC_LIMIT = 3  # the limit of acceleration, this can be calibrated based on the data
Ts = 0.1  # time interval for data sampling for HighD is 0.04 for other datasets are 0.1
de_model = 512
head = 8


# load data
car_following_train_data=np.load('Waymo_train_data.npy', allow_pickle=True)
car_following_val_data=np.load('Waymo_val_data.npy', allow_pickle=True)


max_len = 150 # for HighD dataset for others are 150

class ImitationCarFolData(torch.utils.data.Dataset):
    """
    Dataset class for imitation learning (state -> action fitting) based car-following models.
    """
    def __init__(self, split: str, max_len = max_len):
        if split == 'train':
            self.data = train_data
        elif split == 'validation':
            self.data = val_data
        self.max_len = max_len # Max length of a car following event.

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):

        event = self.data[idx]
        sv = event[1][:self.max_len]
        lv = event[3][:self.max_len]
        spacing = event[0][:self.max_len]
        relSpeed = event[2][:self.max_len]
        inputs = [spacing[:-1], sv[:-1], relSpeed[:-1]]
        acc_label = np.diff(sv)/Ts
        lv_spd = lv
        return {'inputs': inputs, 'label': acc_label, 'lv_spd': lv_spd}



class AGRU_model(nn.Module):
    def __init__(self, input_size = 3, hidden_size = 128, gru_layers = 4, dropout = 0.1):
        super(AGRU_model, self).__init__()
        self.enc_embedding = DataEmbedding(3, de_model)
        self.attention = AttentionLayer(FullAttention(mask_flag=False), de_model, head)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(de_model)
        
        self.encoder = nn.GRU(de_model, hidden_size, gru_layers, batch_first=False, dropout=dropout)
        self.linear = nn.Linear(hidden_size, 1)

        nn.init.normal_(self.linear.weight, 0, .02)
        nn.init.constant_(self.linear.bias, 0.0)

    def forward(self, src):
        # src in the shape seq_len, B, d
        src = src.transpose(0, 1)
        src = self.enc_embedding(src)
        new_x, attns = self.attention(src,src,src,attn_mask=None,tau=None, delta=None) 
        src = src + self.dropout(new_x)  #残差

        src = self.norm(src)
        
        src = src.transpose(0,1)
        enc_x, h_n = self.encoder(src)

        if len(h_n.shape) == 3:
            h_n = h_n[-1] # taking the last layer hidden state

        out = self.linear(h_n)
        out = torch.tanh(out)*ACC_LIMIT
        return out
    
    

# Train
train_data = car_following_train_data
val_data = car_following_val_data

dataset = 'Waymo'
model_type = 'AGRU'
batch_size = 64
total_epochs = 50
train_dataset = ImitationCarFolData(split = 'train')
train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        drop_last=True)

validation_dataset = ImitationCarFolData(split = 'validation')
validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        drop_last=True)



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

his_horizon = 10  # number of time steps as history data
lr = 1e-3  # learning rate
save = f'{model_type}_{dataset}.pt'

model = AGRU_model(input_size = 3).to(device)  # single layer lstm

model_optim = optim.Adam(model.parameters(), lr=lr,betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
criterion = nn.MSELoss()

train_loss_his = []
validation_error_his = []
best_train_loss = None
best_validation_loss = None


print("---")
for epoch in tqdm(range(total_epochs)):
    train_losses = []
    validation_losses = []
    model.train()
    for i, item in enumerate(train_loader):
        # x_data, y_data = item['inputs'].float().to(device), item['label'].float().to(device)
        x_data, y_data = item['inputs'], item['label']
        # Put T into the first dimension, B, T, d -> T, B, d
        x_data = torch.stack(x_data)
        # x_data, y_data = x_data.transpose(0, 2), y_data.transpose(0, 1)
        x_data = x_data.transpose(0, 2).float().to(device)
        y_data = y_data.transpose(0, 1).float().to(device)

        T, B, d = x_data.shape  # (total steps, batch_size, d) as the shape of the data

        y_pre = torch.zeros(T - his_horizon, B).cuda()
        # y_pre = torch.zeros(T - his_horizon, B)
        y_label = y_data[his_horizon:]

        for frame in range(his_horizon, T):
            x = x_data[frame-his_horizon:frame]  # (his_horizon, B, d)
            acc_pre = model(x).squeeze()
            y_pre[frame - his_horizon] = acc_pre

        loss = criterion(y_pre, y_label)

        model_optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        model_optim.step()

        train_losses.append(loss.item())

    train_loss = np.mean(train_losses)

    train_loss_his.append(train_loss)
    print("Epoch: {0}| Train Loss: {1:.7f}".format(epoch + 1, train_loss))

    model.eval()
    error_list = []
    for i, item in enumerate(validation_loader):

        x_data, y_data = item['inputs'], item['label']
        # Put T into the first dimension, B, T, d -> T, B, d
        x_data = torch.stack(x_data)
        # x_data, y_data = x_data.transpose(0, 1), y_data.transpose(0, 1)
        x_data = x_data.transpose(0, 2).float().to(device)
        y_data = y_data.transpose(0, 1).float().to(device)
        lv_spd = item['lv_spd'].float().transpose(0, 1).to(device)
        T, B, d = x_data.shape  # (total steps, batch_size, d) as the shape of the data
        # save a copy of x_data
        x_data_orig = x_data.clone().detach()
        y_label = y_data[his_horizon:]
        for frame in range(his_horizon, T):
            x = x_data[frame-his_horizon:frame] # (his_horizon, B, d)
            acc_pre = model(x).squeeze()
            y_pre[frame - his_horizon] = acc_pre

        validation_loss = criterion(y_pre, y_label)
        model_optim.zero_grad()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        model_optim.step()
        validation_losses.append(validation_loss.item())
    mean_validation_error = np.mean(validation_losses)
    if best_validation_loss is None or best_validation_loss > mean_validation_error:
        best_validation_loss = mean_validation_error
        print("更新模型")
        # save the best model
        with open(save, 'wb') as f:
            torch.save(model, f)

    validation_error_his.append(mean_validation_error)
    print("Epoch: {0}| Validation error: {1:.7f}".format(epoch + 1, mean_validation_error))



