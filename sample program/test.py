from torch import nn, optim
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from functools import partial


ACC_LIMIT = 3  # the limit of acceleration, this can be calibrated based on the data
Ts = 0.1  # time interval for data sampling for HighD is 0.04 for other datasets are 0.1
de_model = 512
head = 8

# load data
car_following_test_data=np.load('Waymo_test_data.npy', allow_pickle=True)

max_len = 150 # for HighD dataset for others are 150

class ImitationCarFolData(torch.utils.data.Dataset):
    """
    Dataset class for imitation learning (state -> action fitting) based car-following models.
    """
    def __init__(self, split: str, max_len = max_len):
        if split == 'test':
            self.data = test_data
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
    
    

# Test
test_data = car_following_test_data
dataset = 'HighD'
model_type = 'Alstm'
batch_size = 32
total_epochs = 20
test_dataset = ImitationCarFolData(split = 'test')
test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

his_horizon = 10  # number of time steps as history data
lr = 1e-3  # learning rate
save = f'{model_type}_{dataset}.pt'

criterion = nn.MSELoss()

test_error_his = []

print("---")
# Testing, closed-loop prediction
# Load the best model saved
model = torch.load("AGRU_HighD_best_acclimt3.pt")
model.eval()

error_list = []
collsion_num = 0
acc_bianhualv = []
TTC_list = []
for i, item in enumerate(test_loader):

    x_data, y_data = item['inputs'], item['label']
    # Put T into the first dimension, B, T, d -> T, B, d
    x_data = torch.stack(x_data)
    # x_data, y_data = x_data.transpose(0, 1), y_data.transpose(0, 1)
    x_data, y_data = x_data.transpose(0, 2).float().to(device), y_data.transpose(0, 1).float().to(device)

    lv_spd = item['lv_spd'].float().to(device).transpose(0, 1)

    T, B, d = x_data.shape # (total steps, batch_size, d) as the shape of the data

    # save a copy of x_data
    x_data_orig = x_data.clone().detach()
    y_pre = torch.zeros(T - his_horizon, B)
    TTC_single = torch.zeros(T - his_horizon, B)

    for frame in range(his_horizon, T):
        x = x_data[frame-his_horizon:frame] # (his_horizon, B, d)

        acc_pre = model(x).squeeze()
        y_pre[frame - his_horizon] = acc_pre


        # update next data
        if frame < T-1:
            sv_spd_ = x_data[frame, :, 1] + acc_pre*Ts   #当frame=10，算的是第11时刻的跟随车速度
            MyDevice = torch.device('cuda:0')
            sv_spd_ = torch.tensor(np.maximum(np.array(sv_spd_.detach().cpu()), [0.001]), device=MyDevice)
            delta_v_ = lv_spd[frame + 1] - sv_spd_  #第11时刻的速度差
            delta_v = x_data[frame, :, -1] #第10时刻的速度差
            spacing_ = x_data[frame, :, 0] + Ts*(delta_v + delta_v_)/2 #计算第11时刻的间距，要用第10时刻的间距去计算，在第10时刻的间距基础上变化

            # update
            next_frame_data = torch.stack((spacing_, sv_spd_, delta_v_)).transpose(0, 1) # B, 3 第11时刻间距 FV速度 速度差
            x_data[frame + 1] = next_frame_data #更新数据，即相当于拿预测出的第11时刻数据导入，拿预测的数据继续预测下一时刻

        TTC_single[frame - his_horizon] = -spacing_ / delta_v_
    # 计算TTC
    TTC_single = TTC_single.transpose(0, 1)
    ttc_list = []
    for m in range(32):
        ttc_single = 0
        for n in range(139):
            if TTC_single[m][n].item() > 0:
                ttc_single = TTC_single[m][n].item()
                break
        for n in range(139):
            if ttc_single == 0:
                break
            if TTC_single[m][n].item() > 0:
                if TTC_single[m][n].item() < ttc_single:
                    ttc_single = TTC_single[m][n].item()
        if ttc_single > 0:
            ttc_list.append(ttc_single)

    if len(ttc_list) > 0:
        TTC_list.append(sum(ttc_list) / len(ttc_list))
    



    # 计算加速度变化率jerk
    y_pre = y_pre.transpose(0, 1)
    jerk = np.diff(y_pre.detach().numpy()) / Ts
    jerk = np.mean(np.abs(jerk))
    acc_bianhualv.append(jerk)





    # Calculating spacing error for the closed-loop simulation
    spacing_pre = x_data[..., 0]
    spacing_obs = x_data_orig[..., 0]
    for m in range(32):
        for n in range(149):
            if spacing_pre[n][m].item() < 0:
                collsion_num += 1
                # print(collsion_num)
                break

    error = criterion(spacing_pre, spacing_obs).item()
    error_list.append(error)
model.train()
mean_test_error = np.mean(error_list)
mean_acc_bianhualv = np.mean(acc_bianhualv)

print('mean_test_error', mean_test_error)
print("collsion_num", collsion_num)
print("acc_bianhualv", mean_acc_bianhualv)
print("miniumu_ttc", np.mean(TTC_list))
