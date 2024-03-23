import os
import pandas as pd
import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
import datetime



#
if not os.name == 'posix' : os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import warnings
import argparse
from eval_pkg import *
import time
warnings.filterwarnings('ignore')


if os.name == 'posix':
#     assert(device == torch.device('cuda'))
    assert(torch.cuda.is_available())


    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--ts", type=int, nargs = '+', required=True)
    parser.add_argument("--u", type=int, nargs = '+', required=True)
    parser.add_argument("--u_plus", type=int, nargs = '+', required=True)
    parser.add_argument("--u_plus2", type=int, nargs = '+', required=True)
    parser.add_argument("--n", type=int, nargs = '+', required=True)
    parser.add_argument("--theta", type=float, nargs = '+', required=True)
    parser.add_argument("--nstack", type=float, nargs = '+', required=True)
    parser.add_argument("--hor", type=int, required=True)
    parser.add_argument("--noi_level", type=int, nargs = '+', required=True)
    parser.add_argument("--noi_std", type=float, nargs = '+', required=True)
    parser.add_argument("--cudanum", type=int)

    args, _ = parser.parse_known_args()
    dataset_name = args.dataset
    ts_list = args.ts
    u_list = args.u
    u_plus_list = args.u_plus
    u_plus2_list = args.u_plus2
    n_list = args.n
    theta_list = args.theta
    nstack_list = args.nstack
    hor_value = args.hor
    noi_level_list = args.noi_level
    noi_std_list = args.noi_std

    if args.cudanum is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        cudanum = args.cudanum
        device = torch.device(f'cuda:{cudanum}' if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_device(cudanum)

else:
    dataset_name = 'Wiki'
    ts_list = [14]
    u_list = [128]
    u_plus_list = [8]
    u_plus2_list = [256]
    n_list = [1]
    theta_list = [0]
    nstack_list = [3]
    hor_value = 1
    noi_level_list = [2]
    noi_std_list = [0.1]
    cudanum = 0
    device = torch.device(f'cuda:{cudanum}' if torch.cuda.is_available() else 'cpu')

    dataset_name = 'Traffic'
    ts_list = [14]
    u_list = [80]
    u_plus_list = [80]
    u_plus2_list = [80]
    n_list = [1]
    theta_list = [1]
    nstack_list = [3]    #3.0,1.0
    hor_value = 1
    noi_level_list = [2]
    noi_std_list = [0.1]
    cudanum = 0
    device = torch.device(f'cuda:{cudanum}' if torch.cuda.is_available() else 'cpu')

    # dataset_name = 'Wiki'
    # ts_list = [14]
    # u_list = [8]
    # u_plus_list = [10]
    # n_list = [2]
    # theta_list = [1]
    # nstack_list = [5]
    # hor_value = 1
    # cudanum = 0
    # device = torch.device(f'cuda:{cudanum}' if torch.cuda.is_available() else 'cpu')


print(device)

def build_evaluation_data(o_data, data, prediction_length):
    index = o_data.index[-prediction_length:]

    new_data = []
    for i in range(data.shape[1]):
        tem_data = pd.DataFrame(data[:, i], columns={'y'})
        tem_data['ds'] = index
        #         tem_data.rename(columns={0:'y'}, inplace=True)
        tem_data = tem_data[['ds', 'y']]
        new_data.append(tem_data)
    new_data2 = pd.concat(new_data)
    new_data2_index = o_data.columns.values.repeat(prediction_length)
    new_data2.index = new_data2_index
    return new_data2

def data_multiple_build(data, dp, DP_label_from_overall, lookback, horizon):
    X_new = []
    X_dp = []
    y_new = []
    z_new = []
    X_DP_label_new = []
    y_DP_label_new = []

    for indices in range(data.shape[0]):

        if indices + lookback + horizon > data.shape[0]:
            break

        X_tem = data[indices:indices + lookback, :]
        X_dp_tem = dp[indices:indices + lookback, :]
        y_tem = data[indices+lookback + horizon - 1 : indices + lookback + horizon, :]
        dp_tem = dp[indices + lookback + horizon - 1 : indices + lookback + horizon, :]
        X_DP_label_tem = DP_label_from_overall[indices:indices + lookback, :]
        y_DP_label_tem = DP_label_from_overall[indices + lookback + horizon - 1 : indices + lookback + horizon, :]


        X_new.append(X_tem)
        X_dp.append(X_dp_tem)
        y_new.append(y_tem)
        z_new.append(dp_tem)
        X_DP_label_new.append(X_DP_label_tem)
        y_DP_label_new.append(y_DP_label_tem)

    return X_new, X_dp, y_new, z_new, X_DP_label_new, y_DP_label_new


class tensor_multiple_data(Dataset):
    def __init__(self, x, x_dp, y, z, X_DP_label_new, y_DP_label_new):
        self.x = x
        self.x_dp = x_dp
        self.y = y
        self.z = z
        self.X_DP_label_new = X_DP_label_new
        self.y_DP_label_new = y_DP_label_new

    def __getitem__(self, item):
        x_i = torch.FloatTensor(self.x[item]).to(device)
        x_dp_i = torch.FloatTensor(self.x_dp[item]).to(device)
        y_i = torch.FloatTensor([self.y[item]]).to(device)  # item？
        z_i = torch.FloatTensor([self.z[item]]).to(device)
        X_DP_label_i = torch.FloatTensor(self.X_DP_label_new[item]).to(device)
        y_DP_label_i = torch.FloatTensor(self.y_DP_label_new[item]).to(device)

        return x_i, x_dp_i, y_i, z_i, X_DP_label_i, y_DP_label_i

    def __len__(self):
        return len(self.x)


def train(data, level_number_list, model, optimizer, loss_func,  epoch_txt, theta, MAE_tag):
    start_time = time.time()
    train_loss = 0
    train_loss_dpd = 0
    train_loss_me = 0
    train_pred = []
    train_true = []
    flag = 0
    data_loader = DataLoader(data, batch_size=1, shuffle=False)
    tqdm_loop = tqdm(data_loader)
    tqdm_loop.set_description(epoch_txt)
    for x_b, X_dp, y_b, z_b, DPfrom, DPfrom_true in tqdm_loop:
        optimizer.zero_grad()
        x_b = x_b.squeeze() # [TS, C]
        X_dp = X_dp.squeeze() # [TS, C]
        X_cat = torch.cat([x_b.T.unsqueeze(-1), X_dp.T.unsqueeze(-1)],dim=2).to(device) # [C, TS, 2]
        y_b = y_b.squeeze() # [C,]
        z_b = z_b.squeeze() # [C,]
        DPfrom = DPfrom.squeeze(0)  # [TS,C,levelcount]
        DPfrom_true = DPfrom_true.squeeze(0)  # [1,C,levelcount]

        y_pred,_,return_loss,x_DPD_overall = model(X_cat, DPfrom)
        loss, loss_dpd, loss_me = loss_func(y_pred, y_b,x_DPD_overall,DPfrom_true, level_number_list, theta, MAE_tag,return_loss)
        flag += 1

        train_loss += loss
        train_loss_dpd += loss_dpd
        train_loss_me += loss_me
        train_pred.append(y_pred.T.squeeze().cpu().detach().numpy().tolist())
        train_true.append(y_b.cpu().detach().numpy().tolist())
        loss.backward()
        optimizer.step()
        running_train_acc = 1 - wmape(y_pred, y_b)
        tqdm_loop.set_postfix(loss=loss.item(), acc=running_train_acc.item())

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(epoch_txt, 'loss = ', train_loss.item(), train_loss_dpd.item(), train_loss_me.item(),
          ',acc = ', running_train_acc.item(),
          ',elapsed_time = ',time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    print('new_acc=', 1 - wmape(torch.tensor(train_pred), torch.tensor(train_true)))

    return train_loss / flag, train_pred, train_loss_dpd / flag, train_loss_me / flag


def test(data, level_number_list, model, loss_func, theta, MAE_tag):
    test_loss = 0
    test_loss_dpd = 0
    test_loss_me = 0
    eval_v = 0
    test_pred = []
    test_true = []
    pred_dec = None
    flag = 0
    data_loader = DataLoader(data, batch_size=1, shuffle=False)
    for x_b, X_dp, y_b, z_b, DPfrom, DPfrom_true in data_loader:
        x_b = x_b.squeeze() # [TS, C]
        X_dp = X_dp.squeeze()  # [TS, C]
        X_cat = torch.cat([x_b.T.unsqueeze(-1), X_dp.T.unsqueeze(-1)], dim=2).to(device)  # [C, TS, 2]
        y_b = y_b.squeeze()
        z_b = z_b.squeeze()
        DPfrom = DPfrom.squeeze(0)  # [TS,C,levelcount]
        DPfrom_true = DPfrom_true.squeeze(0)  # [1,C,levelcount]
        test_true.append(y_b)

        with torch.no_grad():
            y_pred,y_pred_dec,return_loss,x_DPD_overall = model(X_cat, DPfrom)
            loss, loss_dpd, loss_me = loss_func(y_pred, y_b,x_DPD_overall,DPfrom_true, level_number_list, theta, MAE_tag,return_loss)
            #             tem_eval = eval_func(y_pred.cpu().numpy(), y_b.cpu().numpy())

            flag += 1

            test_loss += loss
            test_loss_dpd += loss_dpd
            test_loss_me += loss_me

            test_pred.append(y_pred.T.squeeze().cpu().numpy().tolist())
            if pred_dec is None:
                pred_dec = y_pred_dec.unsqueeze(-1)   # [C, levelcount, 1]
            else:
                #另起一个维度
                pred_dec = torch.cat([pred_dec, y_pred_dec.unsqueeze(-1)],dim=2)  # [C, levelcount, npred]
            # pred_dec[0].append(y_pred_dec[0].T.squeeze().cpu().numpy().tolist())
            # pred_dec[1].append(y_pred_dec[1].T.squeeze().cpu().numpy().tolist())
            # pred_dec[2].append(y_pred_dec[2].T.squeeze().cpu().numpy().tolist())

    return test_loss / flag, test_pred, test_true, test_loss_dpd / flag , test_loss_me / flag  , pred_dec


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, name, patience=7, verbose=False, delta=0, checkpoint='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.checkpoint = checkpoint
        self.name = name  # custom

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif (score <= self.best_score + self.delta) or (np.isnan(score)):
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.checkpoint)  #str(self.name) + 'model_normal.pt'
        self.val_loss_min = val_loss



class get_HTS_strc():
    def __init__(self, level_number_list):
        self.level_number_list = level_number_list

    def get_pred_loc(self):
        now_loc = 0
        pred_loc = []

        level_quan = 1
        pred_loc.append(level_quan)
        while True:
            next_loc = now_loc + level_quan
            if next_loc == len(self.level_number_list):
                level_quan = sum(self.level_number_list[now_loc:])
                pred_loc.append(level_quan)
                break
            else:
                # print(now_loc,now_loc+ level_quan)
                level_quan = sum(self.level_number_list[now_loc:next_loc])
                now_loc = next_loc
                pred_loc.append(level_quan)
        return pred_loc

class MultiHeadedAttention(nn.Module):
    def __init__(self, d_model, d_model_qk,d_model_v, h, hierchy_struc):
        """
        h: head的数量
        """
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        # 定义W^q, W^k, W^v和W^o矩阵。
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model_qk), nn.Linear(d_model, d_model_qk), nn.Linear(d_model, d_model_v), nn.Linear(d_model_v, d_model_v)])
        self.hierchy_struc = hierchy_struc

    def forward(self, x, mask_matrix=None):
        # x:  [C, levelcount, units0]
        # 获取Batch Size
        nbatches = x.size(0)
        levelcount = x.size(1)

        """
        1. 求出Q, K, V，这里是求MultiHead的Q,K,V，所以Shape为(batch, head数, 词数，d_model/head数)
            1.1 首先，通过定义的W^q,W^k,W^v求出SelfAttention的Q,K,V，此时Q,K,V的Shape为(batch, 词数, d_model)
                对应代码为 `linear(x)`
            1.2 分成多头，即将Shape由(batch, 词数, d_model)变为(batch, 词数, head数，d_model/head数)。
                对应代码为 `view(nbatches, -1, self.h, self.d_k)`
            1.3 最终交换“词数”和“head数”这两个维度，将head数放在前面，最终shape变为(batch, head数, 词数，d_model/head数)。
                对应代码为 `transpose(1, 2)`
        """
        query, key, value = [
            linear(x).view(nbatches, levelcount, self.h, -1).transpose(1, 2)
            for linear, x in zip(self.linears, (x, x, x))
        ]

        """
        2. 求出Q,K,V后，通过attention函数计算出Attention结果，
           这里x的shape为(batch, head数, 词数，d_model/head数)
           self.attn的shape为(batch, head数, 词数，词数)
        """
        x,p_attn = self.attention(
            query, key, value, mask_matrix
        )

        """
        3. 将多个head再合并起来，即将x的shape由(batch, head数, 词数，d_model/head数)
           再变为 (batch, 词数，d_model)
           3.1 首先，交换“head数”和“词数”，这两个维度，结果为(batch, 词数, head数, d_model/head数)
               对应代码为：`x.transpose(1, 2).contiguous()`
           3.2 然后将“head数”和“d_model/head数”这两个维度合并，结果为(batch, 词数，d_model)
        """
        x = (
            x.transpose(1, 2)
                .contiguous()
                .view(nbatches, levelcount, -1)
        )

        p_attn = p_attn.transpose(1, 2)   # [C, head, levelcount] -> [C, levelcount, head]
        # 将p_attn中的每一行都重复d_model/head次，最终shape为(C, levelcount, d_model)
        # p_attn_sum = torch.cat([p_attn[:,:,[i]].repeat(1,1,self.d_k) for i in range(p_attn.shape[2])],dim=2)
        # p_attn_sum = self.linears[-1](p_attn_sum).mean(dim=-1)  # [C, levelcount, d_model]
        # p_attn_sum = p_attn_sum.softmax(dim=1)  # [C, levelcount, d_model]
        # p_attn_sum = p_attn_sum  # [C, levelcount]
        # # 如果p_attn_sum的每一行有不同的符号，说明有问题
        # if (((p_attn_sum < 0).sum(axis=1) != 0) * ((p_attn_sum > 0).sum(axis=1) != 0)).sum()!=0:
        #     print('p_attn_sum has different sign')
        #     assert False
        # if abs(p_attn_sum[0,0] - 0.25) < 0.01 and abs(p_attn_sum[0,1] - 0.25) < 0.01 and abs(p_attn_sum[0,2] - 0.25) < 0.01 and abs(p_attn_sum[0,3] - 0.25) < 0.01:
        #     print('p_attn_sum has same value')
        #     assert False
        p_attn_sum = p_attn.mean(dim=2)  # [C, levelcount]
        if (abs(p_attn_sum.sum(dim=1)-1)>0.01).sum()!=0:
            print('p_attn_sum has different sum')
            assert False

        # 最终通过W^o矩阵再执行一次线性变换，得到最终结果。
        return self.linears[-1](x), p_attn_sum

    def attention(self, query, key, value, mask_matrix=None):
        """
        计算Attention的结果。
        这里其实传入的是Q,K,V，而Q,K,V的计算是放在模型中的，请参考后续的MultiHeadedAttention类。

        这里的Q,K,V有两种Shape，如果是Self-Attention，Shape为(batch, 词数, d_model)，
                               但如果是Multi-Head Attention，则Shape为(batch, head数, 词数，d_model/head数)，
        """

        # 获取d_model的值。之所以这样可以获取，是因为query和输入的shape相同，
        # 若为Self-Attention，则最后一维都是词向量的维度，也就是d_model的值。
        # 若为MultiHead Attention，则最后一维是 d_model / h，h为head数
        d_k = query.size(-1)
        # 执行QK^T / √d_k
        scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)       # [C, head, levelcount, levelcount]

        # 执行公式中的Softmax
        # 这里的p_attn是一个方阵
        # 若是Self Attention，则shape为(batch, 词数, 次数)，例如(1, 7, 7)
        # 若是MultiHead Attention，则shape为(batch, head数, 词数，词数)
        if mask_matrix is not None:
            scores = scores + mask_matrix
        p_attn = scores.softmax(dim=-1)
        if mask_matrix is None:
            p_attn = get_attmask(p_attn.shape,self.hierchy_struc) * p_attn

        # 最后再乘以 V。
        # 对于Self Attention来说，结果Shape为(batch, 词数, d_model)。
        # 对于MultiHead Attention来说，结果Shape为(batch, head数, 词数，d_model/head数)
        # 而这不是最终结果，后续还要将head合并，变为(batch, 词数, d_model)。不过这是MultiHeadAttention该做的事情。

        return torch.matmul(p_attn, value),p_attn.sum(dim=2)

def get_attmask(scores_shape,hierchy_struc):
    mask_matrix = torch.zeros(scores_shape).to(device)
    TS_index_now = 0
    for i,j in enumerate(hierchy_struc):
        if len(scores_shape) == 3:
            mask_matrix[TS_index_now:TS_index_now + j, i, :] = 1
        elif len(scores_shape) == 4:
            mask_matrix[TS_index_now:TS_index_now + j, :, i,:] = 1
        else:
            raise ValueError('scores_shape must be 3 or 4')
        TS_index_now += j
    # assert TS_index_now == scores_shape[0]
    return mask_matrix

class Model(nn.Module):
    def __init__(self, input_size, time_step, A, M, units, dropout, nblock, parents_number_list_,level_number_list_from_overall,level_number_list_to_overall, horizon, bottom_num, cor_tag, ratio_loc, nstack):
        super(Model, self).__init__()

        self.nstack = nstack
        self.nblock = nblock
        self.relu = nn.ReLU()
        self.time_step = time_step
        self.horizon = horizon
        self.units = units
        self.dropout = dropout
        self.A = A
        self.M = M
        self.bottom_num = bottom_num
        self.cor_tag = cor_tag
        self.ratio_loc = ratio_loc
        self.parents_number_list_ = parents_number_list_
        self.level_number_list_from_overall = level_number_list_from_overall
        self.level_number_list_to_overall = level_number_list_to_overall

        self.max_child_nodes = np.array(parents_number_list_).max()

        hierchy_struc = get_HTS_strc(self.parents_number_list_).get_pred_loc()
        levelcount = len(hierchy_struc)

        self.acf = nn.Softplus()
        self.acftanh = nn.Tanh()  #tanh

        # self.fc1dpd = nn.ModuleList()
        # self.mulattdpd = nn.ModuleList()
        # self.fc20dpd = nn.ModuleList()
        # for i in range((self.nblock)//2+1):
        #     self.fc1dpd.append(nn.Linear(self.time_step * self.units[1], self.units[0], bias=False))
        #     # self.mulattdpd.append(MultiHeadedAttention(self.units[0], 4, hierchy_struc))
        #     self.fc20dpd.append(nn.Linear(self.units[0],self.time_step, bias=False))
        self.fc2dpd = nn.Linear(self.time_step, self.horizon, bias=False)

        self.fcLSTM = nn.ModuleList()
        self.fcLSTM1 = nn.ModuleList()
        # self.rnn_main = nn.ModuleList()
        # self.rnns = nn.ModuleList()
        # self.fc1 = nn.ModuleList()
        # self.fc11 = nn.ModuleList()
        # self.fc2 = nn.ModuleList()
        self.multihead_attn = nn.ModuleList()
        # self.fcQ = nn.ModuleList()
        # self.fcK = nn.ModuleList()
        # self.fcV = nn.ModuleList()
        self.fc20 = nn.ModuleList()



        for i in range(self.nblock):
            self.fcLSTM.append(nn.Linear(self.time_step, self.units[0], bias=False))
            self.fcLSTM1.append(nn.Linear(self.units[0], self.units[0], bias=False))
            # self.rnn_main.append(nn.LSTM(1, self.units[1], dropout=self.dropout, bias=False, batch_first=True))
            #
            # self.rnns_block = nn.ModuleList()
            # for i in range(5):
            #     self.rnns_block.append(nn.LSTM(1, self.units[1], dropout=self.dropout, bias=False, batch_first=True))
            # self.rnns.append(self.rnns_block)
            #
            # self.fc1.append(nn.Linear(self.time_step * self.units[1], self.units[0], bias=False))
            # self.fc11.append(nn.Linear(self.time_step * self.units[1], 1, bias=False))

            # self.multihead_attn.append(nn.MultiheadAttention(self.units[0], 4, batch_first=True))
            self.multihead_attn.append(MultiHeadedAttention(self.units[0], self.units[1],self.units[2],4, hierchy_struc))
            # self.fcQ.append(nn.Linear(self.units[0], self.units[0], bias=False))
            # self.fcK.append(nn.Linear(self.units[0], self.units[0], bias=False))
            # self.fcV.append(nn.Linear(self.units[0], self.units[0], bias=False))
            # self.fc2lw = nn.Linear(levelcount, 1, bias=False)  #
            if self.nstack != 1.1:
                self.fc20.append(nn.Linear(self.units[2], self.time_step, bias=False))
            else:
                self.fc20.append(nn.Linear(self.units[0], self.time_step, bias=False))

        self.fc2 = (nn.Linear(self.time_step, self.horizon, bias=False))


        #     fc1_append = nn.ModuleList()
        #     fc2_append = nn.ModuleList()
        #     fc3_append = nn.ModuleList()
        #     fc4_append = nn.ModuleList()
        #     fcB1_append = nn.ModuleList()
        #     fcF1_append = nn.ModuleList()
        #
        #     for j in range(3):
        #         fcBF_mulnum = 1 if j == 0 else 1 if j == 1 else self.max_child_nodes
        #     # self.rnns.append(nn.LSTM(self.units[0], units[1], dropout=self.dropout, bias=False, batch_first=True))
        #         fc1_append.append((nn.Linear(self.time_step, self.units[0], bias=False)))
        #         fc2_append.append((nn.Linear(self.units[0], self.units[0], bias=False)))
        #         fc3_append.append((nn.Linear(self.units[0], self.units[0], bias=False)))
        #         fc4_append.append((nn.Linear(self.units[0], self.units[0], bias=False)))
        #         fcB1_append.append((nn.Linear(self.units[0], self.time_step * fcBF_mulnum, bias=False)))
        #         fcF1_append.append((nn.Linear(self.units[0], self.horizon * fcBF_mulnum, bias=False)))
        #         if j==2:
        #             self.fcB2.append((nn.Linear(fcBF_mulnum,1, bias=False)))
        #             # self.fcF2.append((nn.Linear(fcBF_mulnum,1, bias=False)))
        #     self.fc1.append(fc1_append)
        #     self.fc2.append(fc2_append)
        #     self.fc3.append(fc3_append)
        #     self.fc4.append(fc4_append)
        #     self.fcB1.append(fcB1_append)
        #     self.fcF1.append(fcF1_append)


        # self.ar = nn.Linear(self.time_step, self.horizon, bias=False)
        # self.ardpd = nn.Linear(self.time_step, self.horizon, bias=False)

    #         self.ln1 = nn.Linear(units, 1)
    #         self.ln2 = nn.Linear(units, 1)

    def normalization(self, x):
        x = x + 1e-9
        sum_x = x.sum(axis=-1)
        return x / sum_x.unsqueeze(-1)

    def forward(self, x, DPfrom, training=True):

        # TN_LN = top node + leaf nodes
        # LN_BN = leaf nodes + bottom nodes
        # MN = max nodes in all parents nodes
        # C = all nodes
        # parents = number of parents nodes
        parents_number_list_ = self.parents_number_list_
        level_number_list_from_overall = self.level_number_list_from_overall
        level_number_list_to_overall = self.level_number_list_to_overall
        TN_LN = x.shape[0]- np.array(parents_number_list_[-self.bottom_num:]).sum()


        # [bz, TS, C]
        #         print(x.size())

        if len(x.size()) == 2:
            x = x.T.unsqueeze(-1)   # [TS,C] -> [C,TS,1]

        DPfrom = DPfrom.permute(1, 0, 2)  # [TS,C,levelcount_1] -> [C,TS,levelcount_1]DPfrom = DPfrom.permute(1, 0, 2)  # [TS,C,levelcount] -> [C,TS,levelcount]
        # DPfrom_true = DPfrom_true.permute(2, 1, 0)  # [1,C,levelcount] -> [levelcount,C,1]
        x = x[:,:,0:1]  # [C,TS,1]

        hierchy_struc = get_HTS_strc(self.parents_number_list_).get_pred_loc()
        ones_noise = torch.ones_like(x)
        noise_tensor = torch.ones_like(x).normal_(mean=0, std=noi_std)
        noise_level_number = noi_level  # 注意此处指的是现实中的噪音所在层数，从1开始，不是从0开始
        ones_noise[sum(hierchy_struc[:noise_level_number - 1]):sum(hierchy_struc[:noise_level_number]), :] = (
                noise_tensor[sum(hierchy_struc[:noise_level_number - 1]):sum(hierchy_struc[:noise_level_number]),
                :] + 1)
        x = x * ones_noise


        #         forecast = torch.zeros(x.size()[0], x.size()[1], x.size()[2]).to(device)

        self.DPD_overall = torch.zeros(x.size(0)-1,x.size(1)).to(device)



        ## ----- ratio model -----
        hierchy_struc = get_HTS_strc(self.parents_number_list_).get_pred_loc()
        levelcount = len(hierchy_struc)

        DPfrom_now = DPfrom.permute(2, 0, 1)  # [levelcount_1,C,TS]

        # 生成attention mask矩阵
        DPmask = torch.ones(DPfrom_now.size(0), DPfrom_now.size(1), DPfrom_now.size(1)).to(device)  # [levelcount_1, C, C]
        DPmask = DPmask * (-np.inf)
        for level_index in range(levelcount - 1):
            child_index_inTS_end = sum(hierchy_struc[:level_index + 1])
            DPmask[level_index, [k for k in range(child_index_inTS_end)], [k for k in range(child_index_inTS_end)]] = 0
            for i in range(levelcount):
                childcount_in_eachparent = level_number_list_from_overall[level_index][i]
                if childcount_in_eachparent != childcount_in_eachparent:  # is nan
                    continue
                assert len(childcount_in_eachparent) == hierchy_struc[level_index]
                for parent_index, childcount in enumerate(childcount_in_eachparent):
                    child_index_inTS_start = child_index_inTS_end
                    child_index_inTS_end = child_index_inTS_start + childcount
                    DPmask[level_index, child_index_inTS_start:child_index_inTS_end,
                    child_index_inTS_start:child_index_inTS_end] = 0

        DPmask = DPmask.unsqueeze(1)  # [levelcount_1, 1, C, C]

        # DPfrom_now = DPfrom_now.reshape(DPfrom_now.size(0) * DPfrom_now.size(1), DPfrom_now.size(2)).unsqueeze(
        #     -1)  # [levelcount_1,C,TS] -> [levelcount_1*C,TS,1]
        # DPfrom_now = self.rnns[0][0](DPfrom_now)[0]  # [levelcount_1*C,TS,1] -> [levelcount_1, C, TS, units1]
        # DPfrom_now = DPfrom_now.reshape(levelcount - 1, -1, DPfrom_now.size(-1) * DPfrom_now.size(
        #     -2))  # [levelcount_1*C,TS,units1] -> [levelcount_1, C, TS*units1]
        # DPfrom_now = self.fc1dpd[0](DPfrom_now)  # [levelcount_1, C, TS*units1] -> [levelcount_1, C, units1]
        # for i in range((self.nblock) // 2 + 1):
        #
        #     DPfrom = self.mulattdpd[i](DPfrom_now, DPmask)  # [levelcount_1, C, units1]
        #
        #
        #     # DPfrom = (self.fc20dpd[i](DPfrom))  # [levelcount_1, C, units1] -> [levelcount_1, C, 1]
        #     DPfrom = self.acf(DPfrom)
        #     DPfrom_now = DPfrom_now + DPfrom
        DPfrom_now = self.acf(self.fc2dpd(DPfrom_now))
        x_DPD_overall = torch.zeros_like(DPfrom_now)
        #做normalization
        for level_index in range(levelcount - 1):
            child_index_inTS_end = sum(hierchy_struc[:level_index + 1])
            for i in range(levelcount):
                childcount_in_eachparent = level_number_list_from_overall[level_index][i]
                if childcount_in_eachparent != childcount_in_eachparent:  # is nan
                    continue
                assert len(childcount_in_eachparent) == hierchy_struc[level_index]
                for parent_index, childcount in enumerate(childcount_in_eachparent):
                    child_index_inTS_start = child_index_inTS_end
                    child_index_inTS_end = child_index_inTS_start + childcount
                    x_DPD_overall[level_index, child_index_inTS_start:child_index_inTS_end, :] = (DPfrom_now[level_index, child_index_inTS_start:child_index_inTS_end, :]).softmax(dim=0)


        # for level_index in range(levelcount-1):
        #     DPfrom_leveli = DPfrom[:,:,[level_index]]   # [C, TS, 1]
        #     DPfrom_leveli = (self.rnns[0][level_index](DPfrom_leveli)[0]).unsqueeze(0)  # [C, TS, 1] -> [1, C, TS, units1]
        #     if level_index == 0:
        #         x_DPD_overall = DPfrom_leveli
        #     else:
        #         x_DPD_overall = torch.cat([x_DPD_overall, DPfrom_leveli], dim=0)  # [1, C, TS, units1] -> [1+1+level_index, C, TS, units1]
        #
        # x_DPD_overall = x_DPD_overall.reshape(x_DPD_overall.size(0), x_DPD_overall.size(1), x_DPD_overall.size(2) * x_DPD_overall.size(3))  # [1+1+level_index, C, TS*units1]
        #
        # # [1+1+level_index, C, TS, units1] -> [1+1+level_index, C, units0]
        # x_DPD_overall = self.fc11[0](x_DPD_overall)  # [1+1+level_index, C, TS*units1] -> [1+level_index, C, 1]
        # x_DPD_overall = DPfrom_true


        for nblock in range(self.nblock):
            # ------ LSTM ------
            # x_new = (self.rnn_main[nblock](x)[0]).unsqueeze(0)  # [C, TS, 1] -> [1, C, TS, units1]
            # x_new = x_new.reshape(x_new.size(0), x_new.size(1), x_new.size(2) * x_new.size(3))  # [1+1+level_index, C, TS*units1]
            # x_new = x_new[[0]]
            # x_new = self.fc1[nblock](x_new)  # [1+1+level_index, C, TS*units1] -> [1, C, units0]
            x_new = self.acf(self.fcLSTM[nblock](x.squeeze(-1)))  # [C, TS, 1] -> [1, C, units0]
            x_new = self.acf(self.fcLSTM1[nblock](x_new))

            if nstack!=1.1:
                self_index_inTS_end = 0
                x_frommodule_list = []
                for level_index in range(levelcount):   #brunch index
                    self_index_inTS_start = self_index_inTS_end
                    self_index_inTS_end = self_index_inTS_start + hierchy_struc[level_index]
                    x_self = x_new[self_index_inTS_start:self_index_inTS_end,:]  # [TScount_in_this_level, units0]

                    # ------ TD ------
                    if (level_index < levelcount - 1) and ((self.nstack == 3) or (self.nstack == 2.1)):
                        assert self_index_inTS_end < x_new.size(0)
                        TD_index_inTS_start = self_index_inTS_end
                        TD_index_inTS_end = x_new.size(0)
                        x_DPD = x_DPD_overall[level_index,TD_index_inTS_start:TD_index_inTS_end,:].repeat(1,x_self.size(1))  # [TScount_in_TD, units0]

                        # x_parent_of_children
                        x_parent_of_children = torch.zeros_like(x_DPD)  # [TScount_in_TD, units0]
                        child_index_inTS_end = 0
                        for i in range(levelcount):
                            childcount_in_eachparent = level_number_list_from_overall[level_index][i]
                            if childcount_in_eachparent != childcount_in_eachparent:  # is nan
                                continue
                            assert len(childcount_in_eachparent) == x_self.size(0)
                            for parent_index,childcount in enumerate(childcount_in_eachparent):
                                child_index_inTS_start = child_index_inTS_end
                                child_index_inTS_end = child_index_inTS_start + childcount
                                x_parent_of_children[child_index_inTS_start:child_index_inTS_end,:] = x_self[[parent_index],:].repeat(childcount,1)

                        assert child_index_inTS_end == x_parent_of_children.size(0)
                        x_TD = torch.mul(x_DPD, x_parent_of_children)  # [TScount_in_TD, units0]
                    else:
                        # x_TD = torch.zeros(0,x_self.size(1)).to(device)
                        x_TD = x_new[self_index_inTS_end:, :]

                    # ------ BU ------
                    if (level_index > 0) and ((self.nstack == 3) or (self.nstack == 2.2)):
                        assert self_index_inTS_start > 0
                        # BU_index_inTS_start = 0
                        # BU_index_inTS_end = self_index_inTS_start
                        x_BU = torch.zeros(self_index_inTS_start,x_self.size(1)).to(device)  # [TScount_in_BU, units0]

                        parent_index = 0
                        for i in range(levelcount):
                            childcount_for_eachparent = level_number_list_to_overall[level_index][i]
                            if childcount_for_eachparent != childcount_for_eachparent:  # is nan
                                continue
                            assert sum(childcount_for_eachparent) == x_self.size(0)

                            child_index = 0
                            for childcount in childcount_for_eachparent:
                                x_BU[[parent_index],:] = x_self[child_index:child_index+childcount,:].sum(dim=0,keepdim=True)
                                child_index += childcount
                                parent_index += 1
                        assert parent_index == x_BU.size(0)
                    else:
                        # x_BU = torch.zeros(0,x_self.size(1)).to(device)
                        x_BU = x_new[0:self_index_inTS_start, :]

                    x_thismodule = torch.cat([x_BU, x_self, x_TD], dim=0)  # [TScount_in_BU+TScount_in_self+TScount_in_TD, units0
                    assert x_thismodule.size(0) == x_new.size(0)
                    x_frommodule_list.append(x_thismodule)

                x_new = torch.stack(x_frommodule_list, dim=1)  # [C, units0] -> [C, levelcount, units0]

                #做MultiheadAttention
                # x_new = self.multihead_attn[nblock](self.fcQ[nblock](x_new), self.fcK[nblock](x_new), self.fcV[nblock](x_new))[0]  # [C, levelcount, units0] -> [C, levelcount, units0]
                x_new,p_attn_sum = self.multihead_attn[nblock](x_new)

                x_new = self.acf(x_new.sum(dim=1))  # [C, levelcount, units0] -> [C, units0]
                # x_new = x_new.reshape(x_new.size(0), x_new.size(1) * x_new.size(2))  # [C, levelcount, units0] -> [C, levelcount*units0]
                # x_new = x_new.permute(0, 2, 1)
                # x_new = self.acf(self.fc2lw(x_new).squeeze(-1))
            x_new = self.acf(self.fc20[nblock](x_new)).unsqueeze(-1)  #bkill

            #残差结构
            x = x_new + x


        x = self.fc2(x.squeeze(-1))  # [C, units0] -> [C, horizon]

        zeros = torch.zeros(*x.size()).to(device)
        # x = torch.where(x < zeros, zeros, x)

        x = torch.matmul(self.M, x)
        return_x = x
        # 将x_stackBU每一个值都做平方
        return_loss = torch.Tensor([0]).to(device)
        return return_x.squeeze(),p_attn_sum,return_loss,x_DPD_overall # DPD_reverse_[:, -self.horizon:].squeeze()

    def build_DPD(self, x, parents_number_list_, training=True):
        if training:
            # DPD = torch.zeros([len(parents_number_list_), np.array(parents_number_list_).max(),
            #                    self.time_step]).to(device)
            DPD = torch.zeros([len(parents_number_list_), np.array(parents_number_list_).max(),
                                x.size(0)]).to(device)
        else:
            assert 0
        # else:
        #     DPD = torch.zeros([len(parents_number_list_), np.array(parents_number_list_).max(),
        #                        self.horizon]).to(device)



        stop = 0
        for i, j in enumerate(parents_number_list_):
            if i == 0:
                stop = j
                DPD[i, :j, :] = x[:, :stop].T

            else:
                stop = stop + j
                DPD[i, :j, :] = x[:, stop - j:stop].T

        return DPD.permute(2, 0, 1)

    def reverse_DPD(self, x, parents_number_list_):
        reverse_DPD = torch.zeros([x.size(0), sum(parents_number_list_[:])]).to(device)  #self.time_step
        stop = 0
        for i, j in enumerate(parents_number_list_):
            reverse_DPD[:, stop: stop + j] = x[:, i, :j]
            #             print(x[:,i,:j].sum(axis=-1))
            stop += j

        return reverse_DPD.permute(1, 0)

    def loss(self, B1, label1,B2,label2, parents_number_list_, theta, MAE_tag,return_loss):
        #         B1,B2 is prediction, label1,label2 is truth
        if MAE_tag == '_MAE': forecast_loss = nn.L1Loss(reduction='sum')
        elif MAE_tag == '': forecast_loss = nn.MSELoss(reduction='sum')

        #         label2_loss = self.normalization(self.build_DPD(label2[:,1:], level_number_list, training=False))
        #         B1_loss = B1
        #         B2_loss = B2.unsqueeze(0)
        #         label2 = label2.unsqueeze(0)


        if len(label1.size()) == 1:
            label1 = label1.unsqueeze(0)
        else:
            label1 = label1
        # if len(label2.size()) == 1:
        #     label2 = label2.unsqueeze(0)
        # else:
        #     label2 = label2
        if len(B1.size()) == 1:
            B1_loss = B1.unsqueeze(0)
        else:
            B1_loss = B1.T
        # if len(B2.size()) == 1:
        #     B2_loss = B2.unsqueeze(0)
        # else:
        #     B2_loss = B2.T

        label2 = label2.permute(2, 1, 0)
        assert label2.shape == B2.shape

        likelihood = 0
        # # for k in range(B2_loss.shape[0]):
        # stop = 1
        # for i, j in enumerate(parents_number_list_):  #[:4]  [:-self.botton_num]
        #     if len(B2_loss.shape) == 2:
        #         if stop + j == sum(parents_number_list_) + 1:
        #             tem_B2 = B2_loss[:, stop - 1:]
        #             tem_label2 = label2[:, stop:]
        #         else:
        #             tem_B2 = B2_loss[:, stop - 1: stop + j - 1]
        #             tem_label2 = label2[:, stop: stop + j]
        #     else:
        #         print(error)
        #         if stop + j == sum(parents_number_list_) + 1:
        #             tem_B2 = B2_loss[:, :, stop - 1:]
        #             tem_label2 = label2[:, :, stop:]
        #         else:
        #             tem_B2 = B2_loss[:, :, stop - 1: stop + j - 1]
        #             tem_label2 = label2[:, :, stop: stop + j]
        #     tem_dis = torch.distributions.dirichlet.Dirichlet(j * self.normalization(tem_B2))
        #     tem_likelihood = tem_dis.log_prob(self.normalization(tem_label2))
        #
        #     likelihood += tem_likelihood.sum()
        #     stop += j
        # likelihood = likelihood / B2_loss.shape[0]
        # likelihood = return_loss  #torch.zeros([1])

        hierchy_struc = get_HTS_strc(self.parents_number_list_).get_pred_loc()
        levelcount = len(hierchy_struc)
        for level_index in range(levelcount - 1):
            child_index_inTS_end = sum(hierchy_struc[:level_index + 1])
            for i in range(levelcount):
                childcount_in_eachparent = level_number_list_from_overall[level_index][i]
                if childcount_in_eachparent != childcount_in_eachparent:  # is nan
                    continue
                assert len(childcount_in_eachparent) == hierchy_struc[level_index]
                for parent_index, childcount in enumerate(childcount_in_eachparent):
                    child_index_inTS_start = child_index_inTS_end
                    child_index_inTS_end = child_index_inTS_start + childcount
                    tem_B2 = B2[level_index, child_index_inTS_start:child_index_inTS_end, :].squeeze(-1)
                    tem_label2 = label2[level_index, child_index_inTS_start:child_index_inTS_end, :].squeeze(-1)
                    # try:
                    #     assert (torch.round(tem_B2.sum()) == 1)
                    #     assert (torch.round(tem_label2.sum()) == 1)
                    # except AssertionError:
                    #     print(tem_B2.sum())
                    #     print(tem_label2.sum())
                    #     assert 0
                    tem_dis = torch.distributions.dirichlet.Dirichlet(childcount * self.normalization(tem_label2))
                    tem_likelihood = tem_dis.log_prob(self.normalization(tem_B2))
                    likelihood += tem_likelihood.sum()






        # likelihood = forecast_loss(B2, label2)

        HTS_stuc = get_HTS_strc(parents_number_list_)
        pred_loc = HTS_stuc.get_pred_loc()
        ratio_loc = self.ratio_loc #[0.21085779683092262,0.23239679903741464,0.26296517958407817,1.2533606976428853]
        stop = 0
        # loss_sum = 0
        # for i, j in zip(pred_loc,ratio_loc):
        #     tem_Bl = B1_loss[:, stop:i + stop]
        #     tem_label1 = label1[:, stop:i + stop]
        #     stop += i
        #     loss_sum += forecast_loss(tem_Bl, tem_label1) / j
        loss_sum = forecast_loss(B1_loss, label1)

        if len(B1_loss.size()) == 2:
            return likelihood * theta + loss_sum, likelihood * theta, loss_sum   #forecast_loss(B1_loss[:, :11], label1[:, :11])
        else:
            return likelihood * theta + loss_sum, likelihood * theta, loss_sum    #forecast_loss(B1_loss[:, :, :11], label1[:, :, :11])


import matplotlib.pyplot as plt


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.uniform(m.weight, 0, 1)

    elif isinstance(m, nn.LSTM):
        for para in m.parameters():
            nn.init.uniform(para, 0, 1)



def result(data, DP_label, DP_label_from_overall, A, M, level_number_list,level_number_list_from_overall,level_number_list_to_overall, bottom_num, time_step, horizon, units, dropout, epoch, nlayers, lr, theta, patience, cor_tag, name, MAE_tag, nstack):
    torch.random.manual_seed(2020)
    input_size = [data.shape[0], data.shape[1]]
    A = torch.from_numpy(A).float().to(device)
    M = torch.from_numpy(M).float().to(device)

    X_new, X_dp, y_new, z_new, X_DP_label_new, y_DP_label_new = data_multiple_build(data.values, DP_label, DP_label_from_overall, time_step, horizon)
    new_data = tensor_multiple_data(X_new, X_dp, y_new, z_new, X_DP_label_new, y_DP_label_new)
    train_data = Subset(new_data, np.arange(int(len(new_data) * 0.8)))
    valid_data = Subset(new_data, np.arange(int(len(new_data) * 0.8), len(new_data) - 7))
    test_data = Subset(new_data, np.arange(len(new_data) - 7, len(new_data)))

    HTS_stuc = get_HTS_strc(level_number_list)
    pred_loc = HTS_stuc.get_pred_loc()
    true_new_tr = df.iloc[:int(len(df) * 0.8), :].values
    pred_loc_cumsum = [sum(pred_loc[:i + 1]) for i in range(len(pred_loc) - 1)]
    true_new_split = np.hsplit(true_new_tr, pred_loc_cumsum)
    ratio_loc = []
    ratio_loc_dict = {'Overall_STDMEAN': evaluate(true_new_tr, true_new_tr)[5]}
    for i, (j, k) in enumerate(zip(true_new_split, true_new_split)):
        ratio_loc_dict['Level_' + str(i + 1) + '_STDMEAN'] = evaluate(j, k)[5]
        ratio_loc.append(evaluate(j, k)[5])
    print(ratio_loc_dict)

    model = Model(input_size[0], time_step, A, M, units, dropout, nlayers, level_number_list,level_number_list_from_overall,level_number_list_to_overall, 1, bottom_num, cor_tag, ratio_loc, nstack).to(device)  #里面的1是horizon
    # model.apply(weight_init)


    flag = 0

    loss_func = model.loss
    optimizer = torch.optim.RMSprop(model.parameters(), lr)
    my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.1)
    early_stop = EarlyStopping(name, patience, verbose=False,
                               checkpoint=f'./result1/pt/time_step_units_nlayers_cor_theta_{ts}_{u}_{u_plus}_{u_plus2}_{n}_{cor_tag}_{theta}_{name}_model_normal.pt')  # './' + str(name) + 'model_normal.pt'


    train_loss_list = []
    train_loss_dpd_list = []
    train_loss_me_list = []
    valid_loss_list = []
    valid_loss_dpd_list = []
    valid_loss_me_list = []

    for i in range(epoch):
        epoch_txt = f'Epoch [{i + 1}/{epoch}]'
        flag += 1

        train_loss, train_pred, train_loss_dpd, train_loss_me = train(train_data, level_number_list, model, optimizer, loss_func, epoch_txt, theta, MAE_tag)
        valid_loss, valid_pred, valid_true, valid_loss_dpd, valid_loss_me, _ = test(valid_data, level_number_list, model, loss_func, theta, MAE_tag)
        print('Valid, loss=', valid_loss.item(), valid_loss_dpd.item(), valid_loss_me.item())
        train_loss_list.append(train_loss.item())
        train_loss_dpd_list.append(train_loss_dpd.item())
        train_loss_me_list.append(train_loss_me.item())
        valid_loss_list.append(valid_loss.item())
        valid_loss_dpd_list.append(valid_loss_dpd.item())
        valid_loss_me_list.append(valid_loss_me.item())

        #             if (i>150) & ((i + 1) % 20 == 0):
        #                 my_lr_scheduler.step()
        early_stop(valid_loss.item(), model)
        if early_stop.early_stop:
            break

    #         for param in model.named_parameters():
    #             print(param)
    model.load_state_dict(torch.load(f'./result1/pt/time_step_units_nlayers_cor_theta_{ts}_{u}_{u_plus}_{u_plus2}_{n}_{cor_tag}_{theta}_{name}_model_normal.pt', map_location=device))
    test_loss, test_pred, test_true, test_loss_dpd, test_loss_me, pred_dec = test(test_data, level_number_list, model, loss_func, theta, MAE_tag)
    # model.load_state_dict(torch.load('./' + str(name) + 'model_normal.pt'))

    fig, ax1 = plt.subplots()  # subplots一定要带s
    ax1.plot(train_loss_list, label='train',c='r')
    # ax1.set_ylabel('train')
    # ax2 = ax1.twinx()  # twinx将ax1的X轴共用与ax2，这步很重要
    ax1.plot(valid_loss_list, label='valid',c='g')
    # ax2.set_ylabel('valid')
    plt.legend()
    plt.show()
    plt.savefig(f'./result1/time_step_units_nlayers_cor_theta_{ts}_{u}_{u_plus}_{u_plus2}_{n}_{cor_tag}_{theta}_{name}_loss.jpg')
    plt.close()

    fig, ax1 = plt.subplots()  # subplots一定要带s
    line1, = ax1.plot(train_loss_me_list, c='r')
    line2, = ax1.plot(valid_loss_me_list, c='r', linestyle='--')
    ax1.set_ylabel('me')
    ax2 = ax1.twinx()  # twinx将ax1的X轴共用与ax2，这步很重要
    line3, = ax2.plot(train_loss_dpd_list, c='g')
    line4, = ax2.plot(valid_loss_dpd_list, c='g', linestyle='--')
    ax2.set_ylabel('dpd')
    plt.legend((line1, line2, line3, line4), ('train_me', 'valid_me', 'train_dpd', 'valid_dpd'))
    plt.show()
    plt.savefig(f'./result1/time_step_units_nlayers_cor_theta_{ts}_{u}_{u_plus}_{u_plus2}_{n}_{cor_tag}_{theta}_{name}_lossdpd.jpg')
    plt.close()

    return test_pred, test_true, pred_dec



MAE_tag = ''
MAE_tag = '_MAE'
model_name_ori = f'{dataset_name}_HATN_mean2FC_ratioFC_nblock_mymask_simpleloss_ratiodirr_pcwise_tu3_twoFC{MAE_tag}'

if dataset_name == 'M5':
    from Dataset_M5 import *
    for ts in ts_list:  # 14,28
        for u in u_list:   # 16,64
            for u_plus in u_plus_list:
                for u_plus2 in u_plus2_list:
                    for n_ori in n_list:  # 20,50 [2,20]
                        for cor_tag in [1]:
                            for theta in theta_list:  # 0.0 ,6e2, 6
                                for nstack in nstack_list:
                                    for noi_level in noi_level_list:
                                        for noi_std in noi_std_list:
                                            if nstack!=3 and theta!=0: continue
                                            if nstack==3 and theta==0: continue
                                            # nstack_new = 2 if nstack == 3 else nstack
                                            # nstack_new = nstack
                                            # n = int(n_ori / int(nstack_new))
                                            n = n_ori
                                            model_name_in = f'{model_name_ori}_lv_{noi_level}_std_{noi_std}_nstack{nstack}_hor{hor_value}'
                                            print(f'model_{ts}_{u}_{n}_{cor_tag}_{theta}_{model_name_in}')
                                            start_time = time.time()
                                            print('start_time:', time.ctime(start_time))
                                            # if os.path.exists('./result/time_step_units_nlayers_cor_theta_'+ str(ts)+'_'+str(u)+'_'+str(n)+'_'+str(cor_tag)+'_'+str(theta)+'_'+f'{dataset_name}_NHBEATS_re_tu_pingxing_output_fixxstack1{MAE_tag}_nstack{nstack}'+'_pred.csv'):
                                            #     pd.read_csv('./result/time_step_units_nlayers_cor_theta_'+ str(ts)+'_'+str(u)+'_'+str(n)+'_'+str(cor_tag)+'_'+str(theta)+'_'+f'{dataset_name}_NHBEATS_re_tu_pingxing_output_fixxstack1{MAE_tag}_nstack{nstack}'+'_pred.csv',index_col=0).to_csv('./result/time_step_units_nlayers_cor_theta_'+ str(ts)+'_'+str(u)+'_'+str(n)+'_'+str(cor_tag)+'_'+str(theta)+'_'+model_name_in+'_pred.csv')
                                            #     pd.read_csv('./result/time_step_units_nlayers_cor_theta_'+ str(ts)+'_'+str(u)+'_'+str(n)+'_'+str(cor_tag)+'_'+str(theta)+'_'+f'{dataset_name}_NHBEATS_re_tu_pingxing_output_fixxstack1{MAE_tag}_nstack{nstack}'+'_pred_dec_1.csv',index_col=0).to_csv('./result/time_step_units_nlayers_cor_theta_'+ str(ts)+'_'+str(u)+'_'+str(n)+'_'+str(cor_tag)+'_'+str(theta)+'_'+model_name_in+'_pred_dec_1.csv')
                                            #     pd.read_csv('./result/time_step_units_nlayers_cor_theta_'+ str(ts)+'_'+str(u)+'_'+str(n)+'_'+str(cor_tag)+'_'+str(theta)+'_'+f'{dataset_name}_NHBEATS_re_tu_pingxing_output_fixxstack1{MAE_tag}_nstack{nstack}'+'_pred_dec_2.csv',index_col=0).to_csv('./result/time_step_units_nlayers_cor_theta_'+ str(ts)+'_'+str(u)+'_'+str(n)+'_'+str(cor_tag)+'_'+str(theta)+'_'+model_name_in+'_pred_dec_2.csv')
                                            #     pd.read_csv('./result/time_step_units_nlayers_cor_theta_'+ str(ts)+'_'+str(u)+'_'+str(n)+'_'+str(cor_tag)+'_'+str(theta)+'_'+f'{dataset_name}_NHBEATS_re_tu_pingxing_output_fixxstack1{MAE_tag}_nstack{nstack}'+'_pred_dec_3.csv',index_col=0).to_csv('./result/time_step_units_nlayers_cor_theta_'+ str(ts)+'_'+str(u)+'_'+str(n)+'_'+str(cor_tag)+'_'+str(theta)+'_'+model_name_in+'_pred_dec_3.csv')
                                            #     continue
                                            pred, true, pred_dec = result(df_new, DP_label, DP_label_from_overall, A, M, level_number_list,level_number_list_from_overall,level_number_list_to_overall,
                                                                bottom_num=len(item_level_number_list),
                                                                time_step=ts, horizon=hor_value, units=[u, u_plus, u_plus2], dropout=0.0,
                                                                epoch=50, nlayers=n, lr=5*(1e-03), theta=theta, patience=3, cor_tag=cor_tag,
                                                                name=model_name_in, MAE_tag=MAE_tag, nstack=nstack)
                        #patience 10/20/50
                        #lr最初1e-04，后来改成1e-03
                        #MAE MSE

                                            # pred_new = (np.multiply(np.array(pred).squeeze(), scale[1].reshape(1,-1).repeat(7,axis=0)) + scale[0].reshape(1,-1).repeat(7,axis=0)) #.astype(int)
                                            pred_new = np.array(pred).squeeze()
                                            true_new = df.iloc[-7:,:].values

                                            pred_new2 = build_evaluation_data(df_new, pred_new, 7)
                                            true_new2 = build_evaluation_data(df_new, true_new, 7)

                                            pred_dec_all = None
                                            for dec_num in range(pred_dec.shape[1]):
                                                pred_dec_new = build_evaluation_data(df_new, np.array(
                                                    pred_dec[:, dec_num].cpu().numpy()).T, 7).rename(
                                                    columns={'y': f'y_{dec_num + 1}'})
                                                if pred_dec_all is None:
                                                    pred_dec_all = pred_dec_new
                                                else:
                                                    pred_dec_all = pd.concat(
                                                        [pred_dec_all, pred_dec_new[[f'y_{dec_num + 1}']]], axis=1)
                                            pred_dec_all.to_csv(
                                                f'./result1/time_step_units_nlayers_cor_theta_{ts}_{u}_{u_plus}_{u_plus2}_{n}_{cor_tag}_{theta}_{model_name_in}_pred_dec.csv')
                                            # [pred_dec_1, pred_dec_2, pred_dec_3] = pred_dec
                                            # pred_dec_1 = build_evaluation_data(df_new, np.array(pred_dec_1).squeeze(), 7)
                                            # pred_dec_2 = build_evaluation_data(df_new, np.array(pred_dec_2).squeeze(), 7)
                                            # pred_dec_3 = build_evaluation_data(df_new, np.array(pred_dec_3).squeeze(), 7)
                                            # pred_dec_1.to_csv('./result/time_step_units_nlayers_cor_theta_'+ str(ts)+'_'+str(u)+'_'+str(n)+'_'+str(cor_tag)+'_'+str(theta)+'_'+model_name_in+'_pred_dec_1.csv')
                                            # pred_dec_2.to_csv('./result/time_step_units_nlayers_cor_theta_'+ str(ts)+'_'+str(u)+'_'+str(n)+'_'+str(cor_tag)+'_'+str(theta)+'_'+model_name_in+'_pred_dec_2.csv')
                                            # pred_dec_3.to_csv('./result/time_step_units_nlayers_cor_theta_'+ str(ts)+'_'+str(u)+'_'+str(n)+'_'+str(cor_tag)+'_'+str(theta)+'_'+model_name_in+'_pred_dec_3.csv')
                                            pred_new2.to_csv(f'./result1/time_step_units_nlayers_cor_theta_{ts}_{u}_{u_plus}_{u_plus2}_{n}_{cor_tag}_{theta}_{model_name_in}_pred.csv')
                                            end_time = time.time()
                                            elapsed_time = end_time - start_time
                                            print('start_time:', time.ctime(start_time))
                                            print('end_time:', time.ctime(end_time))
                                            print('elapsed_time:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
                                            print(' ')
                                            print(' ')


elif dataset_name == 'Labor':
    from Dataset_Labor import *
    for ts in ts_list: #14,28
        for u in u_list: #8,20
            for u_plus in u_plus_list:
                for u_plus2 in u_plus2_list:
                    for n_ori in n_list:  #10,64,128
                        for cor_tag in [1]:
                            for theta in theta_list:  # , 0.1, 0.9 [6e2,1e2]
                                for nstack in nstack_list:
                                    for noi_level in noi_level_list:
                                        for noi_std in noi_std_list:
                                            if nstack != 3 and theta != 0: continue
                                            if nstack == 3 and theta == 0: continue
                                            # nstack_new = 2 if nstack == 3 else nstack
                                            # nstack_new = nstack
                                            # n = int(n_ori / int(nstack_new))
                                            n = n_ori
                                            model_name_in = f'{model_name_ori}_lv_{noi_level}_std_{noi_std}_nstack{nstack}_hor{hor_value}'
                                            print(f'model_{ts}_{u}_{n}_{cor_tag}_{theta}_{model_name_in}')
                                            start_time = time.time()
                                            print('start_time:', time.ctime(start_time))
                                            pred, true, pred_dec = result(df_new, DP_label, DP_label_from_overall, A, M, level_number_list,level_number_list_from_overall,level_number_list_to_overall, bottom_num=len(l_3),
                                                                time_step=ts, horizon=hor_value, units=[u, u_plus, u_plus2], dropout=0.0,
                                                                epoch=50, nlayers=n, lr=(1e-04), theta=theta, patience=3, cor_tag=cor_tag,
                                                                name=model_name_in, MAE_tag=MAE_tag, nstack=nstack)

                                            # pred_new = (np.multiply(np.array(pred).squeeze(), scale[1].reshape(1,-1).repeat(7,axis=0)) + scale[0].reshape(1,-1).repeat(7,axis=0)) #.astype(int)
                                            pred_new = np.array(pred).squeeze()
                                            true_new = df.iloc[-7:, :].values

                                            pred_new2 = build_evaluation_data(df_new, pred_new, 7)
                                            true_new2 = build_evaluation_data(df_new, true_new, 7)

                                            pred_dec_all = None
                                            for dec_num in range(pred_dec.shape[1]):
                                                pred_dec_new = build_evaluation_data(df_new, np.array(
                                                    pred_dec[:, dec_num].cpu().numpy()).T, 7).rename(
                                                    columns={'y': f'y_{dec_num + 1}'})
                                                if pred_dec_all is None:
                                                    pred_dec_all = pred_dec_new
                                                else:
                                                    pred_dec_all = pd.concat(
                                                        [pred_dec_all, pred_dec_new[[f'y_{dec_num + 1}']]], axis=1)
                                            pred_dec_all.to_csv(
                                                f'./result1/time_step_units_nlayers_cor_theta_{ts}_{u}_{u_plus}_{u_plus2}_{n}_{cor_tag}_{theta}_{model_name_in}_pred_dec.csv')
                                            # [pred_dec_1, pred_dec_2, pred_dec_3] = pred_dec
                                            # pred_dec_1 = build_evaluation_data(df_new, np.array(pred_dec_1).squeeze(), 7)
                                            # pred_dec_2 = build_evaluation_data(df_new, np.array(pred_dec_2).squeeze(), 7)
                                            # pred_dec_3 = build_evaluation_data(df_new, np.array(pred_dec_3).squeeze(), 7)
                                            # pred_dec_1.to_csv(
                                            #     './result/time_step_units_nlayers_cor_theta_' + str(ts) + '_' + str(u) + '_' + str(
                                            #         n) + '_' + str(cor_tag) + '_' + str(
                                            #         theta) + '_' + model_name_in + '_pred_dec_1.csv')
                                            # pred_dec_2.to_csv(
                                            #     './result/time_step_units_nlayers_cor_theta_' + str(ts) + '_' + str(u) + '_' + str(
                                            #         n) + '_' + str(cor_tag) + '_' + str(
                                            #         theta) + '_' + model_name_in + '_pred_dec_2.csv')
                                            # pred_dec_3.to_csv(
                                            #     './result/time_step_units_nlayers_cor_theta_' + str(ts) + '_' + str(u) + '_' + str(
                                            #         n) + '_' + str(cor_tag) + '_' + str(
                                            #         theta) + '_' + model_name_in + '_pred_dec_3.csv')
                                            pred_new2.to_csv(f'./result1/time_step_units_nlayers_cor_theta_{ts}_{u}_{u_plus}_{u_plus2}_{n}_{cor_tag}_{theta}_{model_name_in}_pred.csv')
                                            end_time = time.time()
                                            elapsed_time = end_time - start_time
                                            print('start_time:', time.ctime(start_time))
                                            print('end_time:', time.ctime(end_time))
                                            print('elapsed_time:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
                                            print(' ')
                                            print(' ')

elif dataset_name == 'Traffic':
    from Dataset_Traffic import *
    for ts in ts_list:  #[14,28]
        for u in u_list: # 8,2,64
            for u_plus in u_plus_list:
                for u_plus2 in u_plus2_list:
                    for n_ori in n_list:  # 10,32
                        for cor_tag in [1]:
                            for theta in theta_list:  # , 0.1, 0.9 [6,1]
                                for nstack in nstack_list:
                                    for noi_level in noi_level_list:
                                        for noi_std in noi_std_list:
                                            if nstack != 3 and theta != 0: continue
                                            if nstack == 3 and theta == 0: continue
                                            # nstack_new = 2 if nstack == 3 else nstack
                                            # nstack_new = nstack
                                            # n = int(n_ori / int(nstack_new))
                                            n = n_ori
                                            model_name_in = f'{model_name_ori}_lv_{noi_level}_std_{noi_std}_nstack{nstack}_hor{hor_value}'
                                            print(f'model_{ts}_{u}_{n}_{cor_tag}_{theta}_{model_name_in}')
                                            start_time = time.time()
                                            print('start_time:', time.ctime(start_time))
                                            pred, true, pred_dec = result(df_new, DP_label, DP_label_from_overall, A, M, level_number_list,level_number_list_from_overall,level_number_list_to_overall, bottom_num=len(l_3),
                                                                time_step=ts, horizon=hor_value, units=[u, u_plus, u_plus2], dropout=0.0,
                                                                epoch=50, nlayers=n, lr=0.5*(1e-04), theta=theta, patience=3, cor_tag=cor_tag,
                                                                name=model_name_in, MAE_tag=MAE_tag, nstack=nstack)

                                            # pred_new = (np.multiply(np.array(pred).squeeze(), scale[1].reshape(1,-1).repeat(7,axis=0)) + scale[0].reshape(1,-1).repeat(7,axis=0)) #.astype(int)
                                            pred_new = np.array(pred).squeeze()
                                            true_new = df.iloc[-7:, :].values

                                            pred_new2 = build_evaluation_data(df_new, pred_new, 7)
                                            true_new2 = build_evaluation_data(df_new, true_new, 7)

                                            pred_dec_all = None
                                            for dec_num in range(pred_dec.shape[1]):
                                                pred_dec_new = build_evaluation_data(df_new, np.array(
                                                    pred_dec[:, dec_num].cpu().numpy()).T, 7).rename(
                                                    columns={'y': f'y_{dec_num + 1}'})
                                                if pred_dec_all is None:
                                                    pred_dec_all = pred_dec_new
                                                else:
                                                    pred_dec_all = pd.concat(
                                                        [pred_dec_all, pred_dec_new[[f'y_{dec_num + 1}']]], axis=1)
                                            pred_dec_all.to_csv(
                                                f'./result1/time_step_units_nlayers_cor_theta_{ts}_{u}_{u_plus}_{u_plus2}_{n}_{cor_tag}_{theta}_{model_name_in}_pred_dec.csv')
                                            # [pred_dec_1, pred_dec_2, pred_dec_3] = pred_dec
                                            # pred_dec_1 = build_evaluation_data(df_new, np.array(pred_dec_1).squeeze(), 7)
                                            # pred_dec_2 = build_evaluation_data(df_new, np.array(pred_dec_2).squeeze(), 7)
                                            # pred_dec_3 = build_evaluation_data(df_new, np.array(pred_dec_3).squeeze(), 7)
                                            # pred_dec_1.to_csv(
                                            #     './result/time_step_units_nlayers_cor_theta_' + str(ts) + '_' + str(u) + '_' + str(
                                            #         n) + '_' + str(cor_tag) + '_' + str(
                                            #         theta) + '_' + model_name_in + '_pred_dec_1.csv')
                                            # pred_dec_2.to_csv(
                                            #     './result/time_step_units_nlayers_cor_theta_' + str(ts) + '_' + str(u) + '_' + str(
                                            #         n) + '_' + str(cor_tag) + '_' + str(
                                            #         theta) + '_' + model_name_in + '_pred_dec_2.csv')
                                            # pred_dec_3.to_csv(
                                            #     './result/time_step_units_nlayers_cor_theta_' + str(ts) + '_' + str(u) + '_' + str(
                                            #         n) + '_' + str(cor_tag) + '_' + str(
                                            #         theta) + '_' + model_name_in + '_pred_dec_3.csv')
                                            pred_new2.to_csv(f'./result1/time_step_units_nlayers_cor_theta_{ts}_{u}_{u_plus}_{u_plus2}_{n}_{cor_tag}_{theta}_{model_name_in}_pred.csv')
                                            end_time = time.time()
                                            elapsed_time = end_time - start_time
                                            print('start_time:', time.ctime(start_time))
                                            print('end_time:', time.ctime(end_time))
                                            print('elapsed_time:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
                                            print(' ')
                                            print(' ')

elif dataset_name == 'Wiki':
    from Dataset_Wiki import *
    for ts in ts_list:  #[28]
        for u in u_list: # 6,12
            for u_plus in u_plus_list:
                for u_plus2 in u_plus2_list:
                    for n_ori in n_list:  # 64,128
                        for cor_tag in [1]:
                            for theta in theta_list:  # , 0.1, 0.9 [1e3]
                                for nstack in nstack_list:
                                    for noi_level in noi_level_list:
                                        for noi_std in noi_std_list:
                                            if nstack != 3 and theta != 0: continue
                                            if nstack == 3 and theta == 0: continue
                                            # nstack_new = 2 if nstack == 3 else nstack
                                            # nstack_new = nstack
                                            # n = int(n_ori / int(nstack_new))
                                            n = n_ori
                                            model_name_in = f'{model_name_ori}_lv_{noi_level}_std_{noi_std}_nstack{nstack}_hor{hor_value}'
                                            print(f'model_{ts}_{u}_{n}_{cor_tag}_{theta}_{model_name_in}')
                                            # if os.path.exists(
                                            #     './result/time_step_units_nlayers_cor_theta_' + str(ts) + '_' + str(u) + '_' + str(
                                            #         n) + '_' + str(cor_tag) + '_' + str(theta) + '_' + model_name_in + '_pred.csv'):
                                            #     continue

                                            start_time = time.time()
                                            print('start_time:', time.ctime(start_time))
                                            pred, true, pred_dec = result(df_new, DP_label, DP_label_from_overall, A, M, level_number_list,level_number_list_from_overall,level_number_list_to_overall, bottom_num=len(l_4),
                                                                time_step=ts, horizon=hor_value, units=[u, u_plus, u_plus2], dropout=0.0,
                                                                epoch=50, nlayers=n, lr=5*(1e-04), theta=theta, patience=3, cor_tag=cor_tag,
                                                                name=model_name_in, MAE_tag=MAE_tag, nstack=nstack)

                                            # pred_new = (np.multiply(np.array(pred).squeeze(), scale[1].reshape(1,-1).repeat(7,axis=0)) + scale[0].reshape(1,-1).repeat(7,axis=0)) #.astype(int)
                                            pred_new = np.array(pred).squeeze()
                                            true_new = df.iloc[-7:, :].values

                                            pred_new2 = build_evaluation_data(df_new, pred_new, 7)
                                            true_new2 = build_evaluation_data(df_new, true_new, 7)

                                            pred_dec_all = None
                                            for dec_num in range(pred_dec.shape[1]):
                                                pred_dec_new = build_evaluation_data(df_new, np.array(
                                                    pred_dec[:, dec_num].cpu().numpy()).T, 7).rename(
                                                    columns={'y': f'y_{dec_num + 1}'})
                                                if pred_dec_all is None:
                                                    pred_dec_all = pred_dec_new
                                                else:
                                                    pred_dec_all = pd.concat(
                                                        [pred_dec_all, pred_dec_new[[f'y_{dec_num + 1}']]], axis=1)
                                            pred_dec_all.to_csv(
                                                f'./result1/time_step_units_nlayers_cor_theta_{ts}_{u}_{u_plus}_{u_plus2}_{n}_{cor_tag}_{theta}_{model_name_in}_pred_dec.csv')
                                            # [pred_dec_1, pred_dec_2, pred_dec_3] = pred_dec
                                            # pred_dec_1 = build_evaluation_data(df_new, np.array(pred_dec_1).squeeze(), 7)
                                            # pred_dec_2 = build_evaluation_data(df_new, np.array(pred_dec_2).squeeze(), 7)
                                            # pred_dec_3 = build_evaluation_data(df_new, np.array(pred_dec_3).squeeze(), 7)
                                            # pred_dec_1.to_csv(
                                            #     './result/time_step_units_nlayers_cor_theta_' + str(ts) + '_' + str(u) + '_' + str(
                                            #         n) + '_' + str(cor_tag) + '_' + str(
                                            #         theta) + '_' + model_name_in + '_pred_dec_1.csv')
                                            # pred_dec_2.to_csv(
                                            #     './result/time_step_units_nlayers_cor_theta_' + str(ts) + '_' + str(u) + '_' + str(
                                            #         n) + '_' + str(cor_tag) + '_' + str(
                                            #         theta) + '_' + model_name_in + '_pred_dec_2.csv')
                                            # pred_dec_3.to_csv(
                                            #     './result/time_step_units_nlayers_cor_theta_' + str(ts) + '_' + str(u) + '_' + str(
                                            #         n) + '_' + str(cor_tag) + '_' + str(
                                            #         theta) + '_' + model_name_in + '_pred_dec_3.csv')
                                            pred_new2.to_csv(f'./result1/time_step_units_nlayers_cor_theta_{ts}_{u}_{u_plus}_{u_plus2}_{n}_{cor_tag}_{theta}_{model_name_in}_pred.csv')
                                            end_time = time.time()
                                            elapsed_time = end_time - start_time
                                            print('start_time:', time.ctime(start_time))
                                            print('end_time:', time.ctime(end_time))
                                            print('elapsed_time:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
                                            print(' ')
                                            print(' ')

elif dataset_name == 'Wi4ki':
    from Dataset_Wi4ki import *
    for ts in ts_list:  #[28]
        for u in u_list: # 6,12
            for u_plus in u_plus_list:
                for u_plus2 in u_plus2_list:
                    for n_ori in n_list:  # 64,128
                        for cor_tag in [1]:
                            for theta in theta_list:  # , 0.1, 0.9 [1e3]
                                for nstack in nstack_list:
                                    for noi_level in noi_level_list:
                                        for noi_std in noi_std_list:
                                            if nstack != 3 and theta != 0: continue
                                            if nstack == 3 and theta == 0: continue
                                            # nstack_new = 2 if nstack == 3 else nstack
                                            # nstack_new = nstack
                                            # n = int(n_ori / int(nstack_new))
                                            n = n_ori
                                            model_name_in = f'{model_name_ori}_lv_{noi_level}_std_{noi_std}_nstack{nstack}_hor{hor_value}'
                                            print(f'model_{ts}_{u}_{n}_{cor_tag}_{theta}_{model_name_in}')
                                            # if os.path.exists(
                                            #     './result/time_step_units_nlayers_cor_theta_' + str(ts) + '_' + str(u) + '_' + str(
                                            #         n) + '_' + str(cor_tag) + '_' + str(theta) + '_' + model_name_in + '_pred.csv'):
                                            #     continue

                                            start_time = time.time()
                                            print('start_time:', time.ctime(start_time))
                                            pred, true, pred_dec = result(df_new, DP_label, DP_label_from_overall, A, M, level_number_list,level_number_list_from_overall,level_number_list_to_overall, bottom_num=len(l_3),
                                                                time_step=ts, horizon=hor_value, units=[u, u_plus, u_plus2], dropout=0.0,
                                                                epoch=50, nlayers=n, lr=1e-04, theta=theta, patience=3, cor_tag=cor_tag,
                                                                name=model_name_in, MAE_tag=MAE_tag, nstack=nstack)

                                            # pred_new = (np.multiply(np.array(pred).squeeze(), scale[1].reshape(1,-1).repeat(7,axis=0)) + scale[0].reshape(1,-1).repeat(7,axis=0)) #.astype(int)
                                            pred_new = np.array(pred).squeeze()
                                            true_new = df.iloc[-7:, :].values

                                            pred_new2 = build_evaluation_data(df_new, pred_new, 7)
                                            true_new2 = build_evaluation_data(df_new, true_new, 7)

                                            pred_dec_all = None
                                            for dec_num in range(pred_dec.shape[1]):
                                                pred_dec_new = build_evaluation_data(df_new, np.array(
                                                    pred_dec[:, dec_num].cpu().numpy()).T, 7).rename(
                                                    columns={'y': f'y_{dec_num + 1}'})
                                                if pred_dec_all is None:
                                                    pred_dec_all = pred_dec_new
                                                else:
                                                    pred_dec_all = pd.concat(
                                                        [pred_dec_all, pred_dec_new[[f'y_{dec_num + 1}']]], axis=1)
                                            pred_dec_all.to_csv(
                                                f'./result1/time_step_units_nlayers_cor_theta_{ts}_{u}_{u_plus}_{u_plus2}_{n}_{cor_tag}_{theta}_{model_name_in}_pred_dec.csv')
                                            # [pred_dec_1, pred_dec_2, pred_dec_3] = pred_dec
                                            # pred_dec_1 = build_evaluation_data(df_new, np.array(pred_dec_1).squeeze(), 7)
                                            # pred_dec_2 = build_evaluation_data(df_new, np.array(pred_dec_2).squeeze(), 7)
                                            # pred_dec_3 = build_evaluation_data(df_new, np.array(pred_dec_3).squeeze(), 7)
                                            # pred_dec_1.to_csv(
                                            #     './result/time_step_units_nlayers_cor_theta_' + str(ts) + '_' + str(u) + '_' + str(
                                            #         n) + '_' + str(cor_tag) + '_' + str(
                                            #         theta) + '_' + model_name_in + '_pred_dec_1.csv')
                                            # pred_dec_2.to_csv(
                                            #     './result/time_step_units_nlayers_cor_theta_' + str(ts) + '_' + str(u) + '_' + str(
                                            #         n) + '_' + str(cor_tag) + '_' + str(
                                            #         theta) + '_' + model_name_in + '_pred_dec_2.csv')
                                            # pred_dec_3.to_csv(
                                            #     './result/time_step_units_nlayers_cor_theta_' + str(ts) + '_' + str(u) + '_' + str(
                                            #         n) + '_' + str(cor_tag) + '_' + str(
                                            #         theta) + '_' + model_name_in + '_pred_dec_3.csv')
                                            pred_new2.to_csv(f'./result1/time_step_units_nlayers_cor_theta_{ts}_{u}_{u_plus}_{u_plus2}_{n}_{cor_tag}_{theta}_{model_name_in}_pred.csv')
                                            end_time = time.time()
                                            elapsed_time = end_time - start_time
                                            print('start_time:', time.ctime(start_time))
                                            print('end_time:', time.ctime(end_time))
                                            print('elapsed_time:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
                                            print(' ')
                                            print(' ')