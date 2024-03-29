import os
import pandas as pd
import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
import datetime
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available() : os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import warnings
import argparse
from eval_pkg import *
import time
warnings.filterwarnings('ignore')





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

def data_multiple_build(data, dp, lookback, horizon):
    X_new = []
    X_dp = []
    y_new = []
    z_new = []

    for indices in range(data.shape[0]):

        if indices + lookback + horizon > data.shape[0]:
            break

        X_tem = data[indices:indices + lookback, :]
        X_dp_tem = dp[indices:indices + lookback, :]
        y_tem = data[indices+lookback : indices + lookback + horizon, :]
        dp_tem = dp[indices + lookback : indices + lookback + horizon , :]

        X_new.append(X_tem)
        X_dp.append(X_dp_tem)
        y_new.append(y_tem)
        z_new.append(dp_tem)
    return X_new, X_dp, y_new, z_new


class tensor_multiple_data(Dataset):
    def __init__(self, x, x_dp, y, z):
        self.x = x
        self.x_dp = x_dp
        self.y = y
        self.z = z

    def __getitem__(self, item):
        x_i = torch.FloatTensor(self.x[item])
        x_dp_i = torch.FloatTensor(self.x_dp[item])
        y_i = torch.FloatTensor([self.y[item]])  # item？
        z_i = torch.FloatTensor([self.z[item]])

        return x_i, x_dp_i, y_i, z_i

    def __len__(self):
        return len(self.x)


def train(data, level_number_list, model, optimizer, loss_func,  epoch_txt, theta, MAE_tag):
    start_time = time.time()
    train_loss = 0
    train_loss_dpd = 0
    train_loss_me = 0
    train_pred = []
    flag = 0
    data_loader = DataLoader(data, batch_size=1, shuffle=False)
    tqdm_loop = tqdm(data_loader)
    tqdm_loop.set_description(epoch_txt)
    for x_b, X_dp, y_b, z_b in tqdm_loop:
        optimizer.zero_grad()
        x_b = x_b.squeeze() # [TS, C]
        X_dp = X_dp.squeeze() # [TS, C]
        X_cat = torch.cat([x_b.T.unsqueeze(-1), X_dp.T.unsqueeze(-1)],dim=2).to(device) # [C, TS, 2]
        y_b = y_b.squeeze() # [C,]
        z_b = z_b.squeeze() # [C,]

        y_pred, y_pred_p = model(X_cat, level_number_list)
        loss, loss_dpd, loss_me = loss_func(y_pred.cpu(), y_pred_p.cpu(), y_b, z_b, level_number_list, theta, MAE_tag)
        flag += 1

        train_loss += loss
        train_loss_dpd += loss_dpd
        train_loss_me += loss_me
        train_pred.append(y_pred.T.squeeze().cpu().detach().numpy().tolist())
        loss.backward()
        optimizer.step()
        running_train_acc = 1 - wmape(y_pred.cpu(), y_b)
        tqdm_loop.set_postfix(loss=loss.item(), acc=running_train_acc.item())

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(epoch_txt, 'loss = ', train_loss.item(), train_loss_dpd.item(), train_loss_me.item(),
          ',acc = ', running_train_acc.item(),
          ',elapsed_time = ',time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    return train_loss / flag, train_pred, train_loss_dpd / flag, train_loss_me / flag


def test(data, level_number_list, model, loss_func, theta, MAE_tag):
    test_loss = 0
    test_loss_dpd = 0
    test_loss_me = 0
    eval_v = 0
    test_pred = []
    test_true = []
    flag = 0
    data_loader = DataLoader(data, batch_size=1, shuffle=False)
    for x_b, X_dp, y_b, z_b in data_loader:
        x_b = x_b.squeeze() # [TS, C]
        X_dp = X_dp.squeeze()  # [TS, C]
        X_cat = torch.cat([x_b.T.unsqueeze(-1), X_dp.T.unsqueeze(-1)], dim=2).to(device)  # [C, TS, 2]
        y_b = y_b.squeeze()
        z_b = z_b.squeeze()
        test_true.append(y_b)

        with torch.no_grad():
            y_pred, y_pred_p = model(X_cat, level_number_list)
            loss, loss_dpd, loss_me = loss_func(y_pred.cpu(), y_pred_p.cpu(), y_b, z_b, level_number_list, theta, MAE_tag)
            #             tem_eval = eval_func(y_pred.cpu().numpy(), y_b.cpu().numpy())

            flag += 1

            test_loss += loss
            test_loss_dpd += loss_dpd
            test_loss_me += loss_me

            test_pred.append(y_pred.T.squeeze().cpu().numpy().tolist())

    return test_loss / flag, test_pred, test_true, test_loss_dpd / flag , test_loss_me / flag


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
        elif score <= self.best_score + self.delta:
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

print(device)

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

class Model(nn.Module):
    def __init__(self, input_size, time_step, A, M, units, dropout, nlayer, horizon, bottom_num, cor_tag, ratio_loc):
        super(Model, self).__init__()

        self.nlayer = nlayer
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

        self.acf = nn.Softplus()
        self.proj = nn.Linear(2, units[0], bias=False)

        self.lns1 = nn.ModuleList()
        self.lns2 = nn.ModuleList()
        # self.lns3 = nn.ModuleList()
        self.lns4 = nn.ModuleList()
        self.rnns = nn.ModuleList()
        for i in range(self.nlayer):
            self.rnns.append(nn.LSTM(self.units[0], units[1], dropout=self.dropout, bias=False, batch_first=True))
            self.lns1.append(nn.Linear(self.units[1], 1, bias=False))
            self.lns2.append(nn.Linear(self.units[1], 1, bias=False))
            # self.lns3.append(nn.Linear(self.units[1], 1, bias=False))
            self.lns4.append(nn.Linear(3, 1, bias=False))
        self.ar = nn.Linear(self.time_step, self.horizon, bias=False)
        self.ardpd = nn.Linear(self.time_step, self.horizon, bias=False)

    #         self.ln1 = nn.Linear(units, 1)
    #         self.ln2 = nn.Linear(units, 1)

    def normalization(self, x):
        x = x + 1e-9
        sum_x = x.sum(axis=-1)
        return x / sum_x.unsqueeze(-1)

    def forward(self, x, parents_number_list_, training=True):

        # TN_LN = top node + leaf nodes
        # LN_BN = leaf nodes + bottom nodes
        # MN = max nodes in all parents nodes
        # C = all nodes
        # parents = number of parents nodes

        TN_LN = x.shape[0]- np.array(parents_number_list_[-self.bottom_num:]).sum()

        # [bz, TS, C]
        #         print(x.size())

        if len(x.size()) == 2:
            # [TS,C] -> [C,TS,1]
            x = x.T.unsqueeze(-1)

        #         forecast = torch.zeros(x.size()[0], x.size()[1], x.size()[2]).to(device)

        self.DPD_overall = torch.zeros(x.size(0)-1,x.size(1)).to(device)

        for i in range(self.nlayer):
            # x_max = x.max(axis=1, keepdim=True).values
            # x_min = x.min(axis=1, keepdim=True).values
            # x_demon = x_max - x_min
            # x = torch.div(torch.sub(x, x_min), (x_demon + 1e-9))
            # scale = [x_min, x_demon]

            rnn_inputs = self.proj(x) # [C, TS, units]
            tem_inputs, _ = self.rnns[i](rnn_inputs) # [C, TS, units]

            # B1 forecast
            F = self.acf(self.lns1[i](tem_inputs)) # [C, TS, 1]
            B_DPD = self.acf(self.lns2[i](tem_inputs)).squeeze()  # [C, TS]
            # F = self.lns1[i](tem_inputs)  # [C, TS, 1]
            # B_DPD = self.lns2[i](tem_inputs).abs().squeeze()  # [C, TS]

            # F_self = self.lns3[i](tem_inputs) # [C, TS, 1]

            F_struc = self.build_DPD(F.squeeze()[1:, :].T, parents_number_list_) # [TS, parents, max_child_nodes]
            F_struc_sum = F_struc.sum(axis=-1).T.unsqueeze(-1) # [parents, TS, 1]
            F_BU = torch.cat([F_struc_sum, F[F_struc_sum.size()[0]: , : , :]],dim=0)
            DPD_struc_ = self.normalization(
                self.build_DPD(B_DPD[1:, :].T, parents_number_list_))  # [TS, parents, max_child_nodes]
            input_recon_ = F.repeat(1, 1, np.array(parents_number_list_).max())  # [C, TS, max_child_nodes]
            # obtain the TP_reconcilation forecast by the top nodes and leaf nodes, therefore is 4
            recon_forecast = torch.mul(input_recon_[:TN_LN, :, :].permute(1, 0, 2),
                                       DPD_struc_)  # [TS, parents, max_child_nodes]

            recon_forecast_reverse = self.reverse_DPD(recon_forecast, parents_number_list_).unsqueeze(-1)  # [C-1, TS, 1]
            F_TD = torch.cat([F[0:1 , : , :], recon_forecast_reverse], dim=0) # [C, TS, 1]

            DPD_reverse_ = self.reverse_DPD(DPD_struc_, parents_number_list_)  # [C-1, TS]

            # print(recon_forecast_reverse.size(),F_self[1:, :, :].size())
            x_ln4 = torch.cat([F_TD, F, F_BU], dim=2) # [C, TS, 3]
            x_add = self.lns4[i](x_ln4) # [C, TS, 1]
            # x_add = recon_forecast_reverse
            x_0 = torch.add(x[:,:,0:1], x_add).squeeze() # [C, TS]
            if i ==0:
                self.DPD_overall = DPD_reverse_
            else:
                self.DPD_overall = torch.cat([self.DPD_overall, DPD_reverse_], dim=1)

            # x = torch.add(torch.mul(x.unsqueeze(-1), scale[1]), scale[0]).squeeze()
            if i == self.nlayer -1:
                zeros = torch.zeros(*x_0.size()).to(device)
                x_0 = torch.where(x_0<zeros, zeros, x_0)

                x_0 = torch.matmul(x_0.T, self.M).T
                if self.cor_tag == 0:
                    while int((x_0<-1e-4).sum()) > 0:
                        x_0 = torch.where(x_0<zeros, zeros, x_0)
                        x_0 = torch.matmul(x_0.T, self.M).T
                    # print('aaaaa',torch.matmul(x.squeeze().T, self.A.T).abs().mean())

            x = torch.cat([x_0.unsqueeze(-1), x[:, :, 1:]], dim=2)
            # x = x


        # zeros = torch.zeros(*x.size()).to(device)
        # x = torch.where(x < zeros, zeros, x)
        if training is False:
            correhent_error = torch.matmul(x.squeeze().T, self.A.T).abs().mean()
            assert correhent_error <= 1e-2

        #             forecast += F

        return_x = self.ar(x[:,:,0]) # [C, TS]
        return_DPD = self.DPD_overall

        return return_x.squeeze(), return_DPD.squeeze() # DPD_reverse_[:, -self.horizon:].squeeze()

    def build_DPD(self, x, parents_number_list_, training=True):
        if training:
            DPD = torch.zeros([len(parents_number_list_), np.array(parents_number_list_).max(),
                               self.time_step]).to(device)
        else:
            DPD = torch.zeros([len(parents_number_list_), np.array(parents_number_list_).max(),
                               self.horizon]).to(device)

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
        reverse_DPD = torch.zeros([self.time_step, sum(parents_number_list_[:])]).to(device)
        stop = 0
        for i, j in enumerate(parents_number_list_):
            reverse_DPD[:, stop: stop + j] = x[:, i, :j]
            #             print(x[:,i,:j].sum(axis=-1))
            stop += j

        return reverse_DPD.permute(1, 0)

    def loss(self, B1, B2, label1, label2, parents_number_list_, theta, MAE_tag):
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
        if len(label2.size()) == 1:
            label2 = label2.unsqueeze(0)
        else:
            label2 = label2
        if len(B1.size()) == 1:
            B1_loss = B1.unsqueeze(0)
        else:
            B1_loss = B1.T
        if len(B2.size()) == 1:
            B2_loss = B2.unsqueeze(0)
        else:
            B2_loss = B2.T


        likelihood = 0
        # for k in range(B2_loss.shape[0]):
        stop = 1
        for i, j in enumerate(parents_number_list_):  #[:4]  [:-self.botton_num]
            if len(B2_loss.shape) == 2:
                if stop + j == sum(parents_number_list_) + 1:
                    tem_B2 = B2_loss[:, stop - 1:]
                    tem_label2 = label2[:, stop:]
                else:
                    tem_B2 = B2_loss[:, stop - 1: stop + j - 1]
                    tem_label2 = label2[:, stop: stop + j]
            else:
                print(error)
                if stop + j == sum(parents_number_list_) + 1:
                    tem_B2 = B2_loss[:, :, stop - 1:]
                    tem_label2 = label2[:, :, stop:]
                else:
                    tem_B2 = B2_loss[:, :, stop - 1: stop + j - 1]
                    tem_label2 = label2[:, :, stop: stop + j]
            tem_dis = torch.distributions.dirichlet.Dirichlet(j * self.normalization(tem_B2))
            tem_likelihood = tem_dis.log_prob(self.normalization(tem_label2))

            likelihood += tem_likelihood.sum()
            stop += j
        likelihood = likelihood / B2_loss.shape[0]

        HTS_stuc = get_HTS_strc(parents_number_list_)
        pred_loc = HTS_stuc.get_pred_loc()
        ratio_loc = self.ratio_loc #[0.21085779683092262,0.23239679903741464,0.26296517958407817,1.2533606976428853]
        stop = 0
        loss_sum = 0
        for i, j in zip(pred_loc,ratio_loc):
            tem_Bl = B1_loss[:, stop:i + stop]
            tem_label1 = label1[:, stop:i + stop]
            stop += i
            loss_sum += forecast_loss(tem_Bl, tem_label1) / j

        if len(B1_loss.size()) == 2:
            return -likelihood * theta + loss_sum, -likelihood * theta, loss_sum   #forecast_loss(B1_loss[:, :11], label1[:, :11])
        else:
            return -likelihood * theta + loss_sum, -likelihood * theta, loss_sum    #forecast_loss(B1_loss[:, :, :11], label1[:, :, :11])


import matplotlib.pyplot as plt


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.uniform(m.weight, 0, 1)

    elif isinstance(m, nn.LSTM):
        for para in m.parameters():
            nn.init.uniform(para, 0, 1)



def result(data, DP_label, A, M, level_number_list, bottom_num, time_step, horizon, units, dropout, epoch, nlayers, lr, theta, patience, cor_tag, name, MAE_tag):
    torch.random.manual_seed(2020)
    input_size = [data.shape[0], data.shape[1]]
    A = torch.from_numpy(A).float().to(device)
    M = torch.from_numpy(M).float().to(device)

    X_new, X_dp, y_new, z_new = data_multiple_build(data.values, DP_label, time_step, horizon)
    new_data = tensor_multiple_data(X_new, X_dp, y_new, z_new)
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

    model = Model(input_size[0], time_step, A, M, units, dropout, nlayers, horizon, bottom_num, cor_tag, ratio_loc).to(device)
    # model.apply(weight_init)


    flag = 0

    loss_func = model.loss
    optimizer = torch.optim.RMSprop(model.parameters(), lr)
    my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.1)
    early_stop = EarlyStopping(name, patience, verbose=False,
                               checkpoint='./result/time_step_units_nlayers_cor_theta_' + str(ts) + '_' + str(u) + '_' + str(n) + '_' + str(cor_tag) + '_' + str(theta) + '_' + str(name) + '_model_normal.pt')  # './' + str(name) + 'model_normal.pt'


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
        valid_loss, valid_pred, valid_true, valid_loss_dpd, valid_loss_me = test(valid_data, level_number_list, model, loss_func, theta, MAE_tag)
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

    test_loss, test_pred, test_true, test_loss_dpd, test_loss_me = test(test_data, level_number_list, model, loss_func, theta, MAE_tag)
    # model.load_state_dict(torch.load('./' + str(name) + 'model_normal.pt'))

    fig, ax1 = plt.subplots()  # subplots一定要带s
    ax1.plot(train_loss_list, label='train',c='r')
    # ax1.set_ylabel('train')
    # ax2 = ax1.twinx()  # twinx将ax1的X轴共用与ax2，这步很重要
    ax1.plot(valid_loss_list, label='valid',c='g')
    # ax2.set_ylabel('valid')
    plt.legend()
    plt.show()
    plt.savefig('./result/time_step_units_nlayers_cor_theta_'+ str(ts)+'_'+str(u)+'_'+str(n)+'_'+str(cor_tag)+'_'+str(theta)+'_'+str(name)+'_loss.jpg')
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
    plt.savefig('./result/time_step_units_nlayers_cor_theta_' + str(ts) + '_' + str(u) + '_' + str(n) + '_' + str(
        cor_tag) + '_' + str(theta) + '_' + str(name) + '_lossdpd.jpg')
    plt.close()

    return test_pred, test_true


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--ts", type=int, nargs = '+', required=True)
parser.add_argument("--u", type=int, nargs = '+', required=True)
parser.add_argument("--n", type=int, nargs = '+', required=True)
parser.add_argument("--theta", type=float, nargs = '+', required=True)

args, _ = parser.parse_known_args()
dataset_name = args.dataset
ts_list = args.ts
u_list = args.u
n_list = args.n
theta_list = args.theta
# dataset_name = 'Traffic'
# ts_list = [14]
# u_list = [8]
# n_list = [20]
# theta_list = [0.0]

MAE_tag = ''
MAE_tag = '_MAE'
model_name_in = dataset_name + '_' + 'TDMBULL' + MAE_tag

if dataset_name == 'M5':
    from Dataset_M5 import *
    for ts in ts_list:  # 14,28
        for u in u_list:   # 16,64
            for n in n_list:  # 20,50 [2,20]
                for cor_tag in [1]:
                    for theta in theta_list:  # 0.0 ,6e2, 6
                        print('model_'+str(ts)+'_'+str(u)+'_'+str(n)+'_'+str(cor_tag)+'_'+str(theta))
                        start_time = time.time()
                        print('start_time:', time.ctime(start_time))
                        pred, true = result(df_new, DP_label, A, M, level_number_list,
                                            bottom_num=len(item_level_number_list),
                                            time_step=ts, horizon=1, units=[u, u], dropout=0.0,
                                            epoch=500, nlayers=n, lr=1e-03, theta=theta, patience=3, cor_tag=cor_tag,
                                            name=model_name_in, MAE_tag=MAE_tag)
    #patience 10/20/50
    #lr最初1e-04，后来改成1e-03
    #MAE MSE

                        # pred_new = (np.multiply(np.array(pred).squeeze(), scale[1].reshape(1,-1).repeat(7,axis=0)) + scale[0].reshape(1,-1).repeat(7,axis=0)) #.astype(int)
                        pred_new = np.array(pred).squeeze()
                        true_new = df.iloc[-7:,:].values

                        pred_new2 = build_evaluation_data(df_new, pred_new, 7)
                        true_new2 = build_evaluation_data(df_new, true_new, 7)

                        pred_new2.to_csv('./result/time_step_units_nlayers_cor_theta_'+ str(ts)+'_'+str(u)+'_'+str(n)+'_'+str(cor_tag)+'_'+str(theta)+'_'+model_name_in+'_pred.csv')
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
            for n in n_list:  #10,64,128
                for cor_tag in [1]:
                    for theta in theta_list:  # , 0.1, 0.9 [6e2,1e2]
                        print('model_' + str(ts) + '_' + str(u) + '_' + str(n) + '_' + str(cor_tag) + '_' + str(theta))
                        start_time = time.time()
                        print('start_time:', time.ctime(start_time))
                        pred, true = result(df_new, DP_label, A, M, level_number_list, bottom_num=len(l_3),
                                            time_step=ts, horizon=1, units=[u, u], dropout=0.0,
                                            epoch=500, nlayers=n, lr=1e-04, theta=theta, patience=3, cor_tag=cor_tag,
                                            name=model_name_in, MAE_tag=MAE_tag)

                        # pred_new = (np.multiply(np.array(pred).squeeze(), scale[1].reshape(1,-1).repeat(7,axis=0)) + scale[0].reshape(1,-1).repeat(7,axis=0)) #.astype(int)
                        pred_new = np.array(pred).squeeze()
                        true_new = df.iloc[-7:, :].values

                        pred_new2 = build_evaluation_data(df_new, pred_new, 7)
                        true_new2 = build_evaluation_data(df_new, true_new, 7)

                        pred_new2.to_csv(
                            './result/time_step_units_nlayers_cor_theta_' + str(ts) + '_' + str(u) + '_' + str(
                                n) + '_' + str(cor_tag) + '_' + str(theta) + '_' + model_name_in + '_pred.csv')
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
            for n in n_list:  # 10,32
                for cor_tag in [1]:
                    for theta in theta_list:  # , 0.1, 0.9 [6,1]
                        print('model_' + str(ts) + '_' + str(u) + '_' + str(n) + '_' + str(cor_tag) + '_' + str(theta))
                        start_time = time.time()
                        print('start_time:', time.ctime(start_time))
                        pred, true = result(df_new, DP_label, A, M, level_number_list, bottom_num=len(l_3),
                                            time_step=ts, horizon=1, units=[u, u], dropout=0.0,
                                            epoch=500, nlayers=n, lr=1e-04, theta=theta, patience=3, cor_tag=cor_tag,
                                            name=model_name_in, MAE_tag=MAE_tag)

                        # pred_new = (np.multiply(np.array(pred).squeeze(), scale[1].reshape(1,-1).repeat(7,axis=0)) + scale[0].reshape(1,-1).repeat(7,axis=0)) #.astype(int)
                        pred_new = np.array(pred).squeeze()
                        true_new = df.iloc[-7:, :].values

                        pred_new2 = build_evaluation_data(df_new, pred_new, 7)
                        true_new2 = build_evaluation_data(df_new, true_new, 7)

                        pred_new2.to_csv(
                            './result/time_step_units_nlayers_cor_theta_' + str(ts) + '_' + str(u) + '_' + str(
                                n) + '_' + str(cor_tag) + '_' + str(theta) + '_' + model_name_in + '_pred.csv')
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
            for n in n_list:  # 64,128
                for cor_tag in [1]:
                    for theta in theta_list:  # , 0.1, 0.9 [1e3]
                        print('model_' + str(ts) + '_' + str(u) + '_' + str(n) + '_' + str(cor_tag) + '_' + str(theta))
                        start_time = time.time()
                        print('start_time:', time.ctime(start_time))
                        pred, true = result(df_new, DP_label, A, M, level_number_list, bottom_num=len(l_4),
                                            time_step=ts, horizon=1, units=[u, u], dropout=0.0,
                                            epoch=500, nlayers=n, lr=1e-04, theta=theta, patience=3, cor_tag=cor_tag,
                                            name=model_name_in, MAE_tag=MAE_tag)

                        # pred_new = (np.multiply(np.array(pred).squeeze(), scale[1].reshape(1,-1).repeat(7,axis=0)) + scale[0].reshape(1,-1).repeat(7,axis=0)) #.astype(int)
                        pred_new = np.array(pred).squeeze()
                        true_new = df.iloc[-7:, :].values

                        pred_new2 = build_evaluation_data(df_new, pred_new, 7)
                        true_new2 = build_evaluation_data(df_new, true_new, 7)

                        pred_new2.to_csv(
                            './result/time_step_units_nlayers_cor_theta_' + str(ts) + '_' + str(u) + '_' + str(
                                n) + '_' + str(cor_tag) + '_' + str(theta) + '_' + model_name_in + '_pred.csv')
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        print('start_time:', time.ctime(start_time))
                        print('end_time:', time.ctime(end_time))
                        print('elapsed_time:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
                        print(' ')
                        print(' ')

