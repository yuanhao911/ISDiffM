import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import time
import math
import matplotlib.pyplot as plt
from utils.tools import *
from model_transformer import Transformer
import random
import scipy.io as scio

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class my_dataset(Dataset):
    def __init__(self, enc_input, dec_input, dec_output):
        super().__init__()
        self.enc_input = enc_input
        self.dec_input = dec_input
        self.dec_output = dec_output

    def __getitem__(self, index):
        return self.enc_input[index], self.dec_input[index], self.dec_output[index]

    def __len__(self):
        return self.enc_input.size(0)


def add_noise(signal, snr_db):
    signal_power = np.mean(np.abs(signal) ** 2, axis=1)
    snr = 10 ** (snr_db / 10)
    noise_power = signal_power / snr
    noise = np.sqrt(np.expand_dims(noise_power, axis=1)) * np.random.randn(*signal.shape)
    noisy_signal = signal + noise
    return noisy_signal, noise


def check_snr(signal, noise):
    signal_power = (1 / signal.shape[0]) * np.sum(np.power(signal, 2))  # 0.5722037
    noise_power = (1 / noise.shape[0]) * np.sum(np.power(noise, 2))  # 0.90688
    SNR = 10 * np.log10(signal_power / noise_power)
    return SNR


def train(mask):
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    begin = time.time()
    dataset_all = scio.loadmat('./dataset/dataset_1.mat')
    X_all, Y1_all = dataset_all['x_all'], dataset_all['y1_all']
    num = int(len(X_all) * 0.8)
    split82_index = dataset_all['final_split82_index'][0]  # ori index
    train_index_82 = split82_index[:num]
    X = torch.from_numpy(X_all[train_index_82]).squeeze(0).permute(0, 2, 1)[:, :, :9]
    Y1 = torch.from_numpy(Y1_all[train_index_82]).squeeze(0)
    simple_interval = 10
    X = simple(X, simple_interval)
    root_path = './output/'
    file_path = root_path + "transformer" + '/'
    # create work_dir
    if not os.path.exists(file_path):
        os.mkdir(file_path)

    X_normalized, X_mean, X_std = normalization(X)
    Y_normalized, Y_mean, Y_std = normalization(Y1)
    X_normalized = X_normalized.to(device)
    Y_normalized = Y_normalized.to(device)
    X_mean, X_std, Y_mean, Y_std = X_mean.to(device), X_std.to(device), Y_mean.to(device), Y_std.to(device)

    # X_normalized[:, :, mask] = X_normalized[:, :, mask] * 0
    # X_normalized = torch.cat((X_normalized[:, :, :mask], X_normalized[:, :, mask + 1:]), dim=-1) # valid
    # X_enc_input, Y_dec_input, Y_dec_output = make_data(X_normalized, Y_normalized, Y_mean, Y_std)
    Y_empty = torch.zeros_like(Y_normalized).to(device)
    epoch = 2000
    # X_normalized[:, :, mask] = X_normalized[:, :, 0] * 0 # MASK
    train_iter = DataLoader(my_dataset(X_normalized, Y_empty, Y_normalized), batch_size, shuffle=True)
    # train_iter = DataLoader(my_dataset(X.to(device), Y_empty, Y.to(device)), batch_size, shuffle=True)
    # net = Seq2seq(in_features, hidden_size, dropout_size, num_layers_size).to(device)
    model = Transformer().to(device)
    model = model.to(device)
    learning_rate = 0.01
    # optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    step = math.ceil(len(Y1) / batch_size)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1500*step, 1800*step], gamma=0.1)

    criterion = nn.MSELoss()
    loss = 0
    loss_all = []
    best_loss = 1e5
    loss_log = []
    for i in range(epoch):
        for enc_inputs, dec_inputs, dec_outputs in train_iter:
            LR = scheduler.get_lr()
            enc_inputs, dec_inputs, dec_outputs = enc_inputs, dec_inputs, dec_outputs
            outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
            # outputs: [batch_size * tgt_len, tgt_vocab_size]
            loss = 0
            for j in range(0, len(outputs)):
                # loss += criterion(outputs[j], dec_outputs[j])
                error = outputs[j] - dec_outputs[j]
                loss += torch.sqrt(torch.mean(torch.sum(error ** 2, axis=1), axis=0))
            loss = loss / len(outputs)
            loss_log.append(loss.cpu().detach().numpy())
            # print('Epoch:', '%04d' % (i + 1), 'loss =', '{:.6f}'.format(loss))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            # best_loss = loss/batch_size if best_loss > loss/batch_size else best_loss

        if best_loss > loss and i >= 1500:
            print('saving the {0} th best .pt model and the loss_1 is {1} '.format(i, loss))
            best_loss = loss

            # save_variable(loss_all, save_name + '_loss.txt')
            save_name = file_path + 'diffusion_based_net_best_epoch_' + str(i)
            torch.save(model, save_name + '.pt')
        if (i + 1) % 10 == 0:
            print("epoch {0} loss_51plus2: {1} loss_5plus1: {1} lr {2}".format(i, loss, loss, scheduler.get_lr()[0]))

        if (i + 1) % 100 == 0:
            save_name = file_path + 'diffusion_based_net_epoch_' + str(i)
            torch.save(model, save_name + '.pt')

    time_cost = time.time() - begin
    print('training time cost is {0} s'.format(time_cost))

if __name__ == '__main__':

    hidden_size_list = [128]  # 128 07 3
    dropout_size_list = [0.7]
    num_layers_size_list = [1]
    run_time_list = []
    mask_list = [0,1,2,3, 4, 5, 6, 7, 8]
    SNR_list = [10, 20, 30]

    for mask in mask_list:
        start_time = time.time()
        train(mask)
        end_time = time.time()
        run_time = end_time - start_time
        run_time_list.append(run_time)
    save_variable(run_time_list, 'run_time.txt')