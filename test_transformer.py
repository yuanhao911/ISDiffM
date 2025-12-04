import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import time
# import matplotlib.pyplot as plt
from utils.tools import *
from scipy.io import savemat
import random
import scipy.io as scio

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
np.random.seed(42)
torch.manual_seed(42)

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU训练需要设置这个
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
setup_seed(42)

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

def test_all_batch():
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    dataset_all = scio.loadmat('./dataset/dataset_1.mat')
    X_all, Y1_all = dataset_all['x_all'], dataset_all['y1_all']
    num = int(len(X_all) * 0.8)
    split82_index = dataset_all['final_split82_index'][0]
    train_index_82 = split82_index[:num]
    test_index_82 = split82_index[num:]
    X = torch.from_numpy(X_all[train_index_82]).squeeze(0).permute(0, 2, 1)[:, :, :9]
    Y1 = torch.from_numpy(Y1_all[train_index_82]).squeeze(0)

    X_normalized, X_mean, X_std = normalization(X)
    Y_normalized1, Y_mean1, Y_std1 = normalization(Y1)

    X_test = torch.from_numpy(X_all[test_index_82]).permute(0, 2, 1)[:, :, :9]
    Y1_test = torch.from_numpy(Y1_all[test_index_82])
    simple_interval = 10
    X_test = simple(X_test, simple_interval)

    X_test_normalization = to_normalization(X_test, X_mean, X_std)
    Y_test_normalization = to_normalization(Y1_test, Y_mean1, Y_std1)
    Y_empty = torch.zeros_like(Y_test_normalization)

    model = torch.load(
        'diffusion_based_net_best_epoch_XXXX.pt',
        map_location=device)

    from thop import profile

    net = model
    flops, params = profile(net, (
    X_test_normalization[0].unsqueeze(0).to(device), Y_empty[0].unsqueeze(0).to(device)))
    print('flops: ', flops, 'params: ', params)
    tb = time.time()
    # predict, enc_self_attns, dec_self_attns, dec_enc_attns = model(X_test_normalization[0].unsqueeze(0).to(device), Y_empty[0].unsqueeze(0).to(device))

    predict, enc_self_attns, dec_self_attns, dec_enc_attns = model(X_test_normalization.to(device), Y_empty.to(device))
    time_cost_one = time.time() - tb
    print('test one instance time cost of 100 f is {0} s'.format(time_cost_one))
    loss = 0
    criterion = nn.MSELoss()
    for j in range(0, len(predict)):
        loss += criterion(predict[j], Y_test_normalization[j].to(device))
    predict_Y = denormalization(predict, Y_mean1.to(device), Y_std1.to(device)).detach().cpu().numpy()
    Y_test_array = Y1_test.detach().numpy()

    file_name = './output/transformer/transformer_pred_gt_100_only_y1.mat'
    savemat(file_name, {'gt': np.array(Y_test_array), 'pred': np.array(predict_Y)})

    # rsme
    error = predict_Y - Y_test_array
    rsme_pre_point = np.sqrt(np.mean(np.mean(error ** 2, axis=2), axis=0))
    # rsme_pre_data = np.sqrt(np.mean(np.mean(error ** 2, axis=2), axis=1)) # ori_rsme
    rsme_pre_data = np.sqrt(np.mean(np.sum(error ** 2, axis=2), axis=1))
    rsme_pre_xy = np.sqrt(np.mean(np.mean(error ** 2, axis=1), axis=0))
    rsme_pre_point_xy = np.sqrt(np.mean(error ** 2, axis=0))
    rsme = np.mean(rsme_pre_data)
    # rsme = np.mean(rsme_pre_data[0:30])/30

    rsme = np.median(rsme_pre_data)
    rmse_mean = np.mean(rsme_pre_data)
    print('rsme_median=', rsme)
    print('rsme_mean=', rmse_mean)


time_begin = time.time()
test_all_batch()
time_cost = time.time()-time_begin
print('testing time cost of 180 f is {0} s'.format(time_cost))