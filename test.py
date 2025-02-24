import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from utils.tools import *
from scipy.io import savemat
import math
import scipy.io as scio
import random

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def add_noise(signal, snr_db):
    # 计算信号功率
    a = np.abs(signal) ** 2
    signal_power = np.mean(np.abs(signal) ** 2, axis=1)
    # signal_power = np.mean(signal, axis=-1)

    # 计算噪声功率
    snr = 10 ** (snr_db / 10)  # 将信噪比（单位dB）转换为线性信噪比
    noise_power = signal_power / snr

    # 生成高斯噪声
    a = np.expand_dims(noise_power, axis=1)
    noise = np.sqrt(np.expand_dims(noise_power, axis=1)) * np.random.randn(*signal.shape)

    # 添加噪声后的信号
    noisy_signal = signal + noise
    return noisy_signal, noise


def check_snr(signal, noise):
    """
    :param signal: 原始信号
    :param noise: 生成的高斯噪声
    :return: 返回两者的信噪比
    """
    signal_power = (1 / signal.shape[0]) * np.sum(np.power(signal, 2))  # 0.5722037
    noise_power = (1 / noise.shape[0]) * np.sum(np.power(noise, 2))  # 0.90688
    SNR = 10 * np.log10(signal_power / noise_power)
    return SNR


# forward diffusion
def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod.to(device), t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod.to(device), t, x_start.shape)

    return sqrt_alphas_cumprod_t.to(device) * x_start + sqrt_one_minus_alphas_cumprod_t.to(device) * noise

def extract(a, t, x_shape):
    """extract the appropriate  t  index for a batch of indices"""
    batch_size = t.shape[0]
    out = a.to(device).gather(-1, t.to(torch.int64))
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

# build diffusion
timesteps = 1000
sampling_timesteps = 1

betas = cosine_beta_schedule(timesteps)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
timesteps, = betas.shape
num_timesteps = int(timesteps)

sampling_timesteps0 = default(sampling_timesteps, timesteps)

assert sampling_timesteps0 <= timesteps
is_ddim_sampling = sampling_timesteps0 < timesteps
ddim_sampling_eta = 1.
self_condition = False
scale = 0.1
box_renewal = True
use_ensemble = True

sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
log_one_minus_alphas_cumprod = torch.log(1. - alphas_cumprod)
sqrt_recip_alphas_cumprod = torch.sqrt(1. / alphas_cumprod)
sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / alphas_cumprod - 1)

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
    X_test = simple(X_test)

    # mask = 30
    # noisy_signal, noise = add_noise(X_test.numpy(), snr_db=mask)
    # SNR = check_snr(X_test.numpy(), noise)
    # print(SNR)  # 输出 -1.999999999999991 dB,由于计算机数值计算的问题，存在极小的误差。
    # X_test = torch.from_numpy(noisy_signal)

    X_test_normalization = to_normalization(X_test, X_mean, X_std)
    Y_test_normalization = to_normalization(Y1_test, Y_mean1, Y_std1)
    Y_empty = torch.zeros_like(Y_test_normalization)

    # mask = 8
    # X_test_normalization[:, :, mask] = X_test_normalization[:, :, mask] * 0

    model = torch.load(
        './models/diffusion_based_net_best_dataset_1.pt',
        map_location=device)  # 100s

    from thop import profile

    net = model  # 定义好的网络模型
    inputs = torch.randn(1, 3, 112, 112)
    time_cond = torch.full((1,), 1, device=device, dtype=torch.long)
    flops, params = profile(net, (X_test_normalization[0].unsqueeze(0).to(device), Y_empty[0].unsqueeze(0).to(device), time_cond))
    print('flops: ', flops, 'params: ', params)

    # ddim
    def predict_noise_from_start(x_t, t, x0):
        return (
                (extract(sqrt_recip_alphas_cumprod.to(device), t, x_t.shape) * x_t - x0) /
                extract(sqrt_recipm1_alphas_cumprod.to(device), t, x_t.shape)
        )

    total_timesteps, sampling_timesteps, eta = num_timesteps, sampling_timesteps0, ddim_sampling_eta

    # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
    times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
    times = list(reversed(times.int().tolist()))
    time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

    img = torch.randn((len(Y_empty), 51, 2), device=device)
    batch = len(Y_empty)

    ensemble_score, ensemble_label, ensemble_coord = [], [], []
    x_start = None
    for time0, time_next in time_pairs:
        time_cond = torch.full((batch,), time0, device=device, dtype=torch.long)
        # img = torch.clamp(img, min=-1 * scale, max=scale)
        # img = ((img / scale) + 1) / 2
        time_b = time.time()
        # print(time_b)
        predict_1, enc_self_attns, dec_self_attns, dec_enc_attns = model(X_test_normalization.to(device),
                                                                       img.to(device), time_cond)
        time_cost = time.time() - time_b
        # print(time.time())
        print('testing time cost of is {0} s'.format(time_cost))
        # predict = (predict * 2 - 1.) * scale
        # predict = torch.clamp(predict, min=-1 * scale, max=scale)
        pred_noise = predict_noise_from_start(img, time_cond.to(torch.float32), predict_1)
        alpha = alphas_cumprod[time0]
        alpha_next = alphas_cumprod[time_next]

        sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
        c = (1 - alpha_next - sigma ** 2).sqrt()

        noise = torch.randn_like(img)

        img = pred_noise * alpha_next.sqrt() + \
              c * pred_noise + \
              sigma * noise

    loss = 0
    criterion = nn.MSELoss()
    for j in range(0, len(predict_1)):
        loss += criterion(predict_1[j], Y_test_normalization[j].to(device))
    predict_Y1 = denormalization(predict_1, Y_mean1.to(device), Y_std1.to(device)).detach().cpu().numpy()

    Y_test_array1 = Y1_test.detach().numpy()
    length_signal = X.shape[1]
    root_path = './output/'
    file_name = root_path + 'pred_gt_S' + str(sampling_timesteps) + '_'+ str(length_signal) + '.mat'
    savemat(file_name, {'gt1': np.array(Y_test_array1), 'pred1': np.array(predict_Y1)})

    # rsme
    error = predict_Y1 - Y_test_array1

    rsme_pre_data = np.sqrt(np.mean(np.sum(error ** 2, axis=2), axis=1))
    rsme = np.median(rsme_pre_data)
    rmse_mean = np.mean(rsme_pre_data)
    print('rsme_median=', rsme)
    print('rsme_mean=', rmse_mean)

    need_pic = False
    i = 0
    row = 3
    col = 5
    fig_num = 0
    while need_pic and i < predict.shape[0]:
        plt.figure(fig_num)
        fig, axs = plt.subplots(row, col)
        fig.suptitle('impedance spectrum compare')
        for j in range(row):
            for k in range(col):
                if i >= predict.shape[0]:
                    continue
                axs[j, k].scatter(Y_test_array[i, :, 0], Y_test_array[i, :, 1], color='red')
                axs[j, k].scatter(predict_Y[i, :, 0], predict_Y[i, :, 1], color='blue')
                axs[j, k].legend(labels=['truth', 'predict'])
                axs[j, k].set_xlabel('real part')
                axs[j, k].set_ylabel('imaginary part')
                i = i + 1
        fig_num = fig_num + 1
    plt.show()

time_begin = time.time()
test_all_batch()
time_cost = time.time()-time_begin
