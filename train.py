import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import time
import math
from utils.tools import *
from model_only1_supervize import only_y1_supervize
import random
from scipy.io import savemat
import scipy.io as scio

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

# 设立随机种子，方便复现
np.random.seed(42)
torch.manual_seed(42)

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

def setup_seed(seed):
    #  下面两个常规设置了，用来np和random的话要设置
    np.random.seed(seed)
    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)  # 禁止hash随机化
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # 在cuda 10.2及以上的版本中，需要设置以下环境变量来保证cuda的结果可复现

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU训练需要设置这个
    torch.manual_seed(seed)

    torch.use_deterministic_algorithms(True)  # 一些操作使用了原子操作，不是确定性算法，不能保证可复现，设置这个禁用原子操作，保证使用确定性算法
    torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.enabled = False  # 禁用cudnn使用非确定性算法
    torch.backends.cudnn.benchmark = False  # 与上面一条代码配套使用，True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现。
setup_seed(42)

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

def train():
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    begin = time.time()

    dataset_all = scio.loadmat('./dataset/dataset_1.mat')

    X_all, Y1_all = dataset_all['x_all'], dataset_all['y1_all']
    num = int(len(X_all) * 0.8)

    split82_index = dataset_all['final_split82_index'][0] # ori index
    train_index_82 = split82_index[:num]
    X = torch.from_numpy(X_all[train_index_82]).squeeze(0).permute(0, 2, 1)[:, :, :9]
    Y1 = torch.from_numpy(Y1_all[train_index_82]).squeeze(0)

    X = simple(X)

    # noisy_signal, noise = add_noise(X.numpy(), snr_db=mask)
    # SNR = check_snr(X.numpy(), noise)
    # print(SNR)  # 输出 -1.999999999999991 dB,由于计算机数值计算的问题，存在极小的误差。
    # X = torch.from_numpy(noisy_signal)


    root_path = './output/'
    file_path = root_path + "pre_diffusion" + '/'
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
    train_iter = DataLoader(my_dataset(X_normalized, Y_empty, Y_normalized), batch_size, shuffle=True)

    model = only_y1_supervize().to(device)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    step = math.ceil(len(Y1) / batch_size)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1500*step, 1800*step], gamma=0.1)

    best_loss = 1e5
    loss_log = []
    for i in range(epoch):
        for enc_inputs, dec_inputs, dec_outputs in train_iter:
            targets, x_boxes, noises, t = prepare_targets(dec_outputs)
            dec_inputs = x_boxes

            outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs, t)
            loss = 0
            loss_1 = 0
            for j in range(0, len(outputs)):
                # loss += criterion(outputs[j], dec_outputs[j])
                error = outputs[j] - dec_outputs[j]
                loss += torch.sqrt(torch.mean(torch.sum(error ** 2, axis=1), axis=0))*1
            loss = loss / len(outputs)
            loss_log.append(loss.cpu().detach().numpy())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        if best_loss > loss and i >= 1700:
            print('saving the {0} th best .pt model and the loss_1 is {1} '.format(i, loss))
            best_loss = loss

            # save_variable(loss_all, save_name + '_loss.txt')
            if (i+1) >= 1600:
                save_name = file_path + 'diffusion_based_net_best_epoch_' + str(i)
                torch.save(model, save_name + '.pt')

        if (i+1) % 10 == 0:
            print("epoch {0} loss_51plus2: {1} loss_5plus1: {1} lr {2}".format(i, loss, loss_1, scheduler.get_lr()[0]))


        if (i+1) % 100 == 0:
            save_name = file_path + 'diffusion_based_net_epoch_' + str(i)
            torch.save(model, save_name + '.pt')


    file_name = file_path + 'loss.mat'
    savemat(file_name, {'loss': np.array(loss_log)})
    time_cost = time.time() - begin
    print('training time cost is {0} s'.format(time_cost))

def prepare_targets(targets):
    new_targets = []
    diffused_boxes = []
    noises = []
    ts = []
    for targets_per_image in targets:
        target = {}
        gt_boxes = targets_per_image
        d_boxes, d_noise, d_t = prepare_diffusion_concat(gt_boxes)
        diffused_boxes.append(d_boxes)
        noises.append(d_noise)
        ts.append(d_t)
        target = gt_boxes

        new_targets.append(target)

    return new_targets, torch.stack(diffused_boxes), torch.stack(noises), torch.stack(ts)

def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def prepare_diffusion_concat(gt_boxes):
    """
    :param gt_boxes: (cx, cy, w, h), normalized
    :param num_proposals:
    """
    t = torch.randint(0, num_timesteps, (1,), device=device).long()
    noise = torch.randn(num_proposals, 2, device=device)

    x_start = gt_boxes


    # x_start = (x_start * 2. - 1.) * scale

    # noise sample
    x = q_sample(x_start=x_start, t=t, noise=noise)

    # x = torch.clamp(x, min=-1 * scale, max=scale)
    # x = ((x / scale) + 1) / 2.

    diff_boxes = x

    return diff_boxes, noise, t



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
    out = a.gather(-1, t)
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

sampling_timesteps = default(sampling_timesteps, timesteps)
assert sampling_timesteps <= timesteps
is_ddim_sampling = sampling_timesteps < timesteps
ddim_sampling_eta = 1.
self_condition = False
scale = 0.01
box_renewal = True
use_ensemble = True

sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
log_one_minus_alphas_cumprod = torch.log(1. - alphas_cumprod)
sqrt_recip_alphas_cumprod = torch.sqrt(1. / alphas_cumprod)
sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / alphas_cumprod - 1)

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)
if __name__ == '__main__':
    train()