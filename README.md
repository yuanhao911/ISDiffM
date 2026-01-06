# Diffusion Models Enable High-Fidelity Prediction of Fuel Cell Impedance Spectrum from Short Time-Domain Profiles

![image-20250223142112693](https://github.com/yuanhao911/ISDiffM/blob/main/image/Framework.png)

Hao Yuan, Dayi Tan, Zhihua Zhong, Jiangong Zhu, Pingwen Ming, Xuezhe Wei , Haifeng Dai† (†corresponding authors)

+++

## Environment

The codebase has been tested with the following setup:

- Operating System: Ubuntu 20.04
- Python Version: 3.8
- GPU: 1x NVIDIA RTX 3090 with CUDA version 12.2

## Quick start

   1. **Install Pytorch and torchvision**

      Follow the instruction on https://pytorch.org/get-started/locally/.

      ```
      # an example:
      conda install -c pytorch pytorch torchvision
      ```

   2. **Install Dependencies**

      ```
      pip install -r requirements.txt
      ```

3. **Init output(training model output directory) **

   ```
   mkdir output 
   ```

​       Your directory tree should look like this:

```
${POSE_ROOT}
├── dataset
├── model_only1_supervize.py
├── output
├── utils
├── train.py
├── test.py
├── model.py
├── README.md
└── requirements.txt
```

+++

## Training and Testing

#### Training on dataset_1

```
python train.py 
```

#### Testing on dataset_1 using model zoo's model

```
python test.py
```

------

For comparison, we also implement a Transformer-based baseline model using the same input/output format as the diffusion-based model.

#### Training on dataset_1 via transformer_based baseline

```
python train_transformer.py 
```

#### Testing on dataset_1 via transformer_based baseline

```
python test_transformer.py
```

------

## Transfer Learning to `dataset_2`, `dataset_3`, and `dataset_4`

To evaluate the generalization ability of the diffusion-based model, we further perform transfer learning from `dataset_1` to three additional datasets: `dataset_2`, `dataset_3`, and `dataset_4`.

#### Pre-trained model on `dataset_1`

First, we train the model on `dataset_1` as described above and select the best checkpoint:

```
./models/diffusion_based_net_best_dataset_1.pt # 51-dimensional version
./diffusion_based_dataset1_dim_38.pt # 38-dimensional version
```

**Note:** The 51-dimensional checkpoint is used as the **pre-trained initialization** for all transfer experiments on `dataset_2` and `dataset_3`.  
**For `dataset_4`, whose input dimension is 38 instead of 51, we train an additional variant on `dataset_1` using only the 38-dimensional subset of features, and obtain the checkpoint `diffusion_based_dataset1_dim_38.pt`. This 38-dimensional checkpoint plays the same role as the 51-dimensional one, but is dimension-aligned with `dataset_4`.**

#### Transfer Learning Settings

For each target dataset D∈{dataset_2,dataset_3,dataset_4}, we reload the pre-trained weights (e.g. diffusion_based_net_best_dataset_1.pt) from `dataset_1` and train under **three different transfer learning regimes**:

1. Mode A: Frozen encoder
2. Mode B: Encoder frozen, last fully-connected (FC) layer trainable
3. Mode C: Full fine-tuning (all layers trainable)

#### Transfer Learning on dataset_2

```
python transfer_learning.py --dataset dataset2 --pretrain_weights_path "./models/diffusion_based_net_best_dataset_1.pt" --num_proposals 51 --simple_interval 10 --mode A
```

#### Testing on dataset_2 

```
python test_transfer_learning.py --dataset dataset2 --num_proposals 51 --simple_interval 10 --mode A
```

#### Transfer Learning on dataset_3

```
python transfer_learning.py --dataset dataset3 --pretrain_weights_path "./models/diffusion_based_net_best_dataset_1.pt" --num_proposals 51 --simple_interval 1 --mode A
```

#### Testing on dataset_3 

```
python test_transfer_learning.py --dataset dataset3 --num_proposals 51 --simple_interval 1 --mode A
```

#### Transfer Learning on dataset_4

```
python transfer_learning.py --dataset dataset4 --pretrain_weights_path "./models/diffusion_based_dataset1_dim_38.pt" --num_proposals 38 --simple_interval 1 --mode A
```

#### Testing on dataset_4 

```
python test_transfer_learning.py --dataset dataset4 --num_proposals 38 --simple_interval 1 --mode A
```


## LICENSE
```
This project is licensed under the MIT License. For more details, see the LICENSE file included with this repository.
```

## Citation
```
If you find it useful, please cite our paper:
Hao Yuan, Dayi Tan, Zhihua Zhong, Jiangong Zhu, Pingwen Ming, Xuezhe Wei, Haifeng Dai. Diffusion Models Enable High-Fidelity Prediction of Fuel Cell Impedance Spectrum from Short Time-Domain Profiles, DOI identifier: xxxxx, 2026.
```
