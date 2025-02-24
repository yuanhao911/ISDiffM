# Diffusion Models Enable High-Fidelity Prediction of Fuel Cell Impedance Spectrum from Short Time-Domain Profiles

![image-20250223142112693](C:\Users\dayi\AppData\Roaming\Typora\typora-user-images\image-20250223142112693.png)

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
├── models
├── output
├── utils
├── tools 
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

