# Path to the downloaded CLIP official weights.
# See: https://github.com/openai/CLIP/blob/a9b1bf5920416aaeaec965c25dd9e8f98c864f16/clip/clip.py#L30
CLIP_VIT_B16_PATH = '/media/zhang/data2/zly/m2clip2/ViT-B-16.pt'
CLIP_VIT_B32_PATH = '/slurm-files/wmm/models/ViT-B-32.pt'
CLIP_VIT_L14_PATH = 'https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt'

# Whether cuDNN should be temporarily disable for 3D depthwise convolution.
# For some PyTorch builds the built-in 3D depthwise convolution may be much
# faster than the cuDNN implementation. You may experiment with your specific
# environment to find out the optimal option.
DWCONV3D_DISABLE_CUDNN = True

# Configuration of datasets. The required fields are listed for Something-something-v2 (ssv2)
# and Kinetics-400 (k400). Fill in the values to use the datasets, or add new datasets following
# these examples.
DATASETS = {
    'ssv2': dict(
        TRAIN_ROOT='/media/zhang/data2/zly/ssv2/20bn-something-something-v2',
        VAL_ROOT='/media/zhang/data2/zly/ssv2/20bn-something-something-v2',
        TRAIN_LIST='/media/zhang/data2/zly/m2clip/ssv2_train.txt',
        VAL_LIST='/media/zhang/data2/zly/m2clip/ssv2_val.txt',
        NUM_CLASSES=174,
    ),
    'k400': dict(
        TRAIN_ROOT='/media/zhang/data2/zly/k400_train',
        VAL_ROOT='/media/zhang/data2/zly/k400_test',
        TRAIN_LIST='/media/zhang/data2/zly/m2clip/k400_train2.txt',
        VAL_LIST='/media/zhang/data2/zly/m2clip/k400_val2.txt',
        NUM_CLASSES=400,
    ),
    'k400_128st': dict(
        TRAIN_ROOT='/media/zhang/data2/zly/k400_train',
        VAL_ROOT='/media/zhang/data2/zly/k400_test',
        TRAIN_LIST='/media/zhang/data2/zly/m2clip/k400_train2.txt',
        VAL_LIST='/media/zhang/data2/zly/m2clip/k400_val2.txt',
        NUM_CLASSES=400,
    ),
    'hmdb51': dict(
        TRAIN_ROOT='/media/zhang/data2/hmdb51/',
        VAL_ROOT='/media/zhang/data2/hmdb51/',
        TRAIN_LIST='/media/zhang/data2/zly/m2clip2/ucf101_val.txt',
        VAL_LIST='/media/zhang/data2/zly/m2clip/hmdb51_val.txt',
        NUM_CLASSES=51,
    ),
    'ucf101': dict(
        TRAIN_ROOT='/media/zhang/data2/UCF101_videos/UCF101/',  # 零样本下不用，但占位
        VAL_ROOT='/media/zhang/data2/UCF101_videos/UCF101/',    # 替换为你实际的UCF101视频路径
        TRAIN_LIST='/media/zhang/data2/zly/m2clip2/ucf101_val.txt', # 占位
        VAL_LIST='/media/zhang/data2/zly/m2clip-master/ucf101_val1.txt',     # 指向UCF101的验证集/测试集列表
        NUM_CLASSES=101,
    ),
}
