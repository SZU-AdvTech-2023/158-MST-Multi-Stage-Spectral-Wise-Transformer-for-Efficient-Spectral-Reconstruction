from torch.utils.data import Dataset
import numpy as np
import random
import cv2
import h5py


class TrainDataset(Dataset):
    def __init__(self, data_root, crop_size, arg=True, bgr2rgb=True, stride=8):
        # 初始化数据集并设置指定的参数
        self.crop_size = crop_size
        self.hypers = []  # 存储高光谱数据的列表
        self.bgrs = []  # 存储RGB数据的列表
        self.arg = arg  # 数据增强标志
        h, w = 482, 512  # 图像形状
        self.stride = stride
        # 计算每一块的大小
        self.patch_per_line = (w - crop_size) // stride + 1
        self.patch_per_colum = (h - crop_size) // stride + 1
        self.patch_per_img = self.patch_per_line * self.patch_per_colum

        # 定义高光谱和RGB数据的路径
        hyper_data_path = f'{data_root}/Train_Spec/'
        bgr_data_path = f'{data_root}/Train_RGB/'

        # 读取包含训练数据列表的文件
        with open(f'{data_root}/split_txt/train_list.txt', 'r') as fin:
            # 创建高光谱和RGB文件名的列表
            hyper_list = [line.replace('\n', '.mat') for line in fin]
            bgr_list = [line.replace('mat', 'jpg') for line in hyper_list]

        # 对列表进行排序以确保一致的顺序
        hyper_list.sort()
        bgr_list.sort()

        # 打印数据集中场景的数量
        print(f'ntire2022数据集的高光谱数量:{len(hyper_list)}')
        print(f'ntire2022数据集的RGB数量:{len(bgr_list)}')

        # 遍历场景并加载数据
        for i in range(len(hyper_list)):
            hyper_path = hyper_data_path + hyper_list[i]

            # 跳过非MAT文件
            if 'mat' not in hyper_path:
                continue

            # 从MAT文件中加载高光谱数据,CWH
            with h5py.File(hyper_path, 'r') as mat:
                hyper = np.float32(np.array(mat['cube']))

            # 转置高光谱数据,CWH->CHW
            hyper = np.transpose(hyper, [0, 2, 1])

            # 构建RGB数据的路径
            bgr_path = bgr_data_path + bgr_list[i]

            # 检查高光谱和RGB文件名是否匹配
            assert hyper_list[i].split('.')[0] == bgr_list[i].split('.')[0], '高光谱和RGB来自不同的场景.'

            # 读取和预处理RGB数据,HWC
            bgr = cv2.imread(bgr_path)

            # 如果指定了bgr2rgb，则将BGR转换为RGB
            if bgr2rgb:
                bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            # 将BGR转换为float32并归一化到范围[0, 1]
            bgr = np.float32(bgr)
            bgr = (bgr - bgr.min()) / (bgr.max() - bgr.min())

            # 转置RGB数据,CHW
            bgr = np.transpose(bgr, [2, 0, 1])  # [3, 482, 512]

            # 将数据添加到列表
            self.hypers.append(hyper)
            self.bgrs.append(bgr)

            # 关闭MAT文件
            mat.close()

            # 打印加载成功的消息
            print(f'Ntire2022场景 {i} 已加载.')

        # 设置数据集属性
        self.img_num = len(self.hypers)
        self.length = self.patch_per_img * self.img_num

    def argument(self, img, rot_times, v_flip, h_flip):
        # 数据增强函数
        # 随机旋转
        for j in range(rot_times):
            img = np.rot90(img.copy(), axes=(1, 2))
        # 随机垂直翻转
        for j in range(v_flip):
            img = img[:, :, ::-1].copy()
        # 随机水平翻转
        for j in range(h_flip):
            img = img[:, ::-1, :].copy()
        return img

    def __getitem__(self, idx):
        # 获取指定索引处的项
        stride = self.stride
        crop_size = self.crop_size
        img_idx, patch_idx = idx // self.patch_per_img, idx % self.patch_per_img
        # 计算行索引和列索引
        h_idx, w_idx = patch_idx // self.patch_per_line, patch_idx % self.patch_per_line
        bgr = self.bgrs[img_idx]
        hyper = self.hypers[img_idx]

        # 从RGB和高光谱数据中提取小块
        bgr = bgr[:, h_idx * stride:h_idx * stride + crop_size, w_idx * stride:w_idx * stride + crop_size]
        hyper = hyper[:, h_idx * stride:h_idx * stride + crop_size, w_idx * stride:w_idx * stride + crop_size]

        # 生成随机的增强参数
        rot_times = random.randint(0, 3)
        v_flip = random.randint(0, 1)
        h_flip = random.randint(0, 1)

        # 如果指定了数据增强，则应用
        if self.arg:
            bgr = self.argument(bgr, rot_times, v_flip, h_flip)
            hyper = self.argument(hyper, rot_times, v_flip, h_flip)

        # 将数据作为连续数组返回
        return np.ascontiguousarray(bgr), np.ascontiguousarray(hyper)

    def __len__(self):
        # 返回数据集中的总补丁数量
        return self.patch_per_img * self.img_num

class ValidDataset(Dataset):
    def __init__(self, data_root, bgr2rgb=True):
        self.hypers = []
        self.bgrs = []
        hyper_data_path = f'{data_root}/Train_Spec/'
        bgr_data_path = f'{data_root}/Train_RGB/'
        with open(f'{data_root}/split_txt/valid_list.txt', 'r') as fin:
            hyper_list = [line.replace('\n', '.mat') for line in fin]
            bgr_list = [line.replace('mat','jpg') for line in hyper_list]
        hyper_list.sort()
        bgr_list.sort()
        print(f'len(hyper_valid) of ntire2022 dataset:{len(hyper_list)}')
        print(f'len(bgr_valid) of ntire2022 dataset:{len(bgr_list)}')
        for i in range(len(hyper_list)):
            hyper_path = hyper_data_path + hyper_list[i]
            if 'mat' not in hyper_path:
                continue
            with h5py.File(hyper_path, 'r') as mat:
                hyper = np.float32(np.array(mat['cube']))
            hyper = np.transpose(hyper, [0, 2, 1])
            bgr_path = bgr_data_path + bgr_list[i]
            assert hyper_list[i].split('.')[0] == bgr_list[i].split('.')[0], 'Hyper and RGB come from different scenes.'
            bgr = cv2.imread(bgr_path)
            if bgr2rgb:
                bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            bgr = np.float32(bgr)
            bgr = (bgr - bgr.min()) / (bgr.max() - bgr.min())
            bgr = np.transpose(bgr, [2, 0, 1])
            self.hypers.append(hyper)
            self.bgrs.append(bgr)
            mat.close()
            print(f'Ntire2022 scene {i} is loaded.')

    def __getitem__(self, idx):
        hyper = self.hypers[idx]
        bgr = self.bgrs[idx]
        return np.ascontiguousarray(bgr), np.ascontiguousarray(hyper)

    def __len__(self):
        return len(self.hypers)