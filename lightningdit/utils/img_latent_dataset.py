"""
ImageNet Latent Dataset with safetensors.
"""

import os
import numpy as np
from glob import glob
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

from safetensors import safe_open
from torch.utils.data import DataLoader


class ImgLatentDataset(Dataset):
    def __init__(self, data_dir, latent_norm=True, latent_multiplier=1.0, is_overfit=False, fixed_latent_index=None):
        self.data_dir = data_dir
        self.latent_norm = latent_norm
        self.latent_multiplier = latent_multiplier

        self.files = sorted(glob(os.path.join(data_dir, "*.safetensors")))
        self.img_to_file_map = self.get_img_to_safefile_map()
        
        self.is_overfit = is_overfit
        if is_overfit:
            self.files = self.files[:1]
            self.img_to_file_map = {i: self.img_to_file_map[0] for i in range(1922036)}

            random_indices = np.random.choice(len(self.img_to_file_map), 1024, replace=False)
            # 每len / 1024个取一个
            random_indices = np.arange(len(self.img_to_file_map))[::len(self.img_to_file_map) // 1024]
            print(f'### Overfit mode, only use the fix 1024 images. random_indices={random_indices}, len(random_indices)={len(random_indices)}')
            # self.files = [self.files[i] for i in random_indices]
            self.img_to_file_map_overfit = {}
            for index in range(1024):
                self.img_to_file_map_overfit[index] = self.img_to_file_map[random_indices[index]]
            self.img_to_file_map = self.img_to_file_map_overfit
            print('### Overfit mode, only use fix 1024 images.')

        self.fixed_latent_entry = None
        if fixed_latent_index is not None:
            if fixed_latent_index < 0 or fixed_latent_index >= len(self.img_to_file_map):
                raise ValueError(f"fixed_latent_index {fixed_latent_index} out of range (0-{len(self.img_to_file_map)-1})")
            self.fixed_latent_entry = self.img_to_file_map[fixed_latent_index]

        if latent_norm:
            self._latent_mean, self._latent_std = self.get_latent_stats_lmx()

            # debug VAVAE generation
            # self._latent_mean, self._latent_std = self.get_VAVAE_latent_stats()

    def get_img_to_safefile_map(self):
        img_to_file = {}
        for safe_file in self.files:
            with safe_open(safe_file, framework="pt", device="cpu") as f:
                labels = f.get_slice('labels')
                labels_shape = labels.get_shape()
                num_imgs = labels_shape[0]
                cur_len = len(img_to_file)
                for i in range(num_imgs):
                    img_to_file[cur_len+i] = {
                        'safe_file': safe_file,
                        'idx_in_file': i
                    }
        return img_to_file

    def get_latent_stats(self):
        latent_stats_cache_file = os.path.join(self.data_dir, "latents_stats.pt")
        if not os.path.exists(latent_stats_cache_file):
            latent_stats = self.compute_latent_stats()
            torch.save(latent_stats, latent_stats_cache_file)
        else:
            latent_stats = torch.load(latent_stats_cache_file)
        return latent_stats['mean'], latent_stats['std']
    
    def get_latent_stats_lmx(self):
        latent_stats_cache_file = os.path.join(self.data_dir, "latents_stats.pt")
        if not os.path.exists(latent_stats_cache_file):
            raise RuntimeError('Cannot find stats.')
        else:
            latent_stats = torch.load(latent_stats_cache_file)
        mean = latent_stats['mean'][:, 0, ...]
        std = latent_stats['std'][:, 0, ...]
        assert mean.shape == (1, 832, 16, 16)
        assert std.shape == (1, 832, 16, 16)
        return mean, std
    
    def get_VAVAE_latent_stats(self):
        latent_stats_cache_file = os.path.join("./datasets/latents_stats.pt")
        latent_stats = torch.load(latent_stats_cache_file)
        return latent_stats['mean'], latent_stats['std']
    
    def compute_latent_stats(self, files_per_batch=5):
        """
        在线分批统计 latents 和 latents_flip 的全局 mean / std
        Args:
            files (List[str]): safetensors 文件路径列表
            files_per_batch (int): 每次加载的文件数量
            device (str): 加载张量时所在设备，一般为 'cpu'
        Returns:
            dict(mean=[1,2,C,H,W], std=[1,2,C,H,W])
        """
        global_sum = [None, None]
        global_sum_sq = [None, None]
        global_count = [0, 0]

        num_files = len(self.files)
        total_batches = (num_files + files_per_batch - 1) // files_per_batch

        with torch.no_grad():
            for start in tqdm(range(0, num_files, files_per_batch),
                            total=total_batches,
                            desc="批量加载文件"):
                end = min(start + files_per_batch, num_files)
                batch_files = self.files[start:end]

                # 累积当前批次所有 latents
                latents, latents_flip = [], []
                for path in batch_files:
                    with safe_open(path, framework="pt", device='cpu') as f:
                        lat = f.get_slice('latents')[:].to(torch.float32)       # [B, C, H, W]
                        lat_flip = f.get_slice('latents_flip')[:].to(torch.float32)
                    latents.append(lat)
                    latents_flip.append(lat_flip)

                if len(latents) == 0 or len(latents_flip) == 0:
                    continue

                # 拼接当前批次
                lat = torch.cat(latents, dim=0)
                lat_flip = torch.cat(latents_flip, dim=0)
                del latents, latents_flip

                # 转 double 精度防溢出
                lat64 = lat.to(torch.float64)
                lat_flip64 = lat_flip.to(torch.float64)

                # 当前批次统计
                sum0 = lat64.sum(dim=0, keepdim=True)           # Σx
                sum_sq0 = (lat64 ** 2).sum(dim=0, keepdim=True) # Σx²
                cnt0 = lat64.shape[0]

                sum1 = lat_flip64.sum(dim=0, keepdim=True)
                sum_sq1 = (lat_flip64 ** 2).sum(dim=0, keepdim=True)
                cnt1 = lat_flip64.shape[0]

                # 累加到全局
                if global_sum[0] is None:
                    global_sum[0], global_sum_sq[0] = sum0, sum_sq0
                    global_sum[1], global_sum_sq[1] = sum1, sum_sq1
                else:
                    global_sum[0] += sum0
                    global_sum_sq[0] += sum_sq0
                    global_sum[1] += sum1
                    global_sum_sq[1] += sum_sq1

                global_count[0] += cnt0
                global_count[1] += cnt1

                # 清理内存
                del lat, lat_flip, lat64, lat_flip64, sum0, sum_sq0, sum1, sum_sq1
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # ===== 最终统计 =====
        means, stds = [], []
        for k in (0, 1):
            n = max(global_count[k], 1)
            mean_k = global_sum[k] / n
            var_k = (global_sum_sq[k] / n) - (mean_k ** 2)
            var_k = torch.clamp(var_k, min=0.0)  # 防止数值误差
            std_k = torch.sqrt(var_k)

            means.append(mean_k.to(torch.float32))
            stds.append(std_k.to(torch.float32))

        mean = torch.stack(means, dim=1)  # [1, 2, C, H, W]
        std = torch.stack(stds, dim=1)    # [1, 2, C, H, W]

        print(f"\n在线统计完成：")
        print(f"  - 变体0样本数: {global_count[0]:,}")
        print(f"  - 变体1样本数: {global_count[1]:,}")
        print(f"  - Mean shape: {tuple(mean.shape)}, Std shape: {tuple(std.shape)}")
        print(f"  - Mean范围: [{mean.min():.6f}, {mean.max():.6f}]")
        print(f"  - Std范围:  [{std.min():.6f}, {std.max():.6f}]")

        return {'mean': mean, 'std': std}
        # num_samples = min(min_num, len(self.img_to_file_map))
        # random_indices = np.random.choice(len(self.img_to_file_map), num_samples, replace=False)
        # latents = []
        # for idx in tqdm(random_indices):
        #     img_info = self.img_to_file_map[idx]
        #     safe_file, img_idx = img_info['safe_file'], img_info['idx_in_file']
        #     with safe_open(safe_file, framework="pt", device="cpu") as f:
        #         features = f.get_slice('latents')
        #         feature = features[img_idx:img_idx+1]
        #         latents.append(feature)
        # latents = torch.cat(latents, dim=0)
        # mean = latents.mean(dim=[0, 2, 3], keepdim=True)
        # std = latents.std(dim=[0, 2, 3], keepdim=True)
        # latent_stats = {'mean': mean, 'std': std}
        # print(latent_stats)
        # return latent_stats

    def __len__(self):
        if self.is_overfit:
            assert len(self.img_to_file_map) == 1024
        return len(self.img_to_file_map.keys())

    def __getitem__(self, idx):
        if self.fixed_latent_entry is not None:
            img_info = self.fixed_latent_entry
        else:
            img_info = self.img_to_file_map[idx]
        safe_file, img_idx = img_info['safe_file'], img_info['idx_in_file']
        with safe_open(safe_file, framework="pt", device="cpu") as f:
            tensor_key = "latents" if np.random.uniform(0, 1) > 0.5 else "latents_flip"
            features = f.get_slice(tensor_key)
            labels = f.get_slice('labels')
            feature = features[img_idx:img_idx+1]
            label = labels[img_idx:img_idx+1]

        if self.latent_norm:
            feature = (feature - self._latent_mean) / self._latent_std
        feature = feature * self.latent_multiplier
        
        # remove the first batch dimension (=1) kept by get_slice()
        feature = feature.squeeze(0)
        label = label.squeeze(0)
        return feature, label


if __name__ == "__main__":
    dataset = ImgLatentDataset(
        data_dir="/path/to/stats", 
        latent_norm=False, 
        latent_multiplier=1.0,
        is_overfit=False)
    

