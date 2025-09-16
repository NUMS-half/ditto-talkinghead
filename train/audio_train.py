import os
import sys
import cv2
import torch
import librosa
import numpy as np
from tqdm.auto import tqdm
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, get_worker_info

# 将项目的根目录添加到 Python 路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.models.modules.motion_extractor import MotionExtractor
from core.atomic_components.audio2motion import _cvt_LP_motion_info
from core.atomic_components.wav2feat import Wav2Feat  # 音频处理器
from core.models.modules.LMDM import LMDM as DittoLMDM  # ditto 原生的 LMDM 模型定义，并重命名以区分
from core.models.modules.lmdm_modules.utils import extract  # 扩散过程需要的工具函数


def collate_fn(batch):
    # 过滤掉数据集中返回的 None
    batch = list(filter(lambda x: x is not None, batch))
    # 如果一个批次中所有样本都无效，则返回 None
    if not batch:
        return None, None
    # 使用 PyTorch 默认的打包方式处理有效的样本
    return torch.utils.data.dataloader.default_collate(batch)


# 数据集定义
class DittoDataset(Dataset):
    def __init__(self, data_root, w2f_cfg_template, motion_ckpt, seq_len=80,
                 motion_mean_path="./processed_data/motion_mean.pt",
                 motion_std_path="./processed_data/motion_std.pt"):
        self.data_root = data_root
        self.w2f_cfg_template = w2f_cfg_template  # 保存配置模板
        self.motion_ckpt = motion_ckpt
        self.audio_processor = None  # 将实例初始化为 None，它们将在 worker 进程中被创建
        self.motion_extractor = None
        self.worker_device = None

        self.seq_len = seq_len
        self.audio_dir = os.path.join(data_root, "audio")
        self.video_dir = os.path.join(data_root, "video")
        self.file_list = [os.path.splitext(f)[0] for f in os.listdir(self.audio_dir) if f.endswith('.wav')]

        self.motion_output_names = ["pitch", "yaw", "roll", "t", "exp", "scale", "kp"]

        if os.path.exists(motion_mean_path) and os.path.exists(motion_std_path):
            self.motion_mean = torch.load(motion_mean_path)
            self.motion_std = torch.load(motion_std_path)
            print("成功加载运动潜空间的均值和标准差。")
        else:
            print("未找到均值/标准差文件，将使用 0/1 进行占位。")
            self.motion_mean = torch.zeros(265)
            self.motion_std = torch.ones(265)

        print(f"DittoDataset 初始化完成，共 {len(self.file_list)} 个数据片段。")

    def __len__(self):
        return len(self.file_list)

    def _initialize_worker(self):
        """在每个 worker 进程中独立初始化模型"""
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        # gpu_id = worker_id % torch.cuda.device_count() # 正序调用GPU
        gpu_id = torch.cuda.device_count() - 1 - (worker_id % torch.cuda.device_count()) # 反序调用GPU
        device = f"cuda:{gpu_id}"
        self.worker_device = device

        print(f"Initializing models for worker {worker_id} on device {device}...")

        # 在分配好的 GPU 上创建 MotionExtractor
        self.motion_extractor = MotionExtractor(num_kp=21, backbone="convnextv2_tiny").to(device)
        self.motion_extractor.load_state_dict(torch.load(self.motion_ckpt, map_location=device))
        self.motion_extractor.eval()

        # 基于模板，为当前 worker 创建专属的 w2f_cfg
        w2f_cfg = self.w2f_cfg_template.copy()
        w2f_cfg['device'] = device
        self.audio_processor = Wav2Feat(w2f_cfg=w2f_cfg, w2f_type="hubert")

    def __getitem__(self, index):
        if self.motion_extractor is None or self.audio_processor is None:
            self._initialize_worker()

        base_name = self.file_list[index]
        audio_path = os.path.join(self.audio_dir, f"{base_name}.wav")
        video_path = os.path.join(self.video_dir, f"{base_name}.mp4")

        # 音频处理
        try:
            waveform, sr = librosa.load(audio_path, sr=25000)
            hubert_1024_np = self.audio_processor.wav2feat(waveform, sr=sr)
            hubert_1024 = torch.from_numpy(hubert_1024_np).float()
            padding = torch.zeros(hubert_1024.shape[0], 1103 - 1024)
            audio_features = torch.cat([hubert_1024, padding], dim=1)
        except Exception as e:
            # 音频处理失败
            print(f"警告: 处理音频 {audio_path} 失败: {e}, 将跳过此样本。")
            return None

        # 视频处理
        cap = cv2.VideoCapture(video_path)
        frames = [cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (256, 256)) for ret, f in
                  iter(lambda: cap.read(), (False, None))]
        cap.release()

        if not frames:
            print(f"警告: 无法从 {video_path} 读取帧, 将跳过此样本。")
            return None

        frames_np = np.array(frames) / 255.0
        frames_tensor = torch.from_numpy(frames_np).float().permute(0, 3, 1, 2).to(self.worker_device)
        # 直接在这里实现 extract_motion_dict 的逻辑
        with torch.no_grad():
            # 1. 直接调用模型进行前向传播
            pred_tuple = self.motion_extractor(frames_tensor)

        # 2. 手动将输出的 tuple 打包成字典
        motion_dict_list = {name: pred_tuple[i].cpu().numpy() for i, name in enumerate(self.motion_output_names)}

        # 3. 手动 reshape
        motion_dict_list["exp"] = motion_dict_list["exp"].reshape(-1, 63)
        motion_dict_list["kp"] = motion_dict_list["kp"].reshape(-1, 21, 3)

        motion_latents_list = [_cvt_LP_motion_info({k: v[i] for k, v in motion_dict_list.items()}, 'dic2arr', {'kp'})
                               for i in range(len(frames))]
        motion_latents = torch.from_numpy(np.array(motion_latents_list)).float()

        # 标准化数据，增加安全处理来避免NaN
        epsilon = 1e-4
        clipped_std = torch.clamp(self.motion_std, min=epsilon)
        motion_latents = (motion_latents - self.motion_mean) / clipped_std

        # 对齐与裁剪
        min_len = min(len(audio_features), len(motion_latents))
        if min_len < self.seq_len:
            print(f"警告: 样本 {base_name} 长度不足 ({min_len} < {self.seq_len}), 将跳过。")
            return None

        start_idx = np.random.randint(0, min_len - self.seq_len + 1)
        return motion_latents[start_idx:start_idx + self.seq_len], audio_features[start_idx:start_idx + self.seq_len]


# 主训练器
class DittoAudioMotionTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. 创建模型结构
        self.model = DittoLMDM(motion_feat_dim=265, audio_feat_dim=1103, seq_frames=80).to(self.device)

        # 2. 获取预训练模型的路径
        pretrained_path = self.config.get('pretrained_lmdm_path')
        if pretrained_path and os.path.exists(pretrained_path):
            print(f"正在从 {pretrained_path} 加载预训练权重...")
            try:
                # 3. 加载权重文件
                checkpoint = torch.load(pretrained_path, map_location=self.device)

                # 4. 从 checkpoint 字典中提取 state_dict
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint

                # 5. 将权重载入模型
                self.model.model.load_state_dict(state_dict)
                print("预训练权重加载成功！模型已准备好进行微调。")
            except Exception as e:
                print(f"加载预训练权重失败: {e}。将从头开始训练。")
        else:
            print("未提供或未找到预训练模型路径，将从头开始训练。")

        self.optim = torch.optim.AdamW(self.model.parameters(), lr=config['learning_rate'],
                                       weight_decay=config['weight_decay'])

        self.motion_ckpt = config['motion_ckpt']
        self.w2f_cfg_template = {"model_path": config['audio_ckpt']}
        self.best_loss = float('inf')

    def evaluate_initial_loss(self):
        """
        在训练开始前，评估一次预训练模型在当前数据集上的初始损失。
        """
        print("\n" + "=" * 50)
        print("开始评估预训练模型的初始损失...")

        # 1. 准备数据集和数据加载器 (与训练时使用相同的配置)
        eval_dataset = DittoDataset(data_root=self.config['data_path'],
                                    w2f_cfg_template=self.w2f_cfg_template,
                                    motion_ckpt=self.motion_ckpt,
                                    seq_len=80)
        # 注意：评估时 shuffle 可以为 False，但为了与训练过程保持一致，这里设为 False
        eval_loader = DataLoader(eval_dataset, batch_size=self.config['batch_size'], shuffle=False,
                                 num_workers=self.config['num_workers'], collate_fn=collate_fn, multiprocessing_context='spawn')

        self.model.eval()  # 将模型设置为评估模式
        total_loss = 0
        batch_count = 0

        # 2. 使用 torch.no_grad()，因为我们只评估，不计算梯度
        with torch.no_grad():
            pbar = tqdm(eval_loader, desc="评估初始Loss")
            for x_start, cond in pbar:
                if x_start is None or cond is None:
                    continue

                x_start = x_start.to(self.device)
                cond = cond.to(self.device)

                # 3. 复用训练过程中的损失计算逻辑
                t = torch.randint(0, self.model.n_timestep, (x_start.shape[0],), device=self.device).long()
                noise = torch.randn_like(x_start)
                sqrt_alphas_cumprod_t = extract(self.model.alphas_cumprod.sqrt(), t, x_start.shape)
                sqrt_one_minus_alphas_cumprod_t = extract((1.0 - self.model.alphas_cumprod).sqrt(), t, x_start.shape)
                x_t = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
                dummy_cond_frame = torch.zeros_like(x_start[:, 0, :])
                predicted_noise, _ = self.model.model_predictions(x_t, dummy_cond_frame, cond, t)
                loss = F.mse_loss(predicted_noise, noise)

                if np.isfinite(loss.item()):
                    total_loss += loss.item()
                    batch_count += 1
                pbar.set_postfix(current_loss=loss.item())

        # 4. 计算并打印平均损失
        if batch_count > 0:
            average_loss = total_loss / batch_count
            print(f"预训练模型在你的数据集上的初始平均Loss为: {average_loss:.6f}")
        else:
            print("未能计算初始Loss，可能是所有数据都无效。")

        print("=" * 50 + "\n")

    def train(self):
        # 将配置模板和路径传递给 Dataset
        train_dataset = DittoDataset(data_root=self.config['data_path'],
                                     w2f_cfg_template=self.w2f_cfg_template,
                                     motion_ckpt=self.motion_ckpt,
                                     seq_len=80)
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True, pin_memory=True,
                                  num_workers=self.config['num_workers'], multiprocessing_context='spawn',
                                  collate_fn=collate_fn)

        # 数据缓存
        cached_data = []
        print("首次运行，开始加载并缓存所有数据...")
        # 仅在第一个 epoch 加载数据并存入内存
        caching_pbar = tqdm(train_loader, desc="Caching data")  # 缓存进度
        for batch in caching_pbar:
            cached_data.append(batch)
        print(f"数据缓存完成！共缓存 {len(cached_data)} 个批次。")

        for epoch in range(1, self.config['epochs'] + 1):
            self.model.train()

            # 用于累计当前 epoch 的 loss
            epoch_losses = []

            # 从第二个 epoch 开始，直接从内存中的 cached_data 读取数据
            data_iterator = cached_data
            pbar = tqdm(data_iterator, desc=f"Epoch {epoch}")

            # x_start 是干净的目标数据 (motion_clip), cond 是音频条件 (audio_clip)
            for x_start, cond in pbar:
                # 检查批次是否因为所有样本都无效而变成 None
                if x_start is None or cond is None:
                    continue  # 跳过这个无效的批次

                # 将数据移到正确的设备上
                x_start = x_start.to(self.device)
                cond = cond.to(self.device)

                # 1. 随机采样时间步 t
                t = torch.randint(0, self.model.n_timestep, (x_start.shape[0],), device=self.device).long()

                # 2. 创建高斯噪声
                noise = torch.randn_like(x_start)

                # 3. 制造带噪声的样本 x_t
                sqrt_alphas_cumprod_t = extract(self.model.alphas_cumprod.sqrt(), t, x_start.shape)
                sqrt_one_minus_alphas_cumprod_t = extract((1.0 - self.model.alphas_cumprod).sqrt(), t, x_start.shape)
                x_t = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

                # 4. 模型预测噪声
                dummy_cond_frame = torch.zeros_like(x_start[:, 0, :])
                predicted_noise, _ = self.model.model_predictions(x_t, dummy_cond_frame, cond, t)

                # 5. 计算损失
                loss = F.mse_loss(predicted_noise, noise)

                # 6. 反向传播和优化
                self.optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optim.step()

                # 将当前批次的 loss 值存入列表
                epoch_losses.append(loss.item())
                # 更新进度条，显示瞬时 loss
                pbar.set_postfix(loss=loss.item())

            # 每 epoch 结束后，计算并打印平均 loss
            valid_losses = [l for l in epoch_losses if np.isfinite(l)]
            if valid_losses:
                avg_epoch_loss = np.mean(valid_losses)
                if epoch % 10 == 0 or epoch == self.config['epochs']:
                    print(f"Epoch {epoch} 完成. 平均 Loss: {avg_epoch_loss:.6f}")

                # 检查当前 epoch 的 loss 是否是最低
                if avg_epoch_loss < self.best_loss:
                    self.best_loss = avg_epoch_loss
                    print(f"发现新的最低损失 {self.best_loss:.6f}，保存最佳模型...")

                    os.makedirs(self.config['save_dir'], exist_ok=True)
                    # 使用固定的文件名来保存最佳模型，这样会覆盖上一个最佳模型
                    save_path = os.path.join(self.config['save_dir'], 'lmdm_v0.4_hubert.pth')

                    # 保存骨干网络 MotionDecoder 的权重
                    state_dict = self.model.model.state_dict()
                    checkpoint_to_save = {
                        "model_state_dict": state_dict,
                        "epoch": epoch,
                        "loss": self.best_loss
                    }
                    torch.save(checkpoint_to_save, save_path)
                    print(f"最佳模型已保存至 {save_path}")
            else:
                print(f"Epoch {epoch} 完成. 所有批次的 Loss 均为无效值 (NaN/inf)。")

        print(f"微调训练完成，最佳模型已保存(loss={self.best_loss})")



if __name__ == '__main__':
    training_config = {
        "data_path": "./processed_data",
        "pretrained_lmdm_path": "../checkpoints/ditto_pytorch/models/lmdm_v0.4_hubert.pth",
        "audio_ckpt": "../checkpoints/ditto_onnx/hubert.onnx",
        "motion_ckpt": "../checkpoints/ditto_pytorch/models/motion_extractor.pth",
        "save_dir": "./fine_tuned_checkpoints",
        "epochs": 200,
        "batch_size": 8,
        "learning_rate": 2e-6,
        "weight_decay": 0.0001,
        "num_workers": 4,
    }
    trainer = DittoAudioMotionTrainer(training_config)
    trainer.evaluate_initial_loss()
    trainer.train()