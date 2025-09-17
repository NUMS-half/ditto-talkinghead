import os
import sys
import cv2
import torch
import wandb
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

from utils import set_seed, get_logger

LOG_DIR = "./log"
logger = get_logger('audio_train', LOG_DIR)

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
            logger.info("Successful load mean and std of motion space.")
        else:
            logger.warning("Not found mean or std file, use 0/1 for placeholder.")
            self.motion_mean = torch.zeros(265)
            self.motion_std = torch.ones(265)

        logger.info(f"DittoDataset initialize finished with {len(self.file_list)} data segments in total.")

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

        logger.info(f"Initializing models for worker {worker_id} on device {device}...")

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
            logger.warning(f"Handling audio {audio_path} failed with: {e}, this sample will be skipped.")
            return None

        # 视频处理
        cap = cv2.VideoCapture(video_path)
        frames = [cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (256, 256)) for ret, f in
                  iter(lambda: cap.read(), (False, None))]
        cap.release()

        if not frames:
            logger.warning(f"Cannot read frames from {video_path}, this sample will be skipped.")
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
            logger.warning(f"The length of sample {base_name} is not long enough({min_len} < {self.seq_len}), it will be skipped.")
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
            logger.info(f"Loading pre-trained weights from {pretrained_path}...")
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
                logger.info("Successfully loaded pre-trained weights, model is ready to fine tune.")
            except Exception as e:
                logger.warning(f"Failed to load pre-trained weights with: {e}. The model will be trained from scratch.")
        else:
            logger.error("The path of the pre-trained model was not provided or found. The model will be trained from scratch.")

        self.optim = torch.optim.AdamW(self.model.parameters(), lr=config['learning_rate'],
                                       weight_decay=config['weight_decay'])

        self.motion_ckpt = config['motion_ckpt']
        self.w2f_cfg_template = {"model_path": config['audio_ckpt']}
        self.best_loss = float('inf')

    def evaluate_initial_loss(self):
        """
        在训练开始前，评估一次预训练模型在当前数据集上的初始损失。
        """
        logger.info("\n" + "=" * 100)
        logger.info("Starting to evaluate initial loss of the pre-trained model...")

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
            pbar = tqdm(eval_loader, desc="Evaluating initial loss")
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
            logger.info(f"The initial average Loss of the pre-trained model on current dataset is: {average_loss:.6f}")
            # 将初始 loss 记录到 wandb summary
            wandb.summary["initial_loss"] = average_loss
        else:
            logger.warning("Cannot calculate the initial Loss, which might cause by all the data is invalid.")

        print("=" * 100 + "\n")

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
        logger.info("First run, start loading and caching data...")
        # 仅在第一个 epoch 加载数据并存入内存
        caching_pbar = tqdm(train_loader, desc="Caching data")  # 缓存进度
        for batch in caching_pbar:
            cached_data.append(batch)
        logger.info(f"Caching data finished. Cached {len(cached_data)} batches in total.")

        # 使用 wandb.watch 监控模型梯度和参数
        wandb.watch(self.model, log="all", log_freq=100)

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
                    logger.info(f"Epoch {epoch} finished with Avg Loss: {avg_epoch_loss:.6f}")

                # 记录 epoch loss 到 wandb
                wandb.log({"epoch": epoch, "avg_epoch_loss": avg_epoch_loss})

                # 检查当前 epoch 的 loss 是否是最低
                if avg_epoch_loss < self.best_loss:
                    self.best_loss = avg_epoch_loss
                    logger.info(f"Find the lowest loss {self.best_loss:.6f}, saving the best model...")

                    # 更新 wandb summary 中的 best_loss
                    wandb.summary["best_loss"] = self.best_loss
                    wandb.summary["best_loss_epoch"] = epoch

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
                    logger.info(f"The best model has been saved to {save_path}.")
            else:
                logger.warning(f"Epoch {epoch} finished. the loss of all batched are invalid (NaN/inf).")

        logger.info(f"Fine tuning finished, the best model has been saved (loss={self.best_loss}).")
        # 标记 wandb 运行结束
        wandb.finish()


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

    # 设置随机种子
    seed = 42
    set_seed(seed)
    logger.info(f"Random seed has been set to {seed}.")

    # 初始化 wandb
    wandb.init(
        project="ditto-talkinghead-finetune",
        entity=None,
        mode="offline",
        config=training_config  # 将训练配置上传到 wandb
    )
    # logger.info(f"Wandb initialized. Project: {args.wandb_project}, Entity: {args.wandb_entity}")

    trainer = DittoAudioMotionTrainer(training_config)
    trainer.evaluate_initial_loss()
    trainer.train()