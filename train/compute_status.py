import os
import sys
import cv2
import torch
import numpy as np
from tqdm import tqdm

# 将项目的根目录添加到 Python 路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.models.modules.motion_extractor import MotionExtractor as MotionExtractorModelDefinition
from core.atomic_components.audio2motion import _cvt_LP_motion_info


class MotionStatsCalculator:
    def __init__(self, motion_ckpt_path, device):
        self.device = device
        self.model = MotionExtractorModelDefinition(num_kp=21, backbone="convnextv2_tiny").to(device)
        self.model.load_state_dict(torch.load(motion_ckpt_path, map_location=device))
        self.model.eval()
        self.output_names = ["pitch", "yaw", "roll", "t", "exp", "scale", "kp"]
        print(f"MotionExtractor 已从 '{motion_ckpt_path}' 初始化")

    def _extract_motion_from_video(self, video_path):
        """从单个视频文件中提取原始运动特征"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"警告：无法打开视频文件 {video_path}，跳过。")
            return None

        # 优化帧读取
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (256, 256)))
        cap.release()

        if not frames:
            return None

        frames_np = np.array(frames, dtype=np.float32) / 255.0
        frames_tensor = torch.from_numpy(frames_np).permute(0, 3, 1, 2).to(self.device)

        with torch.no_grad():
            pred_tuple = self.model(frames_tensor)
            outputs = {name: pred_tuple[i].cpu().numpy() for i, name in enumerate(self.output_names)}
            outputs["exp"] = outputs["exp"].reshape(-1, 63)
            outputs["kp"] = outputs["kp"].reshape(-1, 21, 3)

        motion_latents_list = [_cvt_LP_motion_info({k: v[i] for k, v in outputs.items()}, 'dic2arr', {'kp'})
                               for i in range(len(frames))]

        return np.array(motion_latents_list, dtype=np.float32)

    def compute_and_save(self, data_path, mean_path, std_path):
        """遍历数据目录，计算并保存均值和标准差"""
        video_dir = os.path.join(data_path, "video")
        if not os.path.isdir(video_dir):
            print(f"错误：在 '{data_path}' 中未找到 'video' 子目录。请确保数据路径正确。")
            return

        video_files = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith('.mp4')]
        if not video_files:
            print(f"错误：在 '{video_dir}' 中未找到任何 .mp4 文件。")
            return

        all_motion_latents = []
        print(f"发现 {len(video_files)} 个视频文件，开始提取运动特征...")

        for video_path in tqdm(video_files, desc="处理视频"):
            latents = self._extract_motion_from_video(video_path)
            if latents is not None and len(latents) > 0:
                all_motion_latents.append(latents)

        if not all_motion_latents:
            print("错误：未能从任何视频中提取有效的运动特征！")
            return

        # 使用 np.concatenate 效率更高
        full_dataset_latents = np.concatenate(all_motion_latents, axis=0)
        print(f"总共提取了 {full_dataset_latents.shape[0]} 帧的运动数据。")

        # 计算均值和标准差
        mean_val = np.mean(full_dataset_latents, axis=0)
        std_val = np.std(full_dataset_latents, axis=0)

        # 防止标准差为0导致后续除零错误
        std_val[std_val < 1e-6] = 1.0

        # 转换为 Tensor 并保存
        mean_tensor = torch.from_numpy(mean_val)
        std_tensor = torch.from_numpy(std_val)

        # 将文件直接保存在数据目录下，方便管理
        final_mean_path = os.path.join(data_path, mean_path)
        final_std_path = os.path.join(data_path, std_path)

        torch.save(mean_tensor, final_mean_path)
        torch.save(std_tensor, final_std_path)

        print("\n计算完成！")
        print(f"均值文件已保存至: {final_mean_path}")
        print(f"标准差文件已保存至: {final_std_path}")
        print(f"均值张量形状: {mean_tensor.shape}")
        print(f"标准差张量形状: {std_tensor.shape}")


if __name__ == '__main__':
    # 配置
    DATA_DIRECTORY = "processed_data"

    stats_config = {
        "data_path": DATA_DIRECTORY,
        "motion_ckpt": "../checkpoints/ditto_pytorch/models/motion_extractor.pth",
        "mean_path": "motion_mean.pt",  # 文件名
        "std_path": "motion_std.pt",    # 文件名
    }

    if not os.path.isdir(DATA_DIRECTORY):
        print(f"错误：数据目录 '{DATA_DIRECTORY}' 不存在。")
        print("请先运行 `data_process.py` 脚本来生成数据。")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        calculator = MotionStatsCalculator(stats_config['motion_ckpt'], device)
        calculator.compute_and_save(
            stats_config['data_path'],
            stats_config['mean_path'],
            stats_config['std_path']
        )