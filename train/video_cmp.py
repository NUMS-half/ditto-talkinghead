import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def compare_videos(ref_path, test_path, max_frames=None):
    cap_ref = cv2.VideoCapture(ref_path)
    cap_test = cv2.VideoCapture(test_path)

    fps_ref = cap_ref.get(cv2.CAP_PROP_FPS)
    fps_test = cap_test.get(cv2.CAP_PROP_FPS)

    if fps_ref != fps_test:
        print(f"警告: 两个视频的 FPS 不一致 ({fps_ref} vs {fps_test})，可能导致帧错位")

    frame_idx = 0
    psnr_list, ssim_list = [], []

    while True:
        ret1, frame1 = cap_ref.read()
        ret2, frame2 = cap_test.read()

        if not ret1 or not ret2:
            break

        # 转灰度更符合 SSIM 定义，也可以用彩色比较
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # 调整大小（防止分辨率不同）
        if gray1.shape != gray2.shape:
            h = min(gray1.shape[0], gray2.shape[0])
            w = min(gray1.shape[1], gray2.shape[1])
            gray1 = cv2.resize(gray1, (w, h))
            gray2 = cv2.resize(gray2, (w, h))

        # 计算指标
        psnr = peak_signal_noise_ratio(gray1, gray2, data_range=255)
        ssim = structural_similarity(gray1, gray2, data_range=255)

        psnr_list.append(psnr)
        ssim_list.append(ssim)

        frame_idx += 1
        if max_frames and frame_idx >= max_frames:
            break

    cap_ref.release()
    cap_test.release()

    mean_psnr = np.mean(psnr_list) if psnr_list else 0
    mean_ssim = np.mean(ssim_list) if ssim_list else 0

    print(f"对比完成，共比较 {frame_idx} 帧")
    print(f"平均 PSNR: {mean_psnr:.4f} dB") # PSNR 越高越好
    print(f"平均 SSIM: {mean_ssim:.4f}")    # SSIM 越接近 1 越好

    return mean_psnr, mean_ssim


if __name__ == "__main__":
    source_video_path = "../example/source.mp4"
    # output_video_path = "../tmp/target.mp4"
    output_video_path = "../tmp/target_finetune.mp4"
    compare_videos(source_video_path, output_video_path)
