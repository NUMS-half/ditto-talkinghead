import os
import shutil
import subprocess
from tqdm import tqdm
from pydub import AudioSegment
from pydub.silence import detect_silence

# --- 配置参数 ---
VIDEO_PATH = "./raw_data/train_video.mp4"  # 视频路径
OUTPUT_DIR = "./processed_data"
CROP_RECT = "852:480:0:0"  # 裁剪参数 "width:height:x:y"

# 最小片段时长，确保满足模型 seq_len=80 的要求
MIN_SEGMENT_DURATION_SECONDS = 3.2  # 80帧 / 25fps = 3.2秒


def find_speech_segments_from_audio(video_path):
    """
    使用 pydub.silence.detect_silence 从音频中检测非静音片段，
    并返回它们在原始音频中的 (start_sec, end_sec) 时间戳列表。
    """
    print("Step 1: Detecting speech segments from audio using VAD...")

    audio_path = os.path.splitext(video_path)[0] + '.wav'

    if not os.path.exists(audio_path):
        print(f"Error: Corresponding audio file not found at '{audio_path}'. Attempting to extract it...")
        try:
            cmd_extract = ['ffmpeg', '-i', video_path, '-vn', '-ar', '25000', '-ac', '1', '-y', audio_path]
            subprocess.run(cmd_extract, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print("Audio extracted successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to extract audio: {e.stderr.decode()}")
            return []

    try:
        audio_segment = AudioSegment.from_wav(audio_path)
    except Exception as e:
        print(f"Error loading audio file '{audio_path}': {e}")
        return []

    # 1. 检测所有静音片段的时间戳 (单位: 毫秒)
    silence_chunks = detect_silence(
        audio_segment,
        min_silence_len=700,
        silence_thresh=-40
    )

    if not silence_chunks:
        print("Warning: No silence detected. Assuming the entire audio is speech.")
        # 如果没有静音，整个文件就是一个大的语音片段
        return [(0, len(audio_segment) / 1000.0)]

    # 2. 通过“反转”静音时间戳，推断出语音片段的时间戳
    speech_timestamps = []
    last_silence_end = 0
    total_duration_ms = len(audio_segment)

    # 遍历所有静音块
    for start_ms, end_ms in silence_chunks:
        # 上一个静音的结尾到当前静音的开头，是一个语音片段
        if start_ms > last_silence_end:
            speech_timestamps.append((last_silence_end / 1000.0, start_ms / 1000.0))
        last_silence_end = end_ms

    # 检查最后一个静音块之后是否还有语音
    if total_duration_ms > last_silence_end:
        speech_timestamps.append((last_silence_end / 1000.0, total_duration_ms / 1000.0))

    print(f"Found {len(speech_timestamps)} speech segments.")
    return speech_timestamps


def split_and_process_video(video_path, segment_list, output_dir, crop_rect):
    """
    此函数保持不变。
    """
    print("\nStep 3: Splitting, Cropping, and Extracting Audio for valid segments...")
    video_dir = os.path.join(output_dir, "video")
    audio_dir = os.path.join(output_dir, "audio")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(video_path))[0]

    for i, (start_sec, end_sec) in enumerate(tqdm(segment_list, desc="Processing Valid Segments")):
        clip_name = f"{base_name}_clip_{i:04d}"
        output_video_path = os.path.join(video_dir, f"{clip_name}.mp4")
        output_audio_path = os.path.join(audio_dir, f"{clip_name}.wav")

        cmd_video = [
            'ffmpeg', '-i', video_path, '-ss', str(start_sec), '-to', str(end_sec),
            '-vf', f'crop={crop_rect}', '-c:v', 'libx264', '-an', '-y', output_video_path
        ]
        subprocess.run(cmd_video, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        cmd_audio = [
            'ffmpeg', '-i', video_path, '-ss', str(start_sec), '-to', str(end_sec),
            '-vn', '-ar', '25000', '-ac', '1', '-y', output_audio_path
        ]
        subprocess.run(cmd_audio, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    print("\nProcessing finished!")
    print(f"Data saved in: {output_dir}")


if __name__ == '__main__':
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video file not found at '{VIDEO_PATH}'")
    else:
        # 步骤 1: 从音频中查找所有包含语音的片段
        initial_segments = find_speech_segments_from_audio(VIDEO_PATH)

        if initial_segments:
            # 步骤 2: 过滤，只保留长度足够的片段
            print(f"\nStep 2: Filtering segments to be longer than {MIN_SEGMENT_DURATION_SECONDS} seconds...")
            long_enough_segments = []
            for start, end in initial_segments:
                duration = end - start
                if duration >= MIN_SEGMENT_DURATION_SECONDS:
                    long_enough_segments.append((start, end))

            print(f"Found {len(long_enough_segments)} segments that meet the duration requirement.")

            # 步骤 3: 只处理那些长度足够的片段
            if long_enough_segments:
                split_and_process_video(VIDEO_PATH, long_enough_segments, OUTPUT_DIR, CROP_RECT)
                print("\nData preparation complete!")
                print(f"Please update your training script's 'data_path' to: '{OUTPUT_DIR}'")
            else:
                print("\nNo speech segments met the minimum duration requirement.")
        else:
            print(
                "No speech segments detected at all. Please check your audio file or adjust the VAD parameters in the script.")