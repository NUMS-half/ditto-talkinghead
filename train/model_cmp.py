import torch


def compare_pth(model1_path, model2_path):
    # 加载权重
    state_dict1 = torch.load(model1_path, map_location="cpu")
    state_dict2 = torch.load(model2_path, map_location="cpu")

    # 如果包含 state_dict 包装，提取出来
    if "state_dict" in state_dict1:
        state_dict1 = state_dict1["state_dict"]
    if "state_dict" in state_dict2:
        state_dict2 = state_dict2["state_dict"]

    keys1, keys2 = set(state_dict1.keys()), set(state_dict2.keys())

    # 找不同的参数 key
    only_in_1 = keys1 - keys2
    only_in_2 = keys2 - keys1
    common_keys = keys1 & keys2

    print("🔹 仅在模型1中的参数:", only_in_1)
    print("🔹 仅在模型2中的参数:", only_in_2)

    # 对公共参数逐一比较
    diff_params = []
    for key in common_keys:
        t1, t2 = state_dict1[key], state_dict2[key]
        if t1.shape != t2.shape:
            diff_params.append((key, f"Shape 不同: {t1.shape} vs {t2.shape}"))
        else:
            if not torch.equal(t1, t2):  # 精确比较，不设容差
                max_diff = (t1 - t2).abs().max().item()
                diff_params.append((key, f"数值不同 (最大差异: {max_diff:.6f})"))

    if diff_params:
        print("\n⚠️ 公共参数中的差异:")
        for k, v in diff_params:
            print(f"  {k}: {v}")
    else:
        print("\n✅ 公共参数完全一致（逐元素相等）.")


if __name__ == "__main__":
    compare_pth(
        "../checkpoints/ditto_pytorch/models/lmdm_v0.4_hubert.pth",
        "archive_20250916/training_checkpoints_finetune/ditto_lmdm_epoch_100.pth"
    )
