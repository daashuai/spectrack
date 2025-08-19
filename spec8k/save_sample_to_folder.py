import os
import json
import shutil
import pandas as pd
import random

import os
import json
import shutil

def save_sample_to_folder(data_path_1, data_path_2, similarity, json_file_path="data/metadata.json"):
    """
    保存两个数据文件到 data/ 文件夹，并记录相应路径和相似度到 metadata.json。
    文件命名格式为 sample_{id}_data1.txt 和 sample_{id}_data2.txt。
    metadata.json 是一个样本列表，不包含额外的嵌套字段。
    """
    os.makedirs("data", exist_ok=True)

    # 加载 metadata 列表
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = []

    # 自动计算新的 sample_id（避免重复）
    if metadata:
        max_id = max(int(entry["sample_id"]) for entry in metadata)
    else:
        max_id = 0
    sample_id = str(max_id + 1)

    # 构建新文件名
    filename_1 = f"sample_{sample_id}_data_1.txt"
    filename_2 = f"sample_{sample_id}_data_2.txt"
    target_path_1 = os.path.join("data", filename_1)
    target_path_2 = os.path.join("data", filename_2)

    # 复制数据文件
    shutil.copy(data_path_1, target_path_1)
    shutil.copy(data_path_2, target_path_2)

    # 添加新样本记录
    metadata.append({
        "sample_id": sample_id,
        "data_path_1": filename_1,
        "data_path_2": filename_2,
        "similarity": similarity
    })

    # 写入 metadata.json
    with open(json_file_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"Sample {sample_id} saved and metadata updated.")

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save sample to folder and generate Parquet dataset.")
    parser.add_argument("data_path_1", type=str, help="Path to first txt file")
    parser.add_argument("data_path_2", type=str, help="Path to second txt file")
    parser.add_argument("similarity", type=float, help="Similarity score between the two data samples")

    args = parser.parse_args()

    save_sample_to_folder(
        data_path_1=args.data_path_1,
        data_path_2=args.data_path_2,
        similarity=args.similarity
    )


# python save_sample_to_folder.py ./test/20250411光大三厂一期进水.TXT ./test/八涧堡路口西3倍.TXT 84.5
