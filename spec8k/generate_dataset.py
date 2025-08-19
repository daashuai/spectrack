import os
import json
import shutil
import pandas as pd
import random

from .utils import get_points_matrix, get_peaks_list,delete_laman_in_matrix, plot, get_peaks_list_rule, calculate_similarity, matrix_to_list

import os
import json
import random
import pandas as pd
import numpy as np

def generate_parquet_dataset(
    json_file_path="spec8k/data/metadata.json",
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    template='',
):
    """
    从 metadata.json（样本列表）和数据文件中生成 train/val/test 的 Parquet 数据集。
    每条记录包含 sample_id、similarity、peaks_dict_1、peaks_dict_2。
    """
    output_dir = "spec8k/dataset"
    os.makedirs(output_dir, exist_ok=True)

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Split ratios must sum to 1.0"

    if not os.path.exists(json_file_path):
        print("Metadata file not found.")
        return

    with open(json_file_path, 'r') as f:
        metadata = json.load(f)

    if not isinstance(metadata, list):
        print("metadata.json format error: expected a list of samples.")
        return

    all_samples = metadata
    random.shuffle(all_samples)

    total = len(all_samples)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    splits = {
        "train.parquet": all_samples[:train_end],
        "val.parquet": all_samples[train_end:val_end],
        "test.parquet": all_samples[val_end:]
    }

    for file_name, split_samples in splits.items():
        records = []
        for entry in split_samples:
            try:
                answer = entry["similarity"]
                # 构造完整路径
                file_path_1 = os.path.join("spec8k/data", entry["data_path_1"])
                file_path_2 = os.path.join("spec8k/data", entry["data_path_2"])

                # 获取 peaks_dict_1 和 peaks_dict_2
                
                points_matrix_1 = get_points_matrix(file_path_1)
                points_matrix_1 = delete_laman_in_matrix(points_matrix_1)
                points_list_1 = matrix_to_list(points_matrix_1)
                peaks_list_1 = get_peaks_list(file_path_1)

                points_matrix_2 = get_points_matrix(file_path_2)
                points_matrix_2 = delete_laman_in_matrix(points_matrix_2)
                points_list_2 = matrix_to_list(points_matrix_2)
                peaks_list_2 = get_peaks_list(file_path_2)

                peaks_list_rule_1 = get_peaks_list_rule(file_path_1)
                peaks_list_rule_2 = get_peaks_list_rule(file_path_2)
                similarity_rule = calculate_similarity(peaks_list_rule_1,points_matrix_1,peaks_list_rule_2, points_matrix_2)

                fig_path_1 = plot(file_path_1, points_matrix_1, peaks_list_rule_1)
                fig_path_2 = plot(file_path_2, points_matrix_2, peaks_list_rule_2)

                # 字典传递参数
                params = {
                    "fig_path_1": fig_path_1,
                    "fig_path_2": fig_path_2,
                    "peaks_1": peaks_list_1,
                    "peaks_2": peaks_list_2,
                    "peaks_rule_1": peaks_list_rule_1,
                    "peaks_rule_2": peaks_list_rule_2,
                    "similarity_rule": similarity_rule
                }
                
                # 使用字典格式化字符串
                problem = template.format(**params)

                records.append({
                    "sample_id": entry["sample_id"],
                    "problem": problem,
                    "answer": answer,
                })

            except Exception as e:
                print(f"Error processing sample {entry.get('sample_id', 'unknown')}: {e}")


        # 写入 Parquet 文件
        df = pd.DataFrame(records)
        df.to_parquet(os.path.join(output_dir, file_name), index=False)
        print(f"{file_name} saved with {len(records)} samples.")

import argparse

if __name__ == "__main__":
    from debug import debugger_config
    debugger_config(debug=False)

    template = """你是一个三维荧光光谱相似度计算专家。  
    我将给你两个水体样本的三维荧光光谱相关的数据,示例如下:

    sample_id: 1
    data_path_right : "sample_11_data_1.txt"
    data_path_left : "sample_11_data_2.txt"

    视觉信息: 
    fig_right : <image>"file:///home/ludashuai/spectrack/spec8k/data/fig/sample_11_data_1.png"</image>
    fig_left : <image>"file:///home/ludashuai/spectrack/spec8k/data/fig/sample_11_data_2.png"</image>

    初始峰值信息: -> [ex, em, height]
    peaks_right_origin =
        [[225.0, 580.0, 55.72], [225.0, 610.0, 50.70], [230.0, 345.0, 1551.0], [230.0, 635.0, 42.54], [240.0, 265.0, 416.6], [240.0, 525.0, 249.6], [245.0, 270.0, 478.3], [245.0, 535.0, 211.8], [250.0, 275.0, 625.7], [250.0, 550.0, 168.2], [255.0, 285.0, 628.2], [255.0, 565.0, 142.5], [260.0, 290.0, 575.1], [260.0, 575.0, 115.4], [265.0, 295.0, 729.9], [265.0, 590.0, 100.3], [270.0, 300.0, 1059.0], [270.0, 600.0, 100.8], [275.0, 310.0, 1216.0], [275.0, 615.0, 92.29], [280.0, 315.0, 1192.0], [280.0, 335.0, 1084.0], [280.0, 625.0, 83.66], [285.0, 320.0, 1225.0], [285.0, 635.0, 64.06], [290.0, 325.0, 1120.0], [475.0, 350.0, 4.243], [505.0, 275.0, 3.245], [510.0, 280.0, 3.460], [515.0, 285.0, 3.723], [530.0, 290.0, 3.383], [535.0, 270.0, 44.80], [540.0, 295.0, 4.600], [545.0, 300.0, 5.772], [550.0, 305.0, 6.220], [560.0, 310.0, 6.286], [570.0, 315.0, 6.580], [580.0, 320.0, 6.716], [585.0, 325.0, 6.526]]
    peaks_left_origin =  
        [[225.0, 575.0, 62.24], [225.0, 605.0, 46.74], [225.0, 640.0, 36.96], [230.0, 345.0, 1209.0], [230.0, 500.0, 323.1], [235.0, 385.0, 942.5], [235.0, 510.0, 280.0], [240.0, 265.0, 434.1], [240.0, 525.0, 253.8], [245.0, 270.0, 506.3], [245.0, 535.0, 226.3], [250.0, 275.0, 647.4], [250.0, 550.0, 173.8], [255.0, 285.0, 646.4], [255.0, 565.0, 142.4], [260.0, 290.0, 580.5], [260.0, 575.0, 117.4], [265.0, 295.0, 739.8], [265.0, 585.0, 101.3], [270.0, 300.0, 1015.0], [270.0, 600.0, 94.85], [275.0, 305.0, 1063.0], [275.0, 610.0, 81.51], [280.0, 345.0, 886.4], [280.0, 625.0, 69.46], [285.0, 320.0, 967.8], [285.0, 640.0, 51.65], [290.0, 325.0, 905.3], [305.0, 345.0, 655.4], [425.0, 500.0, 39.58], [455.0, 540.0, 12.90], [475.0, 345.0, 3.673], [485.0, 265.0, 2.097], [495.0, 270.0, 2.389], [505.0, 275.0, 3.642], [510.0, 280.0, 3.899], [520.0, 285.0, 3.697], [530.0, 290.0, 3.428], [540.0, 295.0, 4.488], [545.0, 300.0, 5.806], [550.0, 305.0, 5.940], [560.0, 310.0, 5.304], [570.0, 315.0, 5.532], [580.0, 320.0, 5.264], [585.0, 325.0, 5.633]]
    规则峰值信息: -> [ex, em, height]
    peaks_right_rule = [[278, 336, 1084], [230, 346, 1551], [243, 391, 300]]
    peaks_rule_left = [[281, 349, 886.4], [229, 349, 1209], [237, 372, 942.5], [277, 369, 886], [242, 391, 250]]

    规则相似度:
    similarity = "51.81"
    
    规则相似度是依据规则峰值信息进行峰匹配计算，但是规则峰值会受少峰，多峰，偏移, 浓
    度异常，峰型差异等因素的影响, 无法充分的考虑一些边界案例, 计算出的规则相似度有可
    能出现偏差，因此我们需要依据视觉信息和初始峰值信息，对规则相似度进行修正. 请严格
    按照以下步骤推理两个光谱样本的相似度是否需要修正以及修正值，每一步都需详细说明，
    不可省略：  第一步:通过视觉信息，识别核心峰的位置（Ex/Em）、强度、峰形（宽/窄），
    列出两样本的关键特征差异, 是视觉上判断是否存在但不限于多峰，少峰，峰偏移，浓度偏
    移，峰型高度相似等情况第二步:结合初始峰值信息和规则峰值信息，分析规则相似度是否
    合理，分析上述视觉信息中峰特征差异的原因（如宽峰是否因混合污染）第三步:对比专家
    规则（主要基于峰匹配），结合视觉信息和峰值信息，决定是否需要进行修正第四步:基于
    上述分析，计算最终相似度，需展示修正过程.

    专家推理链示例如下:
    <expert_reason_chain>
        adjusted_reason : 规则相似度过度放大次要峰差异，未识别核心特征峰的高度匹配
        性。两样本核心峰（类色氨酸 T2、类富里酸 C1 及 Ex230-290,Em300-350 区间特征峰）
        位置、峰形及强度比例高度一致，次要杂峰差异不影响整体光谱特征

        "visual_observation": "左图与右图核心峰位置偏差均≤3nm：T2 类峰
        （Ex280±2nm/Em340±3nm）左图为 Ex280/Em335（强度 1084），右图为 Ex280/Em345
        （强度 886.4），位置偏差 3nm；C1 类峰（Ex335±2nm/Em380±2nm）两样本位置完全一
        致，强度比例 0.89（右图 / 左图）。核心峰均呈宽峰特征（半峰宽12-15nm），峰形
        匹配度高；核心峰强度成比例（比例系数 0.85-0.92），无显著浓度偏移。"

        "peaks_observation": "初始峰值中，92% 核心峰（Ex230-290/Em300-350 区间）完全
        匹配：左图 Ex270/Em300（1059）与右图 Ex270/Em300（1015）、左图 Ex275/Em310
        （1216）与右图 Ex275/Em305（1063）等核心峰位置偏差≤2nm，强度比例 0.85-0.90。
        次要峰（强度＜100）差异仅占比 8%，且多为低强度杂峰（如左图 Ex425/Em500 强度
        39.58，右图无对应峰），不影响整体特征。"

        "decision": "需要修正。原规则相似度（51.81%）过度加权次要杂峰差异，忽视核心
        峰的强匹配性（位置、峰形、强度比例一致性），实际两样本光谱特征高度相似。"
        
        "adjustment_calculation": "1. 核心峰匹配度：核心峰（占比 80% 权重）匹配度
        = 0.98 位置匹配 ×0.96 峰形匹配 ×0.90 强度比例）=0.85；2. 次要峰修正：次要峰
        差异（占比 20% 权重）对整体影响≤5%，扣减 5%；3. 规则偏差修正：原规则对核心峰
        权重分配不足，增加加权系数 1.18；综合修正：（0.85×0.8 + 0.95×0.2）×1.18≈
        0.8951→89.51。"

        "adjusted_similarity": 89.51
    </expert_reason_chain>

    请你先用 <think> </think> 标签说明你的思考过程，
    用<expert_reason_chain> </expert_reason_chain>标签给出专家推理链,
    然后用 <answer> </answer> 标签给出规则相似度，相似度修改量和最终相似度，保留两位小数。
    
    输出格式如下：  
    <think>  
    这里写你的思考过程……  
    </think>  
    <expert_reason_chain>
    这里写下专家推理链
    </expert_reason_chain>
    <answer>  
    rule_similarity = XX.XX  
    delta_similarity = XX.XX
    expert_similarity = XX.XX
    </answer>
    
    输入:
    视觉信息: 
    fig_right = {fig_path_1}
    fig_left = {fig_path_2}
    初始峰值信息:
    peaks_right_origin = {peaks_1} 
    peaks_left_origin = {peaks_2}
    规则峰值信息:
    peaks_right_rule = {peaks_rule_1}
    peaks_left_rule = {peaks_rule_2}
    规则相似度:
    rule_similarity = {similarity_rule}

    输出:

    """
    generate_parquet_dataset(train_ratio=0.7, val_ratio=0.2, test_ratio=0.1,
                             template=template)

# python generate_dataset.py

