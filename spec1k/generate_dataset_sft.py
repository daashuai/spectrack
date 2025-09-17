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
from debug import debugger
debugger(debug=True)

def generate_parquet_dataset(
    json_file_path="spec1k/data/metadata_sft.json",
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    template_1='',
    template_2='',
):
    """
    从 metadata.json（样本列表）和数据文件中生成 train/val/test 的 Parquet 数据集。
    每条记录包含 sample_id、similarity、peaks_dict_1、peaks_dict_2。
    """
    output_dir = "spec1k/dataset/sft"
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
                file_path_1 = os.path.join("spec1k/data", entry["data_path_1"])
                file_path_2 = os.path.join("spec1k/data", entry["data_path_2"])
                expert_reason_chain = entry["expert_reason_chain"]

                # 获取 peaks_dict_1 和 peaks_dict_2
                
                points_matrix_1 = get_points_matrix(file_path_1)
                points_matrix_1 = delete_laman_in_matrix(points_matrix_1)
                points_list_1 = matrix_to_list(points_matrix_1)
                peaks_list_1 = get_peaks_list(file_path_1)
                len_peaks_1 = len(peaks_list_1)
                keep_count_1 = min(10, len_peaks_1)  # 若长度>10则取10，否则取实际长度
                top_peaks_1 = sorted(peaks_list_1, key=lambda x: x[2], reverse=True)[:keep_count_1]



                points_matrix_2 = get_points_matrix(file_path_2)
                points_matrix_2 = delete_laman_in_matrix(points_matrix_2)
                points_list_2 = matrix_to_list(points_matrix_2)
                peaks_list_2 = get_peaks_list(file_path_2)
                # 对peaks_list_2处理
                len_peaks_2 = len(peaks_list_2)
                keep_count_2 = min(10, len_peaks_2)
                top_peaks_2 = sorted(peaks_list_2, key=lambda x: x[2], reverse=True)[:keep_count_2]

                peaks_list_rule_1 = get_peaks_list_rule(file_path_1)
                peaks_list_rule_2 = get_peaks_list_rule(file_path_2)
                similarity_rule = calculate_similarity(peaks_list_rule_1,points_matrix_1,peaks_list_rule_2, points_matrix_2)

                fig_path_1 = plot(file_path_1, points_matrix_1, peaks_list_rule_1)
                fig_path_2 = plot(file_path_2, points_matrix_2, peaks_list_rule_2)
                fig_path_1 = f"file://{fig_path_1}"
                fig_path_2 = f"file://{fig_path_2}"
                data_source = "daashuai/spec8k"
                # 字典传递参数
                params = {
                    "fig_path_1": fig_path_1,
                    "fig_path_2": fig_path_2,
                    "peaks_1": top_peaks_1,
                    "peaks_2": top_peaks_2,
                    "peaks_rule_1": peaks_list_rule_1,
                    "peaks_rule_2": peaks_list_rule_2,
                    "similarity_rule": similarity_rule
                }
                template_2 = template_2.format(**params)

                record = {
                    "data_source": data_source,
                    "prompt": [
                        {
                            "role": "system",
                            "content": template_1,
                        },
                        {
                            "role": "user",
                            "content": template_2,
                        }
                    ],
                    "images": [
                        # {"image": "file:///home/ludashuai/spectrack/spec8k/data/fig/sample_11_data_1.png"},
                        # {"image": "file:///home/ludashuai/spectrack/spec8k/data/fig/sample_11_data_2.png"},
                        {"image": fig_path_1},
                        {"image": fig_path_2}
                    ],
                    "ability": "spec",
                    "expert_reason_chain": expert_reason_chain,
                    "reward_model": {"style": "rule", "ground_truth": answer},
                    "extra_info": {
                        "sample_id": entry["sample_id"],
                    },
                }
                records.append(record)

            except Exception as e:
                print(f"Error processing sample {entry.get('sample_id', 'unknown')}: {e}")


        # 写入 Parquet 文件
        df = pd.DataFrame(records)
        df.to_parquet(os.path.join(output_dir, file_name), index=False)
        print(f"{file_name} saved with {len(records)} samples.")

import argparse

if __name__ == "__main__":

    template_0 = """你是三维荧光光谱相似度计算专家，需根据提供的视觉信息和峰值数据，
修正规则相似度。我将给你两个水体样本的三维荧光光谱相关的数据,示例如下: 
视觉信息:
    fig_right : <image>file:///home/ludashuai/spectrack/spec8k/data/fig/sample_11_data_1.png 
    fig_left: <image>file:///home/ludashuai/spectrack/spec8k/data/fig/sample_11_data_2.png
初始峰值信息: -> [ex, em, height] 
    peaks_right_origin = [[230.0, 345.0, 1551.0], [240.0,265.0, 416.6], [240.0,
    525.0, 249.6], [245.0, 270.0, 478.3], [245.0, 535.0, 211.8], [250.0, 275.0,
    625.7], [250.0, 550.0, 168.2], [255.0, 285.0, 628.2], [255.0, 565.0, 142.5],
    [260.0, 290.0, 575.1], [260.0, 575.0, 115.4], [265.0, 295.0, 729.9],  [270.0,
    300.0, 1059.0], [275.0, 310.0, 1216.0], [280.0, 315.0, 1192.0], [280.0, 335.0,
    1084.0],[285.0, 320.0, 1225.0], [290.0, 325.0, 1120.0]] 

    peaks_left_origin =  [[230.0, 345.0, 1209.0], [230.0,500.0, 323.1], [235.0,
    385.0, 942.5], [235.0, 510.0, 280.0], [240.0, 265.0, 434.1], [240.0, 525.0,
    253.8], [245.0, 270.0, 506.3], [245.0, 535.0, 226.3], [250.0, 275.0, 647.4],
    [250.0, 550.0, 173.8], [255.0, 285.0, 646.4], [255.0, 565.0, 142.4], [260.0,
    290.0, 580.5], [260.0, 575.0, 117.4], [265.0, 295.0, 739.8], [270.0, 300.0,
    1015.0],[275.0, 305.0, 1063.0], [280.0, 345.0, 886.4], [285.0, 320.0, 967.8],
    [290.0, 325.0,905.3], [305.0, 345.0, 655.4]]
规则峰值信息: -> [ex, em, height] 
    peaks_right_rule = [[278, 336,1084], [230, 346, 1551], [243, 391, 300]] 
    peaks_rule_left = [[281, 349, 886.4],[229, 349, 1209], [237, 372, 942.5], [277, 369, 886], [242, 391, 250]]
规则相似度: similarity = "51.81"
    
规则相似度是依据规则峰值信息进行峰匹配计算，但是规则峰值会受少峰，多峰，偏移, 浓度异
常，峰型差异等因素的影响, 无法充分的考虑一些边界案例, 计算出的规则相似度有可能出现偏
差，因此我们需要依据视觉信息和初始峰值信息，对规则相似度进行修正. 请严格按照以下步骤
推理两个光谱样本的相似度是否需要修正以及修正值，每一步都需详细说明，不可省略：  第一
步:通过视觉信息，识别核心峰的位置（Ex/Em）、强度、峰形（宽/窄），列出两样本的关键特
征差异, 是视觉上判断是否存在但不限于多峰，少峰，峰偏移，浓度偏移，峰型高度相似等情况
第二步:结合初始峰值信息和规则峰值信息，分析规则相似度是否合理，分析上述视觉信息中峰
特征差异的原因（如宽峰是否因混合污染）第三步:对比专家规则（主要基于峰匹配），结合视
觉信息和峰值信息，决定是否需要进行修正第四步:基于上述分析，计算最终相似度，需展示修
正过程.

专家推理链示例如下: 
<expert_reason_chain> 
    adjusted_reason : 规则相似度过度放大次要峰
    差异，未识别核心特征峰的高度匹配性。两样本核心峰（类色氨酸 T2、类富里酸 C1及
    Ex230-290,Em300-350 区间特征峰）位置、峰形及强度比例高度一致，次要杂峰差异不影响
    整体光谱特征

    "visual_observation": "左图与右图核心峰位置偏差均≤3nm：T2 类峰
    （Ex280±2nm/Em340±3nm）左图为 Ex280/Em335（强度 1084），右图为 Ex280/Em345（强度
    886.4），位置偏差 3nm；C1 类峰（Ex335±2nm/Em380±2nm）两样本位置完全一致，强度比
    例 0.89（右图 / 左图）。核心峰均呈宽峰特征（半峰宽12-15nm），峰形匹配度高；核心
    峰强度成比例（比例系数 0.85-0.92），无显著浓度偏移。"

    "peaks_observation": "初始峰值中，92% 核心峰（Ex230-290/Em300-350 区间）完全匹配：
    左图 Ex270/Em300（1059）与右图 Ex270/Em300（1015）、左图 Ex275/Em310（1216）与右
    图 Ex275/Em305（1063）等核心峰位置偏差≤2nm，强度比例 0.85-0.90。次要峰（强度＜
    100）差异仅占比 8%，且多为低强度杂峰（如左图 Ex425/Em500 强度39.58，右图无对应
    峰），不影响整体特征。"

    "decision": "需要修正。原规则相似度（51.81%）过度加权次要杂峰差异，忽视核心峰的 强匹配性（位置、峰形、强度比例一致性），实际两样本光谱特征高度相似。"
        
    "adjustment_calculation": "1. 核心峰匹配度：核心峰（占比 80% 权重）匹配度=0.98
    位置匹配 ×0.96 峰形匹配 ×0.90 强度比例）=0.85；2. 次要峰修正：次要峰差异（占比
    20% 权重）对整体影响≤5%，扣减 5%；3. 规则偏差修正：原规则对核心峰权重分配不足，
    增加加权系数 1.18；综合修正：（0.85×0.8 + 0.95×0.2）×1.18≈0.8951→89.51。"

    "adjusted_similarity": 89.51 
</expert_reason_chain>
    

我将给你两个新的水质三维荧光样本,输出格式如下： 
<expert_reason_chain> [这里写下专家推理链] </expert_reason_chain>
<answer> [这里写下修正后的相似度] </answer> """ 

    template__00 = """你是三维荧光光谱相似度计算专家，需根据提供的视觉信息和峰值数据，
    进行推理并修正规则相似度。请严格按四步推理：1. 视觉分析：核心峰的位置、强度、峰形差异（多峰/少
峰/偏移等）2. 峰值验证：结合初始峰值和规则峰值以及视觉信息，分析规则相似度合理性3. 修正决策：判
断是否需要修正及理由4. 计算结果：展示修正过程，给出最终相似度（保留两位小数）

我将给你两个水质三维荧光样本,输出格式如下： 
<expert_reason_chain> [这里写下专家推理链] </expert_reason_chain>
<answer> XX.XX  </answer> """ 

    template_1 = """你是三维荧光光谱相似度计算专家，需根据两个三维荧光样本的视觉信息
和峰值数据以及规则相似度进行推理并修正规则相似度。规则相似度是依据规则峰值信息进行峰
匹配计算，但是规则峰值会受少峰，多峰，偏移, 浓度异常，峰型差异等因素的影响, 无法充分
的考虑一些边界案例, 计算出的规则相似度有可能出现偏差，因此我们需要依据视觉信息和初始
峰值信息，对规则相似度进行修正. 请严格按照以下步骤推理两个光谱样本的相似度是否需要修
正以及修正值，每一步都需详细说明，不可省略：  第一步:通过视觉信息，识别核心峰的位置
（Ex/Em）、强度、峰形（宽/窄），列出两样本的关键特征差异, 是视觉上判断是否存在但不限
于多峰，少峰，峰偏移，浓度偏移，峰型高度相似等情况第二步:结合初始峰值信息和规则峰值
信息，分析规则相似度是否合理，分析上述视觉信息中峰特征差异的原因（如宽峰是否因混合污
染）第三步:对比专家规则（主要基于峰匹配），结合视觉信息和峰值信息，决定是否需要进行
修正第四步:基于上述分析，计算最终相似度，需展示修正过程.

我将给你两个新的水质三维荧光样本,请在新的样本上进行修正推理，并按照如下的格式输出你
得推理链： 
<expert_reason_chain>
{
    adjusted_reason: , 
    visual_observation: ,
    peaks_observation: , 
    decision: ,
    adjustment_calculation: , 
    adjusted_similarity: 
}  
</expert_reason_chain>
<answer> XX.XX </answer> 
"""


    template_2 = """输入:视觉信息: fig_right = <image>{fig_path_1} fig_left = <image>
    {fig_path_2}初始峰值信息: peaks_right_origin = {peaks_1} peaks_left_origin
    = {peaks_2}规则峰值信息: peaks_right_rule = {peaks_rule_1} peaks_left_rule
    = {peaks_rule_2}规则相似度: rule_similarity = {similarity_rule}

    输出:

    """
    generate_parquet_dataset(train_ratio=0.7, val_ratio=0.2, test_ratio=0.1,
                             template_1=template_1, template_2 = template_2)

# change directory in root work directory
# python -m spec1k.generate_dataset_sft

