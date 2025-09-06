# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re, json
from typing import Dict, List, Tuple, Any

import json
import sys
import re


def format_reward(predict: str) -> float:
    pattern = re.compile(r"<expert_reason_chain>(.*?)</expert_reason_chain>(.*?)<answer>(.*?)</answer>", re.DOTALL)
    format_match = re.fullmatch(pattern, predict)
    return 1.0 if format_match else 0.0


def extract_answer(text: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    return text

def accuracy_reward(
    pred: str, target: float, method: str = "inverse_abs", epsilon: float = 1e-8,
    threshold: float = 2,
) -> float:
    """
    计算两个数字之间的差距并转换为reward。

    参数：
    - pred: 预测值
    - target: 目标值
    - method: 计算差距和转换reward的方法
        - "inverse_abs":  reward = 1 / (|pred - target| + epsilon)
        - "negative_mse": reward = - (pred - target)^2
        - "exp_decay":    reward = exp(-|pred - target|)
    - epsilon: 防止除零

    返回：
    - reward: 数值，差距越小reward越大
    """
    diff = 0
    acc_2 = 0
    acc_5 = 0
    try:
        # pred = extract_answer(pred) if isinstance(pred, str) else target
        pred = extract_answer(pred)
        if not isinstance(pred, (int, float)):
            pred = float(pred)
        if not isinstance(target, (int, float)):
            target = float(target)
        diff = abs(pred - target)
        if method == "inverse_abs":
            # if diff > 10:
            #     reward = -0.01*diff
            # else:
            #     reward = 1 / (diff + epsilon)
            if didd < 5:
                acc_5 = 1

            if diff < 2:
                reward = 1
                acc_2 = 1
            else:
                reward = -0.01*diff

        elif method == "negative_mse":
            reward = -(diff**2)
        elif method == "exp_decay":
            import math

            reward = math.exp(-diff)
        else:
            raise ValueError(f"Unknown method: {method}")

        return acc_2, acc_5, diff, reward
    except Exception as e:
        # print(f"Error in compute_reward: {e}", file=sys.stderr)
        target = float(target)
        diff = 100 - target
        return acc_2, acc_5, diff, -1.5

def compute_score(solution_str, ground_truth, method="strict", format_weight=0.1,
                  score=1.0, data_source=None, extra_info=None):
    """修改后的评分函数，参数与verl保持一致
    
    保留了原有的格式分和准确率分计算逻辑，同时适配verl的参数要求
    """
    # 计算格式分数
    format_score = format_reward(solution_str)
    
    # 计算准确率分数
    acc_2, acc_5, diff, accuracy_score = accuracy_reward(solution_str, ground_truth)
    
    # 计算总分，使用format_weight作为格式分的权重
    # overall_score = (1 - format_weight) * accuracy_score + format_weight * format_score
    
    # 按照verl的逻辑，如果答案提取失败，返回0
    # answer = extract_solution(solution_str, method=method)
    # if answer is None:
    #     return 0.0
    # if overall_score is None:
    #     return 0.0
    metrics={}
    metrics["score"] = accuracy_score
    metrics["acc_2"] = acc_2
    metrics["acc_5"] = acc_5
    metrics["diff"] = diff
    
    return metrics


def compute_score_r1(reward_input: dict[str, Any], format_weight: float = 0.1) -> dict[str, float]:
    if not isinstance(reward_input, dict):
        print(reward_input)
        print(type(reward_input))
        raise ValueError("Please use `reward_type=sequential` for r1 reward function.")
        

    format_score = format_reward(reward_input["response"])
    acc_2, acc_5, accuracy_score = accuracy_reward(reward_input["response"], reward_input["ground_truth"])
    return {
        "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
        "format": format_score,
        "accuracy": accuracy_score,
    }
