import os
import numpy as np
import pandas as pd
from collections import  namedtuple
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter
from collections import namedtuple
import math
from scipy.spatial.distance import cdist
from sklearn.mixture import GaussianMixture
from scipy.optimize import linear_sum_assignment
import matplotlib.patheffects as path_effects

def get_points_matrix(file_path):
    is_matrix = False
    is_header = True
    points = []
    with open(file_path, 'r', encoding='iso-8859-1') as lines:
        for line in lines:
            if is_matrix:
                row = [float(element) for element in line.strip().split()]
                if is_header:
                    row.insert(0, 0)
                points.append(row)
                is_header = False

            if "Data points" in line:
                is_matrix = True
    return points

def matrix_to_list(matrix):
    point_list = []
    row = len(matrix)
    col = len(matrix[1])
    for i in range(1, row):
        for j in range(1, col):
            weight = matrix[i][j]
            if weight < 0:
                weight = 0
            y = matrix[i][0]
            x = matrix[0][j]
            
            point = [x, y, weight]

            point_list.append(point)

    return point_list



def get_peaks_list(file_path):
    is_peak = False
    peaks = []
    with open(file_path, 'r', encoding='iso-8859-1') as lines:
        for line in lines:
            if "Data points" in line:
                is_peak = False 
            if is_peak and line != "\n":
                temp = line.strip().split()
                row = []
                row.append(float(temp[1][0:5]))
                row.append(float(temp[1][6:12]))
                try:
                    weight = float(temp[2])
                except Exception as e:
                    weight = 9999

                row.append(weight)
                peaks.append(row)
            if is_peak and line == "\n":
                is_peak = False

            if "No." in line:
                is_peak = True
    peaks_sorted = sorted(peaks, key=lambda x: x[-1], reverse=True)
    return peaks_sorted
    
def delete_laman_in_matrix(matrix):
    # 遍历矩阵元素（跳过第0行和第0列的表头/坐标信息）
    for i in range(1, len(matrix)):  # 行索引（从1开始，跳过第0行）
        for j in range(1, len(matrix[0])):  # 列索引（从1开始，跳过第0列）
            weight = matrix[i][j]
            if weight < 0:
                weight = 0  # 负值权重处理为0

            # 获取当前元素的y、x坐标（对应矩阵第0列和第0行）
            y = matrix[i][0]
            x = matrix[0][j]
            
            # 剔除一级拉曼散射和二级瑞丽散射区域的点
            # 条件：一级拉曼散射（69y -82x + 1480 <= 0）或 二级瑞丽散射（5y -9x -145 >= 0）
            if 69 * y - 82 * x + 1480 <= 0 or 5 * y - 9 * x - 145 >= 0:
                matrix[i][j] = 0  # 符合条件的点权重设为0

    return matrix  # 返回处理后的矩阵（纯列表形式）


def iterate_txt_folder(folder_path):
    # Check if the provided path is a directory
    if not os.path.isdir(folder_path):
        print("Error: The provided path is not a directory.")
        return
    points_dict = {}
    # peaks_dict = {}

    # Iterate through the files in the folder
    for filename in os.listdir(folder_path):
        # Check if the file has a .txt extension
        if filename.endswith(".TXT"):
            file_path = os.path.join(folder_path, filename)
            print("Processing file:", filename)
            points = get_points_matrix(file_path)
            points = delete_laman_in_matrix(points)
            # peaks = get_peaks(file_path)
            points_dict[filename] = points
            # peaks_dict[filename] = peaks

    # return points_dict, peaks_dict
    return points_dict



def plot(file_path, points_matrix, peaks):
    '''
    peaks: [ex, em, height]
    '''
    points_matrix_without_header = np.array(points_matrix)[1:, 1:]
    
    # 1. 保存目录
    parent_dir = os.path.dirname(file_path)
    fig_dir = os.path.join(parent_dir, 'fig')
    os.makedirs(fig_dir, exist_ok=True)
    
    # 2. 保存文件名
    file_name = os.path.basename(file_path)
    img_name = os.path.splitext(file_name)[0] + '.png'
    save_path = os.path.join(fig_dir, img_name)
    abs_save_path = os.path.abspath(save_path)

    # 3. 波长轴
    excitation_wavelengths = np.linspace(220, 600, 77)  # Y轴
    emission_wavelengths = np.linspace(230, 650, 85)    # X轴

    plt.figure(figsize=(8, 5), dpi=120)  # 加宽一点，给右侧文字留空间
    
    # 4. 伪彩色底图
    im = plt.pcolormesh(
        emission_wavelengths,
        excitation_wavelengths,
        points_matrix_without_header.T,
        cmap='jet',
        shading='gouraud',
        vmin=0,
        vmax=2000,
    )

    # 5. 黑色等高线
    plt.contour(
        emission_wavelengths, 
        excitation_wavelengths, 
        points_matrix_without_header.T,
        levels=25,
        colors='black',
        linewidths=0.4
    )

    # 6. 坐标轴
    plt.xlabel('Emission Wavelength (nm)', fontsize=9)
    plt.ylabel('Excitation Wavelength (nm)', fontsize=9)
    plt.title('Three-Dimensional Fluorescence Spectrum', fontsize=11)

    # 7. 颜色条
    cbar = plt.colorbar(im, label='Intensity', shrink=0.75, aspect=15)
    cbar.ax.set_yticks(range(0, 1001, 250))
    cbar.ax.tick_params(labelsize=7)

    # 8. 轴范围
    plt.xlim(230, 600)
    plt.ylim(220, 550)
    plt.xticks([300, 400, 500, 600], fontsize=8)
    plt.yticks([250, 350, 450, 550], fontsize=8)

    # ⭐ 9. 绘制峰值点
    if peaks:
        peaks = np.array(peaks)
        plt.scatter(
            peaks[:,1],   # emission (X)
            peaks[:,0],   # excitation (Y)
            color='white',
            edgecolor='black',
            s=40, marker='o', zorder=5
        )
        for i, (ex, em, h) in enumerate(peaks, start=1):
            plt.text(em+5, ex+5, f"P{i}", color='white', fontsize=7,
                     path_effects=[path_effects.withStroke(linewidth=1, foreground="black")])

        # ⭐ 10. 在图右侧展示峰值详细信息
        text_lines = [f"P{i}: Ex={ex:.1f}, Em={em:.1f}, I={h:.0f}"
                      for i, (ex, em, h) in enumerate(peaks, start=1)]
        text_str = "\n".join(text_lines)
        
        # 在坐标轴外侧放文字（用 transform=ax.transAxes）
        ax = plt.gca()
        ax.text(0.05, 0.95, text_str, transform=ax.transAxes,
                fontsize=8, va='top', ha='left',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    plt.tight_layout(pad=1)
    
    # 11. 保存图片
    plt.savefig(save_path, bbox_inches='tight', dpi=120)
    plt.close()

    return abs_save_path


    # def plot(file_path, points_matrix, peaks):
    #     '''
    #     peaks: [ex, em, height]
    #     '''
    #     # 1. 处理保存目录：检查同级目录是否有fig文件夹，无则创建
    #     parent_dir = os.path.dirname(file_path)  # 获取file_path的同级目录
    #     fig_dir = os.path.join(parent_dir, 'fig')  # 构建fig目录路径
    #     os.makedirs(fig_dir, exist_ok=True)  # 创建目录（已存在则不报错）
    #     
    #     # 2. 处理保存文件名：原文件名替换后缀为.png
    #     file_name = os.path.basename(file_path)  # 获取文件名（含后缀）
    #     img_name = os.path.splitext(file_name)[0] + '.png'  # 替换后缀为.png
    #     save_path = os.path.join(fig_dir, img_name)  # 完整保存路径
    # 
    #     # 3. 波长轴定义（保持业务逻辑，X=发射波长，Y=激发波长）
    #     excitation_wavelengths = np.linspace(220, 600, 77)  # Y轴：激发波长（220-600nm）
    #     emission_wavelengths = np.linspace(230, 650, 85)    # X轴：发射波长（230-650nm）
    # 
    #     plt.figure(figsize=(7, 5), dpi=120)  # 画布比例匹配示例图
    #     
    #     # 4. 伪彩色底图（绿→黄→红，匹配目标风格）
    #     im = plt.pcolormesh(
    #         emission_wavelengths,        # X轴：发射波长
    #         excitation_wavelengths,      # Y轴：激发波长
    #         points_matrix,                      # 转置数据适配轴（确保维度对齐）
    #         cmap='jet',                  # 核心：jet色卡实现「绿→黄→红」过渡
    #         shading='gouraud',
    #         vmin=0,                      # 强度下限（荧光强度非负）
    #         vmax=2000,                    # 匹配示例图的强度上限（可根据实际数据调整）
    #     )
    # 
    #     # 5. 黑色轮廓线（模拟目标图的黑色等高线）
    #     plt.contour(
    #         emission_wavelengths, 
    #         excitation_wavelengths, 
    #         points_matrix, 
    #         levels=25,                   # 轮廓层数（控制密度，可微调）
    #         colors='black',              # 强制黑色轮廓
    #         linewidths=0.4               # 线宽匹配示例的纤细轮廓
    #     )
    # 
    #     # 6. 坐标轴与刻度（复刻目标图的简洁风格）
    #     plt.xlabel('Emission Wavelength (nm)', fontsize=9)
    #     plt.ylabel('Excitation Wavelength (nm)', fontsize=9)
    #     plt.title('Three-Dimensional Fluorescence Spectrum', fontsize=11)
    # 
    #     # 7. 颜色条（匹配目标图的刻度：0→1000，间隔250）
    #     cbar = plt.colorbar(im, label='Intensity', shrink=0.75, aspect=15)
    #     cbar.ax.set_yticks(range(0, 1001, 250))  # 强制刻度：0,250,500,750,1000
    #     cbar.ax.tick_params(labelsize=7)         # 缩小刻度字体
    # 
    #     # 8. 轴范围与刻度（匹配目标图的显示区间）
    #     plt.xlim(300, 600)  # X轴聚焦300-600nm（匹配示例图的X范围）
    #     plt.ylim(250, 550)  # Y轴聚焦250-550nm（匹配示例图的Y范围）
    #     plt.xticks([300, 400, 500, 600], fontsize=8)
    #     plt.yticks([250, 350, 450, 550], fontsize=8)
    # 
    #     plt.tight_layout(pad=1)  # 紧凑布局，避免元素重叠
    #     
    #     # 9. 保存图片到fig目录
    #     plt.savefig(save_path, bbox_inches='tight', dpi=120)
    #     plt.close()  # 关闭画布释放资源（可选，根据需求决定是否保留plt.show()）
    #     # plt.show()  # 如需显示图片可保留此行


    # def plot(file_path, data):
    #     # 1. 波长轴定义（保持业务逻辑，X=发射波长，Y=激发波长）
    #     excitation_wavelengths = np.linspace(220, 600, 77)  # Y轴：激发波长（220-600nm）
    #     emission_wavelengths = np.linspace(230, 650, 85)    # X轴：发射波长（230-650nm）
    # 
    #     plt.figure(figsize=(7, 5), dpi=120)  # 画布比例匹配示例图
    #     
    #     # 2. 伪彩色底图（绿→黄→红，匹配目标风格）
    #     im = plt.pcolormesh(
    #         emission_wavelengths,        # X轴：发射波长
    #         excitation_wavelengths,      # Y轴：激发波长
    #         data.T,                      # 转置数据适配轴（确保维度对齐）
    #         cmap='jet',                  # 核心：jet色卡实现「绿→黄→红」过渡
    #         shading='gouraud',
    #         vmin=0,                      # 强度下限（荧光强度非负）
    #         vmax=2000,                    # 匹配示例图的强度上限（可根据实际数据调整）
    #     )
    # 
    #     # 3. 黑色轮廓线（模拟目标图的黑色等高线）
    #     plt.contour(
    #         emission_wavelengths, 
    #         excitation_wavelengths, 
    #         data.T, 
    #         levels=25,                   # 轮廓层数（控制密度，可微调）
    #         colors='black',              # 强制黑色轮廓
    #         linewidths=0.4               # 线宽匹配示例的纤细轮廓
    #     )
    # 
    #     # 4. 坐标轴与刻度（复刻目标图的简洁风格）
    #     plt.xlabel('Emission Wavelength (nm)', fontsize=9)
    #     plt.ylabel('Excitation Wavelength (nm)', fontsize=9)
    #     plt.title('Three-Dimensional Fluorescence Spectrum', fontsize=11)
    # 
    #     # 5. 颜色条（匹配目标图的刻度：0→1000，间隔250）
    #     cbar = plt.colorbar(im, label='Intensity', shrink=0.75, aspect=15)
    #     cbar.ax.set_yticks(range(0, 1001, 250))  # 强制刻度：0,250,500,750,1000
    #     cbar.ax.tick_params(labelsize=7)         # 缩小刻度字体
    # 
    #     # 6. 轴范围与刻度（匹配目标图的显示区间）
    #     plt.xlim(300, 600)  # X轴聚焦300-600nm（匹配示例图的X范围）
    #     plt.ylim(250, 550)  # Y轴聚焦250-550nm（匹配示例图的Y范围）
    #     plt.xticks([300, 400, 500, 600], fontsize=8)
    #     plt.yticks([250, 350, 450, 550], fontsize=8)
    # 
    #     plt.tight_layout(pad=1)  # 紧凑布局，避免元素重叠
    #     plt.show()


def cal_distance(point_1, point_2):
    """
    计算两点在x、y维度上的欧氏距离
    参数:
        point_1: 列表，格式为[x1, y1, ...]（只需前两位为x、y）
        point_2: 列表，格式为[x2, y2, ...]（只需前两位为x、y）
    返回:
        两点间的欧氏距离
    """
    a = (point_1[0] - point_2[0]) **2  # x坐标差的平方
    b = (point_1[1] - point_2[1])** 2  # y坐标差的平方
    return (a + b) **0.5  # 欧氏距离

def remove_duplicate_and_close_point(points, distance_threshold=15):
    """
    移除列表中距离过近的点（保留唯一的点）
    参数:
        points: 二维列表，格式为[[x1, y1, z1], [x2, y2, z2], ...]
        distance_threshold: 距离阈值，小于等于该值的点被视为重复
    返回:
        去重后的点列表
    """
    unique_points = []
    for point in points:
        is_unique = True
        # 检查当前点与已保留的点是否过近
        for existing_point in unique_points:
            if cal_distance(point, existing_point) <= distance_threshold:
                is_unique = False
                break
        if is_unique:
            unique_points.append(point)
    return unique_points

def get_center_point(point, peaks):
    """
    计算某个峰周围近邻点的中心坐标（x、y取均值，z取最近点的强度）
    参数:
        point: 列表，格式为[x, y, z]（目标峰坐标）
        peaks: 二维列表，格式为[[x1, y1, z1], [x2, y2, z2], ...]（所有峰的坐标）
    返回:
        中心坐标的整数形式 (x_center, y_center, z_center)
    """
    # 筛选距离当前点小于25的峰
    temps = []
    for peak in peaks:
        if cal_distance(point, peak) < 25:
            temps.append(peak)
    
    # 计算x、y、z的均值（基于近邻点）
    x_sum = 0
    y_sum = 0
    z_sum = 0
    for p in temps:
        x_sum += p[0]  # x坐标累加
        y_sum += p[1]  # y坐标累加
        z_sum += p[2]  # z强度累加
    # 均值计算（避免除零，若temps为空则返回原point的坐标）
    count = len(temps)
    x = x_sum / count if count > 0 else point[0]
    y = y_sum / count if count > 0 else point[1]
    z = z_sum / count if count > 0 else point[2]

    # 找到离中心最近的峰，用其z值作为中心的z
    min_distance = None
    for peak in peaks:
        distance = ((x - peak[0])** 2 + (y - peak[1]) **2)** 0.5
        if min_distance is None or distance < min_distance:
            min_distance = distance
            z = peak[2]  # 更新z为最近峰的强度

    return int(x), int(y), int(z)

def find_peak_weight(peak, points):
    """
    找到与目标峰最近的点，返回该点的强度（z值）
    参数:
        peak: 列表，格式为[x, y, z]（目标峰坐标）
        points: 二维列表，格式为[[x1, y1, z1], [x2, y2, z2], ...]（所有点的坐标及强度）
    返回:
        最近点的强度（z值）
    """
    if not points:  # 处理空列表
        return 0
    
    min_distance = None
    closest_z = 0
    for p in points:
        # 计算当前点与目标峰的距离
        distance = ((peak[0] - p[0])** 2 + (peak[1] - p[1]) **2)** 0.5
        # 更新最近距离和对应的z值
        if min_distance is None or distance < min_distance:
            min_distance = distance
            closest_z = p[2]  # 最近点的强度
    
    return closest_z

def get_peaks_list_rule(file_path, sigma=1.5):
    """
    从指定文件路径提取峰（所有数据格式为列表）
    
    参数:
        file_path: 文件路径
        sigma: 高斯平滑参数，默认1.5
        
    返回:
        list: 峰的列表，每个峰为[x, y, z]列表 [ex, em, height]
    """
    points_matrix = get_points_matrix(file_path)
    points_matrix = delete_laman_in_matrix(points_matrix)
    points_list = matrix_to_list(points_matrix)

    fluorescence_intensity = np.array(points_matrix)[1:, 1:]  # 转换为数组以便平滑处理
    
    # 高斯平滑处理
    fluorescence_intensity_max = fluorescence_intensity.max()
    sigma = 1.5
    if fluorescence_intensity_max < 1000:
        sigma = 2  # 低强度时调整平滑参数
    smoothed = gaussian_filter(fluorescence_intensity, sigma=sigma)
    
    # 生成波长网格（固定范围，与原逻辑一致）
    excitation_wavelengths = np.linspace(220, 600, 77)  # Y轴：激发波长
    emission_wavelengths = np.linspace(230, 650, 85)    # X轴：发射波长
    excitation_grid, emission_grid = np.meshgrid(excitation_wavelengths, emission_wavelengths)
    
    # 两种方向寻找峰（扁平化数组处理）
    peaks_1, _ = find_peaks(
        smoothed.flatten(),
        height=0.1 * fluorescence_intensity_max,
        distance=1,
        threshold=0.0,
        width=1
    )
    peaks_2, _ = find_peaks(
        smoothed.flatten(order='F'),
        height=0.1 * fluorescence_intensity_max,
        distance=1,
        threshold=0.0,
        width=1
    )
    
    peaks_excitation_1 = excitation_grid.flatten()[peaks_1]
    peaks_emission_1 = emission_grid.flatten()[peaks_1]
    peaks_intensity_1 = fluorescence_intensity.flatten()[peaks_1]

    peak_list_1 = []
    for i in range(len(peaks_intensity_1)):
        point = [peaks_excitation_1[i], peaks_emission_1[i],
                      peaks_intensity_1[i]]
        peak_list_1.append(point)

    # Extract the corresponding excitation and emission wavelengths for the peaks
    peaks_excitation_2 = excitation_grid.flatten(order='F')[peaks_2]
    peaks_emission_2 = emission_grid.flatten(order='F')[peaks_2]
    peaks_intensity_2 = fluorescence_intensity.flatten(order='F')[peaks_2]
    
    peak_list_2 = []
    for i in range(len(peaks_intensity_2)):
        point = [peaks_excitation_2[i], peaks_emission_2[i],
                      peaks_intensity_2[i]]
        peak_list_2.append(point)

    # 匹配峰对并计算距离
    peak_pairs = []
    for p1 in list(peak_list_1):
        for p2 in list(peak_list_2):
            dist = cal_distance(p1, p2)
            peak_pairs.append([p1, p2, dist])  # 峰对格式：[峰1列表, 峰2列表, 距离]
    
    # 筛选近距离峰对（距离≤15）并提取所有峰
    sorted_pairs = [p for p in sorted(peak_pairs, key=lambda x: x[2]) if p[2] <= 15]
    all_peaks = [p for pair in sorted_pairs for p in pair[:2]]  # 提取峰1和峰2（均为列表）
    
    # 去重并计算中心峰
    unique_peaks = remove_duplicate_and_close_point(all_peaks)  # 去重（处理列表）
    peaks_center = [get_center_point(p, all_peaks) for p in unique_peaks]  # 中心峰（列表格式）
    peaks_center = remove_duplicate_and_close_point(peaks_center)  # 再次去重
    
    # 过滤低强度峰（z>110）并返回
    valid_peaks = []
    for peak in peaks_center:
        # 计算峰的权重（z值）
        z = find_peak_weight(peak, points_list)
        if peak[2] > 110:  # 用列表索引2访问z值
            valid_peaks.append([peak[0], peak[1], z])  # 存储为[x, y, z]列表
    
    return valid_peaks


#================ 辅助函数 ==================#
def min_index_(peaks_less, peaks_more):
    peaks_less = np.array(peaks_less)
    peaks_more = np.array(peaks_more)
    distance_array = np.zeros((len(peaks_less), len(peaks_more)))
    for i in range(len(peaks_less)):
        peak_1 = peaks_less[i]
        for j in range(len(peaks_more)):
            peak_2 = peaks_more[j]
            dist = np.sqrt((peak_1[0] - peak_2[0]) ** 2 + (peak_1[1] - peak_2[1]) ** 2)
            distance_array[i][j] = dist

    min_index = np.argmin(distance_array)
    min_index_2d = np.unravel_index(min_index, distance_array.shape)
    return distance_array[min_index_2d], min_index_2d

def select_top_n(n, point, points):
    x, y, z = point
    point_list = []
    for temp in points:
        x_, y_, z_ = temp
        dist = np.sqrt((x - x_) ** 2 + (y - y_) ** 2)
        point_list.append([x_, y_, z_, dist])
    point_np = np.array(point_list)
    index = np.argsort(point_np[:, -1])
    return point_np[index][:n, :3]

def feature_vec(points_top_n, feature_dim):
    points_top_n = np.array(points_top_n)
    vec_list = []
    point_num = points_top_n.shape[0] / feature_dim
    count = 0
    for i in range(feature_dim):
        count += int(point_num)
        temp = points_top_n[:count, 2]
        vec_list.append(np.mean(temp))
    vec = np.array(vec_list)
    vec = vec - np.min(vec)
    index = np.argsort(vec)
    return vec[index][::-1]

def gmm_emd(gmm1_means, gmm1_covs, gmm1_weights, gmm2_means, gmm2_covs, gmm2_weights):
    n_components1 = len(gmm1_weights)
    n_components2 = len(gmm2_weights)
    
    # Compute pairwise distances between Gaussian component means
    distances = cdist(gmm1_means, gmm2_means)
    
    # Solve the linear sum assignment problem to find the optimal component matching
    row_ind, col_ind = linear_sum_assignment(distances)
    
    # Compute the EMD as the weighted sum of distances
    emd = np.sum(distances[row_ind, col_ind] * gmm1_weights[row_ind])
    
    return emd

def matrix_to_gmm(matrix):
    point_list = []
    num = 0
    row = len(matrix)
    col = len(matrix[1])
    for i in range(1, row):
        for j in range(1, col):
            weight = matrix[i][j]
            if weight < 0:
                weight = 0
            y = matrix[i][0]
            x = matrix[0][j]
            
            point = [x, y]

            for k in range(int(weight)):
                point_list.append(point)
            # if selected:
            #     for k in range(int(weight)):
            #         point_list.append(point)
    # print(num)
    # print(len(point_list))
    return point_list


#================ 主函数 ==================#
def calculate_similarity(peaks_1, points_matrix_1, peaks_2, points_matrix_2, top_n=300, feature_dim=10):
    """
    输入：
        peaks_list_1, peaks_2   : list [[x, y, weight], ...]
        points_matrix_1, points_matrix_2 :
    输出：
        distance, info_string
    """
    peaks_1 = np.array(peaks_1)
    peaks_2 = np.array(peaks_2)

    peak_num_1 = len(peaks_1)
    peak_num_2 = len(peaks_2)
    if peak_num_1 <= peak_num_2:
        peaks_less = peaks_1
        peaks_more = peaks_2
        peaks_less_ = peaks_1
        peaks_more_ = peaks_2
    else:
        peaks_less = peaks_2
        peaks_more = peaks_1
        peaks_less_ = peaks_2
        peaks_more_ = peaks_1
            

    peaks_less_num = len(peaks_less)
    peaks_more_num = len(peaks_more)

    is_domestic_sewage = False
    domestic_sewage = [[275,325],[233, 342]]
    domestic_sewage_less_match_num = 0
    domestic_sewage_more_match_num = 0
    unmatch = 0
    match_coefficient = 0.1
    e_peak_distance = 0
    e_distribution_distance = 0
    weight_less_total = 0
    weight_more_total = 0
    peak_less_match = []
    peak_more_match = []
    for i in range(peaks_less_num):
        weight = peaks_less[i][2]
        weight_less_total += weight

        # 判断是否为生活污水
        for j in range(len(domestic_sewage)):
            dist_x = (peaks_less[i][0] - domestic_sewage[j][0])**2 
            dist_y = (peaks_less[i][1] - domestic_sewage[j][1])**2
            distance = dist_x + dist_y
            if distance < 40:
                domestic_sewage_less_match_num += 1
                
            


    for i in range(peaks_more_num):
        weight = peaks_more[i][2]
        weight_more_total += weight

        # 判断是否为生活污水
        for j in range(len(domestic_sewage)):
            dist_x = (peaks_more[i][0] - domestic_sewage[j][0])**2 
            dist_y = (peaks_more[i][1] - domestic_sewage[j][1])**2
            distance = dist_x + dist_y
            if distance < 40:
                domestic_sewage_more_match_num += 1
                
    if domestic_sewage_more_match_num == 2 or domestic_sewage_less_match_num==2:
        is_domestic_sewage = True

    if peaks_less_num == 0:
        e_peak_distance += 100

    if peaks_less_num == 0:
        return 0.0, "(0%)"   # 无法匹配，距离设为最大

    for i in range(peaks_less_num):
        min_dist, min_dist_index = min_index_(peaks_less, peaks_more)

        
        peak_index_1 = min_dist_index[0]
        peak_index_2 = min_dist_index[1]
        peak_1 = peaks_less[peak_index_1]
        peak_2 = peaks_more[peak_index_2]

        x_dist = abs(peak_1[0] - peak_2[0])
        y_dist = abs(peak_1[1] - peak_2[1])
        # if min_dist > 25 or x_dist > 7.5 or y_dist > 7.5:
        beta = 12.5
        # 如果很靠近x轴，那么对偏移的容忍度大一些
        if peak_1[0] < 230:
            beta = 15
        
        # 判断是否是生活污水
        # 如果是生活污水，需要严格的对齐，以区分印染废水
        if is_domestic_sewage:
            beta = 5.5

        if min_dist > 25 or x_dist > beta or y_dist > beta:
        # if min_dist > 25:
            unmatch += 1
        else:
            peak_less_match.append(peaks_less[peak_index_1].tolist())
            peak_more_match.append(peaks_more[peak_index_2].tolist())
        peaks_less = np.delete(peaks_less, peak_index_1, axis=0)                
        peaks_more = np.delete(peaks_more, peak_index_2, axis=0)                

        weight_less = peak_1[2]
        weight_more = peak_2[2]
        ratio_less = weight_less/weight_less_total
        ratio_more = weight_more/weight_more_total
        match_coefficient += ratio_less
        match_coefficient += ratio_more
        weight_ratio_simi = 10*abs(ratio_less-ratio_more)
        if weight_ratio_simi <=1:
            weight_ratio_simi = 0
        
        # distance_temp = min_dist*ratio_less
        # distance_temp = min_dist*ratio_less*(weight_ratio_simi)
        distance_temp = ((1.5*weight_ratio_simi)**2+min_dist)*ratio_less

        e_peak_distance += distance_temp

        
        point_1 = [peak_1[0], peak_1[1], peak_1[2]]
        points_1 = matrix_to_list(points_matrix_1)
        points_top_n_1 = select_top_n(top_n, point_1, points_1)
        vec_1 = feature_vec(points_top_n_1, feature_dim)
        vec_sum_1 = abs(vec_1).sum()
        # vec_sum_1 = np.power(vec_1, 2).sum()
        # vec_reg_ratio_1 = vec_sum_1/1000
        vec_reg_ratio_1 = 1
        vec_1 = vec_1/vec_reg_ratio_1

        point_2 = [peak_2[0], peak_2[1], peak_2[2]]
        points_2 = matrix_to_list(points_matrix_2)
        points_top_n_2 = select_top_n(top_n, point_2, points_2)
        vec_2 = feature_vec(points_top_n_2, feature_dim)
        vec_sum_2 = abs(vec_2).sum()
        # vec_sum_2 = np.power(vec_1, 2).sum()
        # vec_reg_ratio_2 = vec_sum_2/1000
        vec_reg_ratio_2 = vec_sum_2
        vec_2 = vec_2/vec_reg_ratio_2

        # if peak_2[2] > peak_1[2]:
        #     scale = peak_1[2]/peak_2[2]
        # else:
        #     scale = peak_2[2]/peak_1[2]

        # e_distribution_distance_i = distance.cdist([vec_1], [vec_2], 'euclidean')
        e_distribution_distance_i = cdist([vec_1], [vec_2], 'euclidean')
        # e_distribution_distance += e_distribution_distance_i/scale
        e_distribution_distance += e_distribution_distance_i
           
    # e_peak_distance = e_peak_distance/peaks_less_num
    # e_distribution_distance = e_distribution_distance/peaks_less_num

    if peaks_less_num == 1:
        gmm_1 = GaussianMixture(n_components=peaks_less_num)
        points_gmm_1 = matrix_to_gmm(points_matrix_1)
        gmm_1.fit(points_gmm_1)
        # labels_1 = gmm_1.predict(value_1)
        means_1 = gmm_1.means_
        weight_1 = gmm_1.weights_
        covariances_1 = gmm_1.covariances_
        gmm_2 = GaussianMixture(n_components=peaks_less_num)
        points_gmm_2 = matrix_to_gmm(points_matrix_2)
        gmm_2.fit(points_gmm_2)
        # labels_1 = gmm_1.predict(value_1)
        means_2 = gmm_2.means_
        weight_2 = gmm_2.weights_
        covariances_2 = gmm_2.covariances_

        emd = gmm_emd(means_1, covariances_1, weight_1, means_2, covariances_2, weight_2)
        if math.sqrt(e_peak_distance) < math.sqrt(emd):
            temp_2 = math.sqrt(emd)
        else:
            temp_2 = math.sqrt(e_peak_distance)
    else:
        temp_2 = math.sqrt(e_peak_distance)


        # temp_1 = peak_1[0]


    # if peaks_less_num == 1 and peaks_more_num <= 2:
    #     ratio_1 = 0.45
    #     ratio_2 = 0.2
    # else:
    #     ratio_1 = 0.2
    #     ratio_2 = 0.45

    # temp_1 = ratio_1*math.sqrt(e_distribution_distance)
    # temp_2 = 1*math.sqrt(e_peak_distance)
    # temp_2 = math.sqrt(e_peak_distance)
    # temp_2 = ratio_2*math.sqrt(e_peak_distance)
    # if unmatch == 0:
    #     temp_3 = 1
    # else:
    #     temp_3 = 1*(unmatch + 1)

    
    # 判断是否有主峰未完成匹配
    list_peaks_less = peaks_less_.tolist()
    list_peaks_more = peaks_more_.tolist()
    
    a = peak_less_match
    b = list(peak_more_match)

    peak_less_unmatch = [item for item in list_peaks_less if item not in peak_less_match]
    peak_more_unmatch = [item for item in list_peaks_more if item not in peak_more_match]
    main_peak_unmatch_less = False
    main_peak_unmatch_more = False
    a = peaks_less
    b = peaks_more
    # average_weight_less = weight_less_total/peaks_less_num
    # average_weight_more = weight_more_total/peaks_more_num
    max_weight_less = max(peaks_less_[:,2])
    max_weight_more = max(peaks_more_[:,2])
    for i in range(len(peak_less_unmatch)):
        if peak_less_unmatch[i][2] > 0.58*max_weight_less:
            main_peak_unmatch_less = True
    for i in range(len(peak_more_unmatch)):
        if peak_more_unmatch[i][2] > 0.58*max_weight_more:
            main_peak_unmatch_more = True



    # 左图中未匹配的点的数目
    temp_3 = peaks_more_num - peaks_less_num + unmatch

    # 左图与右图峰数差异值
    peak_peanty = 1
    if abs(peak_num_1 - peak_num_2) >= 2:
        peak_peanty = abs(peak_num_1 - peak_num_2)
    temp_4 = math.sqrt(peak_peanty)
    # temp_5 = (peaks_less_num - unmatch)/peaks_more_num
    temp_5 = 1.2*peaks_less_num/peaks_more_num
    if temp_5 == 0:
        temp_5 = 0.1

    unmatch_less = unmatch
    match_less = peaks_less_num - unmatch_less
    unmatch_more = peaks_more_num - match_less
    if not main_peak_unmatch_less and not main_peak_unmatch_more:
        if match_less == peaks_less_num :
            temp_6 = 1
        else:
            temp_6 = unmatch + 1
    elif main_peak_unmatch_less or main_peak_unmatch_more:
        temp_6 = unmatch + unmatch_more + 2
    # if unmatch == 0:
    #     temp_6 = 1
    
    
    # temp = temp_3*(temp_1 + temp_2)
    # temp = temp_4*(0.5*1*1*temp_2 + temp_1)/temp_5
    # temp = temp_6*temp_4*(1.5*temp_2)/(temp_5*match_coefficient*0.5)
    temp_7 = 0
    if unmatch_less != 0 or unmatch_more != 0:
        temp_7 = 2
    temp = temp_6*temp_4*(1.5*temp_2)/(match_coefficient) +\
        1.5*(unmatch_less+unmatch_more) + temp_7

    percent = 0
    if temp <= 9.5:
        percent = 100 - temp
    elif temp >= 9.5:
        percent = 100 - 1.5 * temp
    if percent < 0:
        percent = 0

    # info = f"{round(temp, 2)}({round(percent, 2)}%)"

    # return float(temp), info
    return round(percent, 2)

