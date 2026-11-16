import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from PIL import Image
import os


def compute_R_function(image, r_max=20, delta_r=1.0, target_value=None):
    """计算空间关联函数R(r)"""
    height, width = image.shape
    if target_value is None:
        values, counts = np.unique(image, return_counts=True)
        target_value = values[np.argmin(counts)]
    y_coords, x_coords = np.where(image == target_value)
    points = np.column_stack((x_coords, y_coords))
    total_points = len(points)
    print(f"使用的目标像素值为: {target_value}")
    if total_points == 0:
        raise ValueError(f"图像中未找到目标像素值 {target_value}。")

    margin = r_max + delta_r / 2
    valid_mask = (points[:, 0] >= margin) & (points[:, 0] <= width - 1 - margin) & \
                 (points[:, 1] >= margin) & (points[:, 1] <= height - 1 - margin)
    ref_points = points[valid_mask]
    N_ref = len(ref_points)
    if N_ref == 0:
        raise ValueError("没有有效的参考点，请增大图像尺寸或减少 r_max。")

    tree = KDTree(points)
    radii = np.arange(delta_r / 2, r_max + delta_r / 2, delta_r)
    R_values = []
    image_area = height * width

    for r in radii:
        r_low = max(r - delta_r / 2, 0)
        r_high = r + delta_r / 2
        area_ring = np.pi * (r_high ** 2 - r_low ** 2)

        sum_cond = 0
        for point in ref_points:
            indices = tree.query_ball_point(point, r_high)
            count = 0
            for idx in indices:
                neighbor = points[idx]
                if np.array_equal(neighbor, point):
                    continue
                distance = np.linalg.norm(neighbor - point)
                if r_low < distance <= r_high:
                    count += 1
            sum_cond += count
        avg_cond = sum_cond / N_ref

        density = total_points / image_area
        avg_uncond = density * area_ring
        R = avg_cond / avg_uncond if avg_uncond > 0 else 0
        R_values.append(R)

    return radii, R_values


def analyze_R_function(radii, R_values):
    """分析R(r)函数图像特征并生成描述性文本"""
    # 基本统计量
    R_array = np.array(R_values)
    radii_array = np.array(radii)

    # 1. 整体分布特征
    mean_R = np.mean(R_array)
    std_R = np.std(R_array)
    max_R = np.max(R_array)
    min_R = np.min(R_array)

    # 找到最小值对应的半径
    min_r_index = np.argmin(R_array)
    min_r = radii_array[min_r_index]

    # 找到最大值对应的半径
    max_r_index = np.argmax(R_array)
    max_r = radii_array[max_r_index]

    # 2. 聚类/排斥特性
    clustering_regions = find_regions(radii_array, R_array, threshold=1.2, mode='greater')
    repulsion_regions = find_regions(radii_array, R_array, threshold=0.8, mode='less')

    # 3. 特征距离提取
    peaks = find_peaks(R_array)

    # 构建描述文本
    analysis = []

    # 主描述：聚类和排斥特性
    if clustering_regions or repulsion_regions:
        main_desc = "R(r) 空间关联函数分析：呈现不同距离下点的空间关联特征\n"

        # 添加排斥区域描述
        if repulsion_regions:
            repulsion_desc = []
            for region in repulsion_regions:
                r_start, r_end = region
                if r_start == r_end:
                    repulsion_desc.append(f"{r_start:.1f} px")
                else:
                    repulsion_desc.append(f"{r_start:.1f}–{r_end:.1f} px")
            main_desc += f"- 像素点在{', '.join(repulsion_desc)}范围内不易聚集"

            # 如果有聚类区域，添加连接词
            if clustering_regions:
                main_desc += "，"

        # 添加聚类区域描述
        if clustering_regions:
            # 主要峰值描述
            if peaks.size > 0:
                peak_indices = peaks[np.argsort(-R_array[peaks])]  # 按峰值大小排序
                main_peak_r = radii_array[peak_indices[0]]
                main_peak_R = R_array[peak_indices[0]]
                main_desc += f"{main_peak_r:.1f} px（对应空间频率：{1/(2*main_peak_r):.3f}cycles/px）呈现显著聚集峰值"

                # 次级峰值描述
                secondary_peaks = peak_indices[1:]
                if secondary_peaks.size > 0:
                    secondary_desc = []
                    for idx in secondary_peaks:
                        secondary_desc.append(f"{radii_array[idx]:.1f} px")
                    main_desc += f"，{', '.join(secondary_desc)} 呈次聚集态"

            # 如果聚类区域没有被峰值覆盖
            else:
                clustering_desc = []
                for region in clustering_regions:
                    r_start, r_end = region
                    if r_start == r_end:
                        clustering_desc.append(f"{r_start:.1f} px")
                    else:
                        clustering_desc.append(f"{r_start:.1f}–{r_end:.1f} px")
                main_desc += f"{', '.join(clustering_desc)}）呈聚集分布"

        analysis.append(main_desc + "。")

    # 添加整体波动描述
    analysis.append(f"- 整体波动{'较小' if std_R < 0.3 else '中等' if std_R < 0.6 else '较大'}（m={mean_R:.2f}，σ={std_R:.2f}）。")

    # 与理论模型的匹配性
    if min_R < 0.5 and max_R > 1.5 and len(repulsion_regions) > 0 and len(clustering_regions) > 0:
        # 蓝噪声特征描述
        if peaks.size > 0:
            main_peak_r = radii_array[peaks[0]]
            analysis.append(f"- 该“小半径排斥、中半径聚类”的蓝噪声特征表明，图像中最常见的聚类间距约为 {main_peak_r:.1f} px。")
        else:
            analysis.append("- 该“小半径排斥、中半径聚类”的特征符合蓝噪声半色调的典型分布。")
    elif max_R > 1.5 and len(clustering_regions) > 0:
        analysis.append("- 该“整体均匀、局部聚集”的特征符合绿噪声半色调的分布特性。")
    elif np.allclose(R_array, 1, atol=0.2):
        analysis.append("- R(r) 整体接近 1，表明图像中的目标像素分布接近完全随机模式。")

    return "\n".join(analysis)


def find_regions(radii, values, threshold, mode='greater'):
    """查找值大于/小于阈值的连续区域"""
    regions = []
    current_region = None

    compare = lambda x, t: x > t if mode == 'greater' else x < t

    for r, v in zip(radii, values):
        if compare(v, threshold):
            if current_region is None:
                current_region = [r, r]
            else:
                current_region[1] = r
        elif current_region is not None:
            regions.append(current_region)
            current_region = None

    if current_region is not None:
        regions.append(current_region)

    return regions


def find_peaks(y):
    """查找信号中的峰值"""
    peaks = []
    for i in range(1, len(y) - 1):
        if y[i] > y[i - 1] and y[i] > y[i + 1] and y[i] > 1.1:  # 最小峰值阈值
            peaks.append(i)
    return np.array(peaks)


def plot_R_function(radii, R_values, filename=None):
    """绘制R(r)图表"""
    plt.figure(figsize=(4, 3))
    plt.plot(radii, R_values, 'b-', linewidth=2)
    plt.axhline(y=1, color='r', linestyle='--')
    plt.xlabel('Radius r (pixels)')
    plt.ylabel('R(r)')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 2)

    if filename:
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
    else:
        return plt.gcf()


def batch_process_images(input_folder, output_folder, r_max=20, delta_r=0.8):
    """批量处理图像并保存结果"""
    os.makedirs(output_folder, exist_ok=True)
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tif']

    for filename in os.listdir(input_folder):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            try:
                image_path = os.path.join(input_folder, filename)
                image = Image.open(image_path).convert("L")
                image_array = np.array(image)

                radii, R_values = compute_R_function(image_array, r_max, delta_r)

                # 绘制图形
                fig = plot_R_function(radii, R_values)

                # 保存图形
                output_image_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_r.png")
                fig.savefig(output_image_path, bbox_inches='tight')
                plt.close(fig)  # 释放内存

                # 分析R函数并保存分析结果
                analysis_text = analyze_R_function(radii, R_values)
                output_text_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_analysis.txt")
                with open(output_text_path, 'w', encoding='utf-8') as f:
                    f.write(f"图像: {filename}\n\nR(r)函数分析结果:\n")
                    f.write(analysis_text)

                print(f"已处理: {filename} -> {output_image_path}")
                print(f"分析结果已保存至: {output_text_path}")
                print("-" * 50)
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {str(e)}")


if __name__ == "__main__":
    input_folder = "F:\\B"
    output_folder = "F:\\B\\误差扩散-p"
    r_max = 20
    delta_r = 0.8

    batch_process_images(input_folder, output_folder, r_max, delta_r)