import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import logging


# ====================== 算法函数模块 ======================
def compute_D_function(binary_image, r1, r2, delta_a_deg=10, target_value=None):
    """计算归一化的方向分布函数"""
    height, width = binary_image.shape
    if target_value is None:
        values, counts = np.unique(binary_image, return_counts=True)
        target_value = values[np.argmin(counts)]
    y_coords, x_coords = np.where(binary_image == target_value)
    points = np.column_stack((x_coords, y_coords))
    total_points = len(points)
    if total_points == 0:
        raise ValueError(f"图像中未找到目标像素值 {target_value}。")

    delta_a_rad = np.deg2rad(delta_a_deg)
    angle_bins = np.arange(0, 2 * np.pi, delta_a_rad)
    num_bins = len(angle_bins)

    margin = r2
    valid_mask = (points[:, 0] >= margin) & (points[:, 0] <= width - 1 - margin) & \
                 (points[:, 1] >= margin) & (points[:, 1] <= height - 1 - margin)
    ref_points = points[valid_mask]
    N_ref = len(ref_points)
    if N_ref == 0:
        raise ValueError("无有效参考点，请调整参数或图像尺寸。")

    tree = KDTree(points)
    total_counts_all_ref = 0
    D_values_all_refs = np.zeros(num_bins)
    for m in ref_points:
        indices = tree.query_ball_point(m, r2)
        candidates = points[indices]
        distances = np.linalg.norm(candidates - m, axis=1)
        mask = (distances >= r1) & (distances < r2)
        neighbors = candidates[mask]
        neighbors = neighbors[~np.all(neighbors == m, axis=1)]
        if len(neighbors) == 0:
            continue
        total_counts_all_ref += len(neighbors)
        angles = np.arctan2(neighbors[:, 1] - m[1], neighbors[:, 0] - m[0]) % (2 * np.pi)
        for i, angle_start in enumerate(angle_bins):
            angle_end = angle_start + delta_a_rad
            sector_mask = (angles >= angle_start) & (angles < angle_end)
            D_values_all_refs[i] += np.sum(sector_mask)

    area_ring = np.pi * (r2 ** 2 - r1 ** 2)
    area_sector = (delta_a_rad / (2 * np.pi)) * area_ring
    avg_density_total = total_counts_all_ref / (N_ref * area_ring) if area_ring != 0 else 0
    avg_density_refs = D_values_all_refs / (N_ref * area_sector) if avg_density_total != 0 else np.zeros(num_bins)
    D_values = avg_density_refs / avg_density_total
    angles = angle_bins + delta_a_rad / 2
    return angles, D_values


def preprocess_image(image_path, threshold=127):
    """图像二值化处理"""
    image = Image.open(image_path).convert('L')
    binary_image = (np.array(image) < threshold).astype(np.uint8)
    return binary_image


# ====================== 分析函数模块 ======================
def analyze_directional_pattern(angles, D_values, r1, r2):
    """分析方向分布模式并生成文字描述"""
    # 将角度转换为度
    angles_deg = np.rad2deg(angles)
    D_array = np.array(D_values)

    # 计算基本统计量
    mean_D = np.mean(D_array)
    std_D = np.std(D_array)
    max_D = np.max(D_array)
    min_D = np.min(D_array)
    max_angle_idx = np.argmax(D_array)
    min_angle_idx = np.argmin(D_array)
    max_angle = angles_deg[max_angle_idx]
    min_angle = angles_deg[min_angle_idx]

    # 判断方向性强度
    directionality_strength = std_D / mean_D if mean_D > 0 else 0
    directionality_level = "强" if directionality_strength > 0.3 else "中等" if directionality_strength > 0.1 else "弱"

    # 查找主要方向
    significant_directions = []
    for i, (angle, d) in enumerate(zip(angles_deg, D_array)):
        if d > mean_D + 0.3 * std_D:  # 显著高于平均值的方向
            significant_directions.append((angle, d))

    # 排序主要方向
    significant_directions.sort(key=lambda x: -x[1])  # 按D值降序排列

    # 生成描述文本
    description = []

    description.append(f"方向分布函数分析：点分布的方向性特征（半径范围{r1}-{r2}像素）")
    description.append(f"- 整体方向性分布指标：{directionality_strength:.2f}")
    description.append(
        f"- D值范围：{min_D:.2f}至{max_D:.2f}")

    if significant_directions:
        direction_text = []
        for angle, d in significant_directions[:3]:  # 取前3个主要方向
            direction_text.append(f"{angle:.1f}°")
        description.append(f"- 方向性呈现：{', '.join(direction_text)}")
    else:
        description.append("- 未检测到明显的主要方向")


    # 检查是否存在对称模式
    if len(significant_directions) >= 2:
        has_opposite = False
        has_perpendicular = False
        for i in range(len(significant_directions)):
            for j in range(i + 1, len(significant_directions)):
                angle_diff = abs(significant_directions[i][0] - significant_directions[j][0]) % 180
                if abs(angle_diff - 90) < 15:  # 近似垂直
                    has_perpendicular = True
                if abs(angle_diff - 180) < 15 or abs(angle_diff) < 15:  # 近似相反
                    has_opposite = True

    return "\n".join(description)


# ====================== 绘制函数模块 ======================
def save_directional_data(angles, D_values, output_path, r1, r2, delta_a_deg):
    """保存数值数据"""
    with open(output_path, 'w') as f:
        f.write(f"# 方向分布函数数值\n")
        f.write(f"# R1 = {r1}, R2 = {r2}, 角度分段 = {delta_a_deg}度\n")
        f.write("# 角度(度)\tD值\n")
        for angle_rad, d_value in zip(angles, D_values):
            angle_deg = np.rad2deg(angle_rad)
            angle_deg = angle_deg if angle_deg < 360 else 0
            f.write(f"{angle_deg:.2f}\t{d_value:.4f}\n")
    logging.info(f"数值文件已保存: {output_path}")


def plot_polar_diagram(angles, D_values, delta_a_deg, r1, r2):
    """绘制极坐标图并返回图形对象"""
    plt.figure(figsize=(3, 3))
    ax = plt.subplot(111, projection='polar')
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1)
    ax.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
    ax.set_xticklabels(['0°', '45°', '90°', '135°', '180°', '225°', '270°', '315°'])
    ax.set_ylim(0, np.max(D_values) * 1.2)
    ax.set_yticks([0.5, 1.0, 1.5, 2.0])
    ax.set_rlabel_position(90)
    bar_width = np.deg2rad(delta_a_deg)
    for i, (angle, d) in enumerate(zip(angles, D_values)):
        color = (0.8, 0.8, 0.8) if i % 2 == 0 else (0.6, 0.6, 0.6)
        ax.bar(angle, d, width=bar_width, bottom=0.0, color=color, alpha=0.8)
    plt.grid(True, alpha=0.3)
    return plt.gcf()


# ====================== 处理函数模块 ======================
def process_single_image(image_path, output_folder, r1, r2, delta_a_deg, threshold=127):
    """处理单张图像，保存结果和分析"""
    try:
        image_name = image_path.stem
        logging.info(f"处理图像: {image_name}")
        binary_image = preprocess_image(image_path, threshold)
        angles, D_values = compute_D_function(binary_image, r1, r2, delta_a_deg)

        # 保存数值数据
        data_path = output_folder / f"{image_name}_D_values.txt"
        save_directional_data(angles, D_values, data_path, r1, r2, delta_a_deg)

        # 绘制并保存极坐标图
        fig = plot_polar_diagram(angles, D_values, delta_a_deg, r1, r2)
        plot_path = output_folder / f"{image_name}_polar.png"
        fig.savefig(plot_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        logging.info(f"极坐标图已保存: {plot_path}")

        # 分析并保存结果描述
        analysis_text = analyze_directional_pattern(angles, D_values, r1, r2)
        analysis_path = output_folder / f"{image_name}_analysis.txt"
        with open(analysis_path, 'w', encoding='utf-8') as f:
            f.write(analysis_text)
        logging.info(f"分析文本已保存: {analysis_path}")

        return True
    except Exception as e:
        logging.error(f"处理图像 {image_name} 时出错: {str(e)}")
        return False


def batch_process_images(image_folder, output_folder, r1, r2, delta_a_deg, threshold=128):
    """批量处理图像"""
    if r1 >= r2:
        raise ValueError("r1 必须小于 r2")
    if delta_a_deg <= 0 or delta_a_deg >= 360:
        raise ValueError("角度分段宽度必须在 (0, 360) 范围内")

    output_folder.mkdir(parents=True, exist_ok=True)
    logging.info(f"输出文件夹: {output_folder}")

    image_paths = [
        path for path in image_folder.iterdir()
        if path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    ]
    if not image_paths:
        logging.error("未找到任何图片，请检查文件夹路径和图片格式！")
        return

    success_count = 0
    for idx, path in enumerate(image_paths):
        if process_single_image(path, output_folder, r1, r2, delta_a_deg, threshold):
            success_count += 1
    logging.info(f"处理完成！成功: {success_count}/{len(image_paths)}")


# ====================== 主程序 ======================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("directional_analysis.log"),
            logging.StreamHandler()
        ]
    )
    logging.info("=== 方向分布函数分析程序启动 ===")
    image_folder = Path("F:\\学习课本\\函数\\基础\\ETS4")
    output_folder = Path("F:\\学习课本\\函数\\Statistics\\文字\\ETS4\\D")
    R1, R2 = 1, 20
    DELTA_A_DEG = 10
    THRESHOLD = 128

    try:
        batch_process_images(image_folder, output_folder, R1, R2, DELTA_A_DEG, THRESHOLD)
    except Exception as e:
        logging.exception(f"程序执行出错: {str(e)}")
    finally:
        logging.info("=== 方向分布函数分析程序结束 ===")