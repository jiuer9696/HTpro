import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ================ 预处理函数 ================
def preprocess_image(image_path, threshold=128, target_size=None):
    """预处理单张图像：加载、灰度转换、二值化、调整尺寸"""
    try:
        image = Image.open(image_path).convert('L')
        if target_size is not None:
            image = image.resize(target_size)
        img_array = np.array(image)
        binary_image = (img_array > threshold).astype(np.uint8) * 255
        return binary_image
    except Exception as e:
        print(f"图像预处理失败: {str(e)}")
        return None


# ================ 分析函数 ================
def calculate_rapsd(image, delta_rho=0.05, D=1):
    """计算单张图像的径向平均功率谱密度(RAPSD)"""
    if image is None or image.size == 0:
        raise ValueError("输入图像为空")
    if len(image.shape) != 2:
        raise ValueError(f"图像必须是二维数组，当前形状: {image.shape}")

    N = image.shape[0]
    F = np.fft.fftshift(np.fft.fft2(image.astype(float)))
    P = (np.abs(F) ** 2) / (N ** 2)  # 功率谱

    # 生成频率网格
    u = np.arange(N)
    v = np.arange(N)
    u_centered = (u - N // 2) / N
    v_centered = (v - N // 2) / N
    uu, vv = np.meshgrid(u_centered, v_centered)
    f_rho = np.sqrt(uu ** 2 + vv ** 2)

    # 频率分箱
    f_rho_max = np.sqrt(2) / 2 * (1 / D)
    bins = np.arange(0, f_rho_max + delta_rho / 2, delta_rho)
    P_rapsd = np.zeros(len(bins) - 1)
    counts = np.zeros(len(bins) - 1)

    # 计算每个分箱的平均功率
    for i in range(N):
        for j in range(N):
            if i == N // 2 and j == N // 2:
                continue  # 跳过直流分量
            bin_idx = np.digitize(f_rho[i, j], bins) - 1
            if 0 <= bin_idx < len(P_rapsd):
                P_rapsd[bin_idx] += P[i, j]
                counts[bin_idx] += 1

    # 归一化处理
    valid_bins = counts > 0
    P_rapsd[valid_bins] /= counts[valid_bins]
    P_rapsd /= np.mean(P_rapsd[valid_bins]) if np.any(valid_bins) else 1

    f_rho_bins = (bins[:-1] + bins[1:]) / 2
    return f_rho_bins, P_rapsd, counts


def analyze_rapsd(f_rho, P_rapsd, counts):
    """分析RAPSD结果，包含脉冲判断与噪声模式识别"""
    analysis = []
    valid_mask = (counts > 0) & (P_rapsd > 1e-6)  # 有效频率点（过滤低能量）
    if not np.any(valid_mask):
        return "无有效频谱数据", False

    f_valid = f_rho[valid_mask]
    p_valid = P_rapsd[valid_mask]
    total_energy = np.sum(p_valid)

    # 1. 频率成分占比分析
    low_mask = f_valid <= 0.2
    mid_mask = (f_valid > 0.2) & (f_valid <= 0.5)
    high_mask = f_valid > 0.5

    low_energy = np.sum(p_valid[low_mask]) if np.any(low_mask) else 0
    mid_energy = np.sum(p_valid[mid_mask]) if np.any(mid_mask) else 0
    high_energy = np.sum(p_valid[high_mask]) if np.any(high_mask) else 0

    low_ratio = (low_energy / total_energy) * 100 if total_energy > 0 else 0
    mid_ratio = (mid_energy / total_energy) * 100 if total_energy > 0 else 0
    high_ratio = (high_energy / total_energy) * 100 if total_energy > 0 else 0
    analysis.append("径向平均功率谱密度：信号能量在径向频率的分布比率")
    analysis.append(f"- 低频（≤0.2）：{low_ratio:.1f}%")
    analysis.append(f"- 中频（0.2~0.5）：{mid_ratio:.1f}%")
    analysis.append(f"- 高频（>0.5）：{high_ratio:.1f}%")

    # 2. 主导频率判断
    dom_freq_idx = np.argmax(p_valid)
    dom_freq = f_valid[dom_freq_idx]
    dom_energy = p_valid[dom_freq_idx]
    dom_band = "低频" if dom_freq <= 0.2 else "中频" if dom_freq <= 0.5 else "高频"
    analysis.append(f"- 峰值频率：{dom_freq:.3f}：{dom_energy / total_energy * 100:.1f}%")

    # 3. 脉冲式能量分布判断（优化逻辑）
    total_bins = len(valid_mask)
    non_zero_count = np.sum(valid_mask)
    non_zero_ratio = non_zero_count / total_bins

    peak_to_second_ratio = float('inf')
    if len(p_valid) >= 2:
        sorted_peaks = np.sort(p_valid)[::-1]
        peak_to_second_ratio = sorted_peaks[0] / sorted_peaks[1]

    significant_peak_mask = (P_rapsd > 2)  # 高于2倍白噪声基准
    significant_peak_count = np.sum(significant_peak_mask)
    significant_peak_ratio = significant_peak_count / total_bins

    dominant_ratio = dom_energy / total_energy if total_energy > 0 else 0

    # 多条件判断确保极端情况被识别
    is_pulse = (
        (significant_peak_ratio < 0.1 and peak_to_second_ratio > 5) or
        (non_zero_ratio < 0.15) or
        (dominant_ratio > 0.5)
    )
    # 4. 噪声模式识别（仅在有结果时显示）
    noise_patterns = []
    if high_ratio > low_ratio * 1.5 and high_ratio > mid_ratio * 1.2:
        noise_patterns.append("- 符合蓝噪声模式：高频能量占优，细节丰富")
    elif low_ratio > high_ratio * 1.5 and low_ratio > mid_ratio * 1.2:
        noise_patterns.append("- 符合绿噪声模式：低频能量占优，图像较平滑")
    elif abs(low_ratio - high_ratio) < 10 and abs(mid_ratio - high_ratio) < 10:
        noise_patterns.append("- 符合白噪声模式：全频段能量分布均匀")
        """
    elif mid_ratio > low_ratio * 1.2 and mid_ratio > high_ratio * 1.2:
        base_desc = "- 中频主导模式"
        if is_pulse:
            base_desc += "\n    纹理表现为特定频率点的强脉冲式重复"

        noise_patterns.append(base_desc)
"""
    return "\n".join(analysis), is_pulse



# ================ 绘图函数 ================
import matplotlib.pyplot as plt

def plot_rapsd(f_rho, P_rapsd):
    """绘制RAPSD图，显示为400×300像素大小"""
    # 计算得到 400×300 像素的 figsize 和 dpi 组合，这里选择 dpi=100
    plt.figure(figsize=(4, 3), dpi=100)
    plt.plot(f_rho, P_rapsd, 'b-o', markersize=4)
    plt.axhline(1, color='r', linestyle='--', label='白噪声基准')
    plt.axvline(0.2, color='gray', linestyle=':', label='低频/中频分界')
    plt.axvline(0.5, color='gray', linestyle=':', label='中频/高频分界')
    plt.xlabel("径向频率")
    plt.ylabel("RAPSD（归一化）")
    plt.ylim(0, 3)
    plt.xlim(0, 0.7071)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    return plt.gcf()

# ================ 主函数 ================
def process_single_image(image_path, output_folder, threshold=128, target_size=(100, 100),
                         delta_rho_rapsd=0.05, delta_rho_aniso=0.035):
    """处理单张图像，生成RAPSD分析及文字描述"""
    os.makedirs(output_folder, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # 预处理
    binary_image = preprocess_image(image_path, threshold, target_size)
    if binary_image is None:
        return False

    # 保存预处理图像
    preprocessed_path = os.path.join(output_folder, f"{base_name}_preprocessed.png")
    Image.fromarray(binary_image).save(preprocessed_path)

    try:
        # 计算RAPSD
        f_rho_rapsd, P_rapsd, counts = calculate_rapsd(binary_image, delta_rho_rapsd)

        # 绘制RAPSD图
        rapsd_fig = plot_rapsd(f_rho_rapsd, P_rapsd)
        rapsd_path = os.path.join(output_folder, f"{base_name}_rapsd.png")
        rapsd_fig.savefig(rapsd_path)
        plt.close(rapsd_fig)

        # ✅ 生成分析文本（修复写入报错）
        analysis_text, _ = analyze_rapsd(f_rho_rapsd, P_rapsd, counts)
        analysis_path = os.path.join(output_folder, f"{base_name}_analysis.txt")
        with open(analysis_path, 'w', encoding='utf-8') as f:
            f.write(f"图像: {base_name}\n\n径向平均功率谱密度分析结果:\n")
            f.write(analysis_text)
        print(f"分析结果已保存: {analysis_path}")

        return True
    except Exception as e:
        print(f"分析出错: {str(e)}")
        return False


def process_images_in_folder(input_folder, output_folder, **kwargs):
    """批量处理文件夹中的图像"""
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
    image_files = [f for f in os.listdir(input_folder)
                   if os.path.isfile(os.path.join(input_folder, f))
                   and os.path.splitext(f)[1].lower() in image_extensions]

    if not image_files:
        print(f"未找到图像文件 in {input_folder}")
        return

    success_count = 0
    for image_file in image_files:
        if process_single_image(os.path.join(input_folder, image_file), output_folder, **kwargs):
            success_count += 1
    print(f"处理完成: {success_count}/{len(image_files)} 成功")
    return success_count > 0


if __name__ == "__main__":
    input_path = "F:\\B"
    output_folder = "F:\\B\\P2"
    process_images_in_folder(
        input_path, output_folder,
        threshold=128,
        target_size=(100, 100),
        delta_rho_rapsd=0.05
    )