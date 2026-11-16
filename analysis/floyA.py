import numpy as np
from PIL import Image
import os
import io
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ====================== 核心分析函数：生成各向异性文字描述 ======================
def analyze_anisotropy(f_rho, anisotropy):
    """
    分析各向异性结果并生成文字描述（按指定格式输出）
    """
    if len(f_rho) == 0 or len(anisotropy) == 0:
        return "各向异性分析：\n- 未获取有效各向异性数据，无法分析。"

    # 转换为分贝（更直观）
    aniso_db = 10 * np.log10(np.maximum(anisotropy, 1e-12))
    mean_aniso_db = np.mean(aniso_db)

    # 划分频率段（对应不同结构尺度）
    low_mask = f_rho <= 0.2  # 低频：大尺度结构
    mid_mask = (f_rho > 0.2) & (f_rho <= 0.5)  # 中频：中等尺度
    high_mask = f_rho > 0.5  # 高频：小尺度细节

    # 各频段平均各向异性及描述
    low_mean = np.mean(aniso_db[low_mask]) if np.any(low_mask) else -np.inf
    low_desc = "极弱" if low_mean < -30 else "较弱" if low_mean < -15 else "中等" if low_mean < -5 else "较强"

    mid_mean = np.mean(aniso_db[mid_mask]) if np.any(mid_mask) else -np.inf
    mid_desc = "极弱" if mid_mean < -30 else "较弱" if mid_mean < -15 else "中等" if mid_mean < -5 else "较强"

    high_mean = np.mean(aniso_db[high_mask]) if np.any(high_mask) else -np.inf
    high_desc = "极弱" if high_mean < -30 else "较弱" if high_mean < -15 else "中等" if high_mean < -5 else "较强"

    # 显著方向性频率（> -10dB）
    significant_mask = aniso_db > -10
    significant_freqs = f_rho[significant_mask]
    significant_freqs_rounded = [round(f, 3) for f in significant_freqs]
    """
    # 整体方向性描述
    if mean_aniso_db < -30:
        overall_desc = f"整体各向异性极弱（平均{mean_aniso_db:.1f} dB），图像接近各向同性（结构在所有方向分布均匀）。"
    elif mean_aniso_db < -15:
        overall_desc = f"整体各向异性较弱（平均{mean_aniso_db:.1f} dB），方向性特征不明显，可能为均匀纹理或随机分布。"
    elif mean_aniso_db < -5:
        overall_desc = f"整体各向异性中等（平均{mean_aniso_db:.1f} dB），存在一定方向性偏好，但不显著。"
    else:
        overall_desc = f"整体各向异性较强（平均{mean_aniso_db:.1f} dB），图像具有明显的定向结构（如条纹、网格）。"
    """
    # 构建分析文本（严格按指定格式）
    analysis = ["各向异性分析:径向频率的方向性功率分布"]

    # 1. 方向性随频率变化特征
    if np.any(low_mask):
        analysis.append(f"- 低频段（≤0.2）：{low_mean:.1f} dB({low_desc})")
    if np.any(mid_mask):
        analysis.append(f"- 中频段（0.2~0.5）：{mid_mean:.1f} dB({mid_desc})。")
    if np.any(high_mask):
        analysis.append(f"- 高频段（>0.5）：{high_mean:.1f} dB({high_desc})")

    # 2. 显著方向性频率
    if len(significant_freqs) > 0:
        freqs_str = "、".join([f"频率{f}" for f in significant_freqs_rounded])
        analysis.append(f"- {freqs_str}，> -10 dB方向性较明显")
    else:
        analysis.append("- 未有明显方向性特征（所有频段≤-10 dB）")

    # 3. 整体各向异性描述
    #analysis.append(f"- {overall_desc}")

    return "\n".join(analysis)


# ====================== 原有函数：预处理、计算、可视化（保持不变） ======================
def preprocess_image(image_path_or_array, threshold=None, crop_center=True):
    """统一图像预处理：灰度转换、居中裁剪、阈值二值化"""
    if isinstance(image_path_or_array, str):
        image = Image.open(image_path_or_array).convert('L')
    else:
        image = Image.fromarray(image_path_or_array).convert('L')

    np_img = np.array(image)
    h, w = np_img.shape

    if crop_center:
        side = min(h, w)
        top = (h - side) // 2
        left = (w - side) // 2
        cropped = np_img[top:top + side, left:left + side]
    else:
        cropped = np_img

    if threshold is None:
        threshold = threshold_otsu(cropped)
    binary_image = (cropped > threshold).astype(np.uint8)
    return binary_image


def calculate_rapsd_anisotropy(image, delta_rho=0.035, f_rho_min=1e-6, D=1):
    """计算单张图像的各向异性"""
    if image is None or image.size == 0:
        raise ValueError("输入图像为空或处理失败")
    if len(image.shape) != 2:
        raise ValueError(f"图像必须是二维数组，当前形状: {image.shape}")

    N = image.shape[0]
    F = np.fft.fftshift(np.fft.fft2(image.astype(float)))
    P_avg = (np.abs(F) ** 2) / (N ** 2)

    u = np.arange(N)
    v = np.arange(N)
    u_centered = (u - N // 2) / N
    v_centered = (v - N // 2) / N
    uu, vv = np.meshgrid(u_centered, v_centered)
    f_rho = np.sqrt(uu ** 2 + vv ** 2)

    f_rho_max = np.sqrt(2) / 2 * (1 / D)
    bins = np.arange(0, f_rho_max + delta_rho / 2, delta_rho)
    num_bins = len(bins) - 1

    P_rapsd = np.zeros(num_bins)
    Aniso = np.zeros(num_bins)
    counts = np.zeros(num_bins, dtype=int)

    for i in range(N):
        for j in range(N):
            if f_rho[i, j] < f_rho_min or (i == N // 2 and j == N // 2):
                continue
            bin_idx = np.digitize(f_rho[i, j], bins) - 1
            if 0 <= bin_idx < num_bins:
                P_rapsd[bin_idx] += P_avg[i, j]
                counts[bin_idx] += 1

    valid_bins = counts > 0
    P_rapsd[valid_bins] /= counts[valid_bins]
    P_rapsd /= np.mean(P_rapsd[valid_bins]) if np.any(valid_bins) else 1

    for k in range(num_bins):
        if counts[k] > 1 and bins[k] >= f_rho_min:
            mask = np.digitize(f_rho, bins) - 1 == k
            values = P_avg[mask]
            mean_val = P_rapsd[k]
            variance = np.var(values, ddof=1)
            Aniso[k] = variance / (mean_val ** 2 + 1e-12)

    f_rho_bins = (bins[:-1] + bins[1:]) / 2
    valid = f_rho_bins >= f_rho_min
    return f_rho_bins[valid], Aniso[valid]


def plot_anisotropy_db(f_rho, anisotropy, title="各向异性频谱 (dB)", save_path=None):
    """绘制各向异性频谱图"""
    Aniso_db = 10 * np.log10(np.maximum(anisotropy, 1e-12))
    plt.figure(figsize=(4, 3), dpi=300)
    plt.plot(f_rho, Aniso_db, 'b-o', markersize=4, label='各向异性 (dB)')
    plt.axhline(-10, color='r', linestyle='--', label='显著方向性阈值 (-10 dB)')
    plt.xlabel("归一化径向频率 $f_\\rho$")
    plt.ylabel("各向异性 $A(f_\\rho)$ [dB]")
    plt.xlim(0, 0.7071)
    plt.ylim(-50, 20)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    return plt.gcf()


# ====================== 批量处理与结果保存（保持不变） ======================
def process_images_in_folder(folder_path, output_folder=None):
    """处理文件夹中的所有图像，生成分析结果"""
    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"已创建输出文件夹: {output_folder}")

    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
    image_files = [
        f for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
           and os.path.splitext(f)[1].lower() in image_extensions
    ]

    if not image_files:
        print(f"在文件夹 {folder_path} 中未找到图像文件")
        return

    all_anisotropies = []
    f_rho_bins = None

    print(f"开始处理，共发现 {len(image_files)} 张图像...")
    for idx, image_file in enumerate(image_files, 1):
        try:
            image_path = os.path.join(folder_path, image_file)
            print(f"处理第 {idx}/{len(image_files)} 张: {image_file}")

            binary_image = preprocess_image(image_path)
            f_rho, Aniso = calculate_rapsd_anisotropy(binary_image, delta_rho=0.035)
            all_anisotropies.append(Aniso)
            f_rho_bins = f_rho

            # 生成并保存分析文本
            analysis_text = analyze_anisotropy(f_rho, Aniso)
            if output_folder:
                txt_path = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}_analysis.txt")
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(f"图像: {image_file}\n\n{analysis_text}")
                print(f"已保存分析文本: {txt_path}")

            # 保存可视化图
            if output_folder:
                plot_path = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}_anisotropy.png")
                plot_anisotropy_db(f_rho, Aniso, save_path=plot_path)
                print(f"已保存可视化图: {plot_path}")

        except Exception as e:
            print(f"处理图像 {image_file} 时出错: {str(e)}，已跳过")

    # 计算平均各向异性并分析
    if all_anisotropies and f_rho_bins is not None:
        avg_anisotropy = np.mean(all_anisotropies, axis=0)
        avg_analysis = analyze_anisotropy(f_rho_bins, avg_anisotropy)
        if output_folder:
            avg_txt_path = os.path.join(output_folder, "average_analysis.txt")
            with open(avg_txt_path, 'w', encoding='utf-8') as f:
                f.write("所有图像的平均各向异性分析:\n\n" + avg_analysis)
            print(f"已保存平均分析文本: {avg_txt_path}")

            avg_plot_path = os.path.join(output_folder, "average_anisotropy.png")
            plot_anisotropy_db(f_rho_bins, avg_anisotropy, "所有图像的平均各向异性频谱", avg_plot_path)
            print(f"已保存平均可视化图: {avg_plot_path}")

    print(f"\n处理完成！共处理 {len(image_files)} 张图像，成功 {len(all_anisotropies)} 张")


if __name__ == "__main__":
    input_folder = "F:\\学习课本\\函数\\基础\\误差扩散"
    output_folder = "F:\\学习课本\\函数\\Statistics\\文字\\1\\误差扩散-A"

    if not os.path.exists(input_folder):
        print(f"错误：输入文件夹不存在 - {input_folder}")
    else:
        process_images_in_folder(input_folder, output_folder)