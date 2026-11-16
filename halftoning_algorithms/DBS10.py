import numpy as np
import scipy.signal
import os
from PIL import Image
import numpy as np
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']  # 替换为你系统中的中文字体名称
rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题

def nasanen_hvs_filter_precise(size=31, fc=0.2, gamma=0.7):
    """
    更准确的Näsänen人眼感知HVS滤波器
    - size: 滤波器大小 (奇数)
    - fc: 截止频率 (归一化频率，通常0.1 ~ 0.3之间)
    - gamma: 频率加权指数 (通常在0.5~1.0)
    """
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    rr = np.sqrt(xx ** 2 + yy ** 2)

    # 归一化频率 (中心到边缘为0~1)
    f = rr / (size // 2 + 1e-8)

    # 定义滤波器：频率加权 * 高斯衰减
    hvs = (f ** gamma) * np.exp(-(f / fc) ** 2)

    # 中心点设为最大（避免0^gamma导致的问题）
    hvs[size // 2, size // 2] = np.max(hvs)

    # 最后归一化到总能量为1
    hvs /= np.sum(hvs)

    return hvs

# --- 2. HVS感知图像 ---
def perceive(img, hvs_filter):
    """通过HVS滤波器模拟人眼感知图像"""
    return scipy.signal.convolve2d(img, hvs_filter, mode='same', boundary='symm')


# --- 3. 计算感知误差能量 ---
def compute_error_energy(f_perceived, g_perceived):
    """计算感知误差能量"""
    e = f_perceived - g_perceived
    return np.sum(e ** 2)
def floyd_steinberg_dither(img):
    """经典蛇形扫描 + 方向同步反转的 Floyd-Steinberg 误差扩散"""
    img = img.copy()
    h, w = img.shape
    for y in range(h):
        if y % 2 == 0:
            x_range = range(w)
            dir = 1  # 从左到右
        else:
            x_range = range(w-1, -1, -1)
            dir = -1  # 从右到左

        for x in x_range:
            old_pixel = img[y, x]
            new_pixel = np.round(old_pixel)
            img[y, x] = new_pixel
            quant_error = old_pixel - new_pixel

            if 0 <= x + dir < w:
                img[y, x + dir] += quant_error * 7/16
            if y + 1 < h:
                img[y + 1, x] += quant_error * 5/16
                if 0 <= x - dir < w:
                    img[y + 1, x - dir] += quant_error * 3/16
                if 0 <= x + dir < w:
                    img[y + 1, x + dir] += quant_error * 1/16

    return np.clip(img, 0, 1).astype(np.uint8)

def efficient_dbs_with_padding(f, init_g, hvs_filter, max_iters=10):
    """带补零处理+支持像素交换的高效版DBS算法（统计误差时只考虑有效区域）"""
    g = init_g.copy()
    f_perceived = perceive(f, hvs_filter)

    pad_h = hvs_filter.shape[0] // 2
    pad_w = hvs_filter.shape[1] // 2
    g_padded = np.pad(g, ((pad_h, pad_h), (pad_w, pad_w)), mode='symmetric')
    f_perceived_padded = np.pad(f_perceived, ((pad_h, pad_h), (pad_w, pad_w)), mode='symmetric')
    g_perceived = perceive(g_padded, hvs_filter)
    e = f_perceived_padded - g_perceived

    psf_h, psf_w = hvs_filter.shape
    height, width = g.shape

    neighbor_offsets = [(-1, -1), (-1, 0), (-1, 1),
                        (0, -1), (0, 1),
                        (1, -1), (1, 0), (1, 1)]  # 8邻域偏移量

    for iteration in range(max_iters):
        print(f"第 {iteration + 1} 次迭代")
        changed = False

        # 只统计中心有效区域误差
        valid_e = e[pad_h:-pad_h, pad_w:-pad_w]
        error_before = np.sum(valid_e ** 2)
        print(f"  迭代前误差: {error_before}")

        for i in range(height):
            for j in range(width):
                pixel = g[i, j]
                delta_g = 1 - 2 * pixel  # 翻转0<->1

                ip = i + pad_h
                jp = j + pad_w

                i_start = ip - pad_h
                i_end = ip + pad_h + 1
                j_start = jp - pad_w
                j_end = jp + pad_w + 1

                delta_region = hvs_filter
                e_region = e[i_start:i_end, j_start:j_end]

                # 1. 尝试单点翻转
                delta = delta_g * delta_region
                local_error_change = -2 * np.sum(delta * e_region) + np.sum(delta ** 2)

                if local_error_change < 0:
                    g[i, j] = 1 - g[i, j]
                    g_padded[ip, jp] = g[i, j]
                    e[i_start:i_end, j_start:j_end] -= delta
                    changed = True
                    continue  # 翻转成功就不再尝试交换

                # 2. 尝试8邻域像素交换
                for di, dj in neighbor_offsets:
                    ni, nj = i + di, j + dj

                    if 0 <= ni < height and 0 <= nj < width:
                        if g[i, j] != g[ni, nj]:  # 只有不同才有交换意义
                            nip = ni + pad_h
                            njp = nj + pad_w

                            ni_start = nip - pad_h
                            ni_end = nip + pad_h + 1
                            nj_start = njp - pad_w
                            nj_end = njp + pad_w + 1

                            delta1 = (1 - 2 * g[i, j]) * delta_region
                            delta2 = (1 - 2 * g[ni, nj]) * delta_region
                            delta_total = delta1 + delta2

                            e_region_ij = e[i_start:i_end, j_start:j_end]
                            e_region_ninj = e[ni_start:ni_end, nj_start:nj_end]

                            local_error_change_swap = (-2 * np.sum(delta1 * e_region_ij)
                                                       - 2 * np.sum(delta2 * e_region_ninj)
                                                       + np.sum(delta1 ** 2)
                                                       + np.sum(delta2 ** 2))

                            if local_error_change_swap < 0:
                                g[i, j], g[ni, nj] = g[ni, nj], g[i, j]
                                g_padded[ip, jp] = g[i, j]
                                g_padded[nip, njp] = g[ni, nj]

                                e[i_start:i_end, j_start:j_end] -= delta1
                                e[ni_start:ni_end, nj_start:nj_end] -= delta2
                                changed = True
                                break  # 交换成功后不再尝试其他邻居

        # 更新误差后再计算有效区域误差
        valid_e = e[pad_h:-pad_h, pad_w:-pad_w]
        error_after = np.sum(valid_e ** 2)
        print(f"  迭代后误差: {error_after}")
        print(f"  误差减少量: {error_before - error_after}")

        if not changed:
            print("无像素变化，算法收敛。")
            break

    return g


# --- 主程序入口 ---
if __name__ == "__main__":
    input_folder = 'F:/学习课本/函数/基础/jichu'         # 输入图像所在文件夹
    output_folder = 'F:/学习课本/函数/基础/DBS3'       # 输出图像保存文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 生成HVS滤波器（只需一次）
    hvs_filter = nasanen_hvs_filter_precise(size=31)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path).convert('L')  # 转为灰度图
            f = np.array(img) / 255.0  # 归一化

            # 有序抖动生成初始图
            init_g = floyd_steinberg_dither(f)

            # DBS优化
            g_final = efficient_dbs_with_padding(f, init_g, hvs_filter, max_iters=10)

            # 构造保存路径
            base_name = os.path.splitext(filename)[0]
            init_save_path = os.path.join(output_folder, f"{base_name}_init.png")
            final_save_path = os.path.join(output_folder, f"{base_name}_dbs.png")

            # 保存图像
            Image.fromarray((init_g * 255).astype(np.uint8)).save(init_save_path)
            Image.fromarray((g_final * 255).astype(np.uint8)).save(final_save_path)

            print(f"处理完成: {filename}，已保存为 {init_save_path} 和 {final_save_path}")
