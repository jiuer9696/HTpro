import numpy as np
from scipy.ndimage import convolve

def generate_gaussian_kernel(size, sigma):
    """生成高斯滤波器核"""
    ax = np.arange(-size // 2 + 1, size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return kernel / np.sum(kernel)

def pair_correlation_shaping_function(r, r_c, lambda_g, lambda_decay):
    """
    修改后的对相关整形函数 R_bar(r)
    :param r: 像素之间的距离
    :param r_c: 簇的半径
    :param lambda_g: 簇间的平均距离
    :param lambda_decay: 指数衰减常数
    :return: R_bar(r) 的值
    """
    if r < r_c:
        return 1 - r / r_c  # 从 1 降到 0
    elif r < lambda_g:
        return (r - r_c) / (lambda_g - r_c)  # 从 0 上升到 1
    else:
        return 0.5 + 0.5 * np.exp(-(r - lambda_g) / lambda_decay)  # 逐渐下降到 0.5 并保持不变

def bippcca(size, gray_level, r_c, lambda_g, sigma1, sigma2, A, B, lambda_decay, initial_pattern=None, max_iter=10000):
    """
    BIPPCCA算法生成绿噪声抖动模式（支持堆叠约束）
    :param size: 图像大小（size x size）
    :param gray_level: 目标灰度级别（0到15）
    :param r_c: 簇的半径
    :param lambda_g: 簇间的平均距离
    :param sigma1: 高斯函数的标准差（簇内部）
    :param sigma2: 高斯函数的标准差（簇间）
    :param A: 权重系数（簇间高斯函数）
    :param B: 权重系数（指数衰减函数）
    :param lambda_decay: 指数衰减常数
    :param initial_pattern: 初始二值抖动模式（用于堆叠约束）
    :param max_iter: 最大迭代次数
    :return: 生成的二值抖动模式
    """
    height, width = size, size
    total_pixels = height * width
    target_minority_pixels = gray_level  # 直接等于灰度级 g

    # 初始化二值图像
    if initial_pattern is not None:
        Ig = initial_pattern.copy()  # 完全继承前一个灰度级的白色像素
    else:
        Ig = np.zeros((height, width), dtype=int)  # 初始全黑

    U = np.random.rand(height, width)  # 概率矩阵，初始为随机数

    # 生成高斯低通滤波器
    kernel_size = int(2 * r_c) + 1
    H_lp = generate_gaussian_kernel(kernel_size, r_c / 2)

    # 迭代生成抖动模式
    for iteration in range(max_iter):
        # 计算浓度矩阵 CM
        Ig_float = Ig.astype(float)
        concentration = convolve(Ig_float, H_lp, mode='wrap')

        # 归一化浓度矩阵
        concentration = (concentration - np.min(concentration)) / (np.max(concentration) - np.min(concentration))
        CM = 1 - concentration  # 浓度越低，转换概率越高

        # 计算调整后的概率矩阵
        adjusted_U = U * CM

        # 找到概率最大的黑色像素
        flat_adjusted_U = adjusted_U.flatten()
        candidate_indices = np.where(Ig.flatten() == 0)[0]  # 只考虑黑色像素
        if len(candidate_indices) == 0:
            break  # 如果没有黑色像素，终止迭代
        max_index = candidate_indices[np.argmax(flat_adjusted_U[candidate_indices])]
        row, col = np.unravel_index(max_index, (height, width))

        # 将该像素转换为白色像素
        Ig[row, col] = 1

        # 更新概率矩阵
        for i in range(height):
            for j in range(width):
                r = np.sqrt((i - row) ** 2 + (j - col) ** 2)
                U[i, j] *= pair_correlation_shaping_function(r, r_c, lambda_g, lambda_decay)

        # 检查是否达到目标灰度级别
        if np.sum(Ig) >= target_minority_pixels:
            break

    return Ig

def generate_gray_level_sequence(num_levels):
    """
    生成分阶段的灰度级顺序
    :param num_levels: 灰度级数量
    :return: 灰度级顺序列表
    """
    return np.linspace(0, 1, num_levels).tolist()

def construct_threshold_matrix(size=4):
    """
    构造 16*16 阈值矩阵（保证堆叠约束）
    """
    GNM = np.zeros((size, size), dtype=int)  # 初始化阈值矩阵
    I_g_list = []  # 存储所有灰度级的二值抖动模式

    # 设置参数
    r_c = 5  # 簇的半径
    lambda_g = 4  # 簇间的平均距离
    sigma1 = 1  # 高斯函数的标准差（簇内部）
    sigma2 = 2  # 高斯函数的标准差（簇间）
    A = 0.5  # 权重系数（簇间高斯函数）
    B = 0.5  # 权重系数（指数衰减函数）
    lambda_decay = 2  # 指数衰减常数

    # 生成灰度级顺序
    gray_level_sequence = generate_gray_level_sequence(256)

    # 生成 64 个灰度级的二值抖动模式（保证堆叠约束）
    I_g_list.append(np.zeros((size, size), dtype=int))  # 初始全黑
    for g in range(1, 16):
        gray_level = gray_level_sequence[g]  # 使用分阶段生成的灰度级顺序
        I_g = bippcca(size, g, r_c, lambda_g, sigma1, sigma2, A, B, lambda_decay, initial_pattern=I_g_list[g-1])
        I_g_list.append(I_g)

    # 构造阈值矩阵
    for i in range(size):
        for j in range(size):
            for g in range(16):
                if I_g_list[g][i, j] == 1:
                    GNM[i, j] = g
                    break

    return GNM, I_g_list

# 生成并输出阈值矩阵
threshold_matrix, I_g_list = construct_threshold_matrix()
print("4*4 阈值矩阵：")
print(threshold_matrix)

# 验证堆叠约束
for g in range(1, 16):
    if not np.all(I_g_list[g-1] <= I_g_list[g]):
        print(f"堆叠约束未满足：灰度级 {g-1} 和 {g}")
    else:
        print(f"堆叠约束满足：灰度级 {g-1} 和 {g}")