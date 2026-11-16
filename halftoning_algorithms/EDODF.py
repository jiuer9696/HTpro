import numpy as np
from PIL import Image
import os

import numpy as np
from PIL import Image
import os


def edodf_dithering(image, h=1.0, serpentine=False):
    """
    实现 EDODF 算法，接收已打开的图像对象

    参数:
        image (PIL.Image): 输入的灰度图像对象
        h (float): 反馈参数
        serpentine (bool): 是否使用蛇形扫描
    """
    # 确保输入是灰度图像
    if image.mode != 'L':
        image = image.convert('L')

    width, height = image.size
    pixels = np.array(image, dtype=float)

    # 初始化输出图像
    output_pixels = np.zeros((height, width), dtype=np.uint8)

    # 定义误差扩散滤波器
    stucki_weights = np.array([
        [0, 0, 0, 8, 4],
        [2, 4, 8, 4, 2],
        [1, 2, 4, 2, 1]
    ]) / 42.0

    floyd_steinberg_weights = np.array([7, 3, 5, 1]) / 16.0

    # 遍历图像像素
    for y in range(height):
        x_range = range(width) if not serpentine or y % 2 == 0 else reversed(range(width))

        for x in x_range:
            input_value = pixels[y, x]
            feedback = 0

            # 计算反馈项（边界处理）
            if x >= 1 and y >= 1:
                feedback += h * floyd_steinberg_weights[0] * output_pixels[y, x - 1]
                feedback += h * floyd_steinberg_weights[1] * output_pixels[y - 1, x]
                feedback += h * floyd_steinberg_weights[2] * output_pixels[y - 1, x - 1]
                if x + 1 < width:
                    feedback += h * floyd_steinberg_weights[3] * output_pixels[y - 1, x + 1]
            elif x >= 1 and y == 0:
                feedback += h * floyd_steinberg_weights[0] * output_pixels[y, x - 1]
            elif x == 0 and y >= 1:
                feedback += h * floyd_steinberg_weights[1] * output_pixels[y - 1, x]
                if x + 1 < width:
                    feedback += h * floyd_steinberg_weights[3] * output_pixels[y - 1, x + 1]

            # 二值化
            if (input_value + feedback) >= 127:
                output_pixels[y, x] = 255
            else:
                output_pixels[y, x] = 0

            # 误差扩散
            quant_error = input_value - output_pixels[y, x]
            for m in range(3):
                for n in range(-2, 3):
                    if 0 <= x + n < width and 0 <= y + m < height:
                        col_idx = n + 2
                        if 0 <= col_idx < stucki_weights.shape[1]:
                            pixels[y + m, x + n] += quant_error * stucki_weights[m, col_idx]

    # 返回处理后的图像对象
    return Image.fromarray(output_pixels, mode='L')

def process_images_in_folder(input_folder, output_folder, h=1.0, serpentine=False):
    """
    对指定文件夹下的所有图片进行加网处理并保存处理后的图片。

    参数:
        input_folder (str): 输入图片文件夹路径
        output_folder (str): 输出处理后图片文件夹路径
        h (float): 反馈参数
        serpentine (bool): 是否使用蛇形扫描
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            # 调用edodf_dithering并获取结果图像
            result_image = edodf_dithering(input_path, h=h, serpentine=serpentine)
            # 保存结果图像到指定位置
            result_image.save(output_path)
            print(f"Processed and saved: {output_path}")


if __name__ == "__main__":
    input_folder = 'F:/学习课本/函数/基础/jichu'
    output_folder = 'F:/学习课本/函数/基础/edodf'

    process_images_in_folder(
        input_folder=input_folder,
        output_folder=output_folder,
        h=0.5,
        serpentine=False
    )