import os
from PIL import Image
import numpy as np

# 定义4x4 Bayer抖动矩阵
bayer_matrix = np.array([[0, 8, 2, 10],
                         [12, 4, 14, 6],
                         [3, 11, 1, 9],
                         [15, 7, 13, 5]])
bayer_matrix = bayer_matrix / 16.0  # 归一化


def ordered_dither(image):
    """
    使用4x4 Bayer抖动矩阵进行有序抖动二值化处理。

    参数:
    image (PIL.Image): 输入图像

    返回:
    PIL.Image: 二值化后的图像
    """
    img_array = np.array(image.convert('L'), dtype=np.float32) / 255.0  # 转换为灰度并归一化到[0,1]
    height, width = img_array.shape

    result_img = np.zeros((height, width), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            threshold = bayer_matrix[y % 4, x % 4]
            if img_array[y, x] > threshold:
                result_img[y, x] = 255  # 设置为目标颜色（白色）
            else:
                result_img[y, x] = 0  # 设置为背景颜色（黑色）

    return Image.fromarray(result_img)


def process_images_in_folder(input_folder, output_folder):
    """
    处理指定文件夹中的所有图像文件，并将结果保存到输出文件夹。

    参数:
    input_folder (str): 输入图像文件夹路径
    output_folder (str): 输出图像文件夹路径
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):  # 检查是否是图片文件
            print(f"Processing {filename}...")
            image_path = os.path.join(input_folder, filename)
            try:
                with Image.open(image_path) as img:
                    output_img = ordered_dither(img)
                    output_img.save(os.path.join(output_folder, filename))
            except Exception as e:
                print(f"Failed to process {filename}: {e}")


if __name__ == "__main__":
    # 输入图片文件夹路径
    input_dir = "F:/学习课本/函数/Statistics/jichu"
    # 输出图片保存路径
    output_dir = "F:/学习课本/函数/基础/youxu"

    # 调用函数处理文件夹中的所有图像
    process_images_in_folder(input_dir, output_dir)