import os
from PIL import Image
import numpy as np

def floyd_steinberg_dithering(image: Image.Image) -> Image.Image:
    """
    Floyd-Steinberg 误差扩散算法的单通道复现版
    输入:
        image: PIL.Image, 单通道灰度图
    输出:
        halftoned: PIL.Image, 单通道加网图
    """
    assert image.mode == "L", "输入必须是灰度图"

    # 将图像转换为 numpy 数组
    image_np = np.array(image, dtype=np.float32)
    rows, cols = image_np.shape
    out_float = 255.0 - image_np  # 反色，符合 C++ 实现

    for y in range(0, rows, 2):
        # 奇数行左到右
        for x in range(cols):
            old_val = out_float[y, x]
            if old_val >= 127:
                new_val = 255
                image_np[y, x] = 0
            else:
                new_val = 0
                image_np[y, x] = 255
            error = old_val - new_val

            if x + 1 < cols:
                out_float[y, x + 1] += error * 7 / 16
            if y + 1 < rows:
                if x - 1 >= 0:
                    out_float[y + 1, x - 1] += error * 3 / 16
                out_float[y + 1, x] += error * 5 / 16
                if x + 1 < cols:
                    out_float[y + 1, x + 1] += error * 1 / 16

        # 偶数行右到左
        if y + 1 < rows:
            for x in range(cols - 1, -1, -1):
                old_val = out_float[y + 1, x]
                if old_val >= 127:
                    new_val = 255
                    image_np[y + 1, x] = 0
                else:
                    new_val = 0
                    image_np[y + 1, x] = 255
                error = old_val - new_val

                if x - 1 >= 0:
                    out_float[y + 1, x - 1] += error * 7 / 16
                if y + 2 < rows:
                    if x + 1 < cols:
                        out_float[y + 2, x + 1] += error * 3 / 16
                    out_float[y + 2, x] += error * 5 / 16
                    if x - 1 >= 0:
                        out_float[y + 2, x - 1] += error * 1 / 16

    # 将处理后的 numpy 数组转换回 PIL.Image
    result_image = Image.fromarray(image_np.astype(np.uint8))
    return result_image


def process_images_in_folder(input_folder: str, output_folder: str):
    """
    遍历输入文件夹中的所有图片，并对它们应用误差扩散算法。
    :param input_folder: 输入图片文件夹路径
    :param output_folder: 输出加网图像文件夹路径
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            print(f"Processing {filename}...")
            try:
                img_path = os.path.join(input_folder, filename)
                img = Image.open(img_path).convert("L")  # 转灰度

                result_img = floyd_steinberg_dithering(img)  # 直接传入 PIL.Image 对象
                save_path = os.path.join(output_folder, filename)
                result_img.save(save_path)
            except Exception as e:
                print(f"Failed to process {filename}: {e}")


if __name__ == "__main__":
    # 这里写你自己的输入输出文件夹路径
    input_dir = "F:/学习课本/函数/基础/jichu"
    output_dir = "F:/学习课本/函数/基础/误差扩散3"
    process_images_in_folder(input_dir, output_dir)