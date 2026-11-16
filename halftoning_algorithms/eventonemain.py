import cv2
import numpy as np
import os
import ctypes
from PIL import Image  # 导入PIL的Image模块
from .eventone import EvenTone


# 模拟C++固定宽度整数类型
int32 = ctypes.c_int32
uint8 = ctypes.c_uint8
uint16 = ctypes.c_uint16
uint32 = ctypes.c_uint32


def array_to_mat_gray(gray_array, height, width):
    """将字节数组转换为灰度图（保持内存连续性）"""
    gray_np = np.frombuffer(gray_array, dtype=np.uint8)
    gray_np = gray_np.reshape((height, width))
    return gray_np.copy()  # 确保内存连续


def mat_to_array_gray(input_mat):
    """将OpenCV Mat/NumPy数组转换为字节数组（修复连续性判断）"""
    # 对NumPy数组判断连续性（使用np.isfortran判断是否为Fortran顺序，C顺序即为连续）
    if not input_mat.flags.contiguous:
        input_mat = input_mat.copy(order='C')  # 转为C顺序连续数组
    return bytearray(input_mat.data)  # 直接复制底层数据


def perform_halftoning(in_image, xres, yres, bits_num):
    """执行半色调处理（修复行数据提取逻辑），适配PIL.Image输入"""
    # 判断输入是否为PIL.Image类型，若是则转换为numpy数组
    if isinstance(in_image, Image.Image):
        in_image = np.array(in_image)

    # 转换为灰度图
    if len(in_image.shape) == 3 and in_image.shape[2] == 3:
        gray_image = cv2.cvtColor(in_image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = in_image.copy()
        if len(gray_image.shape) == 3:
            gray_image = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)

    height, width = gray_image.shape[:2]
    bits_mask = (1 << bits_num) - 1

    # 初始化EvenTone处理器
    halftoner = EvenTone(
        float(xres), float(yres),
        int32(width).value,
        uint8(bits_num).value
    )

    # 存储处理结果（使用NumPy数组确保内存连续性）
    data = np.zeros((height, width), dtype=np.uint8)

    for i in range(height):
        # 提取当前行（修复reshape逻辑，确保是二维数组）
        row_ptr = gray_image[i:i + 1, :]  # 保持二维结构 (1, width)
        row_array = mat_to_array_gray(row_ptr)  # 转换为字节数组

        # 执行单行半色调处理
        halftone_line = halftoner.dither(row_array, width, False, False)

        # 解析半色调结果（严格复刻C++位运算）
        j = 0
        while j < width:
            byte_idx = (j * bits_num) // 8
            if byte_idx >= len(halftone_line):
                break

            byte_val = uint8(halftone_line[byte_idx]).value
            bit_offset = 8 - bits_num - ((j * bits_num) % 8)
            if bit_offset < 0:
                bit_offset += 8

            pixel_val = (byte_val >> bit_offset) & bits_mask
            data[i, j] = uint8((pixel_val * 255) // bits_mask).value
            j += 1

    # 若需要返回PIL.Image类型，可在此转换
    return data
    # return Image.fromarray(data)  # 如需返回PIL.Image，解开此注释


def main():
    # 直接指定图像文件名（无需命令行参数）
    # 在这里修改为你的图像文件名
    image_name = "PureColors_MS.tif"  # 替换为实际的图像文件名

    # 读取图像（使用PIL读取为PIL.Image类型）
    dir_name = "or_Images"
    image_path = os.path.join(dir_name, image_name)
    in_image = Image.open(image_path).convert('L')  # 转为灰度图的PIL.Image

    # 半色调参数
    xres = 600.0
    yres = 600.0
    bits_num = 1

    # 执行处理
    mat_halftone = perform_halftoning(in_image, xres, yres, bits_num)

    # 若perform_halftoning返回的是numpy数组，需要保存或转回PIL.Image可按需处理
    # 示例：保存为图像（也可用PIL保存）
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "hepburn_PureColors_MS.tiff")

    # 若mat_halftone是numpy数组，用cv2保存（或转回PIL用Image.save保存）
    result = cv2.imwrite(
        output_path,
        mat_halftone,
        [cv2.IMWRITE_TIFF_COMPRESSION, 1]
    )

    if result:
        print(f"图像已保存至: {output_path}")
    else:
        print("保存失败")
        return 1

    return 0


if __name__ == "__main__":
    main()