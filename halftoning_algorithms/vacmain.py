import numpy as np
from PIL import Image


def vac_dithering(image):
    """
    使用Bayer矩阵进行有序抖动算法的二值化处理。

    参数:
    image (PIL.Image): 灰度图像

    返回:
    PIL.Image: 二值化后的图像
    """
    # 定义一个16x16的Bayer矩阵
    bayer_matrix = np.array([
        [1, 34, 166, 208, 114, 85, 197, 45, 211, 95, 146, 55, 178, 216, 148, 192],
        [157, 70, 81, 15, 131, 37, 224, 160, 181, 137, 5, 75, 164, 41, 23, 109],
        [170, 133, 245, 123, 101, 54, 140, 71, 21, 173, 105, 33, 244, 125, 83, 205],
        [214, 20, 189, 220, 202, 0, 191, 151, 241, 234, 87, 195, 222, 93, 141, 236],
        [104, 76, 49, 97, 9, 113, 167, 57, 108, 11, 127, 50, 119, 65, 7, 59],
        [44, 139, 177, 31, 67, 237, 25, 90, 39, 206, 186, 253, 155, 26, 171, 250],
        [182, 149, 225, 163, 198, 229, 124, 62, 218, 80, 145, 18, 212, 200, 194, 230],
        [115, 36, 89, 4, 142, 77, 185, 156, 226, 176, 42, 165, 100, 110, 74, 13],
        [63, 99, 207, 128, 174, 17, 35, 94, 132, 2, 69, 136, 56, 30, 152, 126],
        [190, 242, 248, 52, 215, 247, 112, 201, 240, 254, 117, 221, 238, 91, 249, 217],
        [19, 82, 111, 28, 102, 150, 58, 84, 172, 22, 51, 180, 144, 78, 162, 47],
        [138, 168, 154, 68, 40, 134, 8, 161, 213, 122, 106, 10, 183, 38, 204, 6],
        [231, 175, 223, 184, 121, 199, 233, 48, 72, 32, 88, 228, 129, 235, 196, 61],
        [73, 210, 24, 12, 219, 143, 246, 193, 169, 188, 209, 153, 159, 27, 103, 120],
        [116, 96, 147, 43, 64, 92, 107, 79, 130, 16, 60, 98, 66, 14, 135, 86],
        [239, 53, 251, 179, 232, 158, 3, 29, 252, 118, 227, 243, 203, 46, 255, 187]
    ])
    n = bayer_matrix.shape[0]
    normalized_bayer_matrix = (bayer_matrix + 0.5) / (n * n)

    # 将图像转换为灰度图像并归一化到 [0, 1]
    grayscale_image = image.convert("L")
    image_array = np.array(grayscale_image, dtype=float) / 255.0
    # 获取图像的行和列
    rows, cols = image_array.shape
    # 初始化二值图像数组
    binary_image = np.zeros((rows, cols), dtype=np.uint8)

    # 遍历图像
    for y in range(rows):
        for x in range(cols):
            threshold = normalized_bayer_matrix[y % n, x % n]
            binary_image[y, x] = 255 if image_array[y, x] >= threshold else 0

    # 再次确保所有像素都是0或255
    binary_image = np.clip(binary_image, 0, 255).astype(np.uint8)

    # 将处理后的数组转换为图像
    dithered_image = Image.fromarray(binary_image)
    return dithered_image

if __name__ == "__main__":
    # 打开原始图像
    image_path = 'small.png'  # 替换为你的图像路径
    image = Image.open(image_path)

    # 调用有序抖动算法
    dithered_image = vac_dithering(image)

    # 保存二值化后的图像为 PNG 格式以避免压缩伪影
    output_path = 'ditheredgray_image16.png'
    dithered_image.save(output_path, format='PNG')
    print(f"二值化后的图像已保存到 {output_path}")

    # 可选：验证图像是否为二值图像
    final_image_array = np.array(dithered_image)
    unique_values = np.unique(final_image_array)
    print(f"图像中的唯一像素值: {unique_values}")