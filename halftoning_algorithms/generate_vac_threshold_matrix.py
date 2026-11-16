import struct
import cv2
import numpy as np
from PIL import ImageCms
from PIL import Image
import matplotlib.pyplot as plt
import math
from scipy.signal import convolve2d
from scipy import ndimage
def VAC(guss_kernel, Area, graynum):
    '''
    1、随机产生一个16*16的矩阵 内有25个像素为黑
    2、用视觉核对该二值图进行处理 得到最密集、最稀疏的位置
    3、第一阶段 对这25个黑色点比较其密集度 在DA抖动矩阵中排序 数值大的编号大 0-25
    4、第二阶段 对最稀疏部分进行排序 将对应位置写入DA抖动矩阵中 从26到128  并将选中的像素位置改为1
    5、第三阶段 将第二阶段处理完的图像再继续取最空隙的点 再依次加入1 从129到255
    4、如果切换回白色的黑像素与最初切换成黑色的白像素相同 代表已经收敛
    '''
    # Area取15 graynum取25
    matrix = np.zeros((Area, Area), dtype=int)
    black_positions = np.random.choice(Area * Area, graynum, replace=False)
    for pos in black_positions:
        row, col = divmod(pos, Area)
        matrix[row, col] = 1

    # 视觉核 guss_kernel取5
    ax = np.arange(-guss_kernel // 2 + 1, guss_kernel // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * 1.5 ** 2))
    guss = kernel / np.sum(kernel)

    # 用视觉核卷积生成的矩阵
    guss_image = convolve2d(matrix, guss, mode='same')
    guss_image_copy = guss_image.copy()
    # DA抖动矩阵
    DA_matrix = np.zeros_like(guss_image)

    for i in range(Area * Area):
        max_index_flat = np.argmax(guss_image_copy)
        max_index_2d = np.unravel_index(max_index_flat, guss_image_copy.shape)
        DA_matrix[max_index_2d[0], max_index_2d[1]] = Area * Area - 1 - i
        guss_image_copy[max_index_2d[0], max_index_2d[1]] = 0

    # h,w,c=binary_image.shape
    # h_area=h//Area
    # w_area=w//Area
    # record=np.zeros_like(binary_image)
    # for ch in range(4):
    #     for i in range(0,h_area):
    #         for j in range(0,w_area):
    #             Area_image=binary_image[i*Area:(i+1)+Area,j*Area:(j+1)*Area,ch]
    #             count=np.sum(Area_image > 1)
    #             for _ in range(count):

    #                 guss_image = convolve2d(Area_image, guss, mode='same')

    #                 # 找到最大值和最小值的位置（线性索引）
    #                 max_index = np.unravel_index(np.argmax(guss_image), guss_image.shape)
    #                 min_index = np.unravel_index(np.argmin(guss_image), guss_image.shape)

    #                 Area_image[max_index[0],max_index[1]]=np.abs(Area_image[max_index[0],max_index[1]-255])
    #                 # 转换为全图的坐标
    #                 max_pixel_position = (i*Area + max_index[0], j*Area + max_index[1])
    #                 min_pixel_position = (i*Area + min_index[0], j*Area + min_index[1])
    #                 if record[max_pixel_position[0],max_pixel_position[1],ch]==1 or record[min_pixel_position[0],min_pixel_position[1],ch]==1:
    #                     break
    #                 #转化值
    #                 binary_image[max_pixel_position[0],max_pixel_position[1],ch]=np.abs(binary_image[max_pixel_position[0],max_pixel_position[1],ch]-255)
    #                 binary_image[min_pixel_position[0],min_pixel_position[1],ch]=np.abs(binary_image[min_pixel_position[0],min_pixel_position[1],ch]-255)
    #                 #记录是否已经转换
    #                 record[max_pixel_position[0],max_pixel_position[1],ch]=1
    #                 record[min_pixel_position[0],min_pixel_position[1],ch]=1
    return DA_matrix


# 寻找聚集点
def FindLargestVoid(BinaryPattern, StandardDeviation):
    """This function returns the indices of the largest void in the given binary
       pattern as defined by Ulichney.
      \param BinaryPattern A boolean array (should be two-dimensional although the
             implementation works in arbitrary dimensions).
      \param StandardDeviation The standard deviation used for the Gaussian filter
             in pixels. This can be a single float for an isotropic Gaussian or a
             tuple with one float per dimension for an anisotropic Gaussian.
      \return A flat index i such that BinaryPattern.flat[i] corresponds to the
              largest void. By definition this is a majority pixel.
      \sa GetVoidAndClusterBlueNoise"""
    # The minority value is always True for convenience
    if (np.count_nonzero(BinaryPattern) * 2 >= np.size(BinaryPattern)):
        BinaryPattern = np.logical_not(BinaryPattern)
    # Apply the Gaussian. We do not want to cut off the Gaussian at all because even
    # the tiniest difference can change the ranking. Therefore we apply the Gaussian
    # through a fast Fourier transform by means of the convolution theorem.
    FilteredArray = np.fft.ifftn(
        ndimage.fourier.fourier_gaussian(np.fft.fftn(np.where(BinaryPattern, 1.0, 0.0)), StandardDeviation)).real;
    # Find the largest void
    return np.argmin(np.where(BinaryPattern, 2.0, FilteredArray))


# 寻找松散点
def FindTightestCluster(BinaryPattern, StandardDeviation):
    """Like FindLargestVoid() but finds the tightest cluster which is a minority
       pixel by definition.
      \sa GetVoidAndClusterBlueNoise"""
    if (np.count_nonzero(BinaryPattern) * 2 >= np.size(BinaryPattern)):
        BinaryPattern = np.logical_not(BinaryPattern)
    FilteredArray = np.fft.ifftn(
        ndimage.fourier.fourier_gaussian(np.fft.fftn(np.where(BinaryPattern, 1.0, 0.0)), StandardDeviation)).real;
    return np.argmax(np.where(BinaryPattern, FilteredArray, -1.0))


# 加网
def GetVoidAndClusterBlueNoise(OutputShape, StandardDeviation=1.5, InitialSeedFraction=0.1):
    """Generates a blue noise dither array of the given shape using the method
       proposed by Ulichney [1993] in "The void-and-cluster method for dither array
       generation" published in Proc. SPIE 1913.
      \param OutputShape The shape of the output array. This function works in
             arbitrary dimension, i.e. OutputShape can have arbitrary length. Though
             it is only tested for the 2D case where you should pass a tuple
             (Height,Width).
      \param StandardDeviation The standard deviation in pixels used for the
             Gaussian filter defining largest voids and tightest clusters. Larger
             values lead to more low-frequency content but better isotropy. Small
             values lead to more ordered patterns with less low-frequency content.
             Ulichney proposes to use a value of 1.5. If you want an anisotropic
             Gaussian, you can pass a tuple of length len(OutputShape) with one
             standard deviation per dimension.
      \param InitialSeedFraction The only non-deterministic step in the algorithm
             marks a small number of pixels in the grid randomly. This parameter
             defines the fraction of such points. It has to be positive but less
             than 0.5. Very small values lead to ordered patterns, beyond that there
             is little change.
      \return An integer array of shape OutputShape containing each integer from 0
              to np.prod(OutputShape)-1 exactly once."""
    nRank = np.prod(OutputShape)
    # Generate the initial binary pattern with a prescribed number of ones
    nInitialOne = max(1, min(int((nRank - 1) / 2), int(nRank * InitialSeedFraction)))
    # Start from white noise (this is the only randomized step)
    InitialBinaryPattern = np.zeros(OutputShape, dtype=bool)
    InitialBinaryPattern.flat = np.random.permutation(np.arange(nRank)) < nInitialOne
    # Swap ones from tightest clusters to largest voids iteratively until convergence
    while (True):
        iTightestCluster = FindTightestCluster(InitialBinaryPattern, StandardDeviation)
        InitialBinaryPattern.flat[iTightestCluster] = False
        iLargestVoid = FindLargestVoid(InitialBinaryPattern, StandardDeviation)
        if (iLargestVoid == iTightestCluster):
            InitialBinaryPattern.flat[iTightestCluster] = True
            # Nothing has changed, so we have converged
            break
        else:
            InitialBinaryPattern.flat[iLargestVoid] = True
    # Rank all pixels
    DitherArray = np.zeros(OutputShape, dtype=int)
    # Phase 1: Rank minority pixels in the initial binary pattern
    BinaryPattern = np.copy(InitialBinaryPattern)
    for Rank in range(nInitialOne - 1, -1, -1):
        iTightestCluster = FindTightestCluster(BinaryPattern, StandardDeviation)
        BinaryPattern.flat[iTightestCluster] = False
        DitherArray.flat[iTightestCluster] = Rank
    # Phase 2: Rank the remainder of the first half of all pixels
    BinaryPattern = InitialBinaryPattern
    for Rank in range(nInitialOne, int((nRank + 1) / 2)):
        iLargestVoid = FindLargestVoid(BinaryPattern, StandardDeviation)
        BinaryPattern.flat[iLargestVoid] = True
        DitherArray.flat[iLargestVoid] = Rank
    # Phase 3: Rank the last half of pixels
    for Rank in range(int((nRank + 1) / 2), nRank):
        iTightestCluster = FindTightestCluster(BinaryPattern, StandardDeviation)
        BinaryPattern.flat[iTightestCluster] = True
        DitherArray.flat[iTightestCluster] = Rank
    return DitherArray



# 定义输出形状
output_shape = (8, 8)

# 调用函数生成蓝噪声抖动数组
dither_array = GetVoidAndClusterBlueNoise(output_shape)

# 输出结果
print(dither_array)