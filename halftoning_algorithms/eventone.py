import math
import ctypes  # 用于模拟C++固定宽度整数
import numpy as np

# 模拟C++基础数据类型（确保运算精度和溢出行为一致）
int32 = ctypes.c_int32
uint8 = ctypes.c_uint8
uint16 = ctypes.c_uint16
uint32 = ctypes.c_uint32


class EvenTone:
    class Distance:
        def __init__(self):
            self.dx = int32(0).value  # 用int32确保运算宽度
            self.dy = int32(0).value
            self.r_sq = int32(0).value

        # 深拷贝方法（修复C++对象赋值的浅拷贝差异）
        def copy(self):
            new_dist = EvenTone.Distance()
            new_dist.dx = self.dx
            new_dist.dy = self.dy
            new_dist.r_sq = self.r_sq
            return new_dist

    class ShadeSegment:
        def __init__(self, width):
            self.dis = EvenTone.Distance()
            # 每个Distance都是独立对象（修复C++数组初始化差异）
            self.et_dis = [EvenTone.Distance() for _ in range(width)]

    MAX_SPREAD = 32
    EVEN_C1 = 256
    EVEN_C2 = EVEN_C1 * math.sqrt(3.0) / 2.0  # Sin60

    def __init__(self, xres: float, yres: float, dst_width: int, dev_bits: int):
        self.bits = int(dev_bits)
        self.v = int32(0).value  # 误差累积变量（用int32模拟C++ int）
        self.row = int32(0).value
        self.last_line_was_empty = int32(0).value
        self.width = int(dst_width)
        self.d2x = int32(0).value
        self.d2y = int32(0).value
        self.aspect = float(0.0)
        self.out_buff = None
        self.lut = [int32(0).value for _ in range(256)]  # LUT用int32存储

        # 计算缓冲区大小 (按8字节对齐，与C++一致)
        size = 2 * EvenTone.MAX_SPREAD + ((dst_width + 7) & (~7))
        self.errors = [int32(0).value for _ in range(size)]  # 误差数组用int32

        # 处理设备位数 (3位特殊处理为4位)
        self.obits = 4 if dev_bits == 3 else dev_bits

        # 计算宽高比参数（严格复刻C++整数除法）
        xa = int32(int(xres / yres)).value
        if xa == 0:
            xa = int32(1).value
        d_sq = EvenTone.Distance()
        d_sq.dx = int32(xa * xa).value
        self.d2x = int32(2 * d_sq.dx).value

        ya = int32(int(yres / xres)).value
        if ya == 0:
            ya = int32(1).value
        d_sq.dy = int32(ya * ya).value
        self.d2y = int32(2 * d_sq.dy).value
        d_sq.r_sq = int32(0).value

        # 初始化宽高比和阴影段（修复对象引用问题）
        self.aspect = EvenTone.EVEN_C2 / (xa * ya)
        self.shade = EvenTone.ShadeSegment(dst_width)
        self.shade.dis = d_sq.copy()  # 深拷贝而非引用
        for i in range(dst_width):
            self.shade.et_dis[i] = d_sq.copy()  # 每个元素独立拷贝

        # 初始化输出缓冲区（字节对齐与C++一致）
        buff_size = (dst_width * self.obits + 31) // 32 * 4
        self.out_buff = bytearray(buff_size)

        # 初始化LUT (8位到16位映射，用整数运算复刻C++)
        for i in range(256):
            # 严格模拟C++的整数除法（+127是为了四舍五入）
            self.lut[i] = int32((i * 65535 + 127) // 255).value

    def set_lut(self, src_lut: list) -> None:
        """设置查找表（确保类型匹配）"""
        for i in range(256):
            self.lut[i] = int32(src_lut[i]).value

    def set_gamma(self, gamma: float) -> None:
        """根据gamma值生成查找表（修复浮点运算精度）"""
        for i in range(256):
            val = 1.0 - (i / 255.0)
            # 用C++风格的浮点到整数转换（截断而非四舍五入）
            self.lut[i] = int32(65535 - int(65535 * math.pow(val, gamma))).value

    def _init_et(self, duplicate_line: bool, zero_line: bool) -> bool:
        """初始化变量（修复逻辑分支与C++一致）"""
        if not duplicate_line:
            if not zero_line:
                self.last_line_was_empty = int32(0).value
            else:
                self.last_line_was_empty = int32(self.last_line_was_empty + 1).value
        elif self.last_line_was_empty != 0:
            self.last_line_was_empty = int32(self.last_line_was_empty + 1).value

        if self.last_line_was_empty >= 5:
            return False
        elif self.last_line_was_empty == 4:
            size = 2 * EvenTone.MAX_SPREAD + ((self.width + 7) & (~7))
            self.errors = [int32(0).value for _ in range(size)]
            return False

        self.v = int32(0).value
        return True

    def _advance_eventone_pre(self, x: int) -> None:
        """更新距离信息（前向）- 修复对象赋值引用问题"""
        etd = self.shade.et_dis[x]
        # 用int32运算确保溢出行为与C++一致
        t = int32(self.shade.dis.r_sq + self.shade.dis.dx).value

        if t <= etd.r_sq:
            self.shade.dis.r_sq = t
            self.shade.dis.dx = int32(self.shade.dis.dx + self.d2x).value
        else:
            # 深拷贝而非引用赋值（关键修复！）
            self.shade.dis = etd.copy()

    def _eventone_update(self, x: int) -> None:
        """更新距离信息（后向）- 修复距离计算精度"""
        etd = self.shade.et_dis[x]
        # 严格用int32计算，避免Python自动类型扩展
        t = int32(etd.r_sq + etd.dy).value
        u = int32(self.shade.dis.r_sq + self.shade.dis.dy).value

        if u < t:
            t = u
            etd.dx = self.shade.dis.dx
            etd.dy = self.shade.dis.dy

        etd.dy = int32(etd.dy + self.d2y).value
        # 截断逻辑与C++一致（超过65535则钳位）
        if t > 65535:
            t = 65535
        etd.r_sq = int32(t).value

    def _diffuse_error(self, x: int, direction: int) -> None:
        """误差扩散（Floyd-Steinberg算法）- 修复整数除法"""
        # 误差分配系数严格复刻C++（3/16, 5/16, 1/16）
        self.errors[x + EvenTone.MAX_SPREAD] = int32(self.v * 3).value
        self.errors[x + EvenTone.MAX_SPREAD - direction] = int32(
            self.errors[x + EvenTone.MAX_SPREAD - direction] + self.v * 5
        ).value
        self.errors[x + EvenTone.MAX_SPREAD - (direction * 2)] = int32(
            self.errors[x + EvenTone.MAX_SPREAD - (direction * 2)] + self.v * 1
        ).value
        # 整数除法用//而非/（C++风格截断）
        self.v = int32(self.v - self.v * 9 // 16).value

    def _eventone_adjust(self, dither_point: int, desired: int) -> int:
        """调整抖动点 - 修复浮点转整数的类型错误"""
        dither_point = int32(dither_point).value
        desired = int32(desired).value

        if dither_point <= 0:
            return 0
        if dither_point >= 65535:
            return 65535

        if desired == 0:
            return 0

        # 修复：先计算浮点结果，再用C++风格的截断转换为整数
        float_term1 = self.shade.dis.r_sq * self.aspect  # 浮点运算
        term1 = int32(int(float_term1)).value  # 模拟C++的浮点→整数截断（而非四舍五入）
        term2 = int32((EvenTone.EVEN_C1 * 65535) // desired).value
        adjusted = int32(dither_point + term1 - term2).value

        return max(0, min(65535, adjusted))

    def dither(self, image: bytes, src_width: int, dup_line: bool, is_zero: bool) -> bytearray:
        """核心抖动处理函数 - 修复指针访问和位运算"""
        # 蛇形扫描方向（用整数&运算复刻C++）
        direction = int32(-1).value if (self.row & 1) else int32(1).value
        dev_bit = int32(self.obits).value
        # 计算max_ink（32位整数，高16位和低16位拆分）
        max_ink = int32((((1 << self.bits) - 2) << 16) | 0xFFFF).value

        # 位操作掩码（与C++完全一致）
        mask_bits = [0, 7, 3, 1, 1]
        mask_shift = [0, 3, 2, 1, 1]

        # 初始化检查
        if not self._init_et(dup_line, is_zero):
            self.row = int32(self.row + 1).value
            return self.out_buff

        # 设置扫描范围（修复指针初始化逻辑）
        if direction == 1:
            x = int32(0).value
            terminate = int32(self.width).value
            img_ptr = int32(0).value  # 正向指针
        else:
            x = int32(self.width - 1).value
            terminate = int32(-1).value
            img_ptr = int32(src_width - 1).value  # 反向指针

        # 计算步长和偏移（整数除法与C++一致）
        xstep = int32(src_width // self.width).value
        xmod = int32(src_width % self.width).value
        xerror = int32((xmod * x) % self.width).value

        # 清空输出缓冲区
        buff_size = (self.width * dev_bit + 31) // 32 * 4
        self.out_buff = bytearray(buff_size)

        # 逐像素处理（用while循环复刻C++的for循环逻辑）
        while x != terminate:
            # 距离更新
            self._advance_eventone_pre(x)

            # 计算墨量范围（修复图像像素访问）
            # 用byte访问模拟C++的uint8_t指针（关键修复！）
            pixel_val = uint8(image[img_ptr]).value
            # LUT查找和范围计算（严格整数运算）
            range_point = int32((max_ink * self.lut[pixel_val]) // 65535).value
            out_bit = uint8((range_point >> 16) & 0xFF).value  # 高16位提取
            range_point = int32(range_point & 0xFFFF).value  # 低16位保留

            # 误差累加（用int32防止溢出）
            self.v = int32(
                self.v + 2 * range_point + (self.errors[x + EvenTone.MAX_SPREAD] + 8) // 16
            ).value
            inkspot = int32(self.v - range_point).value

            # 调整抖动点
            point_error = self._eventone_adjust(inkspot, range_point)
            comparison = int32(32768).value  # 中点阈值

            if point_error >= comparison:
                point_error = int32(point_error - 65535).value
                out_bit = uint8(out_bit + 1).value
                self.v = int32(self.v - 131070).value  # 2 * 65535
                self.shade.dis = EvenTone.Distance()  # 重置距离

            # 更新距离和误差扩散
            self._eventone_update(x)
            self._diffuse_error(x, direction)

            # 写入输出缓冲区（修复位偏移计算）
            shift_amount = int32(
                (mask_bits[dev_bit] - (x & mask_bits[dev_bit])) * dev_bit
            ).value
            out_bit_shifted = uint8((out_bit << shift_amount) & 0xFF).value
            buff_index = int32(x >> mask_shift[dev_bit]).value
            if buff_index < len(self.out_buff):
                self.out_buff[buff_index] = bytearray([
                    uint8(self.out_buff[buff_index] | out_bit_shifted).value
                ])[0]

            # 移动指针（复刻C++的步进逻辑）
            if direction == 1:
                img_ptr = int32(img_ptr + xstep).value
                xerror = int32(xerror + xmod).value
                if xerror >= self.width:
                    xerror = int32(xerror - self.width).value
                    img_ptr = int32(img_ptr + 1).value
            else:
                img_ptr = int32(img_ptr - xstep).value
                xerror = int32(xerror - xmod).value
                if xerror < 0:
                    xerror = int32(xerror + self.width).value
                    img_ptr = int32(img_ptr - 1).value

            x = int32(x + direction).value

        self.row = int32(self.row + 1).value
        return self.out_buff