# 第四章作业设计文档

## 概述

本设计文档描述了第四章频率域滤波作业的技术实现方案。系统包括两个主要程序：
1. **作业1_二维FFT变换.py** - 实现2D FFT及逆变换
2. **作业2_高斯低通滤波.py** - 实现频率域高斯低通滤波

设计遵循第三章作业的代码风格，使用Python + NumPy + Matplotlib实现。

## 架构

### 整体架构

```
实验/第四章/
├── 代码/
│   ├── 作业1_二维FFT变换.py
│   └── 作业2_高斯低通滤波.py
├── image.png (输入图像)
└── 输出结果/
    ├── 作业1_二维FFT变换/
    │   ├── 完整结果.png
    │   ├── 频谱图.png
    │   └── 重构图像.png
    └── 作业2_高斯低通滤波/
        ├── 完整结果.png
        ├── 滤波后图像.png
        └── 频谱对比.png
```

### 模块设计

#### 作业1: 二维FFT变换

**核心模块:**
- `fft2d()` - 二维FFT实现（使用NumPy的FFT）
- `ifft2d()` - 二维逆FFT实现
- `fft_shift()` - 频谱中心化
- `compute_magnitude_spectrum()` - 计算幅度谱
- `compute_phase_spectrum()` - 计算相位谱
- `visualize_fft()` - 可视化FFT结果

#### 作业2: 高斯低通滤波

**核心模块:**
- `create_gaussian_filter()` - 生成高斯低通滤波器
- `apply_frequency_filter()` - 在频率域应用滤波器
- `calculate_energy_ratio()` - 计算能量保留率
- `find_cutoff_for_energy()` - 自动寻找95%能量对应的截止频率
- `visualize_filtering()` - 可视化完整滤波流程

## 组件和接口

### 1. 图像加载模块

```python
def load_image(image_path: str) -> np.ndarray:
    """
    加载图像并转换为灰度图
    
    Args:
        image_path: 图像文件路径
        
    Returns:
        image: 灰度图像numpy数组
    """
```

### 2. 图像预处理模块

```python
def pad_to_power_of_2(image: np.ndarray) -> tuple:
    """
    将图像填充到2的幂次尺寸
    
    Args:
        image: 输入图像
        
    Returns:
        padded_image: 填充后的图像
        original_shape: 原始图像尺寸
    """
```

### 3. FFT变换模块

```python
def fft2d(image: np.ndarray) -> np.ndarray:
    """
    执行二维快速傅里叶变换
    
    Args:
        image: 输入灰度图像
        
    Returns:
        fft_result: 复数频谱（未中心化）
    """

def ifft2d(fft_result: np.ndarray) -> np.ndarray:
    """
    执行二维逆快速傅里叶变换
    
    Args:
        fft_result: 频域复数数组
        
    Returns:
        reconstructed: 重构的空间域图像
    """

def fft_shift(fft_result: np.ndarray) -> np.ndarray:
    """
    将零频率分量移到频谱中心
    
    Args:
        fft_result: FFT结果
        
    Returns:
        shifted: 中心化的频谱
    """
```

### 4. 频谱分析模块

```python
def compute_magnitude_spectrum(fft_result: np.ndarray, 
                               log_scale: bool = True) -> np.ndarray:
    """
    计算幅度谱
    
    Args:
        fft_result: FFT复数结果
        log_scale: 是否使用对数尺度
        
    Returns:
        magnitude: 幅度谱（用于可视化）
    """

def compute_phase_spectrum(fft_result: np.ndarray) -> np.ndarray:
    """
    计算相位谱
    
    Args:
        fft_result: FFT复数结果
        
    Returns:
        phase: 相位谱（弧度）
    """
```

### 5. 滤波器生成模块

```python
def create_gaussian_filter(shape: tuple, 
                          cutoff_freq: float,
                          center: tuple = None) -> np.ndarray:
    """
    创建高斯低通滤波器
    
    Args:
        shape: 滤波器尺寸 (H, W)
        cutoff_freq: 截止频率（标准差）
        center: 中心位置，默认为图像中心
        
    Returns:
        filter: 高斯低通滤波器
        
    公式: H(u,v) = exp(-D²(u,v) / (2*D0²))
    其中 D(u,v) 是到中心的距离，D0 是截止频率
    """
```

### 6. 频域滤波模块

```python
def apply_frequency_filter(fft_result: np.ndarray, 
                          filter_mask: np.ndarray) -> np.ndarray:
    """
    在频率域应用滤波器
    
    Args:
        fft_result: 中心化的FFT结果
        filter_mask: 滤波器掩码
        
    Returns:
        filtered_fft: 滤波后的频谱
    """

def calculate_energy_ratio(original_fft: np.ndarray,
                          filtered_fft: np.ndarray) -> float:
    """
    计算滤波后能量保留率
    
    Args:
        original_fft: 原始频谱
        filtered_fft: 滤波后频谱
        
    Returns:
        ratio: 能量保留率（0-1之间）
        
    能量定义: E = Σ|F(u,v)|²
    """

def find_cutoff_for_energy(image: np.ndarray,
                          target_energy: float = 0.95,
                          tolerance: float = 0.01) -> float:
    """
    自动寻找达到目标能量保留率的截止频率
    
    Args:
        image: 输入图像
        target_energy: 目标能量保留率（默认0.95）
        tolerance: 容差范围
        
    Returns:
        cutoff_freq: 最优截止频率
        
    使用二分搜索或迭代方法寻找最优截止频率
    """
```

### 7. 可视化模块

```python
def visualize_fft(original: np.ndarray,
                 magnitude: np.ndarray,
                 phase: np.ndarray,
                 reconstructed: np.ndarray) -> plt.Figure:
    """
    可视化FFT变换结果
    
    显示: 原始图像 | 幅度谱 | 相位谱 | 重构图像
    """

def visualize_filtering(original: np.ndarray,
                       original_spectrum: np.ndarray,
                       filter_mask: np.ndarray,
                       filtered_spectrum: np.ndarray,
                       filtered_image: np.ndarray,
                       energy_ratio: float) -> plt.Figure:
    """
    可视化完整滤波流程（类似图4.35）
    
    布局:
    第一行: 原始图像 | 原始频谱 | 高斯滤波器
    第二行: 滤波后频谱 | 滤波后图像 | 能量信息
    """
```

## 数据模型

### 图像数据流

```
输入图像 (H×W, uint8)
    ↓
灰度转换 (如需要)
    ↓
零填充至2^P × 2^P (如需要)
    ↓
FFT变换 → 复数频谱 (H×W, complex128)
    ↓
中心化 (fftshift)
    ↓
应用滤波器 (元素乘法)
    ↓
逆中心化 (ifftshift)
    ↓
逆FFT变换
    ↓
取实部并裁剪
    ↓
输出图像 (H×W, uint8)
```

### 频谱表示

- **复数频谱**: `F(u,v) = R(u,v) + j*I(u,v)`
- **幅度谱**: `|F(u,v)| = sqrt(R² + I²)`
- **相位谱**: `φ(u,v) = arctan(I/R)`
- **功率谱**: `P(u,v) = |F(u,v)|²`

### 高斯滤波器

```
H(u,v) = exp(-D²(u,v) / (2*D0²))

其中:
- D(u,v) = sqrt((u-M/2)² + (v-N/2)²) 是到中心的欧氏距离
- D0 是截止频率（标准差）
- 当 D(u,v) = D0 时，H = exp(-0.5) ≈ 0.607
```

## 错误处理

### 输入验证

1. **图像文件检查**
   - 验证文件存在性
   - 验证文件格式（支持常见图像格式）
   - 提供清晰的错误消息

2. **图像尺寸处理**
   - 自动检测是否为2的幂次
   - 自动填充非2的幂次尺寸
   - 记录原始尺寸用于裁剪

3. **参数验证**
   - 截止频率范围检查（> 0）
   - 能量保留率范围检查（0 < ratio < 1）

### 数值稳定性

1. **对数变换保护**
   ```python
   magnitude_log = np.log(magnitude + 1)  # 避免log(0)
   ```

2. **除零保护**
   ```python
   energy_ratio = filtered_energy / (original_energy + 1e-10)
   ```

3. **数据类型转换**
   ```python
   # FFT使用float64以保证精度
   image_float = image.astype(np.float64)
   # 输出转换为uint8
   output = np.clip(result.real, 0, 255).astype(np.uint8)
   ```

## 测试策略

### 单元测试

1. **FFT正确性验证**
   - 测试: FFT → IFFT 应恢复原图
   - 验证: MSE < 1e-10（数值误差范围内）

2. **滤波器生成验证**
   - 测试: 中心值应为1
   - 测试: 边缘值应接近0
   - 测试: 对称性检查

3. **能量计算验证**
   - 测试: 无滤波时能量保留率应为100%
   - 测试: 完全滤除时能量保留率应接近0%

### 集成测试

1. **完整流程测试**
   - 使用标准测试图像（如Lena）
   - 验证输出图像质量
   - 检查可视化结果

2. **边界条件测试**
   - 小尺寸图像（如8×8）
   - 大尺寸图像（如1024×1024）
   - 非方形图像

### 视觉验证

1. **频谱特征**
   - 低频集中在中心
   - 高频分布在边缘
   - 对称性（实数图像的频谱共轭对称）

2. **滤波效果**
   - 低通滤波应产生平滑效果
   - 边缘应变模糊
   - 噪声应被抑制

## 性能考虑

### 算法复杂度

- **FFT**: O(N² log N) 对于 N×N 图像
- **滤波器应用**: O(N²) 元素乘法
- **能量计算**: O(N²) 求和操作

### 优化策略

1. **使用NumPy的FFT**
   - 利用高度优化的FFTPACK库
   - 避免手动实现FFT

2. **向量化操作**
   - 使用NumPy广播避免循环
   - 批量处理像素操作

3. **内存管理**
   - 复用数组避免重复分配
   - 及时释放大型临时数组

## 实现注意事项

### 1. 频谱中心化

```python
# 正确的中心化流程
fft_result = np.fft.fft2(image)
fft_shifted = np.fft.fftshift(fft_result)  # 零频率移到中心

# 逆变换前需要逆中心化
fft_unshifted = np.fft.ifftshift(fft_shifted)
reconstructed = np.fft.ifft2(fft_unshifted)
```

### 2. 幅度谱可视化

```python
# 使用对数尺度增强显示
magnitude = np.abs(fft_shifted)
magnitude_log = np.log(magnitude + 1)  # +1避免log(0)
```

### 3. 能量保留率计算

```python
# 使用功率谱（幅度平方）计算能量
original_energy = np.sum(np.abs(original_fft) ** 2)
filtered_energy = np.sum(np.abs(filtered_fft) ** 2)
energy_ratio = filtered_energy / original_energy
```

### 4. 截止频率搜索

使用二分搜索或简单迭代：
```python
# 从小到大尝试不同的截止频率
for D0 in range(10, 200, 5):
    filter = create_gaussian_filter(shape, D0)
    ratio = calculate_energy_ratio(...)
    if abs(ratio - 0.95) < 0.01:
        return D0
```

## 环境要求

- Python 3.x
- Conda环境: `ml`
- 依赖库: numpy, matplotlib, pillow

运行前激活环境：
```bash
conda activate ml
```

## 参考资料

- 教材第4章：频率域滤波
- 教材图4.35：高斯低通滤波示例
- NumPy FFT文档：https://numpy.org/doc/stable/reference/routines.fft.html
- 第三章作业代码风格参考
