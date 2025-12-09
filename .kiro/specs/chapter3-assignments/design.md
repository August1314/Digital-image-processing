# Design Document

## Overview

本设计文档描述了第三章3个编程作业的技术实现方案。每个作业都将实现为独立的Python程序，使用NumPy进行数值计算，使用Matplotlib进行可视化展示。设计遵循模块化原则，确保代码清晰、可维护且易于理解。

## Architecture

### 整体架构

采用模块化设计，每个作业为独立的Python脚本：
- `作业1_直方图均衡化.py` - 直方图均衡化实现
- `作业2_拉普拉斯锐化.py` - 拉普拉斯算子锐化实现  
- `作业3_位平面分析.py` - 位平面分解与分析实现

每个程序包含以下核心组件：
1. 图像加载模块
2. 算法实现模块
3. 可视化展示模块
4. 结果保存模块

### 技术栈

- Python 3.x (使用现有的 conda 环境: ml)
- NumPy - 数值计算和数组操作
- Matplotlib - 图表和图像可视化
- PIL/Pillow - 图像读取和保存
- OpenCV (可选) - 辅助图像处理

### 环境配置

使用现有的 conda 环境 `ml`：
```bash
conda activate ml
```

所需的依赖包应该已经在 ml 环境中安装。如果缺少任何包，可以使用：
```bash
pip install numpy matplotlib pillow opencv-python
```

## Components and Interfaces

### 作业1: 直方图均衡化

#### 核心函数

```python
def calculate_histogram(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]
    """计算图像直方图"""
    
def histogram_equalization(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]
    """执行直方图均衡化，返回增强图像和转换函数"""
    
def visualize_histogram_equalization(original, hist_orig, transform_func, 
                                     enhanced, hist_enhanced) -> None
    """可视化所有结果"""
```

#### 算法实现细节

1. 直方图计算：
   - 统计每个灰度级的像素数量
   - 归一化得到概率密度函数 p(r_k)

2. 累积分布函数（CDF）：
   - 计算 CDF: s_k = Σ p(r_j), j=0 to k
   - 这就是转换函数 T(r)

3. 映射变换：
   - 新像素值 = round(CDF[原像素值] × (L-1))
   - L为灰度级数（通常为256）


### 作业2: 拉普拉斯锐化

#### 核心函数

```python
def laplacian_filter(image: np.ndarray, kernel: np.ndarray) -> np.ndarray
    """应用拉普拉斯滤波器"""
    
def laplacian_sharpen(image: np.ndarray, kernel: np.ndarray) -> tuple[np.ndarray, np.ndarray]
    """执行拉普拉斯锐化，返回拉普拉斯响应和锐化图像"""
    
def visualize_laplacian_sharpening(original, laplacian_response, 
                                   sharpened) -> None
    """可视化锐化结果"""
```

#### 算法实现细节

1. 拉普拉斯核（根据作业要求明确指定）：
   ```
   [-1  -1  -1]
   [-1   8  -1]
   [-1  -1  -1]
   ```
   这是一个包含对角线的拉普拉斯核，中心系数为8，周围8个邻域系数均为-1。

2. 卷积操作：
   - 使用零填充或反射填充处理边界
   - 对每个像素应用3×3卷积核
   - 计算拉普拉斯响应：∇²f(x,y)

3. 锐化公式（式3.54）：
   - 由于使用的是正核（中心为+8），锐化公式为：
   - g(x,y) = f(x,y) + ∇²f(x,y)
   - 或者可以理解为：g(x,y) = f(x,y) + c × ∇²f(x,y)，其中c=1
   - 结果需要裁剪到[0, 255]范围以避免溢出

### 作业3: 位平面分析

#### 核心函数

```python
def extract_bit_planes(image: np.ndarray) -> list[np.ndarray]
    """提取所有8个位平面"""
    
def reconstruct_from_bit_planes(bit_planes: list[np.ndarray], 
                                plane_indices: list[int]) -> np.ndarray
    """从选定的位平面重构图像"""
    
def visualize_bit_planes(original, bit_planes) -> None
    """可视化所有位平面"""
```

#### 算法实现细节

1. 位平面提取：
   - 对于每个位位置 i (0-7)：
   - bit_plane_i = (image >> i) & 1
   - 将二值结果缩放到0-255以便可视化

2. 位平面分析：
   - Bit 7（MSB）：包含最重要的视觉信息
   - Bit 0（LSB）：类似随机噪声
   - 中间位平面：包含边缘和纹理信息

3. 重构实验：
   - 仅使用高位平面（如bit 7-6）重构
   - 观察图像质量与使用的位平面数量的关系

## Data Models

### 图像数据结构

```python
# 灰度图像
image: np.ndarray  # shape: (height, width), dtype: uint8

# 直方图数据
histogram: np.ndarray  # shape: (256,), dtype: int
bins: np.ndarray  # shape: (256,), dtype: int

# 位平面数据
bit_planes: list[np.ndarray]  # 8个元素，每个shape: (height, width)
```

## Error Handling

1. 文件加载错误：
   - 检查文件是否存在
   - 验证图像格式是否支持
   - 提供清晰的错误消息

2. 图像格式处理：
   - 自动转换彩色图像为灰度图
   - 确保像素值在有效范围内

3. 数值计算保护：
   - 防止除零错误
   - 裁剪结果到有效范围[0, 255]
   - 处理边界条件

## Testing Strategy

### 单元测试

1. 直方图计算验证：
   - 测试已知分布的图像
   - 验证直方图总和等于像素总数

2. 拉普拉斯滤波验证：
   - 测试简单图案（如边缘）
   - 验证卷积计算正确性

3. 位平面提取验证：
   - 测试已知像素值
   - 验证重构的正确性

### 集成测试

1. 端到端测试：
   - 使用提供的测试图像
   - 验证输出结果的完整性

2. 可视化测试：
   - 确保所有图表正确显示
   - 验证标签和标题的准确性

## Visualization Design

### 作业1布局（2行3列）

```
[原始图像]     [原始直方图]     [转换函数]
[增强图像]     [增强直方图]     [对比说明]
```

### 作业2布局（1行3列）

```
[原始图像]  [拉普拉斯响应]  [锐化图像]
```

### 作业3布局（3行3列）

```
[原始图像]  [Bit 7]  [Bit 6]
[Bit 5]     [Bit 4]  [Bit 3]
[Bit 2]     [Bit 1]  [Bit 0]
```

## Performance Considerations

1. 图像大小：
   - 对大图像进行适当缩放以加快处理
   - 保持纵横比

2. 内存管理：
   - 避免不必要的图像副本
   - 及时释放临时数组

3. 计算优化：
   - 使用NumPy向量化操作
   - 避免显式循环（除非必要）
