import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage

plt.rcParams['font.family'] = ['Arial Unicode MS']

def robert_gradient(image):
    """
    Robert交叉梯度算子
    对应PPT中的2x2掩模计算
    """
    # Robert算子模板
    robert_x = np.array([[0, 1], [-1, 0]], dtype=np.float32)
    robert_y = np.array([[1, 0], [0, -1]], dtype=np.float32)
    
    # 计算x方向和y方向的梯度
    gx = cv2.filter2D(image.astype(np.float32), -1, robert_x)
    gy = cv2.filter2D(image.astype(np.float32), -1, robert_y)
    
    # 使用L1范数计算梯度幅度：M(x,y) ≈ |gx| + |gy|
    gradient_magnitude = np.abs(gx) + np.abs(gy)
    
    return gradient_magnitude, gx, gy

def sobel_gradient(image):
    """
    Sobel算子
    对应PPT中的3x3掩模计算
    """
    # Sobel算子模板
    sobel_x = np.array([[-1, 0, 1], 
                        [-2, 0, 2], 
                        [-1, 0, 1]], dtype=np.float32)
    
    sobel_y = np.array([[-1, -2, -1], 
                        [0, 0, 0], 
                        [1, 2, 1]], dtype=np.float32)
    
    # 计算x方向和y方向的梯度
    gx = cv2.filter2D(image.astype(np.float32), -1, sobel_x)
    gy = cv2.filter2D(image.astype(np.float32), -1, sobel_y)
    
    # 使用L1范数计算梯度幅度
    gradient_magnitude = np.abs(gx) + np.abs(gy)
    
    return gradient_magnitude, gx, gy

def laplacian_sharpen(image, kernel_type='standard'):
    """
    拉普拉斯锐化
    对应PPT中的四种拉普拉斯滤波器
    """
    # 定义四种拉普拉斯核
    kernels = {
        'standard': np.array([[0, 1, 0],
                             [1, -4, 1],
                             [0, 1, 0]], dtype=np.float32),
        
        'with_diagonal': np.array([[1, 1, 1],
                                  [1, -8, 1],
                                  [1, 1, 1]], dtype=np.float32),
        
        'negative_standard': np.array([[0, -1, 0],
                                      [-1, 4, -1],
                                      [0, -1, 0]], dtype=np.float32),
        
        'negative_diagonal': np.array([[-1, -1, -1],
                                      [-1, 8, -1],
                                      [-1, -1, -1]], dtype=np.float32)
    }
    
    laplacian_kernel = kernels[kernel_type]
    
    # 应用拉普拉斯滤波
    laplacian = cv2.filter2D(image.astype(np.float32), -1, laplacian_kernel)
    
    # 锐化图像：原图 - 拉普拉斯结果（根据核的中心符号调整）
    if kernel_type.startswith('negative'):
        sharpened = image.astype(np.float32) + laplacian
    else:
        sharpened = image.astype(np.float32) - laplacian
    
    # 确保值在0-255范围内
    sharpened = np.clip(sharpened, 0, 255)
    
    return sharpened.astype(np.uint8), laplacian

def first_order_sharpen(image, method='sobel', alpha=0.5):
    """
    一阶微分锐化：将梯度信息加到原图上实现锐化
    """
    if method == 'sobel':
        gradient_magnitude, _, _ = sobel_gradient(image)
    else:  # robert
        gradient_magnitude, _, _ = robert_gradient(image)
    
    # 将梯度图归一化到0-255
    gradient_normalized = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX)
    
    # 将梯度信息加到原图上实现锐化
    sharpened = image.astype(np.float32) + alpha * gradient_normalized
    sharpened = np.clip(sharpened, 0, 255)
    
    return sharpened.astype(np.uint8), gradient_normalized.astype(np.uint8)

def create_test_image():
    """创建测试图像，包含各种边缘类型"""
    size = 256
    image = np.ones((size, size), dtype=np.uint8) * 128  # 灰色背景
    
    # 添加不同对比度的边缘
    image[50:100, 50:100] = 200    # 亮方块
    image[150:200, 150:200] = 50   # 暗方块
    
    # 添加细线
    image[100:102, 30:226] = 255   # 水平白线
    image[30:226, 100:102] = 0     # 垂直线
    
    # 添加噪声点
    np.random.seed(42)
    noise_positions = np.random.randint(0, size, (20, 2))
    for pos in noise_positions:
        image[pos[0], pos[1]] = 255 if np.random.random() > 0.5 else 0
    
    return image

# 主演示程序
def main():
    # 创建测试图像
    original_image = create_test_image()
    
    # 一阶微分锐化演示
    print("一阶微分锐化演示:")
    print("=" * 50)
    
    # Robert算子
    robert_sharp, robert_grad = first_order_sharpen(original_image, 'robert', 0.3)
    robert_magnitude, robert_gx, robert_gy = robert_gradient(original_image)
    
    # Sobel算子
    sobel_sharp, sobel_grad = first_order_sharpen(original_image, 'sobel', 0.3)
    sobel_magnitude, sobel_gx, sobel_gy = sobel_gradient(original_image)
    
    # 二阶微分（拉普拉斯）锐化演示
    print("\n二阶微分锐化演示:")
    print("=" * 50)
    
    # 四种拉普拉斯核的锐化效果
    laplacian_results = {}
    for kernel_type in ['standard', 'with_diagonal', 'negative_standard', 'negative_diagonal']:
        sharpened, laplacian = laplacian_sharpen(original_image, kernel_type)
        laplacian_results[kernel_type] = (sharpened, laplacian)
    
    # 可视化结果
    plt.figure(figsize=(20, 15))
    
    # 原始图像
    plt.subplot(3, 5, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title('原始图像')
    plt.axis('off')
    
    # Robert算子结果
    plt.subplot(3, 5, 2)
    plt.imshow(robert_grad, cmap='gray')
    plt.title('Robert梯度图')
    plt.axis('off')
    
    plt.subplot(3, 5, 3)
    plt.imshow(robert_sharp, cmap='gray')
    plt.title('Robert锐化结果')
    plt.axis('off')
    
    # Sobel算子结果
    plt.subplot(3, 5, 4)
    plt.imshow(sobel_grad, cmap='gray')
    plt.title('Sobel梯度图')
    plt.axis('off')
    
    plt.subplot(3, 5, 5)
    plt.imshow(sobel_sharp, cmap='gray')
    plt.title('Sobel锐化结果')
    plt.axis('off')
    
    # 拉普拉斯锐化结果
    titles = ['标准拉普拉斯', '含对角线拉普拉斯', 
              '负标准拉普拉斯', '负对角线拉普拉斯']
    
    for i, (kernel_type, title) in enumerate(zip(laplacian_results.keys(), titles)):
        sharpened, laplacian = laplacian_results[kernel_type]
        
        plt.subplot(3, 5, 6 + i)
        plt.imshow(laplacian, cmap='gray')
        plt.title(f'{title}\n拉普拉斯图')
        plt.axis('off')
        
        plt.subplot(3, 5, 11 + i)
        plt.imshow(sharpened, cmap='gray')
        plt.title(f'{title}\n锐化结果')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # 打印算法说明
    print("\n算法原理说明:")
    print("=" * 50)
    print("一阶微分锐化（梯度）:")
    print("- Robert算子：2×2掩模，M(x,y) ≈ |z₉-z₅| + |z₈-z₆|")
    print("- Sobel算子：3×3掩模，考虑邻域平滑，对噪声更鲁棒")
    print("- 使用L1范数：M(x,y) ≈ |gx| + |gy|，计算简单")
    print("- 锐化方法：原图 + α × 梯度图")
    
    print("\n二阶微分锐化（拉普拉斯）:")
    print("- 标准核：[0,1,0; 1,-4,1; 0,1,0]")
    print("- 含对角线核：[1,1,1; 1,-8,1; 1,1,1]")
    print("- 特点：各向同性、中心对称、和为零")
    print("- 锐化方法：原图 - ∇²f（增强边缘和细节）")
    
    # 显示卷积核
    print("\n使用的卷积核:")
    print("Robert X方向:") 
    print(np.array([[0, 1], [-1, 0]]))
    print("Robert Y方向:")
    print(np.array([[1, 0], [0, -1]]))
    print("\nSobel X方向:")
    print(np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]))
    print("Sobel Y方向:")
    print(np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]))

# 实际图像测试
def test_real_image():
    """用真实图像测试"""
    # 如果系统中有图像文件，可以取消注释以下代码
    # image_path = 'your_image.jpg'
    # original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # 这里我们创建一个更复杂的测试图像
    size = 300
    real_test = np.ones((size, size), dtype=np.uint8) * 128
    
    # 创建更丰富的测试图案
    x, y = np.meshgrid(np.linspace(0, 4*np.pi, size), np.linspace(0, 4*np.pi, size))
    pattern = (np.sin(x) * np.cos(y) * 127 + 128).astype(np.uint8)
    real_test = cv2.addWeighted(real_test, 0.7, pattern, 0.3, 0)
    
    # 添加文字形状的边缘
    cv2.putText(real_test, 'EDGE', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, 255, 5)
    cv2.putText(real_test, 'TEST', (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 3, 0, 5)
    
    # 测试Sobel和拉普拉斯
    sobel_sharp, sobel_grad = first_order_sharpen(real_test, 'sobel', 0.4)
    laplacian_sharp, laplacian_map = laplacian_sharpen(real_test, 'with_diagonal')
    
    # 显示结果
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 4, 1)
    plt.imshow(real_test, cmap='gray')
    plt.title('测试图像')
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.imshow(sobel_grad, cmap='gray')
    plt.title('Sobel梯度')
    plt.axis('off')
    
    plt.subplot(1, 4, 3)
    plt.imshow(sobel_sharp, cmap='gray')
    plt.title('Sobel锐化')
    plt.axis('off')
    
    plt.subplot(1, 4, 4)
    plt.imshow(laplacian_sharp, cmap='gray')
    plt.title('拉普拉斯锐化')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
    test_real_image()