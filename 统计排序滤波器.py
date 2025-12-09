import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.rcParams['font.family'] = ['Arial Unicode MS']

def median_filter(image, kernel_size=3):
    """
    中值滤波器实现
    Args:
        image: 输入图像（灰度图）
        kernel_size: 滤波器大小，必须是奇数
    Returns:
        滤波后的图像
    """
    # 获取图像尺寸
    height, width = image.shape
    
    # 创建输出图像
    filtered_image = np.zeros_like(image)
    
    # 计算填充大小
    pad = kernel_size // 2
    
    # 对图像进行边界填充
    # 使用反射填充能更好地保留边缘（对比 constant=0 的硬边）
    padded_image = np.pad(image, pad, mode='reflect')
    
    # 应用中值滤波
    for i in range(height):
        for j in range(width):
            # 提取邻域
            neighborhood = padded_image[i:i+kernel_size, j:j+kernel_size]
            
            # 将邻域像素展平并排序
            sorted_pixels = np.sort(neighborhood.flatten())
            
            # 取中值（根据PPT中的公式）
            n = kernel_size * kernel_size
            if n % 2 == 1:  # n为奇数
                median_value = sorted_pixels[n // 2]
            else:  # n为偶数
                median_value = (sorted_pixels[n // 2 - 1] + sorted_pixels[n // 2]) // 2
            
            filtered_image[i, j] = median_value
    
    return filtered_image

def add_salt_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
    """
    添加椒盐噪声
    """
    noisy_image = image.copy()
    height, width = image.shape
    
    # 添加盐噪声（白点）
    salt_mask = np.random.random((height, width)) < salt_prob
    noisy_image[salt_mask] = 255
    
    # 添加椒噪声（黑点）
    pepper_mask = np.random.random((height, width)) < pepper_prob
    noisy_image[pepper_mask] = 0
    
    return noisy_image

def visualize_median_process(image, noisy_image, filtered_image, i=None, j=None, kernel_size=3):
    """
    可视化指定像素位置的中值滤波过程：邻域 -> 排序 -> 取中值
    image/noisy/filtered 为灰度图 (H, W)
    i, j 为要观察的像素位置；默认取图像中心
    """
    h, w = noisy_image.shape
    if i is None:
        i = h // 2
    if j is None:
        j = w // 2

    pad = kernel_size // 2
    padded = np.pad(noisy_image, pad, mode='reflect')
    neighborhood = padded[i:i+kernel_size, j:j+kernel_size]
    sorted_pixels = np.sort(neighborhood.flatten())
    n = kernel_size * kernel_size
    if n % 2 == 1:
        median_value = sorted_pixels[n // 2]
    else:
        median_value = (sorted_pixels[n // 2 - 1] + sorted_pixels[n // 2]) // 2

    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1])

    ax0 = fig.add_subplot(gs[:, 0])
    ax0.imshow(noisy_image, cmap='gray', vmin=0, vmax=255)
    ax0.set_title(f'含噪图像（红框为 {kernel_size}×{kernel_size} 邻域）')
    ax0.add_patch(plt.Rectangle((j-pad-0.5, i-pad-0.5), kernel_size, kernel_size,
                                fill=False, edgecolor='red', linewidth=2))
    ax0.scatter([j], [i], s=30, c='yellow', marker='x')
    ax0.axis('off')

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(neighborhood, cmap='gray', vmin=0, vmax=255)
    ax1.set_title('邻域矩阵')
    for ii in range(kernel_size):
        for jj in range(kernel_size):
            ax1.text(jj, ii, f"{int(neighborhood[ii, jj])}", color='cyan', ha='center', va='center', fontsize=9)
    ax1.set_xticks([]); ax1.set_yticks([])

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(sorted_pixels, 'o-', color='tab:blue', label='排序后的像素')
    mid_idx = n // 2
    ax2.axvline(mid_idx, color='orange', linestyle='--', label='中位索引')
    ax2.axhline(median_value, color='red', linestyle=':', label=f'中值={int(median_value)}')
    ax2.set_title('排序结果与中值')
    ax2.set_xlabel('索引'); ax2.set_ylabel('像素值')
    ax2.legend()

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.imshow(filtered_image, cmap='gray', vmin=0, vmax=255)
    ax3.set_title('中值滤波结果（黄色为观察像素）')
    ax3.scatter([j], [i], s=30, c='yellow', marker='x')
    ax3.axis('off')

    ax4 = fig.add_subplot(gs[1, 2])
    ax4.bar(['原始', '含噪', '滤波'], [image[i, j], noisy_image[i, j], filtered_image[i, j]], color=['gray', 'tab:blue', 'tab:green'])
    ax4.set_ylim(0, 255)
    ax4.set_title(f'像素({i},{j}) 值变化')
    ax4.set_ylabel('灰度')

    plt.tight_layout()
    return fig


def animate_median_filter(image, kernel_size=3, interval_ms=150):
    """
    使用动画演示中值滤波：
    - 左上：当前含噪图像并高亮当前窗口
    - 右上：当前窗口邻域矩阵（叠加数字）
    - 右下：当前窗口像素排序曲线并高亮中值
    - 左下：输出图像的构建进度（已计算位置更新为中值）
    """
    h, w = image.shape
    pad = kernel_size // 2
    noisy = add_salt_pepper_noise(image, salt_prob=0.05, pepper_prob=0.05)
    padded = np.pad(noisy, pad, mode='reflect')
    out = np.zeros_like(image, dtype=np.uint8)

    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])

    ax_noisy = fig.add_subplot(gs[0, 0])
    ax_neigh = fig.add_subplot(gs[0, 1])
    ax_out = fig.add_subplot(gs[1, 0])
    ax_sort = fig.add_subplot(gs[1, 1])

    im_noisy = ax_noisy.imshow(noisy, cmap='gray', vmin=0, vmax=255)
    ax_noisy.set_title('含噪图像（红框为当前窗口）')
    rect = plt.Rectangle((0, 0), kernel_size, kernel_size, fill=False, edgecolor='red', linewidth=2)
    ax_noisy.add_patch(rect)
    ax_noisy.set_xticks([]); ax_noisy.set_yticks([])

    im_neigh = ax_neigh.imshow(np.zeros((kernel_size, kernel_size)), cmap='gray', vmin=0, vmax=255)
    ax_neigh.set_title('邻域矩阵')
    txts = [ax_neigh.text(j, i, '', color='cyan', ha='center', va='center', fontsize=9)
            for i in range(kernel_size) for j in range(kernel_size)]
    ax_neigh.set_xticks([]); ax_neigh.set_yticks([])

    im_out = ax_out.imshow(out, cmap='gray', vmin=0, vmax=255)
    ax_out.set_title('输出图像（构建中）')
    ax_out.set_xticks([]); ax_out.set_yticks([])

    line_sort, = ax_sort.plot([], [], 'o-', color='tab:blue', label='排序后的像素')
    mid_line_v = ax_sort.axvline(0, color='orange', linestyle='--', label='中位索引')
    mid_line_h = ax_sort.axhline(0, color='red', linestyle=':', label='中值')
    ax_sort.set_xlim(0, kernel_size * kernel_size - 1)
    ax_sort.set_ylim(0, 255)
    ax_sort.set_title('排序结果与中值')
    ax_sort.set_xlabel('索引'); ax_sort.set_ylabel('像素值')
    ax_sort.legend(loc='upper left')

    coords = [(i, j) for i in range(h) for j in range(w)]

    def update(frame_idx):
        i, j = coords[frame_idx]
        # 更新红框位置
        rect.set_xy((j - pad - 0.5, i - pad - 0.5))

        # 邻域与中值
        neigh = padded[i:i + kernel_size, j:j + kernel_size]
        im_neigh.set_data(neigh)
        for idx, t in enumerate(txts):
            rr = idx // kernel_size
            cc = idx % kernel_size
            t.set_text(str(int(neigh[rr, cc])))

        sp = np.sort(neigh.flatten())
        n = kernel_size * kernel_size
        if n % 2 == 1:
            median_value = sp[n // 2]
            mid_idx = n // 2
        else:
            median_value = (sp[n // 2 - 1] + sp[n // 2]) // 2
            mid_idx = n // 2
        line_sort.set_data(np.arange(n), sp)
        mid_line_v.set_xdata([mid_idx, mid_idx])
        mid_line_h.set_ydata([median_value, median_value])

        # 输出图像写入当前位置
        out[i, j] = median_value
        im_out.set_data(out)

        return im_noisy, rect, im_neigh, *txts, line_sort, mid_line_v, mid_line_h, im_out

    ani = FuncAnimation(fig, update, frames=len(coords), interval=interval_ms, blit=False, repeat=False)
    plt.tight_layout()
    return fig, ani

# 示例使用
if __name__ == "__main__":
    # 创建一个简单的测试图像
    test_image = np.ones((120, 120), dtype=np.uint8) * 128  # 灰色背景
    
    # 添加一些测试图案
    test_image[20:40, 20:40] = 200  # 亮色方块
    test_image[60:80, 60:80] = 50   # 暗色方块
    
    # 添加椒盐噪声
    noisy_image = add_salt_pepper_noise(test_image, salt_prob=0.05, pepper_prob=0.05)
    
    # 应用中值滤波
    ksize = 3
    filtered_image = median_filter(noisy_image, kernel_size=ksize)
    
    # 显示结果
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(test_image, cmap='gray', vmin=0, vmax=255)
    plt.title('原始图像')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(noisy_image, cmap='gray', vmin=0, vmax=255)
    plt.title('添加椒盐噪声后')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(filtered_image, cmap='gray', vmin=0, vmax=255)
    plt.title(f'中值滤波后 (k={ksize})')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # 打印算法步骤说明
    print("中值滤波算法步骤：")
    print("1. 定义邻域大小（如3×3）")
    print("2. 对图像每个像素点：")
    print("   - 提取邻域内所有像素值")
    print("   - 将像素值按大小排序：x₁ ≤ x₂ ≤ ... ≤ xₙ")
    print("   - 取中值作为该点新值")
    print("3. 根据PPT公式：")
    print("   - n为奇数时：y = x₍ₙ₊₁₎/₂")
    print("   - n为偶数时：y = ½[x₍ₙ/₂₎ + x₍ₙ/₂₊₁₎]")

    # 可视化某个像素位置的处理过程
    fig = visualize_median_process(test_image, noisy_image, filtered_image, i=None, j=None, kernel_size=ksize)
    plt.show()

    # 动画演示：窗口扫描整幅图像
    fig_anim, ani = animate_median_filter(test_image, kernel_size=ksize, interval_ms=100)
    plt.show()