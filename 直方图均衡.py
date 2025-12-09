import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
matplotlib.use('TkAgg')

plt.rcParams['font.family'] = ['Arial Unicode MS']

# 可选的拖放支持：tkinterdnd2
try:
    from tkinterdnd2 import TkinterDnD, DND_FILES  # type: ignore
    DND_AVAILABLE = True
except Exception:
    TkinterDnD = None  # type: ignore
    DND_FILES = None  # type: ignore
    DND_AVAILABLE = False

class HistogramEqualizationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("直方图均衡化演示")
        self.root.geometry("1000x700")
        self.dnd_available = DND_AVAILABLE
        
        # 创建主框架
        main_frame = ttk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 创建拖放区域
        self.create_drop_area(main_frame)
        
        # 创建结果显示区域
        self.create_display_area(main_frame)
        
        # 当前图像
        self.original_image = None
        self.equalized_image = None
        
    def create_drop_area(self, parent):
        drop_frame = ttk.LabelFrame(parent, text="拖放图像区域", padding=10)
        drop_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 拖放标签
        # 使用 tk.Label 而非 ttk.Label，确保 tkinterdnd2 的 dnd_bind 可正常工作
        self.drop_label = tk.Label(
            drop_frame,
            text="将图像文件拖放到此处或点击选择文件",
            bg="lightgray",
            relief="solid",
            padx=20,
            pady=20
        )
        self.drop_label.pack(fill=tk.BOTH, expand=True)
        self.drop_label.bind("<Button-1>", self.select_file)
        
        # 绑定拖放事件
        if self.dnd_available:
            try:
                self.drop_label.drop_target_register(DND_FILES)
                self.drop_label.dnd_bind('<<Drop>>', self.on_drop)
                # 进入/离开高亮反馈
                self.drop_label.dnd_bind('<<DragEnter>>', lambda e: self.drop_label.config(bg="#d0ebff"))
                self.drop_label.dnd_bind('<<DragLeave>>', lambda e: self.drop_label.config(bg="lightgray"))
            except Exception:
                # 若运行时仍不可用，则降级为仅点击
                pass
        else:
            # 无拖放支持时，仅保留点击选择
            pass
        
    def create_display_area(self, parent):
        display_frame = ttk.Frame(parent)
        display_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建左右两个显示区域
        left_frame = ttk.LabelFrame(display_frame, text="原始图像", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        right_frame = ttk.LabelFrame(display_frame, text="均衡化后图像", padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # 图像显示区域
        self.original_canvas = tk.Canvas(left_frame, bg="white", width=300, height=300)
        self.original_canvas.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.equalized_canvas = tk.Canvas(right_frame, bg="white", width=300, height=300)
        self.equalized_canvas.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # 直方图显示区域
        self.create_histogram_area(left_frame, right_frame)
        
    def create_histogram_area(self, left_frame, right_frame):
        # 原始图像直方图
        fig_original, ax_original = plt.subplots(figsize=(4, 2))
        ax_original.set_title("原始直方图")
        ax_original.set_xlabel("灰度值")
        ax_original.set_ylabel("频数")
        self.hist_original = FigureCanvasTkAgg(fig_original, left_frame)
        self.hist_original.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 均衡化后直方图
        fig_equalized, ax_equalized = plt.subplots(figsize=(4, 2))
        ax_equalized.set_title("均衡化后直方图")
        ax_equalized.set_xlabel("灰度值")
        ax_equalized.set_ylabel("频数")
        self.hist_equalized = FigureCanvasTkAgg(fig_equalized, right_frame)
        self.hist_equalized.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.ax_original = ax_original
        self.ax_equalized = ax_equalized
        self.fig_original = fig_original
        self.fig_equalized = fig_equalized
        
    def select_file(self, event=None):
        file_path = filedialog.askopenfilename(
            title="选择图像文件",
            filetypes=[("图像文件", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
        )
        if file_path:
            self.process_image(file_path)
    
    def on_drop(self, event):
        # 解析拖放数据（macOS 上可能为 "{/path with space}"，且可能包含多个文件）
        try:
            paths = self.root.splitlist(event.data)
        except Exception:
            paths = [event.data]

        file_path = paths[0] if paths else None
        # 恢复背景色
        try:
            self.drop_label.config(bg="lightgray")
        except Exception:
            pass

        if file_path:
            self.process_image(file_path)
    
    def process_image(self, file_path):
        try:
            # 打开并处理图像
            image = Image.open(file_path)
            
            # 转换为灰度图像
            if image.mode != 'L':
                image = image.convert('L')
            
            # 调整图像大小以适应显示区域
            max_size = 300
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            self.original_image = np.array(image)
            
            # 应用直方图均衡化
            self.equalized_image = self.histogram_equalization(self.original_image)
            
            # 显示图像和直方图
            self.display_images()
            self.display_histograms()
            
        except Exception as e:
            tk.messagebox.showerror("错误", f"处理图像时出错: {str(e)}")
    
    def histogram_equalization(self, image):
        """实现直方图均衡化算法"""
        # 获取图像尺寸
        h, w = image.shape
        
        # 计算直方图
        hist, bins = np.histogram(image.flatten(), 256, [0, 256])
        
        # 计算累积分布函数 (CDF)
        cdf = hist.cumsum()
        
        # 归一化CDF到[0, 255]范围
        cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
        
        # 使用线性插值计算新的像素值
        equalized_image = np.interp(image.flatten(), bins[:-1], cdf_normalized)
        
        # 重塑为原始图像形状
        equalized_image = equalized_image.reshape(image.shape)
        
        return equalized_image.astype(np.uint8)
    
    def display_images(self):
        # 显示原始图像
        original_pil = Image.fromarray(self.original_image)
        original_photo = ImageTk.PhotoImage(original_pil)
        
        self.original_canvas.delete("all")
        self.original_canvas.create_image(
            self.original_canvas.winfo_width()//2, 
            self.original_canvas.winfo_height()//2, 
            image=original_photo, 
            anchor=tk.CENTER
        )
        self.original_canvas.image = original_photo  # 保持引用
        
        # 显示均衡化后图像
        equalized_pil = Image.fromarray(self.equalized_image)
        equalized_photo = ImageTk.PhotoImage(equalized_pil)
        
        self.equalized_canvas.delete("all")
        self.equalized_canvas.create_image(
            self.equalized_canvas.winfo_width()//2, 
            self.equalized_canvas.winfo_height()//2, 
            image=equalized_photo, 
            anchor=tk.CENTER
        )
        self.equalized_canvas.image = equalized_photo  # 保持引用
    
    def display_histograms(self):
        # 清除之前的直方图
        self.ax_original.clear()
        self.ax_equalized.clear()
        
        # 计算并显示原始图像直方图
        hist_orig, bins_orig = np.histogram(self.original_image.flatten(), 256, [0, 256])
        self.ax_original.bar(bins_orig[:-1], hist_orig, width=1, color='gray', alpha=0.7)
        self.ax_original.set_title("原始直方图")
        self.ax_original.set_xlabel("灰度值")
        self.ax_original.set_ylabel("频数")
        
        # 计算并显示均衡化后直方图
        hist_eq, bins_eq = np.histogram(self.equalized_image.flatten(), 256, [0, 256])
        self.ax_equalized.bar(bins_eq[:-1], hist_eq, width=1, color='blue', alpha=0.7)
        self.ax_equalized.set_title("均衡化后直方图")
        self.ax_equalized.set_xlabel("灰度值")
        self.ax_equalized.set_ylabel("频数")
        
        # 更新画布
        self.hist_original.draw()
        self.hist_equalized.draw()

if __name__ == "__main__":
    # 若可用则使用 TkinterDnD 的 Tk，以启用拖放
    if DND_AVAILABLE and TkinterDnD is not None:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()
    app = HistogramEqualizationApp(root)
    root.mainloop()