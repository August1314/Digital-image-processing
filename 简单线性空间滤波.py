import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import numpy as np



class SimpleSpatialFilterDemo:
    def __init__(self, root):
        self.root = root
        self.root.title("简单线性空间滤波 - 演示")
        self.root.geometry("900x600")

        self.original_image = None  # numpy array (H, W) grayscale
        self.filtered_image = None   # numpy array (H, W) grayscale

        self._build_ui()

    def _build_ui(self):
        top_bar = ttk.Frame(self.root)
        top_bar.pack(fill=tk.X, padx=10, pady=10)

        load_btn = ttk.Button(top_bar, text="选择图片", command=self.on_select_image)
        load_btn.pack(side=tk.LEFT)

        ttk.Label(top_bar, text="滤波器:").pack(side=tk.LEFT, padx=(15, 5))
        self.filter_var = tk.StringVar(value="均值")
        filter_box = ttk.Combobox(top_bar, textvariable=self.filter_var, width=10,
                                   values=["均值", "Gaussian", "锐化"], state="readonly")
        filter_box.pack(side=tk.LEFT)
        filter_box.bind("<<ComboboxSelected>>", lambda e: self.apply_and_show())

        ttk.Label(top_bar, text="核大小:").pack(side=tk.LEFT, padx=(15, 5))
        self.ksize_var = tk.IntVar(value=3)
        ksize_box = ttk.Combobox(top_bar, textvariable=self.ksize_var, width=5,
                                  values=[3, 5, 7], state="readonly")
        ksize_box.pack(side=tk.LEFT)
        ksize_box.bind("<<ComboboxSelected>>", lambda e: self.apply_and_show())

        ttk.Label(top_bar, text="Gaussian σ:").pack(side=tk.LEFT, padx=(15, 5))
        self.sigma_var = tk.DoubleVar(value=1.0)
        sigma = ttk.Scale(top_bar, from_=0.5, to=3.0, value=1.0, orient=tk.HORIZONTAL,
                          command=lambda _v: self.on_sigma_change())
        sigma.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        # display area
        display = ttk.Frame(self.root)
        display.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        left = ttk.LabelFrame(display, text="原图")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        right = ttk.LabelFrame(display, text="滤波后")
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        self.canvas_orig = tk.Canvas(left, bg="white")
        self.canvas_orig.pack(fill=tk.BOTH, expand=True)
        self.canvas_filt = tk.Canvas(right, bg="white")
        self.canvas_filt.pack(fill=tk.BOTH, expand=True)

    def on_sigma_change(self):
        # 在调整过程中若已载入图像，实时应用
        if self.original_image is not None and self.filter_var.get() == "Gaussian":
            self.apply_and_show()

    def on_select_image(self):
        file_path = filedialog.askopenfilename(
            title="选择图像文件",
            filetypes=[("图像文件", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
        )
        if not file_path:
            return
        image = Image.open(file_path)
        if image.mode != "L":
            image = image.convert("L")
        # 缩放到合适大小以便演示
        max_side = 400
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        self.original_image = np.array(image)
        self.apply_and_show()

    def apply_and_show(self):
        if self.original_image is None:
            return
        k = int(self.ksize_var.get())
        if k % 2 == 0:
            k += 1
        filt_type = self.filter_var.get()

        if filt_type == "均值":
            kernel = np.ones((k, k), dtype=np.float64) / (k * k)
        elif filt_type == "Gaussian":
            sigma = float(self.sigma_var.get())
            kernel = self._gaussian_kernel(k, sigma)
        else:  # 锐化：拉普拉斯增强（简单示例）
            # 先做均值平滑再与原图做增强（unsharp masking 的简化版本）
            blur_kernel = np.ones((k, k), dtype=np.float64) / (k * k)
            blurred = self._convolve2d(self.original_image, blur_kernel)
            sharpened = np.clip(self.original_image.astype(np.float64) * 1.5 - blurred * 0.5, 0, 255)
            self.filtered_image = sharpened.astype(np.uint8)
            self._render()
            return

        self.filtered_image = self._convolve2d(self.original_image, kernel)
        self._render()

    def _gaussian_kernel(self, ksize: int, sigma: float) -> np.ndarray:
        half = ksize // 2
        y, x = np.mgrid[-half:half + 1, -half:half + 1]
        g = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        g /= g.sum()
        return g

    def _convolve2d(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        # 反射填充，保持边缘效果更自然
        k = kernel.shape[0]
        pad = k // 2
        padded = np.pad(image.astype(np.float64), pad_width=pad, mode="reflect")
        h, w = image.shape
        out = np.zeros_like(image, dtype=np.float64)
        # 朴素实现，足够用于演示
        for i in range(h):
            for j in range(w):
                region = padded[i:i + k, j:j + k]
                out[i, j] = np.sum(region * kernel)
        out = np.clip(out, 0, 255).astype(np.uint8)
        return out

    def _render(self):
        # 原图
        orig_pil = Image.fromarray(self.original_image)
        orig_photo = ImageTk.PhotoImage(orig_pil)
        self.canvas_orig.delete("all")
        self.canvas_orig.create_image(self.canvas_orig.winfo_width() // 2,
                                      self.canvas_orig.winfo_height() // 2,
                                      image=orig_photo, anchor=tk.CENTER)
        self.canvas_orig.image = orig_photo

        # 滤波后
        filt_pil = Image.fromarray(self.filtered_image)
        filt_photo = ImageTk.PhotoImage(filt_pil)
        self.canvas_filt.delete("all")
        self.canvas_filt.create_image(self.canvas_filt.winfo_width() // 2,
                                      self.canvas_filt.winfo_height() // 2,
                                      image=filt_photo, anchor=tk.CENTER)
        self.canvas_filt.image = filt_photo


if __name__ == "__main__":
    root = tk.Tk()
    app = SimpleSpatialFilterDemo(root)
    root.mainloop()


