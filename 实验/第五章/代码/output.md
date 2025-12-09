(base) lianglihang@MacBook-Air-5 Digital-image-processing % /opt/anaconda3/envs/ml/bin/python /Users/lianglihang/Downloads/Digita
l-image-processing/实验/第五章/代码/作业1_椒盐噪声与中值滤波.py
======================================================================
                         第五章作业 #1：椒盐噪声 + 中值滤波                         
======================================================================
加载原始图像: /Users/lianglihang/Downloads/Digital-image-processing/实验/第五章/图a.png
图像尺寸: (448, 464)
加载参考图像: /Users/lianglihang/Downloads/Digital-image-processing/实验/第五章/5.10b.png
参考图像尺寸与原图不一致，执行双线性缩放以便比较。

添加椒盐噪声: Salt & Pepper Noise (Pa=0.10, Pb=0.10, salt=255, pepper=0)

应用中值滤波: Median Filter (kernel=3, pad=reflect)

中值滤波后 vs 5.10(b): MSE=1802.52, PSNR=15.57 dB
噪声图像 vs 5.10(b): PSNR=10.75 dB

全部结果已保存至: /Users/lianglihang/Downloads/Digital-image-processing/实验/第五章/输出结果/作业1_椒盐噪声与中值滤波

与 5.10(b) 的主要区别：
- 中值滤波后的图像与参考图 5.10(b) 存在 MSE=1802.52 的残余差异，
  主要来自高概率噪声导致的亮/暗斑点仍有少量残留。
- PSNR 为 15.57 dB，说明中值滤波消除了大部分离群像素，
  但在边缘区域仍出现轻微模糊，这是 5.10(b) 中较为锐利细节所没有的。
- 相比 5.10(b)，本实验的差异图显示局部导线/芯片脚区域的细节对噪声更敏感，
  说明在 Pa=Pb=0.2 的极端噪声下，仍可考虑更大的窗口或多次滤波以进一步接近参考结果。
======================================================================

(ml) lianglihang@MacBook-Air-5 Digital-image-processing % /opt/anaconda3/envs/ml/bin/python /Users/lianglihang/Downloads/Digital-
image-processing/实验/第五章/代码/作业2_运动模糊与维纳复原.py
======================================================================
                          第五章作业 #2：运动模糊与维纳复原                          
======================================================================
加载原始图像: /Users/lianglihang/Downloads/Digital-image-processing/实验/第五章/图b.png
图像尺寸: (688, 688)

应用运动模糊: Motion Blur (T=1, a=0.08, b=0.08)
添加高斯噪声: Gaussian Noise (mean=0.0, var=10.0)

执行维纳复原: Wiener Filter (k=0.02)

指标: PSNR_blur=12.30 dB, PSNR_noisy=12.29 dB, PSNR_restored=18.70 dB

所有结果已保存至: /Users/lianglihang/Downloads/Digital-image-processing/实验/第五章/输出结果/作业2_运动模糊与维纳复原

实验观察：
- 运动模糊后图像的 PSNR 下降到 12.30 dB，细节沿 +45° 方向被拖尾。
- 加入方差为 10 的高斯噪声进一步将 PSNR 降至 12.29 dB，噪点在暗背景尤为明显。
- 维纳滤波 (K=0.01) 将 PSNR 提升至 18.70 dB，能有效抑制噪声并重建文字边缘，
  但由于频谱零点仍有限制，残留的条纹和轻微过度增强仍存在。
======================================================================