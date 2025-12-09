#!/bin/bash
# 第三章作业批量运行脚本

echo "======================================"
echo "第三章作业批量运行"
echo "======================================"

# 激活conda环境
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate ml

# 进入代码目录
cd "$(dirname "$0")/代码"

echo ""
echo "运行作业1: 直方图均衡化..."
python 作业1_直方图均衡化.py

echo ""
echo "运行作业2: 拉普拉斯锐化..."
python 作业2_拉普拉斯锐化.py

echo ""
echo "运行作业3: 位平面分析..."
python 作业3_位平面分析.py

echo ""
echo "======================================"
echo "所有作业运行完成！"
echo "结果保存在: 实验/第三章/输出结果/"
echo "======================================"
