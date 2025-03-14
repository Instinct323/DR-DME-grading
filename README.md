# 项目结构

```
├── engine: 训练相关的引擎
    ├── trainer: 神经网络训练器
    ├── loss: 损失函数 (对比损失，焦点损失)
    ├── crosstab: 混淆矩阵 (多分类)
    ├── evolve: 超参数进化算法
    ├── scaling: EfficientNet 论文中的模型复合缩放
    ├── extension: 模型预训练方案 (e.g., SimCLR, MAE)、线性探测
    └── result: 训练过程信息的结构化存储方法
├── model: 计算机视觉模型
    ├── common: CNN、ViT 网络单元
    ├── model: yaml 文件配置的模型
    ├── fourier: 傅里叶特征映射
    ├── utils: 网络单元的注册方法，局部变量的传递方法
    └── ema: Mean Teacher 的维护方法、半监督学习方法
├── deploy: torch 模型转换
    ├── onnx_run: ONNX 模型管理
    └── openvino: OpenVINO 模型管理
├── utils: 拓展工具包
    ├── utils: 通用工具箱
    ├── imgtf: 图像处理方法 (e.g., 颜色失真, 边界填充)
    ├── data: 数据集相关处理方法 (e.g., 留出法, 欠采样, 数据池)
    ├── gradcam: 梯度加权的类激活映射 (i.e., Grad-CAM)
    ├── prune: 非结构化剪枝
    ├── plot: 可视化的基础函数, 以及部分高阶函数 (e.g., 参数利用率分析)
    ├── rfield: 网络感受野可视化
    └── teacher: 快速知识蒸馏 (FKD) 的知识管理系统
├── config: yaml 配置文件
    ├── vovnet: VoVNet 的模型配置文件、权重文件
    └── ml: SVM、决策树的超参数文件
├── data: 图像裁剪及存储策略, 基于混淆矩阵的性能评估
├── train: 多标签分类的损失函数, 训练方案
└── ml: VoVNet 实现的特征降维, 降维数据的分析, 机器学习算法评估
```