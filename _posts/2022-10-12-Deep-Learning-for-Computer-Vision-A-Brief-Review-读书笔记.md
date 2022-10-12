---
layout: post
title: Deep Learning for Computer Vision A Brief Review 读书笔记
categories: 学习记录 AI 论文阅读
tags: AI CNN 编码器 玻尔兹曼机 读书笔记
---
# Deep Learning for Computer Vision: A Brief Review 读书笔记

该论文介绍了卷积神经网络、深度玻尔兹曼机（Deep Boltzmann Machines）、深度信念网络（Deep Belief Networks）以及堆叠去噪自编码器（Stacked Denoising Autoencoders）的特点和架构，并说明了其优缺点。

本笔记主要详细介绍了RBM：受限玻尔兹曼机，对于其他方法做大概介绍。

## 卷积神经网络

卷积三大要素：卷积，池化，全连接层。

卷积层的数学原理：事实上是对全连接神经元的一个简化，给定网络在第d层输出的特征图$\mathbf{y}^{(d)}$，可以知道其计算公式如下：

$\mathbf{y}^{(d)}=\sigma(\mathbf{W}\mathbf{y}^{(d-1)}+\mathbf{b})$

由于$\mathbf{y}^{(d)}$是由若干通道构成的，每个通道下各有一个特征图，因此可以得出权重矩阵$\mathbf{W}$的形式：

$$\left[\begin{array}{cccc}
\mathbf{w} & 0 & \cdots & 0 \\
0 & \mathbf{w} & \cdots & 0 \\
\vdots & \cdots & \ddots & \vdots \\
0 & \cdots & 0 & \mathbf{w}
\end{array}\right]$$

$\mathbf{w} $是一个矩阵，大小与$\mathbf{y^{(d-1)}}$中的感受野大小一致。

考虑到$N\times N$ 大小的一个特征图，经过$m\times m$的卷积核处理后大小变为$N-m+1\times N-m+1$那么特征图中的各个元素计算如下：

$$\mathbf{y}_{i j}^{(d)}=\sigma\left(x_{i j}^{(d)}+b\right)$$

其中x的计算式为：

$$x_{i j}^{(d)}=\sum\limits_{\alpha=0}^{m-1} \sum_\limits{b=0}^{m-1} w_{\alpha b} \mathbf{y}_{(i+\alpha)(j+b)}^{(d-1)}$$

池化层的作用是降低数据的维度，也常常被称作是下采样（subsampling），一般来说，使用最大池化可以使得收敛速度加快。在池化方法上还有随机池化，空间金字塔池化以及定义池化等。

>Also there are a number of other variations of the pooling layer in the literature, each inspired by different motivations and serving distinct needs, for example, stochastic pooling [27], spatial pyramid pooling [28, 29], and def-pooling [30].

事实上CNN的提出是采用了以下三个想法：

1. 局部感受野（local receptive fields）

   局部感受野的存在使得边缘，角点等局部特征被检测到，在后续的卷积过程中被提取出更高维度的抽象。

2. 绑定权重（tied weights）

   在每一个卷积层中，每个单元权重都是一致的。

3. 空间下采样（spatial subsampling）

在处理神经网络的过拟合问题时，使用随机池化（stochastic pooling），dropout ，数据增强等方法。还使用预训练等方法来加快训练速度。

## 深度信念网络

### 受限玻尔兹曼机（RBM）

结构是一个二分图，一部分为可见层（v）另一部分是隐藏层（h）。两个层之间互相连接。首先定义了系统的能量：

$$E(\mathbf{v}, \mathbf{h} ; \theta)=-\sum_{i=1}^{D} \sum_{j=1}^{F} W_{i j} v_{i} h_{j}-\sum_{i=1}^{D} b_{i} v_{i}-\sum_{j=1}^{F} \alpha_{j} h_{j}$$

$$E=e^{b^Tv+C^Th+v^TWh}$$

在一个视频里举了个很棒很直观的例子：

![image-20221007160400695](https://lh-picbed.oss-cn-chengdu.aliyuncs.com/image-20221007160400695.png)

我们要用概率来衡量事情发生的可能性，给出概率的计算公式：

$$\begin{array}{l}
P(\mathbf{v}, \mathbf{h} ; \theta)=\frac{1}{\mathscr{Z}(\theta)} \exp (-E(\mathbf{v}, \mathbf{h} ; \theta)) \\
\mathscr{Z}(\theta)=\sum_{\mathbf{v}} \sum_{\mathbf{h}} \exp (-E(\mathbf{v}, \mathbf{h} ; \theta))
\end{array}$$

事实上在计算概率的时候引入指数，很大程度上是因为为了消除负数的影响。在此基础上，我们的目标是将真实出现的情况确定下来，也就是$P(v)$最大。

![image-20221007162305650](https://lh-picbed.oss-cn-chengdu.aliyuncs.com/image-20221007162305650.png)

在这一过程中需要计算$v,h$的条件概率，以下关于RBM的讲解来自于[这个视频](https://www.youtube.com/watch?v=FJ0z3Ubagt4)

![image-20221007215659328](https://lh-picbed.oss-cn-chengdu.aliyuncs.com/image-20221007215659328.png)

第二部到第三部的步骤为：

$$\begin{aligned}分母&=\frac{1}{Z}\sum\limits_he^{b^Tv+C^Th+v^TWh}&=\frac{1}{Z}e^{b^T}\sum_he^{C^Th+v^TWh}\end{aligned}$$

其中$e^{b^T}$和$\frac{1}{Z}$都可以被消除。并且玲分母余下的求和项为$Z'$就得到推导的式子。事实上最后的乘积形式证明了$h$是独立的。 注意$W_{:j}$代表的是第j列。我所疑惑的向量形式的概率事实上是随机变量形成向量组的特殊情况，为了表达方便还是写作向量的形式，本质上还是很多随机变量的联合概率（joint probability）。

![image-20221007221216608](https://lh-picbed.oss-cn-chengdu.aliyuncs.com/image-20221007221216608.png)

在$h_j$等于0的情况下，得到sigmoid函数。

可以使用吉布斯(Gibbs)采样的方法来获得采样数据训练参数。该方法由于是独立的，可以做到并行计算。

![image-20221007222032998](https://lh-picbed.oss-cn-chengdu.aliyuncs.com/image-20221007222032998.png)

采样的概率可以使用下面的方法进行：假设$p(h_j|v)=0.2$那么可以生成$[0,1]$之间的随机数来控制采样与否，如果小于$p$的话就采样。

在学习参数方面：使用概率论中的极大似然法来更新权重，具体方法如下：

$$\begin{aligned}
\ell(\text { W.b.c }) &=\sum_{t=1}^{n} \log P\left(\mathbf{v}^{(t)}\right) \\
&=\sum_{t=1}^{n} \log \sum_{\mathbf{h}} P\left(\mathbf{v}^{(t)}, \mathbf{h}\right) \\
&\left.=\sum_{t=1}^{n} \log \sum_{\mathbf{h}} \exp \left\{-E\left(\mathbf{v}^{(t)}, \mathbf{h}\right)\right\}\right)-n \log Z \\
&\left.=\sum_{t=1}^{n} \log \sum_{\mathbf{h}} \exp \left\{-E\left(\mathbf{v}^{(t)}, \mathbf{h}\right)\right\}\right)-n \log \sum_{\mathbf{v} \cdot \mathbf{h}} \exp \{-E(\mathbf{v} \cdot \mathbf{h})\}
\end{aligned}$$

使用极大似然法，需要对参数$\theta$求导，参数在这里是$b,c,W$,给出以下式子：

$$\begin{aligned}
\theta=\{\mathbf{b}, \mathbf{c}, W\}: \\
\ell(\theta)=&\left.\sum_{t=1}^{n} \log \sum_{\mathbf{h}} \exp \left\{-E\left(\mathbf{v}^{(t)}, \mathbf{h}\right)\right\}\right)-n \log \sum_{\mathbf{v}, \mathbf{h}} \exp \{-E(\mathbf{v}, \mathbf{h})\} \\
\nabla_{\theta} \ell(\theta)=&\left.\nabla_{\theta} \sum_{t=1}^{n} \log \sum_{\mathbf{h}} \exp \left\{-E\left(\mathbf{v}^{(t)}, \mathbf{h}\right)\right\}\right)-n \nabla_{\theta} \log \sum_{\mathbf{v}, \mathbf{h}} \exp \{-E(\mathbf{v}, \mathbf{h})\} \\
=& \sum_{t=1}^{n} \frac{\sum_{\mathbf{h}} \exp \left\{-E\left(\mathbf{v}^{(t)}, \mathbf{h}\right)\right\} \nabla \theta-E\left(\mathbf{v}^{(t)}, \mathbf{h}\right)}{\sum_{\mathbf{h}} \exp \left\{-E\left(\mathbf{v}^{(t)}, \mathbf{h}\right)\right\}} \\
&-n \frac{\sum_{\mathbf{v}, \mathbf{h}} \exp \{-E(\mathbf{v}, \mathbf{h})\} \nabla \theta-E(\mathbf{v}, \mathbf{h})}{\sum_{\mathbf{v}, \mathbf{h}} \exp \{-E(\mathbf{v}, \mathbf{h})\}} \\
=& \sum_{t=1}^{n} \mathbb{E}_{P(\mathbf{h}, \mathbf{v}(\mathbf{t}))}\left[\nabla_{\theta}-E\left(\mathbf{v}^{(t)}, \mathbf{h}\right)\right]-n \mathbb{E}_{P(\mathbf{h}, \mathbf{v})}\left[\nabla_{\theta}-E(\mathbf{v}, \mathbf{h})\right]
\end{aligned}$$

事实上求导后的两项可以视为期望。注意到极大似然的过程中是让导数值最小，使得第一项小而第二项（不含符号）大，第一项和现在观察的数据情况相关，而第二项同模型整体相关。期望$\mathbb{E}(x)$中的$x$可以写为以下式子：

$$\begin{aligned}
\nabla_{w}-E(\mathbf{v}, \mathbf{h}) &=\frac{\partial}{\partial W}\left(\mathbf{b}^{T} \mathbf{v}+\mathbf{c}^{T} h+\mathbf{v}^{\top} W \mathbf{h}\right) \\
&=\mathbf{h v}^{T} \\
\nabla_{\mathbf{b}}-E(\mathbf{v}, \mathbf{h}) &=\frac{\partial}{\partial \mathbf{b}}\left(\mathbf{b}^{T} \mathbf{v}+\mathbf{c}^{\top} h+\mathbf{v}^{T} W \mathbf{h}\right) \\
&=\mathbf{v} \\
\nabla_{\mathbf{c}}-E(\mathbf{v}, \mathbf{h}) &=\frac{\partial}{\partial \mathbf{c}}\left(\mathbf{b}^{\top} \mathbf{v}+\mathbf{c}^{\top} h+\mathbf{v}^{\top} W \mathbf{h}\right) \\
&=\mathbf{h}
\end{aligned}$$

在推导的过程中发现计算期望的过程非常困难，使用点估计（point estmate）来取代期望。具体的方法是使用$v^{(t)}$得到$p(h|v^{t})$，计算$h^{t+1}$随后用$t+1$来计算后续的值，迭代后最终得到$\tilde{h},\tilde{v}$，并作为期望：

$$\mathbb{E}_{P(\mathbf{h}, \mathbf{v})}\left[\nabla_{\theta}-E(\mathbf{v}, \mathbf{h})\right] \approx \nabla_{\theta}-\left.E(\mathbf{v}, \mathbf{h})\right|_{\mathbf{v}=\tilde{\mathbf{v}}, \mathbf{h}=\tilde{\mathbf{h}}}$$

![image-20221007231935865](https://lh-picbed.oss-cn-chengdu.aliyuncs.com/image-20221007231935865.png)

![image-20221007231956117](https://lh-picbed.oss-cn-chengdu.aliyuncs.com/image-20221007231956117.png)

![image-20221007232116913](https://lh-picbed.oss-cn-chengdu.aliyuncs.com/image-20221007232116913.png)

在后续的发展中，受限玻尔兹曼机被叠加为多层，这种结构也被称作是深度信念网络。在早期时候，玻尔兹曼机也被用于作为自动编码器（auto-encoder）。

自动编码器是一个类似于沙漏的形式，能够将高维度的数据编码为低维度，并且还原回来。每层之间都用权重相连接。

<img src="https://lh-picbed.oss-cn-chengdu.aliyuncs.com/image-20221008204729992.png" alt="image-20221008204729992" style="zoom: 25%;" />



the basis that this auto encoder computes is spam the same space as eigenvectors in PCA

编码器计算的基础是与PCA张成的向量空间维数相同。需要注意PCA和编码器之间的关系。如果是一个中间较为宽的自动编码器形状，并且加入噪音层$\tilde{x}$，那么便可以构成降噪自动编码器（denoising auto-encoder）,这样$h$层中的冗余信息便可以被利用到。

<img src="https://lh-picbed.oss-cn-chengdu.aliyuncs.com/image-20221008210439662.png" alt="image-20221008210439662" style="zoom:25%;" />

### 深度信念网络（DBN）

如同前文所述，深度信念网络是受限玻尔兹曼机的堆叠，但是头两层之外的层只有隐藏层到可见层的单向链接。将在后面的工作中给出如何训练一个深度信念网络。为了提升在CV方面的泛用性，使用卷积深度信念网络来利用相邻的像素信息，作为一个新的方法。

![image-20221008213301420](https://lh-picbed.oss-cn-chengdu.aliyuncs.com/image-20221008213301420.png)

### 深度玻尔兹曼机（DBM）

深度玻尔兹曼机的是由玻尔兹曼机堆叠而成的，正因此奇数和偶数层之间是条件独立的。其训练过程是通过逐步堆叠贪心训练实现的，在文章中有体现[^1]

[^1]:R. Salakhutdinov and H. Larochelle, “Efficient learning of deep Boltzmann machines,” in Proceedings of the AISTATS, 2010.
DBM的一大优势为：可以捕获输入数据的多层复杂表示，由于是无监督学习的方法，对于数据的标签并没有要求。其优化参数的过程是联合优化的，对于跨模态的数据训练很有用。

其缺点是数据集过大时，推理时需要的计算资源消耗太大。

## 堆叠编码器

在RBM中已经对编码器有了介绍，输入和输出之间的误差被称之为重构误差（reconstruction error），目标就是使这一误差最小化。如果隐藏层是线性的，那么就是将输入进行投影变换，与PCA相似。如果隐藏层非线性，则可以捕获输入的多模态。均方误差常常被用于衡量重构的误差。

<img src="https://lh-picbed.oss-cn-chengdu.aliyuncs.com/image-20221008220758788.png" alt="image-20221008220758788" style="zoom:25%;" />

而当输入是位向量或者是概率值的时候，可以使用交叉熵来计算其loss：

$$L=-\sum_{i} \mathbf{x}_{i} \log \mathbf{f}_{i}(\mathbf{r}(\mathbf{x}))+\left(1-\mathbf{x}_{i}\right) \log \left(1-\mathbf{f}_{i}(\mathbf{r}(\mathbf{x}))\right) $$

### 去噪编码器

去噪编码器的原理是让输入的信息产生噪声，但是目的是还原出原始信息。还原的过程是由编码器的统计依赖关系确定的。在应用方面上首先直接用于去噪[^2]

[^2]: P. Gallinari, Y. LeCun, S.Thiria, and F. Fogelman-Soulie, “Mem-oires associatives distribuees,” in Proceedings of the in Proceed-ings of COGNITIVA 87, Paris, La Villette, 1987.

，引入了自动编码器链接到生成模型，在深层无监督预训练这一领域发挥了作用。 如果是将输入的某些信息随机抹除，那么当去噪编码器学习到其分布特性的时候，去噪编码器能够填补缺失的输入信息。[^3]

[^3]: P. Vincent, H. Larochelle, Y. Bengio, and P.-A. Manzagol, “Ex-tracting and composing robust features with denoising autoen-coders,” in in Proceedings of the Twenty-fifth International Conference on Machine Learning (ICML’08), W. W. Cohen, A. McCallum, and S. T. Roweis, Eds., pp. 1096–1103, ACM, 2008.

这一论文的一大优点是，成功使用该方法进行深度架构的无监督预训练并将去噪自动编码器与生成模型联系起来。

### 堆叠的去噪编码器

将去噪编码器堆叠后便可以使用这种方式训练出来一个深度的网络，训练过程可以逐层判断输入输出的误差来进行预训练。

在完成预训练后，需要进行参数微调（fine-tuning）。该步骤的目的是为了将预测的误差最小化，最后在输出层上添加了一个逻辑回归层。

在表现方面上，堆叠去噪编码器和深度信念网络之间差异不大，但是堆叠去噪编码器对于参数化数据的支持更好。

## 总结

在视觉方面，CNN在MNIST上取得了较好的表现，当输入为非视觉数据时（nonvisual）DBN取得了较好的效果，但是难于准确估计联合概率模型以及计算代价过高是该算法的两个缺点。而CNN也并非完美无缺，他的优秀表现是基于标注的数据，也就是所谓的“ground truth”

对于堆叠去噪编码器而言，可以做到较快的训练速度以获得稳定的模型。对于CNN而言，平移，缩放和视角的不变性给其带来了很高的拓展性。

## 应用

CNN中用于目标检测，主要思想是使用候选窗口，提取特征后分类检测是否有对象存在。[^4]还有一些联合语义分割（joint object dection - semantic segmentation）的检测方法被提出。

[^4]: Y. Zhu, R. Urtasun, R. Salakhutdinov, and S. Fidler, “SegDeepM:Exploiting segmentation and context in deep neural networks for object detection,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2015, pp. 4703–4711, USA, June 2015.

其它方面也有一些对象识别的例子，如引入自动堆叠编码器来进行检测。

在姿态检测方面，CV引入了显著性映射（saliency map）来探测和定位事件，深度学习方法用于预训练特征来检测关键帧和对应的事件（如何解决？）。

在动作识别方面还引入了多模态等领域知识，以完成多目标的检测工作。首先对单一的事件特征进行识别，随后使用“与或”来组合特征。

人体姿态估计，根据传感器所提供的的信息数据来获得姿态信息。由于人体的轮廓和外观不同，复杂的光照条件和杂乱的背景也是困难之处。

目前主流的思路是以全局或是以部分作为方法的两种处理方式。前者主要的代表是DeepPose，但是由于姿态向量的不准确性，这一工作准确性不高。

基于局部的方法，训练的素材是基于局部的行为独立训练网络，随后用更为高级的模型来判别整体动作。

