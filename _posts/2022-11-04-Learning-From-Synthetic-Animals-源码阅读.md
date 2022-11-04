---
layout: post
title: Learning From Synthetic Animals 源码阅读
categories: 论文阅读 学习记录 CV
tags: 数据处理 代码阅读
---
# Learning From Synthetic Animals 源码阅读

为了分析一个文章的实现思路，接触一个完整的深度学习模型，因此就这篇文章的代码部分进行分析阅读。

## get_cropped_Tigdog文件

### main 函数

![image-20221104102532864](https://lh-picbed.oss-cn-chengdu.aliyuncs.com/image-20221104102532864.png)

首先使用`load_animal`函数顺次读取每个动物的图像列表，标记列表。

随后使用`dataset_filter`函数来获得有效标记较多的标注数据。

创建文件夹，存储需要用的数据：

![image-20221104105730968](https://lh-picbed.oss-cn-chengdu.aliyuncs.com/image-20221104105730968.png)



### load_animal 函数

该部分进行数据读取，输出如下，确切来说，annolist应当是Nx18x3的数据，因为每个图像都会有18个特征，他也是这么处理的：

![image-20221103201617391](https://lh-picbed.oss-cn-chengdu.aliyuncs.com/image-20221103201617391.png)

由于数据集中的`range_file`是matlab的文件格式，因此使用loadmat方法加载。rangefile是视频的元数据？

随后加载shot_id 所对应的 landmark文件，这个文件嵌套了很深的层数，无法知晓是有何含义，只能继续分析。

![image-20221103210508724](https://lh-picbed.oss-cn-chengdu.aliyuncs.com/image-20221103210508724.png)

进入`frame`的for循环，逐一扫描视频帧，也就是下面的注解：

![image-20221103211538645](https://lh-picbed.oss-cn-chengdu.aliyuncs.com/image-20221103211538645.png)

扫描视频帧，目的是读取关键点（19个）以及其是否可见，拼接后，放入列表中，如下：

![image-20221104101721602](https://lh-picbed.oss-cn-chengdu.aliyuncs.com/image-20221104101721602.png)

随后抽取前面十八个特征，作为返回。

### dataset_filter函数

输入为每一类动物的标记函数，输出是筛选过后的标记数据。筛选的标准是超过1/2的标记点是可见的。并且把其序号给设置为img_idxs变量。

### get_cropped_dataset(img_folder, img_list, anno_list, img_idxs, animal) 函数

img_folder是图片的文件夹，img_list, anno_list分别为原始图片的列表，以及标记列表，img_idxs指满足条件的序号。animal是当前处理的动物，包含马和老虎。

![image-20221104110519412](https://lh-picbed.oss-cn-chengdu.aliyuncs.com/image-20221104110519412.png)

这里的tqdm是一个进度条的库，可以对迭代过程进行可视化，一个非常简单的例子：

![image-20221104111408550](https://lh-picbed.oss-cn-chengdu.aliyuncs.com/image-20221104111408550.png)

读取图片时使用的是scipy.misc.imread(os.path.join(img_folder, 'behaviorDiscovery2.0/', animal, img_list[i][0]), mode='RGB')方法，这里又在读取图片后引入一个`im_to_torch`函数，完成两个工作：

1. 转置轴，变为CxWxH的形式。这里使用np.transpose方式实现：

   ![image-20221104114553531](https://lh-picbed.oss-cn-chengdu.aliyuncs.com/image-20221104114553531.png)

2. 归一化，化为0-1之间的图片。

在这里读取了非0的最小值（坐标不重要，关注是什么值），他的处理方法是：

![image-20221104145540359](https://lh-picbed.oss-cn-chengdu.aliyuncs.com/image-20221104145540359.png)

可以将其理解为如下的简单形式：

![image-20221104145925482](https://lh-picbed.oss-cn-chengdu.aliyuncs.com/image-20221104145925482.png)

就仅仅是将符合条件的下标对应的数字提取出来。

![image-20221104150440995](https://lh-picbed.oss-cn-chengdu.aliyuncs.com/image-20221104150440995.png)

随后调用了crop函数，（`inp = crop(img, c, s, [256, 256], rot)`），第三个参数是res，

#### crop函数（transforms.py文件）

$$s= max(x_{max}-x_{min},y_{max}-y_{min})/200.0*1.5$$

$$\begin{align}sf&=s*200.0/res[0]\\&=max(x_{max}-x_{min},y_{max}-y_{min})*1.5/res[0]\\ &(res[0]=256)\end{align}$$

![image-20221104151147597](https://lh-picbed.oss-cn-chengdu.aliyuncs.com/image-20221104151147597.png)

如果$sf<2$那么对应着$max(x_{max}-x_{min},y_{max}-y_{min})$小于341时，将sf设置为1.留待后续处理，如果是大于的话，进行处理，计算新的尺寸，缩小后进行下一步处理。（为什么？）

$$\begin{aligned}new\_size &= max(w,h)/sf\\&=\frac{max(w,h)}{max(x_{max}-x_{min},y_{max}-y_{min})*1.5/res[0]}\\\end{aligned}$$

在进行处理时，用了transform函数来获得左上和右下（ul,br）两个点。

```python
    # Upper left point
    ul = np.array(transform([0, 0], center, scale, res, invert=1))
    # Bottom right point
    br = np.array(transform(res, center, scale, res, invert=1))
```

下面来看这个函数。

#### transform

`transform(pt, center, scale, res, invert=0, rot=0)`，首先进入下一层函数：`get_transform` 。

这里的接收参数为：

$$pt = [0,0](在左上时) \quad res(右下)$$

$$center =(\frac{x_{max}-x_{min}}{2},\frac{y_{max}-y_{min}}{2})$$

$$scale=s$$

$$res=[256,256]$$

$$invert=1$$

函数`get_transform(center, scale, res, rot=rot)`计算一个类似卷积的3x3算子，计算得到如下的矩阵：

$$t=\begin{bmatrix}
\frac{res[1]}{200 * scale} & 0 & res[1] * (-float(center[0]) / h + .5)\\
0 & \frac{res[0]}{200 * scale} & res[0] * (-float(center[1]) / h + .5)\\0&0&1\\\end{bmatrix}$$

返回后，由于invert参数为1，所以求其逆矩阵（`numpy.linalg.inv`），
