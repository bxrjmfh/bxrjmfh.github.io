---
layout: post
title: Ubuntu + win 启动不经过gurb引导直接进入Ubuntu的解决办法
categories: 系统适配
tags: Ubuntu Windows 开机 BUG
---
###### Ubuntu + win 启动不经过gurb引导直接进入Ubuntu的解决办法

引用：https://blog.csdn.net/qq_45968493/article/details/121801853

首先在终端执行如下命令打开grub文件：

```sh
sudo gedit /etc/default/grub
```

可以看见以下内容：

![](https://lh-picbed.oss-cn-chengdu.aliyuncs.com/20220621210553.png)

其中的GRUB_TIMEOUT_STYLE 这一栏是hidden ,说明是隐藏的状态，此时将其注释，改为`menu`即可。

![](https://lh-picbed.oss-cn-chengdu.aliyuncs.com/20220621210617.png)

此时保存，随后更新引导即可：

```sh
sudo update-grub
```

![](https://lh-picbed.oss-cn-chengdu.aliyuncs.com/20220621211302.png)

重启，解决问题。
