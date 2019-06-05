"""
参考链接:
https://blog.csdn.net/u014796085/article/details/83478583
https://blog.csdn.net/mao_kun/article/details/50576003

1. 使用 Efficient Graph-Based Image Segmentation 的方法获取原始分割区域 R={r1,r2,…,rn}.
2. 初始化相似度集合 S=∅
3. 计算两两相邻区域之间的相似度, 将其添加到相似度集合 S 中
4. 从相似度集合 S 中找出, 相似度最大的两个区域 ri 和 rj, 将其合并成为一个区域 rt,
从相似度集合中除去原先与 ri 和 rj 相邻区域之间计算的相似度, 计算 rt 与其相邻区域 (原先与 ri 或 rj 相邻的区域) 的相似度,
将其结果添加的到相似度集合 S 中. 同时将新区域 rt 添加区域集合 R 中.
5. 重复步骤 4, 直到 S=∅, 即最后一个新区域 rt 为整幅图像.
6. 获取 R 中每个区域的 Bounding Boxes, 去除像素数量小于 2000, 以及宽高比大于 1.2 的, 剩余的框就是物体位置的可能结果 L.
"""























