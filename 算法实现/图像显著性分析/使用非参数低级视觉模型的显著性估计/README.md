## [使用非参数化低级视觉模型的显著性估计](http://www.cvc.uab.es/people/aparraga/Publications/CVPR/Murray_etal_CVPR2011.PDF)

Naila Murray, Maria Vanrell, Xavier Otazu 和 C.Alejandro Parraga,  
计算机科学系计算机视觉中心, 巴塞罗那大学奥特诺玛分校, 塞丹约拉德尔瓦勒斯, 巴塞罗那, 西班牙  
nmurray@cvc.uab.es, xotazu@cvc.uab.es, mvanrell@cvc.uab.es, aparraga@cvc.uab.es  

### 摘要
许多成功的场景注意力预测模型都涉及三个主要步骤: (1) 使用一组卷积滤波器得到不同的特征图.
(2) 对特征图应用中心外围机制以生成显著图(激活图). (3) 空间池化以构造显著图.
但是, 整合空间信息并证明选择各种参数值的合理性仍然是未解决的问题.
本文提出一种可以概括出一种有效的人类视觉色彩外观的模型, 其中包含参数的合理选择以及固有空间池化机制,
从而可以获得优于最先进的显著性模型.
尺度积分是通过对尺度加权的中心环绕响应集进行逆小波变换来实现的.
已对尺度加权函数(称为 ECSF) 进行了优化, 以更好地复制颜色外观方面的心理物理数据,
并且通过在眼动数据上训练高斯混合模型来调整中心外围抑制窗口的适合大小, 从而避免了临时性参数选择.
此外, 我们得出结论, 将颜色外观模开模扩展为显著性估计, 为不同视觉任务的通用低级视觉前端增加了证据.

### 介绍
眼跳运动可能是人类视觉系统最重要的特征之一, 它使用们能够通过改变注视点来快速采样图像.
尽管许多因素可能决定我们的注意力过程选择或丢弃哪些图像特征, 但将这些特征分为两类因素仍然有用:
自下而上和自上而下. 前者包括自动驱动的(瞬时)过程, 而后者包括依赖于生物体的内部状态(例如眼前的视觉任务或受试者的背景).
尽管通常通过在先验知识上训练的机器学习技术来解决理解内部状态的困难, 图像驱动过程通常通过构建显著图来解决,
从低水平的生物学过程中获得灵感, 它比令人难以捉摸的自上而下的机制更容易理解.
显著图是场景中视觉上显著的地形图 (给定位置的显著性又取决于该位置与其周围的颜色, 方向, 运动, 深度等有何不同).
计算这些地图仍然是一个悬而未决的问题, 其对计算机视觉的兴趣正在增长.

已经提出了几种计算模型来预测人的注视现象, 其中一些是受生物学机制(通常是众所周知的低级过程)启发的, 
而另一些则基于直接从注视数据中训练的学习技术.

在显著性的生物学启发模型中, Itti 等人的模型是最有影响力的模型之一.
对不同空间频率和方向的特征图的尺度空间中心外围激励响应求和, 并将结果馈入神经网络, 其输出可测量显著性.
高等人, 在某处采用显著性作为描述该位置的一组特征的区分能力, 以区分该区域及其周围区域.
Bruce & Tsotsos 认为该位置的显著性要通过该位置相对于其周围环境的自信息来量化整个图像, 或更局部的像素区域.
张等人, 也提出了一种基于自我信息的方法, 但是使用空间金字塔来产生局部特征 (上下文统计是从自然图像的集合
而不是像素或单个图像的局部邻域生成的).
Seo & Milanfar 使用自相似机制来计算显著性, 在该显著性中, 与周围环境相比曲率不同的区域被指定为高度突出.
在典型的基于学习的方法中, 使用眼动数据学习并结合了显著性特征, 而学习技术则用于减少必须调整的模型参数的数量.
在最常见的自下而上建模框架中, 场景中的注意力涉及使用一组线性滤波器对输入图像进行比例空间分解,
对分解后的图像进行中心环绕操作以及某种空间池化以构建最终显著图. 但是, 此方法的核心有的两个主要问题仍未解决:
(a) 如何整合从分解的多个尺度得出的信息, 以及 (b) 如何调整各种参数以获得通用机制.
集成不同尺度的信息特别重要, 因为场景中的显著特征以及不同场景中的显著特征可能占据不同的空间频率, 如图 1 所示.
因此, 在空间金字塔的不同层次上定位显著特征并将这些特征组合成最终地图的机制至关重要.

在本文中, 我们提出一种显著性的计算模型, 该模型遵循上述典型的三步体系结构, 同时尝试通过简单的,
在神经上似乎可行的机制的组合来回答上述问题, 该机制几乎消除了所有任意变量.
在本文, 我们的提议并概括了为预测颜色外观而开发的特定低级模型, 它具有三个主要级别:
1. 第一步: 视觉刺激的处理方式与已知的人类早期视觉途径一致(颜色对比和亮度通道, 然后进行多尺度分解).
在生物学上讲, 使用的滤波器组(类 Gabor 小波) 和一定范围的空间尺度 (以八度为单位) 是合理的, 通常用于低级视觉建模.
2. 第二步: 我们的模型包括模拟视觉皮层细胞中存在的抑制机制, 该机制可以有效地规范其对刺激对比的反应.
其中央和归一化环绕窗的大小是通过眼动数据训练的高斯混合模型(GMM) 来确定的.
3. 第三步: 我们的模型通过直接对皮质输出的非线性化计算出的权重执行逆小波变换, 在多个尺度上集成信息.
这种非线性积分是通过类似于 Otazu 等人提出的加权函数完成的. 并命名为 "扩展对比度敏感度函数 (ECSF)",
但经过优化以适合不同空间范围的心理物理色彩匹配数据.

合适的 ECSF 是我们提案的核心, 代表了其最新颖的组成部分. 以前已经通过拟合相同的低层模型对其进行了调整,
以预测人类观察者对颜色感应图案的匹配. 该函数还可以对显著性进行建模, 这一事实为针对不同视觉任务的独特底层机制的假设提供了支持.
可以对该机制建模, 以预测颜色外观 (通过将逆小波变换应用于由 ECSF 权重调制的分解系数) 或
视觉显著性 (通过将变换应用于权重本身). 此外, 我们介绍一种新颖的方法来选择标准化窗口的大小,
这减少了必须以临时方式设置的参数数量.

我们的两个主要贡献可以总结如下:
1. 通过一组加权的中心外围输出的逆小波变换对比例进行积分的框架.
2. 临时参数的减少. 通过引入有关颜色外观的注视心理物理数据的训练步骤来完成此操作.
本文的其余部分安排如下.
在第 2 节中, 我们介绍了低级色觉模型和拟合的 ECSF.
在第 3 节中, 我们使用模型得出的权重来计算显著性,
在第 3.1 节中, 我们评估模型的性能.
在第 3.2 节中, 总结了结果,
在第 4 节中, 讨论进一步的工作.

### 2. 一个低级的视觉模型
我们在这项工作中提出的显著性估计方法是从 Otazu 等人开发的统一颜色归纳模型得出的低级视觉表示的扩展.
在他们的模型中, 作者提出了分别预测亮度和颜色外观的多分辨率模型.
色彩感知是多种适应机制的结果, 这些适应机制会导致同一色块根据其周围环境而有所不同. 
图 2 中两个图像的区域 A 和 B 分别被感知为具有不同的亮度 (在面板 a 中) 
和/或具有不同的颜色 (在面板 c 中), 虽然, 实际上 A 和 B 在物理上是相同的. 
(强度和 RGB 颜色通道配置文件在对应的面板 (b) 和 (d) 中以实线绘制). 
这些错觉 1 是由 Otazu 等人的颜色模型预测的, 如图 2 中的虚线所示 (面板 (b) 和 (d)). 
例如, 区域 A 在图形 (b) 中较暗, 而区域 B 在图形 (d) 中较橙. 

#### 第一阶段
在 Otazu 等人模型的第一阶段, 使用多分辨率小波变换将图像与一组滤镜进行卷积. 
生成的空间金字塔包含水平(h), 垂直(v) 或对角线(d) 定向的小波平面. 
使用小波变换获得的空间金字塔的系数可以认为是局部取向对比度的估计. 
对于给定的图像 I, 小波变换表示为:   
![wavelet_transform](http://chart.googleapis.com/chart?cht=tx&chl=\Large%20WT(I_{c})%20=%20\{\omega_{s,%20o}\}_{s=1,2,....,n;%20o=h,v,d})

其中: ![ω_{s,o}](http://chart.googleapis.com/chart?cht=tx&chl=\Large%20\omega_{s,%20o}) 
是空间尺度 s 和方向的小波平面 o, ![I_{c}](http://chart.googleapis.com/chart?cht=tx&chl=\Large%20I_{c}) 
表示图像 I 的相对通道 O1, O2 和 O3 之一. 使用小波变换 WT 将每个对立通道分解为一个空间金字塔. 
此变换包含类似 Gabor 的基函数, 并且对于最大尺寸为 D 的图像, 分解中使用的比例数由
 ![n=log_{2}D](http://chart.googleapis.com/chart?cht=tx&chl=\Large%20n%20=%20log_{2}{D}) 给出. 


**批注:**  
**第 1 部分**  
实际上是将图像的各个通道应用小波变换, 得到三个通道, 每个通道都有很多张大小不同的图, 其实就是经过变换后的特征图.
因为是从图像的各个通道提出特征, 因此可以想到所使用的图像为 Lab 空间下的三个通道,
L 作为亮度特征图, 并应用小波进行进一步提取.
a, b 作为色彩特征图, 并应用小波进行进一步提取.
方向特征图是通过带方向的小波基来提出的.
其多尺度特性则是通过小波的多尺度特性来实现. 通过小波变换得到小波系数形成特征图.
第一阶段在于计算图像在各尺度方向下的小波系数.  

**需要理解小波变换:**  
一个向量, 其实是一串数字. 好比一个函数 f(x), 其定义域为 [-∞, +∞].
在定义域 f(x) 对应无穷个数值, 这无穷个数值则可以看作是一个无穷(n)维的向量.
道理上存在同样级别的无穷(n)个单位正交向量能够完全表示这个无穷(n)维向量.
而每一个单位向量都将是无穷(n)维向量, 同时注意到, 这些无穷维向量同样可以表示成一个函数 g(x).
所以, 这表示我们可以用无穷多个函数作为基函数来表示另一个函数.
这无穷多个基函数应分别满足
1. 其范数为 1, 这是其作为单位向量的性质要求. 表示为: ∫g(x)dx = 1. 从负无穷积到正无穷.
2. 任意两个函数应该是正交的, 即两个向量的内积为 0, 通常其表达为:
 ![\sum_{i=0}^{n-1}(a_{i}b_{i})=0](http://chart.googleapis.com/chart?cht=tx&chl=\Large%20\sum_{i=0}^{n-1}(a_{i}b_{i})=0)  
但对于此处, 每一个 g(x) 都是一个函数. 因此上式变为:  
 ![\int_{-\infty}^{\infty}g_{i}(x)g_{j}(x)dx=0](http://chart.googleapis.com/chart?cht=tx&chl=\Large%20\int_{-\infty}^{\infty}g_{i}(x)g_{j}(x)dx=0)  
其中: ![g_{i}(x)](http://chart.googleapis.com/chart?cht=tx&chl=\Large%20g_{i}(x)), 
![g_{j}(x)](http://chart.googleapis.com/chart?cht=tx&chl=\Large%20g_{j}(x)) 
表示任意的两个基函数.

在实际应用中, 由于我们不能使用无穷个基函数去表示另一个函数, 但是我们可以预先估计这个目标函数的性质,
然后设计出基函数, 使得目标函数可以分解到少量可数个数的正交的基函数上, 并具有可以忽略的信息损失.
这也就是小波变换的方法, 设置基小波, 将信号分解到基小波上, 用来观察信号中各基小波的成分比例.
而信号分解的方法, 其实就是向量向基向量投影的过程, 即向量内积, 同时其实也就是卷积.
卷积后得到的值表示的就是信号在卷积核所表示的基向量(信号)上的投影大小, 值越大, 则表示越相近, 成分越高.

**第 2 部分**  
在于各像素点处计算中心外围对比度, 具体做法: 计算中心和外围区域的小波系数能量,
然后做除法来体现它们之间的对比度, 得到中心外围的对比系数 Z_{x,y}.
提出频率调制函数 ECSF, ECSF 是 Z_{x,y} 和尺度 S (尺度就是频率的反映, 二者成反比) 的函数,
ECSF 中的参数根据心理学数据通过最小二乘拟合的方式确定.
通过 ECSF, 可以得到各个尺度下 Z_{x, y} 的权重系数 α_{x,y}, 这个权重系数就是中心外围对比度受频率的"影响",
也就是前面所说的人眼观察时所受"影响"的量化表示.
如果把 α_{x,y} 与 小波系数相乘后一起做小波逆变换, 就可以得到人眼实际观察到的图像.

**第 3 部分**  
显著图的构建.
第 2 部分中求出了某特征通道, 某尺度和某方向上的权重系数 α_{x,y}, 接下来只用它做小波逆变换,
不要小波系数, 就得到一个子显著图. 子显著图之间的融合采用正则化后, 线性叠加, 具体顺序:
在各特征通道下, 按照尺度由大到小的顺序, 先在各方向上叠加, 然后在各尺度上叠加, 求出各通道的显著图后,
以平方和根的方式将其融合成最终的显著图.



#### 第二阶段
在第二阶段, 围绕小波系数 ω_{x,y} 的对比能量 a_{x,y} 以位置 x 为中心, 通过将局部区域与二值滤波器 h 卷积来估算 y.
滤波器的形状随其操作的小波平面的方向而变化, 如图 5 所示.
例如, 对于水平小波平面, a_{x, y} 由下式计算:  
![a_{x,y}=\sum_{j}\omega_{x-j,y}2h_{j}](http://chart.googleapis.com/chart?cht=tx&chl=\Large%20a_{x,y}=\sum_{j}\omega_{x-j,y}2h_{j})  
其中, h_{j} 是一维滤波器 h 的第 j 个系数.

```angular2html
批注:
\omega _{x-j,y} 表示在当前位置 (x, y) 附近的 (x-j, y) 处的一个小波系数,
h_{j} 是滤波器上的一个权重值, 小波系数乘以该权重, 再相加, 得到 a_{x, y}.
由于滤波器是对称的, 所以乘以 2. (感觉还是不对. 但是解释不通啊).
```

系数的对比能量在所有空间位置和空间比例下进行计算.
滤波器 ![h_{j}](http://chart.googleapis.com/chart?cht=tx&chl=\Large%20h_{j}) 
定义了计算中心小波系数 ![\omega_{x, y}](http://chart.googleapis.com/chart?cht=tx&chl=\Large%20\omega_{x,%20y}) 
的激活度 ![a_{x,y}](http://chart.googleapis.com/chart?cht=tx&chl=\Large%20a_{x,y})周围的区域.
该中心区域与周围区域之间的相互作用产生了中心环绕效果.
为了建模中心环绕效应, 使用以下方法比较了中心区域
![a_{x,y}^{cen}](http://chart.googleapis.com/chart?cht=tx&chl=\Large%20a_{x,y}^{cen}) 
和周围区域
![a_{x,y}^{sur}](http://chart.googleapis.com/chart?cht=tx&chl=\Large%20a_{x,y}^{sur})
的能量.  
![r_{x,y}=\frac{(a_{x,y}^{cen})^{2}}{(a_{x,y}^{sur})^{2}}](http://chart.googleapis.com/chart?cht=tx&chl=\Large%20r_{x,y}=\frac{(a_{x,y}^{cen})^{2}}{(a_{x,y}^{sur})^{2}})  


周围区域的能量
![a_{x,y}^{sur}](http://chart.googleapis.com/chart?cht=tx&chl=\Large%20a_{x,y}^{sur})
的计算方式类似于
![a_{x,y}^{cen}](http://chart.googleapis.com/chart?cht=tx&chl=\Large%20a_{x,y}^{cen}), 
唯一的区别是滤波器 h 的定义, 也如图 5 所示.


```angular2html
批注:
在图 5 中定义了展的中心环绕滤镜的红色部分对应于中央滤镜, 而蓝色部分对应于环绕滤镜.
它其实是一个如图中形状的滤波器, 在计算中心/外围, 不同的 a_{x,y} 时, 使用不同的取值范围.
```

执行
![r_{x,y}](http://chart.googleapis.com/chart?cht=tx&chl=\Large%20r_{x,y})
的非线性缩放以产生最终的中心环绕能量度量
![z_{x,y}](http://chart.googleapis.com/chart?cht=tx&chl=\Large%20z_{x,y}):  
![z_{x,y}=\frac{r_{x,y}^{2}}{1+r_{x,y}^{2}}](http://chart.googleapis.com/chart?cht=tx&chl=\Large%20z_{x,y}=\frac{r_{x,y}^{2}}{1\add%20r_{x,y}^{2}})  
因此, ![z_{x,y}\in[0,1]](http://chart.googleapis.com/chart?cht=tx&chl=\Large%20z_{x,y}%20\in%20[0,%201]). 
当 
![z_{x,y}\rightarrow0](http://chart.googleapis.com/chart?cht=tx&chl=\Large%20z_{x,y}\rightarrow0), 
中心激活
![a_{x,y}^{cen}](http://chart.googleapis.com/chart?cht=tx&chl=\Large%20a_{x,y}^{cen})
远远小于外围激活 ![a_{x,y}^{sur}](http://chart.googleapis.com/chart?cht=tx&chl=\Large%20a_{x,y}^{sur}).
相似地, 当
![z_{x,y}\rightarrow1](http://chart.googleapis.com/chart?cht=tx&chl=\Large%20z_{x,y}\rightarrow1),
中心激活则远大于外围激活.
因此, 
![r_{x,y}](http://chart.googleapis.com/chart?cht=tx&chl=\Large%20r_{x,y})
可以解释为相对中心活动
![a_{x,y}^{cen}](http://chart.googleapis.com/chart?cht=tx&chl=\Large%20a_{x,y}^{cen})
的饱和近似.
中心区域和周围区域的大小用于定义相应的
![h_{j}](http://chart.googleapis.com/chart?cht=tx&chl=\Large%20h_{j}) 
滤波器的大小.


众所周知, 颜色外观取决于空间频率.
Mullen 用广义的对比敏感度函数(CSF) 描述了人类对色彩独立通道中局部对比度的敏感度, 这是空间频率的函数.
采用这种想法, Otazu 等人. 定义了一个扩展的对比敏感度函数(ECSF), 该函数由空间比例 s 和中心外围对比能量参数化.
空间尺度与空间频率 v 成反比, 因此 
![s=log_{2}(\frac{1}{v})=log_{2}(T)](http://chart.googleapis.com/chart?cht=tx&chl=\Large%20s=log_{2}(\frac{1}{v})=log_{2}(T))
, 其中 T 表示一个以像素为单位的频率周期.
```angular2html
批注: 图像越大, 则 T 越大, 则 s 越大, s 越大则下式中的 g(s) 越小, 则 ECSF 函数返回值越小.
这表示的是, 当对图像比较大时, 细节上的中心外围差异是不容易被人注意到的, 因此具有更小的权重
```
ECSF 函数定义如下:
![ECSF_{z,s}=z\cdotg(s)\addk(s)](http://chart.googleapis.com/chart?cht=tx&chl=\Large%20ECSF_{z,s}=z%20\cdot%20g(s)\add%20k(s))
其中函数 g(s) 定义如下:
g(s) = 
![\beta%20e^{-\frac{s^{2}}{2\delta_{1}^{2}}](http://chart.googleapis.com/chart?cht=tx&chl=\Large%20\beta%20e^{-\frac{s^{2}}{2\delta_{1}^{2}}) 
if 
![s\leq%20s_{0}^{g}](http://chart.googleapis.com/chart?cht=tx&chl=\Large%20s\leq%20s_{0}^{g}) 
else 
![\beta%20e^{-\frac{s^{2}}{2\delta_{2}^{2}}](http://chart.googleapis.com/chart?cht=tx&chl=\Large%20\beta%20e^{-\frac{s^{2}}{2\delta_{2}^{2}})
此处 s 表示要处理的小波平面的空间比例, 是比例常数, 而 
![delta_{1}](http://chart.googleapis.com/chart?cht=tx&chl=\Large%20\delta_{1}) 
和 
![delta_{2}](http://chart.googleapis.com/chart?cht=tx&chl=\Large%20\delta_{2}) 
定义 g(s) 的空间灵敏度的展宽.
![s_{0}^{g}](http://chart.googleapis.com/chart?cht=tx&chl=\Large%20s_{0}^{g})
参数定义 g(s) 的峰值空间尺度灵敏度.
在等式 5 中, 小波系数的中心外围激活 z 由 g(s) 调制.
附加函数 k(s) 确保 ECSF(z,s) 的下限为非零值:

k(s) = 
![e^{-\frac{s^{2}}{2%20\delta_{2}^{2}}](http://chart.googleapis.com/chart?cht=tx&chl=\Large%20e^{-\frac{s^{2}}{2%20\delta_{2}^{2}}) 
if ![s\leq%20s_{0}^{k}](http://chart.googleapis.com/chart?cht=tx&chl=\Large%20s\leq%20s_{0}^{k})
else 1  

这里, 
![delta_{2}](http://chart.googleapis.com/chart?cht=tx&chl=\Large%20\delta_{2}) 
定义了 k(s) 的空间灵敏度的分布, 
![s_{0}{k}](http://chart.googleapis.com/chart?cht=tx&chl=\Large%20s_{0}{k})
定义了 k(s) 的峰值空间尺度灵敏度.
函数 ECSF 用于加权某个位置的中心外围对比度能量 
![z_{x,y}](http://chart.googleapis.com/chart?cht=tx&chl=\Large%20z_{x,y})
, 产生最终响应 
![\alpha%20_{x,y}](http://chart.googleapis.com/chart?cht=tx&chl=\Large%20\alpha%20_{x,y})
:
![\alpha_{x,y}=ECSF(z_{x,y},s_{x,y})](http://chart.googleapis.com/chart?cht=tx&chl=\Large%20\alpha_{x,y}=ECSF(z_{x,y},s_{x,y}))  
![\alpha_{x,y}](http://chart.googleapis.com/chart?cht=tx&chl=\Large%20\alpha_{x,y})
是调制小波系数 
![\omega_{x,y}](http://chart.googleapis.com/chart?cht=tx&chl=\Large%20\omega_{x,y})
的权重.

包含色彩外观错觉的感知图像通道 
![I_{c}^{preceived}](http://chart.googleapis.com/chart?cht=tx&chl=\Large%20I_{c}^{preceived})
是通过在每个位置, 比例和方向上对小波系数 
![\omega_{x,y}](http://chart.googleapis.com/chart?cht=tx&chl=\Large%20\omega_{x,y})
进行逆小波变换后得到,
该系数已通过 
![\alpha_{x,y}](http://chart.googleapis.com/chart?cht=tx&chl=\Large%20\alpha_{x,y}) 
响应在局部进行加权:
![I_{c}^{perceived}(x,y)=\sum_{s}\sum_{o}\alpha_{x,y,s,o}\cdot\omega_{x,y,s,o}+C_{r}](http://chart.googleapis.com/chart?cht=tx&chl=\Large%20I_{c}^{perceived}(x,y)=\sum_{s}\sum_{o}\alpha_{x,y,s,o}\cdot\omega_{x,y,s,o}+C_{r})  
在此, o 表示小波平面的方向; 
![C_{r}](http://chart.googleapis.com/chart?cht=tx&chl=\Large%20C_{r})
表示从 WT 获得的残留像平面.

Otazu 等人的模型能够复制从两个单独的实验中获得心理物理数据.
在第一个实验中 (由 Blakeslee 等人进行), 观察者执行不对称的亮度匹配任务, 以匹配刺激区域中存在的错觉.
图 3(a) 显示了一些示例高度刺激. 第二个实验是由 Otazu 等人进行的. 以类似于方式进行,
但是观察者执行不对称的颜色匹配任务, 而不是执行涉及亮度的任务. 一些实验中使用的一些示例色彩刺激如图 3(a) 所示.

我们的显著性估计模型基于我们刚刚描述的前一个步骤. 但是, 为了获得强度和颜色 ECSF(z,s) 函数的参数,
我们使用了Blakeslee 等人和Otazu 等人的心理物理数据来进行最小二乘回归, 以便选择函数的参数.
我们的结果在表 1 中给出. 两个拟合的 ECSF(z,s) 函数与颜色和亮度心理物理数据保持较高的相关率(r=0.9), 如图 3(b) 所示.
注意, 两个色度通道共享相同的 ECSF(z,s) 函数. 生成的针对亮度的色度通道的优化 ECSF(x,s) 函数的配置文件如图 4 所示.
该功能增强了窄通带中的对比能量响应, 并在低空间比例(高空间频率) 下抑制了对比能量.
增强或抑制的幅度随中心外围对比度能量 z 的幅度增加而增加.



### 3. 构建显著图
在上一节中, 我们描述了预测颜色外观现象的低级视觉表示. 该模型以等式 9 得出结论, 可以将其重写为:  
![I_{c}^{perceived}(x,y)=WT^{-1}\{\alpha_{x,y,s,o}\cdot\omega_{x,y,s,o}\}](http://chart.googleapis.com/chart?cht=tx&chl=\Large%20I_{c}^{perceived}(x,y)=WT^{-1}\{\alpha_{x,y,s,o}\cdot\omega_{x,y,s,o}\})  

其中 ![I_{c}^{perceived}](http://chart.googleapis.com/chart?cht=tx&chl=\Large%20I_{c}^{perceived}) 
是图像原始通道的新版本, 其中图像局部位置可能已通过权重进行了修改(通过模糊或增强效果).
修改后的位置的颜色要么被同化(取平均值), 以变得与周围的颜色更加相似, 要么被增强对比(被锐化), 以变得与周围颜色更加不相似.
为了获得使用这种颜色表示的显著性预测, 我们假设经过增强的图像位置是显著的, 而经过模糊处理的图像位置是不显著的.
从这个意义上讲, 我们可以通过权重的逆小波变换来定义特定图像通道的显著图.
因此, 图像通道 ![I_{c}](http://chart.googleapis.com/chart?cht=tx&chl=\Large%20I_{c}) 在位置 (x,y) 的显著性图
![S_{c}](http://chart.googleapis.com/chart?cht=tx&chl=\Large%20S_{c})
可以很容易地估计为:  
![S_{c}(x,y)=WT^{-1}\{\alpha_{x,y,s,o}\}](http://chart.googleapis.com/chart?cht=tx&chl=\Large%20S_{c}(x,y)=WT^{-1}\{\alpha_{x,y,s,o}\})  

通过去除小波系数 ![\omega_{x,y,s,o}](http://chart.googleapis.com/chart?cht=tx&chl=\Large%20\omega_{x,y,s,o}) 
并仅对在每个图像位置计算出的权重执行逆变换,
我们提供了一种优雅而直接的方法, 用于根据广义的低级视觉表示来估计图像显著性.

为了将每个通道的图合并到最终的显著图 S 中, 我们计算欧几里得范数
![S=\sqrt{S_{O1}^{2}\add%20S_{O2}^{2}\add%20S_{O3}^{2}}](http://chart.googleapis.com/chart?cht=tx&chl=\Large%20S=\sqrt{S_{O1}^{2}\add%20S_{O2}^{2}\add%20S_{O3}^{2}})
显著性模型的计算步骤如图 5 所示.
在此过程将颜色外观模型概括为一种估计显著性的模型.
我们的方法的主要优点是通过逆小波变换集成多尺度信息, 并使 ECSF 函数, 其参数在生物学上是合理的.
但是, 仍有一些参数需要设置. 由等式 4 定义的中心环绕抑制的强度高度取决于二进制
![h_{j}](http://chart.googleapis.com/chart?cht=tx&chl=\Large%20h_{j})
滤波器的大小,
该二进制
![h_{j}](http://chart.googleapis.com/chart?cht=tx&chl=\Large%20h_{j})
滤波器定义了小波系数
![\omega_{x,y}](http://chart.googleapis.com/chart?cht=tx&chl=\Large%20\omega_{x,y})
的局部中心和周围区域的范围.
我们认为, 特征周围的中心区域应该跨越小波平面中对该特征的响应.
但是, 小波对特征的响应程度随每个空间尺度而不同.
因此, 我们将中心区域的大小设计为刚好足够大, 可以在最显著的空间尺度上跨越小波响应.
最显著的空间尺度被认为是两个 ECSF(z,s) 函数最敏感的尺度, 大约 s=4.
我们认为包含特征的中心区域是感兴趣区域 (ROI).
因此, 我们通过确定此数据集的 ROI 的典型大小来估计所需的中心区域大小.
为了确定这个尺寸, 我们首先创建了一个高斯混合模型(GMM), 该模型是来自 Bruce & Tsotsos 数据集的 20 个图像的子集的眼运位置,
这是我们用来评估模型的数据集之一. GMM 包含 5 个组件, 每个组件都将一组眼动的位置聚集在一起, 因此代表了 ROI.
因此, 将每个组件的标准偏差解释为 ROI 的半径. 在所有图像上, 高斯分量的平均半径为 53 个像素.
在 s=4 时, 半径将是 
![2(53/2^{s-1})=2(53/8)=13.25](http://chart.googleapis.com/chart?cht=tx&chl=\Large%202(53/2^{s-1})=2(53/8)=13.25)
因此, 我们将局部中心区域的大小设置为该半径的两倍, 即 27 个像素.
周围区域的大小设置为 54 个像素, 是中央区域大小的两倍.
在对描述我们的实验结果的下一部分中介绍的两个数据集进行评估时, 将使用中心区域和周围区域的大小.

需要考虑的重要一点是 ECSF 函数的峰值空间尺度(大约 s=4) 是否与彩色和消色差通道的人类 CSF 的峰值空间频率一致,
据估计, 每个频率约为 2 和4 个周期度(cpd). 也就是说, 通过 ECSF 功能增强的空间比例是否与人类最喜欢的空间频率一致?
如果我们假设 ROI 跨度具有 2 到 4 cpd 的空间频率的特征, 则 106 个像素包含 2-4 个空间周期.
对应于 2 cpd 的空间比例为 
![s=log_{2}(T)=log_{2}(106/2)=5.7](http://chart.googleapis.com/chart?cht=tx&chl=\Large%20s=log_{2}(T)=log_{2}(106/2)=5.7)
, 而 4 cpd 对应于 s=4.7
这些空间比例确实与最小二乘回归获得的 ECSF 峰值空间比例一致.
