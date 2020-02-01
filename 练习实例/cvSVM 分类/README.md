### 使用 OpenCV 的 SVM 算法进行分类训练

**目的:**  
在做用 HOG+SVM 进行目标检测时, 始终不能得到较好的较果, 因此将 SVM 进行单独的训练, 以检查 SVM 部分的效果. 
效果很好, 精度可以达到 95% 以上. 

**SVM 用法的参考链接:** 
```txt
https://blog.csdn.net/fengbingchun/article/details/78353140
https://blog.csdn.net/bigfatcat_tom/article/details/95170340
https://docs.opencv.org/2.4/modules/ml/doc/support_vector_machines.html
https://docs.opencv.org/2.4/doc/tutorials/ml/introduction_to_svm/introduction_to_svm.html
```

**参数:**  
* svm_type: svm 类型, 使用 setType() 进行设置, 有以下几种, 默认为 SVM::C_SVC. C_SVC=100, NU_SVC=101, ONE_CLASS=102, EPS_SVR=103, NU_SVR=104. C_SVC: C 类支持向量分类机. n 类分组 n>=2, 允许用异常惩罚因子 C 进行不完全分类. NU_SVC: ν 类支持向量分类机. n 类似然不完全分类的分类器. 参数为 ν 取代 C (其值在区间 [0, 1] 中, ν 越大, 决策边界越平滑). ONE_CLASS: 单分类器, 所有的训练数据提取自同一个类里, 然后 SVM 建立了一个分界线以分割该类在特征空间中所占区域和其它类在特征空间中所占区域. EPS_SVR: ϵ 类支持向量机. 训练集中的特征向量和拟合出来的超平面的距离需要小于 P. 异常值惩罚因子 C 被采用. NU_SVR: ν 类支持向量回归机. ν 代替了 P. 
* Gamma: PLOY, RBF, SIGMOID, CHI2 核函数的参数, 使用 setGamma() 进行设置, 默认为 1. 
* coef0: PLOY 或 SIGMOID 核函数的参数, 使用 setCoef0() 进行设置, 默认为 0. 
* Degree: PLOY 核函数的参数, 默认为 0. 
* C: C_SVC, EPS_SVR, NU_SVR 类型的 SVM 的参数, 默认为 0. C 是支持向量机的正则化参数, 最大的值使模型在训练集上拟合得更好. 
* Nu: NU_SVC, ONE_CLASS, NU_SVR 类型 SVM 的参数, 默认为 0. 
* P: EPS_SVR 类型的 SVM 的参数 epsilon, 默认为 0. 
* ClassWeights: C_SVC SVM 中的可选权重, 默认是一个 empty Mat. setClassWeights(const cv::Mat &val). 赋给指定的类, 乘以 C 以后变成 class_weights_i * C. 所以这些权重影响不同类别的错误分类惩罚项. 权重越大, 某一类别的误分类数据的惩罚项就越大. 
* TermCriteria: 迭代 SVM 训练过程的终止准则, 解决了约束二次优化问题的局部情形(机翻). TermCriteria(type, maxCount, epsilon).  默认为: TermCriteria( TermCriteria::MAX_ITER + TermCriteria::EPS, 1000, FLT_EPSILON ). type 为终止件条: COUNT, EPS 或 COUNT + EPS. maxCount: 最大迭代次数. epsilon: 所需的精度. 
* Kernel: 核函数类, 有两种初始化方式: (1) setKernel(), 使用 SVM::KernelTypes 中的核函数, 默认值为 SVM:: RBF, (2) setCustomKernel(), 使用自定义内核. 核函数的类型为: CUSTOM=-1, LINEAR=0, PLOY=1, RBF=2, CHI2=4, INTER=5. LINEAR: 线性内核, 没有任何向映射至高维空间, 线性区分(或回归) 在原始特征空间中被完成, 这是最快的选择, . PLOY: 多项式内核, . RBF: 基于径向的函数, 对于大多数情况都是一个较好的选择, . SIGMOID: sigmoid 函数内核, . 

