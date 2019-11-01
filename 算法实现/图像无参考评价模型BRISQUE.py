"""
参考链接:
https://blog.csdn.net/sinat_36438332/article/details/88743536

原理看不懂啊.
"""
import cv2 as cv
import numpy as np
from scipy.special import gamma
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV


def gaussian_2d_kernel(kernel_size, sigma=None):
    """
    获取高斯核.
    参考链接: https://blog.csdn.net/qq_16013649/article/details/78784791
    :param kernel_size:
    :param sigma:
    :return:
    """
    kernel = np.zeros(shape=(kernel_size, kernel_size))
    center = kernel_size // 2
    if sigma is None:
        sigma = ((kernel_size - 1) * 0.5 - 1) * 0.3 + 0.8
    s = 2 * (sigma ** 2)
    sum_value = 0
    for i in range(kernel_size):
        for j in range(kernel_size):
            x = i - center
            y = j - center
            kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / s)
            sum_value += kernel[i, j]
    sum_value = 1 / sum_value
    return kernel * sum_value


def correlation(image, kernel):
    """
    使用 kernel 对 image 进行 'SAME' 卷积操作.
    cv.
    :param image:
    :param kernel:
    :return:
    """
    kernel_height, kernel_width = kernel.shape[:2]
    h = kernel_height // 2
    w = kernel_width // 2
    # 边界补全
    image = np.pad(image, pad_width=((h, h), (w, w)), mode='constant')
    cor_height = image.shape[0] - kernel_height + 1
    cor_width = image.shape[1] - kernel_width + 1
    result = np.zeros(shape=(cor_height, cor_width), dtype=np.float64)
    for i in range(cor_height):
        for j in range(cor_width):
            result[i, j] = np.sum(image[i: i + kernel_height, j: j + kernel_width] * kernel)
    return result


def estimate_ggd_parameters(vector):
    gam = np.arange(0.2, 10.0, 0.001)
    r_gam = (gamma(1 / gam) * gamma(3 / gam)) / ((gamma(2 / gam)) ** 2)   # 根据候选的 γ 计算 r(γ)
    sigma_sq = np.mean(vector ** 2)
    sigma = np.sqrt(sigma_sq)   # 方差估计
    e = np.mean(np.abs(vector))
    r = sigma_sq / (e / 2)    # 根据 sigma 和 e 计算 r(γ)
    diff = np.abs(r - r_gam)
    gamma_param = gam[np.argmin(diff, axis=0)]
    return [gamma_param, sigma_sq]


def estimate_aggd_parameters(vec):
    alpha = np.arange(0.2, 10.0, 0.001)    # 产生候选的α
    r_alpha = ((gamma(2/alpha))**2)/(gamma(1/alpha)*gamma(3/alpha))    # 根据候选的γ计算r(α)
    sigma_l = np.sqrt(np.mean(vec[vec < 0]**2))
    sigma_r = np.sqrt(np.mean(vec[vec > 0]**2))
    gamma_ = sigma_l/sigma_r
    u2 = np.mean(vec**2)
    m1 = np.mean(np.abs(vec))
    r_ = m1**2/u2
    R_ = r_*(gamma_**3+1)*(gamma_+1)/((gamma_**2+1)**2)
    diff = (R_-r_alpha)**2
    alpha_param=alpha[np.argmin(diff, axis=0)]
    const1 = np.sqrt(gamma(1 / alpha_param) / gamma(3 / alpha_param))
    const2 = gamma(2 / alpha_param) / gamma(1 / alpha_param)
    eta = (sigma_r-sigma_l)*const1*const2
    return [alpha_param, eta, sigma_l**2, sigma_r**2]


def brisque_feature(dis_image):
    dis_image = dis_image.astype(np.float32)    # 类型转换十分重要
    kernel = gaussian_2d_kernel(7, 7/6)
    ux = correlation(dis_image, kernel)
    ux_sq = ux*ux
    sigma = np.sqrt(np.abs(correlation(dis_image**2, kernel)-ux_sq))
    mscn = (dis_image-ux)/(1+sigma)
    f1_2 = estimate_ggd_parameters(mscn)
    H = mscn*np.roll(mscn, 1, axis=1)
    V = mscn*np.roll(mscn, 1, axis=0)
    D1 = mscn*np.roll(np.roll(mscn, 1, axis=1), 1, axis=0)
    D2 = mscn*np.roll(np.roll(mscn, -1, axis=1), -1, axis=0)
    f3_6 = estimate_aggd_parameters(H)
    f7_10 = estimate_aggd_parameters(V)
    f11_14 = estimate_aggd_parameters(D1)
    f15_18 = estimate_aggd_parameters(D2)
    return f1_2+f3_6+f7_10+f11_14+f15_18


image_path = '../dataset/data/image_sample/lena.png'
image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
feat=brisque_feature(image)
size = image.shape

# 下采样
image = cv.resize(image, (int(size[1] / 2), int(size[0]/ 2)), cv.INTER_NEAREST)
feat += brisque_feature(image)
print(feat)


# C、γ的搜索范围都是从1e-4到1e4，取9个候选值
parameters = {"kernel": ("linear", 'rbf'), "C": np.logspace(-4, 4, 9), "gamma": np.logspace(-4, 4,9)}
svr = GridSearchCV(SVR(), param_grid=parameters, cv=4)    # 4折交叉验证
# 不知道它的 score 是怎么来的.
svr.fit(feat, score)
print('The parameters of the best model are: ')
print(svr.best_params_)













