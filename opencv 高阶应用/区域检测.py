"""
参考链接:
https://blog.csdn.net/zhaocj/article/details/40742191
https://github.com/makelove/OpenCV-Python-Tutorial/blob/master/cv-MSER%E5%8C%BA%E5%9F%9F%E6%A3%80%E6%B5%8B/MSER_create1.py

MSER 全称叫作 "最大稳定极值区域" (MSER - Maximally Stable Extermal Regions),
主要是基于分水岭的思想来做图像中斑点检测.

MSER 对灰度图像做二值化, 其阈值从 0 到 255 之间递增, 就像是一片土地上做水平面的上升,
随着水平面上升, 有些较矮的丘陵会被淹没, 如果从天空往下看, 则大地分为陆地和水域两部分,
在得到的所有二值图像中, 图像中的某些连通区域变化很小, 甚至没有变化, 则该区域就被称为最大稳定极值区域.

在水面上升时, 有些被水淹没的地方的面积没有变化, 它的数学定义为:
q(i) = |Q_{i+1} - Q_{i-1}| / |Q_{i}|
其中 Q_{i} 表示第 i 个阈值时的某一连通区域,
q(i) 为第 i 个阈值时, 区域 Q_{i} 的变化率.
当 q(i) 为局部极小值时, 则 Q_{i} 为最大稳定极值区域 (即, 如果在水位上升时, 此处出现了一个台阶, 则此台阶就是一个区域).

需要说明的是, 上述做法只能检测出灰度图像的黑色区域, 不能检测出白色的区域, 因此还需要对原图进行反转,
然后再进行阈值从 0-255 的二值化处理过程. 这两种操作又分别称为 MSER+ 和 MSER-.

MSER 具有以下特点:
1. 对图像灰度具有仿射变换的不变性
2. 稳定性: 具有相同值范围内所支持的区域才会被选择.
3. 无需任何平滑处理就可以实现多尺度检测, 即小的和大的结构都可以被检测到.

MSER 的原理比较简单, 但要更快更好地实现它, 是需要一定的算法, 数据结构和编程技巧的.
David Nister 等人于 2008 年提出了 Linear Time Maximally Stable Exteral Regions 算法,
该算法要比原著提出的算法快, opencv 就是利用该算法实现 MSER 的,
但这里要说明一点的是, opencv 不是利用公式 1 计算 MSER 的, 而是利用更易于实现的改进方法:
q(i) = |Q_{i} - Q_{i-1}| / |Q_{i-1}|
David Nister 提出的算法是基于改进的分水岭算法, 即当往一个固定的地方注水的时候,
只有当该地方的沟壑被水填满以后, 水才会向其四周溢出, 随着注水量的不断增加, 各个沟壑也逐渐被水淹没.
但各个沟壑的水面不是同时上升的, 它是根据水漫过地方的先后顺序, 一个沟壑一个沟壑地填满水,
只有当相邻两个沟壑被水连通在一起以后, 水面对于这两个沟壑来说才是同时上升的.

该算法的具体步骤如下:
1. 初始化栈和堆, 栈用于存储组块 (组块就是区域, 就相当于水面, 水漫过的地方就会出现水面,
水面的高度就是图像的灰度值, 因此, 用灰度值来表示组块的值), 堆用于存储组块的边界像素, 相当于水域的岸边,
岸边要高于水面的, 因此边界像素的灰度值一定不小于它所包围的区域 (即组块) 的灰度值.
首先向栈内放入一个虚假的组块, 当该组块被弹出时意味着程序的结束.
2. 把图像中的任意一个像素 (一般选取图像的左上角像素) 作为源像素, 标注该像素为已访问过, 并且把该像素的灰度值作为当前值.
这一步相当于往源像素这一地点注水.
3. 向栈内放入一个空组块, 该组块的值是当前值.
4. 按照顺序搜索当前值的 4 邻域内剩余的边缘, 对每一个邻域检测它是否已经被访问过, 如果没有,
则标注它已访问过并检测索它的灰度值, 如果灰度值不小于当前值, 则把它放入用于存放边界像素的堆中.
另一方面, 如果邻域灰度值小于当前值, 则把当前值放入堆中, 而把邻域值作为当前值, 并回到步骤 3.
5. 累计栈顶组块的像素个数, 即计算区域面积, 这是通过循环累计得到的, 这一步相当于水面的饱和.
6. 弹出堆中的边界像素. 如果堆是空的, 则程序结束, 如果弹出的边界像素的灰度值等于当前值, 则回到步骤 4.
7. 从堆中得到的像素值会大于当前值, 因此我们需要处理栈中所有的组块,
直到栈中的组块的灰度值大于当前边界像素灰度值为止. 然后回到步骤 4.

至于如何处理组块, 则需要进入处理栈子模块中, 传入该子模块的值为步骤 7 中, 从堆中提取得到的边界像素灰度值.
子模块的具体步骤为:
1. 处理栈顶的组块, 即根据公式 2 计算最大稳定区域,判断其是否为极值区域.
2. 如果边界像素灰度值小于距栈顶第二个组块之间还有组块没有被检测处理,
因此我们需要改变栈顶组块的灰度值为边界像素灰度值 (相当于这两层的组块进行了合并),
并回到主程序, 再次搜索组块.
3. 弹出栈顶组块, 并与止前栈顶组块合并.
4. 如果边界像素灰度值大于栈顶组块的灰度值, 则回到步骤 1.


"""
import cv2 as cv


def show_image(image, win_name='input image'):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


def demo1():
    image_path = '../dataset/data/airport/airport1.jpg'
    image = cv.imread(image_path)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    mser = cv.MSER_create()
    contours, bboxes = mser.detectRegions(gray)

    hulls = [cv.convexHull(contour) for contour in contours]

    for i, _ in enumerate(hulls):
        cv.drawContours(image, hulls, i, (0, 0, 255), 2)

    show_image(image)
    return


def demo2():
    image_path = '../dataset/data/airport/airport1.jpg'
    image = cv.imread(image_path)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    mser = cv.MSER_create()
    contours, bboxes = mser.detectRegions(gray)

    for box in bboxes:
        x, y, w, h = box
        cv.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    show_image(image)
    return


def demo3():
    image_path = '../dataset/data/airport/airport1.jpg'
    image = cv.imread(image_path)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    mser = cv.MSER_create()
    contours, bboxes = mser.detectRegions(gray)

    for i, _ in enumerate(contours):
        cv.drawContours(image, contours, i, (0, 0, 255), 2)

    show_image(image)
    return



if __name__ == '__main__':
    demo1()
