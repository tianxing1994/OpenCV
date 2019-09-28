"""
参考链接:
https://blog.csdn.net/qq_37667364/article/details/81071190

每一个字的 4 个隐状态:
start(该字是一个词的开头)
middle (该字是一个词的中间部分)
end (该字是一个词的结束)
single (该字是单独字符)
"""
import numpy as np


class HMM4MLE(object):
    def __init__(self):
        self._X = None

        self._pi = None
        self._A = None
        self._B = None

    def fit(self, X):
        """
        :param X: 用于进行分词训练的文本的列表或其可迭代对像. 如: ['以', '昂扬', '的', '斗志', '迎来', '虎年']
        """
        self._X = X
        pi, A, B = self._fit()
        self._pi, self._A, self._B = pi, A, B
        return

    def _fit(self):
        # 4 种隐状态, pi 统计每种状态 (0 - start, 1 - middle, 2 - end, 3 - single) 出现的次数. 之后再转换为每种隐状态的初始概率.
        pi = np.zeros(4)
        A = np.zeros(shape=(4, 4))
        B = np.zeros(shape=(4, 65536))  # unicode 字符 65536 个.

        # last_token: 单个字符. 存储当前 token 的前一个 token 用于对比.
        last_word = self._X[0]

        for word in self._X:
            word = word.strip()
            word_len = len(word)

            # 根据前一个词是单个文字或多个来定义 last_token_state.
            last_token_state = 3 if len(last_word) == 1 else 2

            if word_len == 0:
                continue

            if word_len == 1:
                # 单文字的词.
                pi[3] += 1
                A[last_token_state, 3] += 1
                B[3, ord(word)] += 1
            elif word_len == 2:
                # 两个文字的词.
                # start, end 加 1.
                pi[0] += 1
                pi[2] += 1
                # last_token_state 转 0-start 加 1, 0-start 转 2-end 加 1.
                A[last_token_state, 0] += 1
                A[0, 2] += 1
                # 更新发射概率.
                B[0, ord(word[0])] += 1
                B[2, ord(word[1])] += 1
            else:
                # 多文字的词.
                # 包含 start - middle, middle - middle, middle - end. 三种转换.
                pi[0] += 1
                pi[1] += word_len - 2  # 长度减首尾两个字符, 得到 middle 的数量.
                pi[2] += 1
                #
                A[last_token_state, 0] += 1
                A[0, 1] += 1
                A[1, 1] += word_len - 3  # middle-middle 的次数.
                A[1, 2] += 1
                #
                B[0, ord(word[0])] += 1
                for i in range(1, word_len - 1):
                    B[1, ord(word[i])] += 1
                B[2, ord(word[-1])] += 1
            last_word = word

        pi = pi / np.sum(pi)
        A = A / np.sum(A, axis=1, keepdims=True)
        B = B / np.sum(B, axis=1, keepdims=True)
        return pi, A, B

    def predict(self, X):
        """
        对文本 X 进行分词.
        :param X: 需要进行分词的文本.
        :return: Z, words. Z: 通过 O 预测出的隐状态转换的序列. words: 向每个分词后追加 '|' 后的文本.
        """
        if self._pi is None or \
            self._A is None or \
            self._B is None:
            raise RuntimeError("please fit model before call predict function.")
        O = self.word2ord(X)
        Z = self.viterbi_log(self._pi, self._A, self._B, O)
        words = self.output_words(Z, X)
        return Z, words

    @staticmethod
    def viterbi(pi, A, B, O):
        """
        维特比算法.
        """
        n = len(O)
        sigma_1 = pi * B[:, O[0]]
        sigma_list = list()
        sigma_list.append(sigma_1)

        # 计算 sigma
        for i in range(1, n):
            sigma = np.max(A * np.expand_dims(sigma_list[-1], axis=1) * B[:, O[i]], axis=0)
            sigma_list.append(sigma)

        # 最优路径的终点隐状态 i_T.
        i_T = np.argmax(sigma_list[-1])

        # 由最优路径的终点 i_T, 逆向求 I.
        I = list()
        I.append(i_T)
        for i in range(n - 2, -1, -1):
            sigma = sigma_list[i]
            I_t = np.argmax((A * np.expand_dims(sigma, axis=1))[:, I[-1]])
            I.append(I_t)
        result = list(reversed(I))
        return result

    @staticmethod
    def viterbi_log(pi, A, B, O):
        """
        维特比算法. (对数).
        概率都是小于 1 的数, 多次相乘后, 值变得很小, 会溢出. 本方法. 对初始向量, 转换概率矩阵, 发射矩阵的概率取对数.
        :param pi: 概率取对数后的 pi
        :param A: 概率取对数后的 A
        :param B: 概率取对数后的 B
        :param O:
        :return:
        """
        pi = np.log(pi + 1e-20)
        A = np.log(A + 1e-20)
        B = np.log(B + 1e-20)

        n = len(O)
        sigma_1 = pi + B[:, O[0]]
        sigma_list = list()
        sigma_list.append(sigma_1)

        # 计算 sigma
        for i in range(1, n):
            sigma = np.max(A + np.expand_dims(sigma_list[-1], axis=1) + B[:, O[i]], axis=0)
            sigma_list.append(sigma)

        # 最优路径的终点隐状态 i_T.
        i_T = np.argmax(sigma_list[-1])

        # 由最优路径的终点 i_T, 逆向求 I.
        I = list()
        I.append(i_T)
        for i in range(n-2, -1, -1):
            sigma = sigma_list[i]
            I_t = np.argmax((A + np.expand_dims(sigma, axis=1))[:, I[-1]])
            I.append(I_t)
        result = np.array(list(reversed(I)))
        return result

    @staticmethod
    def word2ord(O):
        return np.array([ord(s) for s in O])

    @staticmethod
    def output_words(Z_predict, X):
        result = ''
        for z, x in zip(Z_predict, X):
            # 如果预测当前字符最有可能的状态是end或者single就分词
            if z == 2 or z == 3:
                result += x + '|'
            else:
                result += x
        return result


def demo():
    with open("../dataset/nlp/hmm_corpus/pku_training.utf8", "r", encoding="utf-8") as f:
        # pku_training = f.readlines()
        pku_training = f.read()[3:]
    X = pku_training.split("  ")

    with open("../dataset/nlp/hmm_corpus/novel.txt", "r+", encoding="utf-8") as f:
        # result = f.readlines()
        novel = f.read()[3:]
    novel = novel.strip()

    mle = HMM4MLE()
    mle.fit(X=X)
    Z, words = mle.predict(X=novel)
    print(Z)
    print(words)
    return


if __name__ == '__main__':
    demo()
