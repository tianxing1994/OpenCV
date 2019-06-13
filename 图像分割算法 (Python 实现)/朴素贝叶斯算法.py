"""
参考链接:
https://www.cnblogs.com/ahu-lichang/p/7157855.html
"""
import numpy as np


documents = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
           ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
           ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
           ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
           ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
           ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]

label = [0, 1, 0, 1, 0, 1]


def bag_of_words(documents):
    bag = set()
    for document in documents:
        bag = bag | set(document)
    return list(bag)


def word2vector(document, bag_of_words):
    vector = [0] * len(bag_of_words)
    for word in document:
        if word in bag_of_words:
            vector[bag_of_words.index(word)] += 1
    return vector


def documents2vector(documents, bag_of_words):
    vecters = []
    for document in documents:
        vector = word2vector(document, bag_of_words)
        vecters.append(vector)
    return vecters


def fit():
    return





def load_dataset():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec


def createVocablist(dataset):
    vocabset = set()
    for document in dataset:
        vocabset = vocabset | set(document)
    return list(vocabset)


def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        else:
            print("the word %s is not in my Vocablary!" % word)
    return returnVec


def trainNBO(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num / p1Denom)
    p0Vect = np.log(p0Num / p0Denom)

    return p0Vect, p1Vect, pAbusive


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = np.sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = np.sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def testingNB():
    listOPosts, listClasses = load_dataset()
    myVocabList = createVocablist(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNBO(np.array(trainMat), np.array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    testDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
    return


if __name__ == '__main__':
    testingNB()


























