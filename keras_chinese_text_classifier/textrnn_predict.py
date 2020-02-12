import json
import numpy as np
import keras
from keras.preprocessing import sequence
from sklearn import metrics
from tensorflow.keras.models import load_model

MAX_LEN = 100
num_classes = 3


def read_corpus(corpus_path):
    data, labels = [], []
    with open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()
        for line in lines:
            data.append(line.strip().split("_!_")[-1])
            labels.append(line.strip().split("_!_")[0])
    return data, labels


def _labelToIndex(labels, label2idx):
    """
    将标签转换成索引表示
    """
    labelIds = [[label2idx[label] for label in labels1] for labels1 in labels]
    return labelIds


def _wordToIndex(reviews, word2idx):
    """
    将词转换成索引
    """
    reviewIds = [[word2idx.get(item, word2idx["UNK"]) for item in review] for review in
                 reviews]  # get(key)=value unk不在单词表中，我们通过去停词和低频词已经去除掉一些单词了
    return reviewIds


def gen_data(filePathTrain):
    """
    初始化训练集和验证集
    """
    # 初始化数据集
    reviews, labels = read_corpus(filePathTrain)
    # 初始化词汇-索引映射表和词向量矩阵
    with open("word2idx.json", "r", encoding="utf-8") as f:
        word2idx = json.load(f)

    with open("label2idx.json", "r", encoding="utf-8") as f:
        label2idx = json.load(f)
    # 将标签和句子数值化
    labelIds = _labelToIndex(labels, label2idx)
    reviewIds = _wordToIndex(reviews, word2idx)
    # 初始化训练集和测试集
    return labelIds, reviewIds


labelIds, reviewIds = gen_data(filePathTrain=r"/Users/caowenli/Desktop/nlp/chineseGLUEdatasets.v0.0.1/inews/test.txt")
ont_hot_labelIds = keras.utils.to_categorical(labelIds, num_classes=num_classes)
reviewIds = sequence.pad_sequences(reviewIds, maxlen=MAX_LEN)
print(reviewIds.shape)
print(len(labelIds))
model = load_model('textrnn_model.h5')
result = model.predict(x=reviewIds)  # 预测样本属于每个类别的概率
result_labels = np.argmax(result, axis=1)  # 获得最大概率对应的标签
y_predict = list(map(int, result_labels))
print(y_predict)
labelIds_res = []
for i in range(len(labelIds)):
    labelIds_res.append(labelIds[i][0])
print(labelIds_res)
print('准确率', metrics.accuracy_score(labelIds, y_predict))
print('平均f1-score:', metrics.f1_score(labelIds, y_predict, average='weighted'))
