import json
import numpy as np
import keras
from tensorflow.keras.preprocessing import sequence
from sklearn import metrics
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Embedding, Dense, Conv1D, GlobalMaxPooling1D, Concatenate, Dropout
from tensorflow.keras.models import load_model

filePathTrain = r"/Users/caowenli/Desktop/nlp/chineseGLUEdatasets.v0.0.1/inews/train.txt"
maxlen = 100
embedding_dims = 200
batch_size = 128
epochs = 1
num_classes = 3


def read_corpus(corpus_path):
    data, labels = [], []
    with open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()
        for line in lines:
            data.append(line.strip().split("_!_")[-1])
            labels.append(line.strip().split("_!_")[0])
    return data, labels


def _genVocabulary(reviews, labels):
    """
    生成词向量和词汇-索引映射字典，可以用全数据集
    """
    allWords = [word for review in reviews for word in review]
    uniqueWords = list(set(allWords))
    uniqueWords.append("PAD")  # 句子长度
    uniqueWords.append("UNK")  # 词不在句子中
    print("所有的单词", len(uniqueWords))
    word2idx = dict(zip(uniqueWords, list(range(len(uniqueWords)))))
    uniqueword_len = len(uniqueWords)
    allLabels = [label for labels1 in labels for label in labels1]
    uniqueLabel = list(set(allLabels))
    print("类别数", len(uniqueLabel))
    label2idx = dict(zip(uniqueLabel, list(range(len(uniqueLabel)))))
    # 将词汇-索引映射表保存为json数据，之后做inference时直接加载来处理数据
    with open("word2idx.json", "w", encoding="utf-8") as f:
        json.dump(word2idx, f)
    with open("label2idx.json", "w", encoding="utf-8") as f:
        json.dump(label2idx, f)
    return word2idx, label2idx, uniqueword_len


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
    print(reviews[0])
    print(labels[0])
    # 初始化词汇-索引映射表和词向量矩阵
    word2idx, label2idx, uniqueword_len = _genVocabulary(reviews, labels)
    # 将标签和句子数值化
    labelIds = _labelToIndex(labels, label2idx)
    reviewIds = _wordToIndex(reviews, word2idx)
    # 初始化训练集和测试集
    return labelIds, reviewIds, uniqueword_len


from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Embedding, Dense, Dropout, LSTM

# class TextRNN(object):
#     def __init__(self, maxlen, max_features, embedding_dims,
#                  class_num,
#                  last_activation='softmax'):
#         self.maxlen = maxlen
#         self.max_features = max_features
#         self.embedding_dims = embedding_dims
#         self.class_num = class_num
#         self.last_activation = last_activation
#
#     def get_model(self):
#         input = Input((self.maxlen,))
#         embedding = Embedding(self.max_features, self.embedding_dims, input_length=self.maxlen)(input)
#         x = LSTM(128)(embedding)
#         output = Dense(self.class_num, activation=self.last_activation)(x)
#         model = Model(inputs=input, outputs=output)
#         return model

from tensorflow.keras import Input, Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Embedding, Dense, SimpleRNN, Lambda, Concatenate, Conv1D, GlobalMaxPooling1D


class RCNN(object):
    def __init__(self, maxlen, max_features, embedding_dims,
                 class_num,
                 last_activation='softmax'):
        self.maxlen = maxlen
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.class_num = class_num
        self.last_activation = last_activation

    def get_model(self):
        input_current = Input((self.maxlen,))
        input_left = Input((self.maxlen,))
        input_right = Input((self.maxlen,))
        embedder = Embedding(self.max_features, self.embedding_dims, input_length=self.maxlen)
        embedding_current = embedder(input_current)
        embedding_left = embedder(input_left)
        embedding_right = embedder(input_right)

        x_left = SimpleRNN(128, return_sequences=True)(embedding_left)
        x_right = SimpleRNN(128, return_sequences=True, go_backwards=True)(embedding_right)
        x_right = Lambda(lambda x: K.reverse(x, axes=1))(x_right)
        x = Concatenate(axis=2)([x_left, embedding_current, x_right])
        x = Conv1D(64, kernel_size=1, activation='tanh')(x)
        x = GlobalMaxPooling1D()(x)
        output = Dense(self.class_num, activation=self.last_activation)(x)
        model = Model(inputs=[input_current, input_left, input_right], outputs=output)
        return model


labelIds, reviewIds, uniqueword_len = gen_data(filePathTrain)
ont_hot_labelIds = keras.utils.to_categorical(labelIds, num_classes)
reviewIds = sequence.pad_sequences(reviewIds, maxlen)
print(reviewIds.shape)
print(ont_hot_labelIds.shape)
print(type(reviewIds))
textrcnn = RCNN(maxlen, uniqueword_len, embedding_dims, num_classes)
model = textrcnn.get_model()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

left_temp = np.array([reviewIds[:, 0]])
left_temp2 = np.c_[left_temp.T, np.array(reviewIds)]
left = left_temp2[:, 0:-1]
right_temp = np.array([reviewIds[:, -1]])
right_temp2 = np.c_[right_temp.T, np.array(reviewIds)]
right = right_temp2[:, 1:]
model.fit(x=[left, reviewIds, right], y=ont_hot_labelIds, batch_size=batch_size, epochs=epochs)
model.save("textrcnn_model.h5")
rnnmodel = load_model('textrcnn_model.h5')
result = rnnmodel.predict(
    x=[left, reviewIds, right])  # 预测样本属于每个类别的概率
result_labels = np.argmax(result, axis=1)  # 获得最大概率对应的标签
y_predict = list(map(int, result_labels))
print(y_predict)
labelIds_res = []
for i in range(len(labelIds)):
    labelIds_res.append(labelIds[i][0])
print(labelIds_res)
print('训练数据准确率', metrics.accuracy_score(labelIds, y_predict))
print('训练数据平均f1-score:', metrics.f1_score(labelIds, y_predict, average='weighted'))
