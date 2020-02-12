import json
import numpy as np
import keras
import re
from tensorflow.keras.preprocessing import sequence
from sklearn import metrics
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Embedding, Dense, Conv1D, GlobalMaxPooling1D, Concatenate, Dropout
from tensorflow.keras.models import load_model

filePathTrain = r"/Users/caowenli/Desktop/nlp/chineseGLUEdatasets.v0.0.1/inews/train.txt"
maxlen = 10
maxlen_sentence = 10
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
    # 初始化词汇-索引映射表和词向量矩阵
    word2idx, label2idx, uniqueword_len = _genVocabulary(reviews, labels)
    # 将标签和句子数值化
    labelIds = _labelToIndex(labels, label2idx)
    reviewIds = _wordToIndex(reviews, word2idx)
    # 初始化训练集和测试集
    return labelIds, reviewIds, uniqueword_len


from tensorflow.keras import backend as K
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras.layers import Layer


class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            # 1
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
            # next add a Dense layer (for classification/regression) or whatever...
            # 2
            hidden = LSTM(64, return_sequences=True)(words)
            sentence = Attention()(hidden)
            # next add a Dense layer (for classification/regression) or whatever...
        """
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0

        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight(shape=(input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        e = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))),
                      (-1, step_dim))  # e = K.dot(x, self.W)
        if self.bias:
            e += self.b
        e = K.tanh(e)

        a = K.exp(e)
        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())
        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)

        c = K.sum(a * x, axis=1)
        return c

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim


from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Embedding, Dense, Dropout, Bidirectional, LSTM, TimeDistributed


class HAN(object):
    def __init__(self, maxlen_sentence, maxlen_word, max_features, embedding_dims,
                 class_num,
                 last_activation='softmax'):
        self.maxlen_sentence = maxlen_sentence
        self.maxlen_word = maxlen_word
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.class_num = class_num
        self.last_activation = last_activation

    def get_model(self):
        # Word part
        input_word = Input(shape=(self.maxlen_word,))
        x_word = Embedding(self.max_features, self.embedding_dims, input_length=self.maxlen_word)(input_word)
        x_word = Bidirectional(LSTM(128, return_sequences=True))(x_word)  # LSTM or GRU
        x_word = Attention(self.maxlen_word)(x_word)
        model_word = Model(input_word, x_word)

        # Sentence part
        input = Input(shape=(self.maxlen_sentence, self.maxlen_word))
        x_sentence = TimeDistributed(model_word)(input)
        x_sentence = Bidirectional(LSTM(128, return_sequences=True))(x_sentence)  # LSTM or GRU
        x_sentence = Attention(self.maxlen_sentence)(x_sentence)
        output = Dense(self.class_num, activation=self.last_activation)(x_sentence)
        model = Model(inputs=input, outputs=output)
        return model


labelIds, reviewIds, uniqueword_len = gen_data(filePathTrain)
ont_hot_labelIds = keras.utils.to_categorical(labelIds, num_classes)
reviewIds = sequence.pad_sequences(reviewIds, 100)
print(reviewIds.shape)
reviewIds = reviewIds.reshape((-1, 10, 10))
print(reviewIds.shape)
textrnn = HAN(maxlen_sentence, maxlen, uniqueword_len, embedding_dims, num_classes)
model = textrnn.get_model()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x=reviewIds, y=ont_hot_labelIds, batch_size=batch_size, epochs=epochs)
model.save_weights("HAN_model.h5")
model.load_weights('HAN_model.h5')
result = model.predict(x=reviewIds)  # 预测样本属于每个类别的概率
result_labels = np.argmax(result, axis=1)  # 获得最大概率对应的标签
y_predict = list(map(int, result_labels))
print(y_predict)
labelIds_res = []
for i in range(len(labelIds)):
    labelIds_res.append(labelIds[i][0])
print(labelIds_res)
print('训练数据准确率', metrics.accuracy_score(labelIds, y_predict))
print('训练数据平均f1-score:', metrics.f1_score(labelIds, y_predict, average='weighted'))
