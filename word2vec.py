import csv
import re
import numpy as np
import ast
import pickle
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression

EMBEDDINGS = 100

def read_input(text_path):
    with open(text_path,'r') as file:
        text = []
        for line in file:
            text.append(ast.literal_eval(line))
    return text

def multiple_replace(dict, text):
  regex = re.compile("(%s)" % "|".join(map(re.escape, dict.keys())))
  return regex.sub(lambda mo: dict[mo.string[mo.start():mo.end()]], text)

def preprocess(y_train_path, x_train_path, x_test_path):
    y_train = []
    with open(y_train_path, 'r') as content_file:
        content = content_file.read()
        dictn = {"S":"0", "P":"1"}
        new_y = multiple_replace(dictn, content)
        for line in new_y:
            y_train.append(line)

    pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')

    x_train = []
    file = open(x_train_path, encoding='utf-8')
    for line in file:
        x_train.append(pattern.findall(line.lower()))

    x_test = []
    file = open(x_test_path, encoding='utf-8')
    for line in file:
        x_test.append(pattern.findall(line.lower()))

    return y_train, x_train, x_test

def preprocess_run(y_train_path, x_train_path, x_test_path):
    y_train, x_train, x_test = preprocess(y_train_path, x_train_path, x_test_path)
    with open('x_train.txt', 'w') as filehandle:
        for line in x_train:
            filehandle.write(str(line) + '\n')
    with open('y_train.txt', 'w') as filehandle:
        for line in y_train:
            filehandle.write(line)
    with open('x_test.txt', 'w') as filehandle:
        for line in x_test:
            filehandle.write(str(line) + '\n')

def train_model(x_train_path, x_test_path):
    x_train = read_input(x_train_path)
    x_test = read_input(x_test_path)
    vocab = x_train + x_test
    model = Word2Vec(vocab, min_count=1, size=EMBEDDINGS, workers=4,sg=1)
    model.save("word2vec.model")

    vect_doc = []
    for line in x_train:
        temp = np.zeros((EMBEDDINGS))
        for word in line:
            temp = temp + model[word]
            temp = temp / len(line)
        vect_doc.append(temp)
    X = np.array(vect_doc)
    y = np.loadtxt('y_train.txt')

    word2vec_log_reg = LogisticRegression()
    word2vec_log_reg.fit(X, y)

    with open('word2vec_log_reg.pkl', 'wb') as fid:
        pickle.dump(word2vec_log_reg, fid)


def evaluate(x_test_path):
    model = Word2Vec.load("word2vec.model")
    with open('word2vec_log_reg.pkl', 'rb') as fid:
        word2vec_log_reg = pickle.load(fid)

    x_test = read_input(x_test_path)
    vect_doc = []
    for line in x_test:
        temp = np.zeros((EMBEDDINGS))
        for word in line:
            temp = temp + model[word]
            temp = temp / len(line)
        vect_doc.append(temp)
    x_test = np.array(vect_doc)

    predicted = word2vec_log_reg.predict(x_test)

    with open('../dev-0/out.txt', 'w') as filehandle:
        for line in predicted.astype(int):
            filehandle.write(str(line)+'\n')

preprocess_run("../dev-0/expected.tsv", "../dev-0/in.tsv", "../dev-0/in.tsv")
train_model("x_train.txt", "x_test.txt")
evaluate("x_test.txt")




