import jieba
from nltk import word_tokenize
import subword_nmt

if __name__=="__main__":
    # en 
    with open('data/NEU.en', "r") as f1:
        with open('data/NEU.en.tok', "w") as f2:
            for line in f1:
                words = word_tokenize(line)
                f2.write(' '.join(words) + '\n')    

    # zh
    with open('data/NEU.zh', "r") as f1:
        with open('data/NEU.zh.tok', "w") as f2:
            for line in f1:
                words = list(jieba.cut(line))
                f2.write(' '.join(words))
