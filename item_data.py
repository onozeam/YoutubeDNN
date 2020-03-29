#!/usr/bin/env python3
import gensim
import glob
import MeCab
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import urllib.request


# 流れ:
## 文書をmecabで分かち書き
## bag of wordを作成 (各単語の出現回数ベクトル)
## tf-idfで重み付けして, 上からN個の単語だけ取り出す.
## Word2Vecの学修済みモデルで各単語をベクトル化し、平均をとる.
# TODO:
## - 未知語を辞書に追加
## - 文書の特徴表現に, Word2Vecの平均ではなく, SCVDという手法を試す: https://www.m3tech.blog/entry/similarity-with-scdv
## - titleを情報に含める


def get_item_vector(item_size=100):
    print('creating item data...')
    np.random.seed(0)
    text_paths = glob.glob('livedoor_news_corpus/text/**/*.txt')
    # print(len(text_paths))  # 7376
    model = gensim.models.Word2Vec.load('ja.bin')

    def analyzer(text, mecab, stopwords=[], target_part_of_speech=['proper_noun', 'noun', 'verb', 'adjective']):
        node = mecab.parseToNode(text)
        words = []
        while node:
            features = node.feature.split(',')
            surface = features[6]
            if (surface == '*') or (len(surface) < 2) or (surface in stopwords):
                node = node.next
                continue
            noun_flag = (features[0] == '名詞')
            proper_noun_flag = (features[0] == '名詞') & (features[1] == '固有名詞')
            verb_flag = (features[0] == '動詞') & (features[1] == '自立')
            adjective_flag = (features[0] == '形容詞') & (features[1] == '自立')
            if ('proper_noun' in target_part_of_speech) & proper_noun_flag:
                words.append(surface)
            elif ('noun' in target_part_of_speech) & noun_flag:
                words.append(surface)
            elif ('verb' in target_part_of_speech) & verb_flag:
                words.append(surface)
            elif ('adjective' in target_part_of_speech) & adjective_flag:
                words.append(surface)
            node = node.next
        return words

    req = urllib.request.Request('http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/Japanese.txt')
    with urllib.request.urlopen(req) as res:
        stopwords = res.read().decode('utf-8').split('\r\n')
    while '' in stopwords:
        stopwords.remove('')

    words_list = []
    for tp in text_paths[:item_size]:
        text = open(tp, 'r').read()
        text = text.split('\n')
        # title = text[2]
        text = ' '.join(text[3:])
        mecab = MeCab.Tagger('-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')
        words = analyzer(text, mecab, stopwords=stopwords, target_part_of_speech=['noun', 'proper_noun'])
        words = filter(lambda x: x in model.wv.vocab, words)  # 未知語を除外
        words_list.append(' '.join(words))

    docs = np.asarray(words_list)
    count = CountVectorizer()
    bags = count.fit_transform(docs)
    bags.toarray()  # (2, N)  全単語の出現回数 0を含む
    features = count.get_feature_names()

    # features: データ内の全単語 (F)
    # docs: 各文章の単語を' 'でつなげたもの
    # bags: featuresのidxにしたがって, 出現回数をカウントしたもの (N, F)
    tfidf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
    np.set_printoptions(precision=2)
    tf_idf = tfidf.fit_transform(bags)
    tf_idf = tf_idf.toarray()
    features = np.asarray(features)
    sorted_features = features[tf_idf.argsort(axis=1)][:, -10:]  # 各item内におけるtf-idf top10の単語

    item_vecs = []
    for f in sorted_features:
        item_vecs.append(model[f])
    item_vecs = np.asarray(item_vecs)  # (記事数, 各記事から取得する単語数, 1単語あたりのベクトル長)
    return item_vecs.mean(1)  # (記事数, 1単語あたりのベクトル長)  # 記事を合成した1つの単語ベクトルで表している


if __name__ == '__main__':
    item = get_item_vector()
    print(item)
    print(item.shape)
