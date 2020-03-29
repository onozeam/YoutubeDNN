# [Deep Neural Networks for YouTube Recommendations](https://static.googleusercontent.com/media/research.google.com/ja//pubs/archive/45531.pdf)

- download learned word2vec model from (https://github.com/Kyubyong/wordvectors).


## Usage
download learned word2vec model from (https://github.com/Kyubyong/wordvectors). and put in project dir, for example
```
$ mv ja.bin ~/../youtube_recommendation/.
```

get mecab
```
$ brew install mecab mecab-ipadic
$ git clone --depth 1 https://github.com/neologd/mecab-ipadic-neologd.git
$ cd mecab-ipadic-neologd
$ ./bin/install-mecab-ipadic-neologd -n
```
get liverdoor news corpus
```
$ pwd 
~/../youtube_recommendation
$ wget https://www.rondhuit.com/download/ldcc-20140209.tar.gz
$ tar xvzf ldcc-20140209.tar.gz
```
train
```
$ python3 main.py
```
