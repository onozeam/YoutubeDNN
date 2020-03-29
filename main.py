#!/usr/bin/env python3
"""
# Youtube's video recommendation model
# https://static.googleusercontent.com/media/research.google.com/ja//pubs/archive/45530.pdf

# Architecture
## candidate generation model
- user_and_context_vector = DNN(user_feature, context_feature)
- out = user_and_context_vector * video_corpus = ratings_matrix
- loss := SigmoidCrossEntropy
- label := real rating matrix
- selection :=  get top N item from rating_matrix.
## ranking model
- out = LogisticRegression( DNN(watch_time, etc) )
- loss = SigmoidCrossEntropy
- label = (sum of watch_time / num of negatipve impression)
- selection := get top M items after sort videos by ranking score.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from item_data import get_item_vector


class CandidateGeneration(nn.Module):
    def __init__(self, embed_item_size, hidden_size):
        super(CandidateGeneration, self).__init__()
        self.personal_fc = nn.Linear(1, embed_item_size)
        self.fc1 = nn.Linear(embed_item_size, hidden_size)  # noqa: E226
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, embed_item_size)

    def forward(self, context_src, personal_src, item_src):
        personal_h = self.personal_fc(personal_src)  # (batch_size, n_personal, embed_item_size)
        h = torch.cat((context_src, personal_h), 1)  # (batch_size, 1+n_personal, embed_item_size)
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        personal_context = F.relu(self.fc3(h))  # (batch_size, 1+n_personal, embed_item_size)
        # personal_contextを(batch_size, embed_item_size)の形にavgをとって, item_srcと内積を取る
        out = torch.matmul(personal_context.mean(axis=1), item_src.t())  # (batch_size, n_item) = (batch_size, embed_item_size) * (embed_item_size, n_item)  # noqa: E501
        return out


class BatchIterator:
    """ dummy dataでbatchを作成する """
    def __init__(self, embed_item_size, batch_size=100):
        self.batch_size = batch_size
        self.i = 0
        n_user = 10000
        ages = torch.randint(0, 100, (n_user, 1, 1), dtype=torch.float)  # (n_user, 1, 1)
        gender = torch.randint(0, 2, (n_user, 1, 1), dtype=torch.float)  # (n_user, 1, 1)
        self.personal = torch.cat((ages, gender), 1)  # (n_user, n_personal)  # [[age, sex], [age, sex], ...]
        self.watches = torch.randn(n_user, 1, embed_item_size)  # 視聴した全ての動画の特徴量ベクトルを平均したものと仮定. つまり↓3行のを行ったのと等価.
        # wathces = [[id, id, id], [id], [id, id], ...]  (n_user,  n_each_watch)
        # wathces = [[embed_item, embed_item, embed_item], [embed_item], [embed_item, embed_item], ...]  (n_user,  n_each_watch, embed_item)
        # watches = wathces.mean(0)  (n_user, embed_item)

    def __iter__(self):
        return self

    def __next__(self):
        if self.i * self.batch_size == len(self.watches):
            raise StopIteration()
        watches = self.watches[self.i * self.batch_size: (self.i + 1) * self.batch_size]
        personal = self.personal[self.i * self.batch_size: (self.i + 1) * self.batch_size]
        self.i += 1
        return watches, personal


def main():
    # data
    batch_size = 100  # n_userをbatch_sizeに分割する
    embed_item_size = 100  # 一記事を表現するembeddingサイズ
    hidden_size = 1000
    n_item = 10  # 全記事数.
    item = torch.tensor(get_item_vector(n_item))  # 記事データ. shape is (n_item, embed_item_size).
    embed_item_size = item.size(1)  # 一記事を表現するembeddingサイズ
    # model
    model = CandidateGeneration(embed_item_size, hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.98), eps=1e-9)
    epochs = 3
    # train
    for epoch in range(epochs):
        iterator = BatchIterator(embed_item_size, batch_size)
        total_loss = 0
        for iter_, (mini_watches, mini_personal) in enumerate(iterator):
            out = model(mini_watches, mini_personal, item)
            optimizer.zero_grad()
            train_label = torch.randint(0, 10, (batch_size, n_item), dtype=torch.float)
            loss = nn.MSELoss(reduction='sum')(out, train_label)  # todo: use sigmoid cross entropy loss
            total_loss += loss.item()
            if iter_ != 0 and iter_ % 10 == 0:
                print(f'epoch: {epoch}, iter: {iter_}, loss: {total_loss/10}')
                total_loss = 0
            loss.backward()
            optimizer.step()


if __name__ == '__main__':
    main()
