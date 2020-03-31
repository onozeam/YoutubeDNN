#!/usr/bin/env python3
"""
# Youtube's video recommendation model
# https://static.googleusercontent.com/media/research.google.com/ja//pubs/archive/45530.pdf

# Architecture
## candidate generation model
- user_and_context_vector = DNN(user_feature, context_feature)
- out = user_and_context_vector * video_corpus = ratings_matrix - loss := SigmoidCrossEntropy
- label := real rating matrix
- selection :=  get top N item from rating_matrix.
## ranking model
- out = LogisticRegression( DNN(watch_time, etc) )
- loss = SigmoidCrossEntropy
- label = (sum of watch_time / num of negatipve impression)
- selection := get top M items after sort videos by ranking score.
"""
import argparse
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


class Ranking(nn.Module):
    def __init__(self, watch_time_feature_size, hidden_size, candidate_size):
        super(Ranking, self).__init__()
        self.fc1 = nn.Linear(watch_time_feature_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, src):
        """
        input is (batch_size, n_item, watch_time_feature_size), and output is (batch_size, n_item).
        """
        h = F.relu(self.fc1(src))
        h = F.relu(self.fc2(h))
        out = F.relu(self.fc3(h))
        return out.squeeze(-1)


class BatchIterator:
    def __init__(self, x, y, batch_size):
        self.batch_size = batch_size
        self.i = 0
        self.x = x
        self.y = y

    def __iter__(self):
        return self

    def __next__(self):
        if self.i * self.batch_size == len(self.y):
            raise StopIteration()
        mini_x = self.x[self.i * self.batch_size: (self.i + 1) * self.batch_size]
        mini_y = self.y[self.i * self.batch_size: (self.i + 1) * self.batch_size]
        self.i += 1
        return mini_x, mini_y


class CandidateBatchIterator(BatchIterator):
    def __init__(self, x1, x2, y, batch_size):
        self.batch_size = batch_size
        self.i = 0
        self.x1 = x1
        self.x2 = x2
        self.y = y

    def __next__(self):
        if self.i * self.batch_size == len(self.y):
            raise StopIteration()
        mini_x1 = self.x1[self.i * self.batch_size: (self.i + 1) * self.batch_size]
        mini_x2 = self.x2[self.i * self.batch_size: (self.i + 1) * self.batch_size]
        mini_y = self.y[self.i * self.batch_size: (self.i + 1) * self.batch_size]
        self.i += 1
        return mini_x1, mini_x2, mini_y


def train_candidate_generation(model, get_batch_iter, item, batch_size):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.98), eps=1e-9)
    epochs = 10
    for epoch in range(epochs):
        total_loss = 0
        batch_iter = get_batch_iter()
        for iter_, (mini_personal, mini_watches, mini_label) in enumerate(batch_iter):
            out = model(mini_watches, mini_personal, item)
            optimizer.zero_grad()
            loss = nn.MSELoss(reduction='sum')(out, mini_label)  # todo: use sigmoid cross entropy loss
            total_loss += loss.item()
            if iter_ != 0 and (iter_ + 1) % 10 == 0:
                print(f'epoch: {epoch + 1}, iter: {iter_ + 1}, loss: {total_loss/10}')
                total_loss = 0
            loss.backward()
            optimizer.step()


def train_ranking(model, get_batch_iter, batch_size):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.98), eps=1e-9)
    epochs = 3
    for epoch in range(epochs):
        total_loss = 0
        batch_iter = get_batch_iter()
        for iter_, (mini_x, mini_label) in enumerate(batch_iter):
            out = model(mini_x)  # (batch_size, n_item)
            optimizer.zero_grad()
            loss = nn.MSELoss(reduction='sum')(out, mini_label)  # todo: use sigmoid cross entropy loss
            total_loss += loss.item()
            if iter_ != 0 and (iter_ + 1) % 10 == 0:
                print(f'epoch: {epoch + 1}, iter: {iter_ + 1}, loss: {total_loss/10}')
                total_loss = 0
            loss.backward()
            optimizer.step()


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Initialize training parameter.')
    parser.add_argument('-target_model', required=True, type=str,
                        help='"c" or "r". Please choose the training target model. "c" is candidate generation model, "r" is ranking model.')
    args = parser.parse_args()

    # const
    n_item = 20  # trainingに使用する記事数.
    n_user = 10000
    batch_size = 100  # n_userをbatch_sizeに分割する

    # candidate generation
    if args.target_model == 'c':
        # const
        item = torch.tensor(get_item_vector(n_item))  # 記事データ. shape is (n_item, embed_item_size).
        embed_item_size = item.size(1)  # 一記事を表現するembeddingサイズ
        candidate_hidden_size = 600  # candidate_modelの隠れ層サイズ
        candidate_size = 10  # 候補に出す記事数
        # data
        ages = torch.randint(0, 100, (n_user, 1, 1), dtype=torch.float)  # (n_user, 1, 1)
        gender = torch.randint(0, 2, (n_user, 1, 1), dtype=torch.float)  # (n_user, 1, 1)
        personal = torch.cat((ages, gender), 1)  # (n_user, n_personal)  # [[age, sex], [age, sex], ...]
        watches = torch.randn(n_user, 1, embed_item_size)  # 視聴した全ての動画の特徴量ベクトルを平均したものと仮定. つまり↓3行のを行ったのと等価.
        """
        wathces = [[id, id, id], [id], [id, id], ...]  (n_user,  n_each_watch)
        wathces = [[embed_item, embed_item, embed_item], [embed_item], [embed_item, embed_item], ...]  (n_user,  n_each_watch, embed_item)
        watches = wathces.mean(0)  (n_user, embed_item)
        """
        candidate_train_label = torch.randint(0, 10, (n_user, n_item), dtype=torch.float)  # (user, video) matrix. value is num of clicks.
        cbatch_iter = lambda: CandidateBatchIterator(personal, watches, candidate_train_label, batch_size)  # noqa: E731
        # model
        cmodel = CandidateGeneration(embed_item_size, candidate_hidden_size)
        train_candidate_generation(cmodel, cbatch_iter, item, batch_size)
    if args.target_model == 'r':
        # const
        watch_time_feature_size = 124
        ranking_hidden_size = 248
        candidate_size = 10
        # data
        watch_time_vector = torch.rand(n_user * n_item, n_item, watch_time_feature_size)
        real_impression_matrix = torch.randint(3, 9, (n_user * n_item, n_item), dtype=torch.float)
        real_watch_time_matrix = torch.empty(n_user * n_item, n_item).uniform_(0, 10)
        ranking_train_label = F.softmax(real_watch_time_matrix / real_impression_matrix, dim=-1)  # (n_user*n_item, n_item)
        rbatch_iter = lambda: BatchIterator(watch_time_vector, ranking_train_label, batch_size)  # noqa: E731
        # model
        rmodel = Ranking(watch_time_feature_size, ranking_hidden_size, candidate_size)
        train_ranking(rmodel, rbatch_iter, batch_size)


if __name__ == '__main__':
    main()
