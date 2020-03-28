#!/usr/bin/env python3
"""
# Youtube's video recommendation model
# https://static.googleusercontent.com/media/research.google.com/ja//pubs/archive/45530.pdf


# architecture
## candidate generation model
### flow
- DNN(user_feature, context_feature) = user_and_context_vector
- user_and_context_vector * video_corpus = ratings_matrix
- get top N item from rating_matrix.
### data
user_feature, video_corpus
- user_feature: video_watches, serch_tokens, geographic, age, and gender.
- video_corpus: title and description of videos.
## ranking model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
# from item_data import get_item_vector


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
        out = torch.matmul(personal_context.mean(axis=1), item_src.t())  # (batch_size, embed_item_size) * (embed_item_size, n_item)
        return out


def main():
    # item_vector = get_item_vector()
    # user data
    ## video watches: 各userに対して, 記事IDをランダムに生成して, idに対応するitem_vectorを取得して, ffに通して, avgをとる.
    ## age, gender: 2次元のベクトルにして, ffに通す.
    batch_size = 333  # n_userをbatch_sizeに分割する
    # n_item_feature = 120  # 1記事から使う単語数
    # embed_feature_size = 100  # 一単語の埋め込みサイズ
    embed_item_size = 100  # 一記事を表現するembeddingサイズ
    hidden_size = 1000
    n_item = 1500  # 記事数
    # dummy data
    item = torch.randn(n_item, embed_item_size)  # 1記事で全単語をavgしてたと仮定.
    video_watches = torch.randn(batch_size, 1, embed_item_size)   # 全ての記事の特徴量ベクトルのavgをとったものと仮定
    ages = torch.randint(0, 100, (batch_size, 1, 1), dtype=torch.float)  # (batch_size, 1, 1)
    gender = torch.randint(0, 2, (batch_size, 1, 1), dtype=torch.float)  # (batch_size, 1, 1)
    personal = torch.cat((ages, gender), 1)  # (batch_size, n_personal)

    model = CandidateGeneration(embed_item_size, hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.98), eps=1e-9)
    out = model(video_watches, personal, item)
    print(out.shape)
    optimizer.zero_grad()
    train_label = torch.random.randn()
    loss = nn.CrossEntropyLoss(reduction='none')
    loss(loss, train_label)
    # loss = nn.CrossEntropyLoss(ignore_index=0)(out.view(-1, out.size(-1)), tgt_label.view(-1))
    loss
    # loss += 0.001*(avg_n_updates + avg_remainders)
    # del out, tgt_label
    # loss.backward()
    # optimizer.step()


if __name__ == '__main__':
    main()
