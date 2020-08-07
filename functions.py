import re

from pymorphy2 import MorphAnalyzer

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext import data, vocab


def clearing(text):
    """
    Удаляет все кроме русских символов
    :param text: необработанный текст
    :return: чистый текст
    """
    result = ' '.join(re.sub(r'[^а-яА-ЯёЁ]', ' ', text).split())
    if len(result) == 0:
        result = None
    return result


morph = MorphAnalyzer()
def tokenizer(text: str) -> list:
    """
    Функция приводит каждое слово в строке к нормально словоформе
    :param text: необработанный
    :return: лемматизированные текст
    """
    lemm_list = list(map(lambda x: morph.parse(x)[0].normal_form, text.split()))
    return lemm_list


class DataFrameDataset(data.Dataset):
    """
    Класс создает датасет из типа DataFrame
    """
    def __init__(self, df, fields, is_test=False, **kwargs):
        examples = []
        for i, row in df.iterrows():
            label = row.target
            text = row.text
            examples.append(data.Example.fromlist([text, label], fields))

        super().__init__(examples, fields, **kwargs)

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    @classmethod
    def splits(cls, fields, train_df, val_df=None, test_df=None, **kwargs):
        train_data, val_data, test_data = (None, None, None)
        data_field = fields

        if train_df is not None:
            train_data = cls(train_df.copy(), data_field, **kwargs)
        if val_df is not None:
            val_data = cls(val_df.copy(), data_field, **kwargs)
        if test_df is not None:
            test_data = cls(test_df.copy(), data_field, True, **kwargs)

        return tuple(d for d in (train_data, val_data, test_data) if d is not None)


class LSTM_net(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout)

        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):
        # Получение эмбеддинга
        embedded = self.embedding(text)
        # Упаковка последовательности
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        # Конкатинация последнего и предпоследнего выхода из LSTM слоя
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        # FeedForward
        output = F.relu(self.fc1(hidden))
        output = self.fc2(output)

        return output