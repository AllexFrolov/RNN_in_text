from functions import clearing, tokenizer, DataFrameDataset, LSTM_net
import pandas as pd
import torch
from torchtext import data, vocab
import pickle


def load_vocab(path):
    file = open(path, 'rb')
    out = pickle.load(file)
    file.close()
    return out


def load_model(path='model.torch'):
    """
    Загружает аргументы и веса модели
    :param path: Путь к файлу
    :return: Модель
    """
    checkpoint = torch.load(path)
    model = LSTM_net(checkpoint['input_dim'],
                     checkpoint['embedding_dim'],
                     checkpoint['hidden_dim'],
                     checkpoint['output_dim'],
                     checkpoint['n_layers'],
                     checkpoint['bidirectional'],
                     checkpoint['dropout'],
                     checkpoint['pad_idx'])

    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def predict(model, loader):
    """
    Делает предсказания
    :param model: Модель
    :param loader: Итератор данных
    """
    model.eval()
    with torch.no_grad():
        for ind, batch in enumerate(loader):
            text, text_lengths = batch.text
            predictions = model(text, text_lengths).squeeze(1)
            if torch.round(torch.sigmoid(predictions)).item() == 1:
                print('Позитив')
            else:
                print('Негатив')


def prepair_text(text: list):
    """
    Обрабатывает сырой текст. Создает итератор
    :param text: Необработанный лист с тектом
    :return: Загрузчик
    """
    text_df = pd.DataFrame(text, columns=['text'])
    text_df['target'] = 0
    text_df['text'] = text_df['text'].apply(clearing)

    field = data.Field(tokenize=tokenizer, include_lengths=True)
    field.vocab = load_vocab('vocab.pkl')
    label = data.LabelField(dtype=torch.float)
    label.vocab = vocab.Vocab({'<unk>': 0, '<pad>': 1, 0: 2, 1: 1})
    fields = [('text', field), ('label', label)]
    text_ds = DataFrameDataset(text_df, fields)

    text_loader = data.BucketIterator(text_ds,
                                      sort_within_batch=True,
                                      batch_size=1)
    print()
    print(text[0])
    print(text_df['text'].values)
    print(vars(text_ds[0])['text'])

    return text_loader


def load_text(path):
    with open(path, 'r', encoding='utf-8') as f:
        out = list(map(str.strip, f.read().split('\n')))
    return out


if __name__ == '__main__':
    # Загрузим тексты из файла
    texts = load_text('texts.txt')
    # Пройдемся по каждому тексту
    for text in texts:
        loader = prepair_text([text])
        model = load_model()
        predict(model, loader)