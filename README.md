# RNN_in_text
Для корректной работы программы необходим установить библиотеки:
pandas, re, pymorphy2, tqdm, scikit-learn, PyTorch, torchtext, pickle.

__Основные файлы__
Файл тренировки:  rnn_in_text.jpynb  
Файл тестирования: testing_model.py   
Для работы файла testing_model.py необходим словарь vocab.txt и модель model.torch. Они создаются в процессе тренировки

Вспомогательные файлы:
Данные для тестирования: texts.txt - Вносятся тексты для анализа 
Словарь: vocab.txt - Создаётся после выполнения файла rnn_in_text.jpynb
Модель : model.torch - Создаётся после выполнения файла rnn_in_text.jpynb
Вспомогательные функции: functions.py
В папке Data находятся данные для обучения модели
