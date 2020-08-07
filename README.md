# RNN_in_text
## Постановка задачи заказчиком:
Нейросеть на питоне для анализа тональности текста.


Для корректной работы программы необходим установить [библиотеки](https://github.com/AllexFrolov/RNN_in_text/blob/master/requirements.txt)

__Основные файлы__  
Файл тренировки [rnn_in_text.jpynb](https://github.com/AllexFrolov/RNN_in_text/blob/master/rnn_in_text.ipynb)  
Файл тестирования [testing_model.py](https://github.com/AllexFrolov/RNN_in_text/blob/master/testing_model.py)   
Для работы файла testing_model.py необходим словарь vocab.txt и модель model.torch. Они создаются в процессе тренировки

Вспомогательные файлы:
Данные для тестирования: texts.txt - Вносятся тексты для анализа 
Словарь: vocab.txt - Создаётся после выполнения файла rnn_in_text.jpynb
Модель : model.torch - Создаётся после выполнения файла rnn_in_text.jpynb
Вспомогательные функции: [functions.py](https://github.com/AllexFrolov/RNN_in_text/blob/master/functions.py)
В папке Data находятся данные для обучения модели
