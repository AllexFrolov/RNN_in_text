# RNN_in_text
__Постановка задачи заказчиком:__   
Реализовать рекурентную нейронную сеть для анализа тональности текста. Тренировочные данные состоят из двух файлов positive.txt и negative.txt с позитивными и негативными текстами соответственно. Важен сам pipeline и описание действий. 
## Этапы выполнения проекта
1. __Анализ и подготовка данных__
  * Объеденины данные, добавлен целевой признак 'Target' со значениями 1 - негативный текст, 0 - позитивный. Использовалась библиотека Pandas.
  * Текст отчищен от лишних знаков и сивмолов. Оставлены только буквы. Использована библиотека для регулярных выражений re.
  * При помощи библиотеки pymorphy2 слова приведены к их нормальной словарной форме:
    + для существительных — именительный падеж, единственное число;
    + для прилагательных — именительный падеж, единственное число, мужской род;
    + для глаголов, причастий, деепричастий — глагол в инфинитиве несовершенного вида.
  * Данные разделены на тренировочную, валидационную и тестовую выборки. Использовалась библиотека Scikit-Learn
  * Текст векторизирован. В качестве алгоритма использовался мешок слов, размер словаря 73567. Для этого была использована библиотекой PyTorch.
2. __Создание модели и тренировка__
  * Реализованы функции для тренировки модели. В качесте метрики выбрана и реализована функция F1-score.
  * Создана следующая модель Input __-> Embedding -> LSTM -> Dropout -> Linear -> ReLU -> Linear ->__ Class. Использована библиотека PyTorch
  * В качестве функции потерь была взята BCEWithLogLoss. Так было решено сделать различный learning rate для различных слоев модели.  
3. __Тестирование модели__
  * Проведена оценка модели на тестовой выборке.
  * Обученная модель сохранена для дальнейшего использования

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
