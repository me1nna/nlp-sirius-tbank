# Русскоязычная QA система для отбора на смену по ML в Сириусе от T-банка
### Ссылка на демо тг-бота: <a href='https://t.me/guru_qa_bot'>тык</a>

## What? 

### Функциональность

Требуется реализовать свою русскоязычную QA систему. Система должна отвечать на вопросы, при условии, что ответ уже есть в заданном вопросе.

### Метрики успеха
1. **Точность ответов**: Процент правильных ответов на тестовых данных.
2. **Время ответа**: Среднее время, за которое система генерирует ответ на вопрос.

## How? 


1. **Разбраотка Retrieval-Augmented Generation (RAG)**: Использование моделей для поиска релевантного контекста и генерации ответов на основе этого контекста.
   
2. **Использование модели T5**: А именно, `AAQG-QA-QG-FRED-T5-1.7B`, для генерации ответов на основе предоставленного контекста. Это `ai-forever/FRED-T5-1.7B` обученная модель на задачах **Question-Answering**, **Question-Generation** и **Answer-Aware Question Generation** на русскоязычном датасете `hivaze/ru-AAQG-QA-QG`. В моем решении используется способность к **Question-Answering** этой модели, т.е. генерация ответа на вопрос по предоставленному контексту (контексту, который находится с помощью другой модели по заданному вопросу).

3. **Разбиение на чанки**: Перед созданием эмбеддингов контекста, он бьется на чанки. Регулируя размер чанков, можно настроить размер контекста, поиск которого происходит для ответа на вопрос. Это может влиять на поведение модели.

4. **kNN**: Для нахождения релевантного контекста используется метод ближайших соседей, в качестве метрики используется косинусное сходство. 

---

### Данные
1. **Датасет SberQuAD**: Используется для предоставления данных для контекста. 
2. **Дополнительные данные**: Возможно, в дальнейшем потребуется дополнительный датасет для улучшения точности.

---

### Дополнительные фичи

В ходе разработки системы мною было принято решение добавить полезные фичи к решению, а именно:

1. **Детекция токсичности**: Для реализации используется модель `cointegrated/rubert-tiny-toxicity`. Если система считает ответ токсичным, не отвечает на вопрос и требует сменить тон общения

2. **Проверка орфографии**: Для этого используется `UrukHan/t5-russian-spell`. Вопрос проверяется на орфографию перед подачей в непосредственно саму систему


## Оценка результата


Я протестировала работоспособность модели на своих вопросах, и вот что она ответила:

```markdown
> Какой город является столицой Франции, известный как Париж?
> Париж
```

```markdown
> Как зовут создателя теории относительности Альберта Эйнштейна?
> Альберт Эйнштейн
```

```markdown
> Как зовут студентку Инну Вакуленко?
> Инна Вакуленко
```

```markdown
> Какое государство является родиной фьордов и известно как Норвегия?
> Норвегия
```

```markdown
> Как зовут знаменитого детектива, созданного Артуром Конаном Дойлем, известного как Шерлок Холмс?
> Шерлок Холмс
```

```markdown
> Какое оборудование использовалось в кинематографе, если киноплёнка шириной 35 мм и более считается профессиональной, а более узкая — любительской?
> Киноплёнка, съёмочные камеры, проекторы и монтажные столы.
```

```markdown
> Почему более 80 % российских менеджеров недовольны своими системами оценки, если они считают, что отсутствует связь между планами, исполнением, результатом и мотивацией?
> Потому что нет связи между планами, исполнением, результатом и мотивацией.
```

```markdown
> Как зовут первого человека, ступившего на Луну в 1969 году, известного как Нил Армстронг?
> Нил Армстронг
```
<br>
Иногда ошибается 😪:

```markdown
> Как называется самая высокая гора в мире, известная как Эверест?
> Гималаи (инд. Обитель снегов)
```

## How to run? 🏃‍♂️




## Что дальше?

1. Поэкспериментировать с размером чанков, на которые бьется контекст перед созданием эмбеддингов.
3. Расширить индекс контекстов, где ищется релевантный контекст.
4. Использовать для ответа не один релевантный контекст, а, например, несколько.



## Заключение

Создание русскоязычной системы ответов на вопросы на основе RAG и модели T5 позволит эффективно решать проблему поиска релевантной информации и улучшить пользовательский опыт. Интеграция с Telegram-ботом и Streamlit обеспечит удобное взаимодействие с пользователями и позволит быстро получать ответы на их вопросы. 
