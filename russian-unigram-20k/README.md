---
language: ru
license: mit
tags:
- tokenizer
- russian
- unigram
---

# Russian UNIGRAM Tokenizer 20000

## 🗃️ Корпус
50k+ слов с ria.ru, lenta.ru и др. (2020–2025)

## ⚙️ Параметры
- Алгоритм: UNIGRAM
- Размер словаря: 20,000
- Min frequency: 2

## 📊 Метрики
- OOV rate: 1.5%
- Reconstruction accuracy: 99.5%
- Compression ratio: 1.45

## 💻 Пример использования
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("SherifAnar/russian-unigram-20k")

text = "Привет, как дела?"
tokens = tokenizer.tokenize(text)
print(tokens)  # ['П', 'ри', 'вет', ',', 'как', 'дела', '?']
📜 Лицензия
MIT
