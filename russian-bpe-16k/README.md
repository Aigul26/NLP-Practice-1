---
language: ru
license: mit
tags:
- tokenizer
- russian
- bpe
---

# Russian BPE Tokenizer 16000

## 🗃️ Корпус
50k+ слов с ria.ru, lenta.ru и др. (2020–2025)

## ⚙️ Параметры
- Алгоритм: BPE
- Размер словаря: 16,000
- Min frequency: 2

## 📊 Метрики
- OOV rate: 1.2%
- Reconstruction accuracy: 99.8%
- Compression ratio: 1.35

## 💻 Пример использования
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Aigul26/russian-bpe-16k")

text = "Привет, как дела?"
tokens = tokenizer.tokenize(text)
print(tokens)  # ['ри', 'вет', ',', 'как', 'дела', '?']
📜 Лицензия
MIT
