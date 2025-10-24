import re
import json
import time
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Загрузка ресурсов NLTK для русского языка
def ensure_nltk_resources():
    """Загрузка или проверка ресурсов NLTK."""
    try:
        nltk.data.find('tokenizers/punkt_tab')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("Загружаем необходимые ресурсы NLTK...")
        nltk.download('punkt_tab')
        nltk.download('stopwords')

ensure_nltk_resources()

# Инициализация стоп-слов для русского языка
stop_words = set(stopwords.words('russian'))
# Дополнительные стоп-слова для новостных сайтов
stop_words.update(['тасс', 'риа', 'новости', 'лента', 'коммерсант'])

def clean_text(text, to_lower=True, remove_stopwords=True):
    """
    Очистка и нормализация текста.

    Args:
        text (str): Исходный текст.
        to_lower (bool): Приводить ли текст к нижнему регистру.
        remove_stopwords (bool): Удалять ли стоп-слова.

    Returns:
        str: Очищенный и нормализованный текст или None при ошибке.
    """
    try:
        # Удаление HTML-разметки
        soup = BeautifulSoup(text, 'html.parser')
        text = soup.get_text(separator=' ')

        # Удаление URL
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

        # Удаление служебных символов, эмодзи и специальных символов
        text = re.sub(r'[^\w\s.,!?]', '', text)

        # Удаление рекламных фраз
        ad_patterns = [
            r'подписывайтесь на наш канал',
            r'читайте также',
            r'поделиться в соцсетях',
            r'источник tass',
            r'риа новости',
            r'лента новостей',
            r'перейти в раздел',
            r'реклама',
            r'подписаться',
            r'больше новостей в'
        ]
        for pattern in ad_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        # Стандартизация пробельных символов
        text = re.sub(r'\s+', ' ', text).strip()

        # Пропуск пустого текста
        if not text:
            return None

        # Приведение к нижнему регистру
        if to_lower:
            text = text.lower()

        # Удаление стоп-слова
        if remove_stopwords:
            try:
                tokens = word_tokenize(text, language='russian')
                tokens = [token for token in tokens if token not in stop_words and token.strip()]
                text = ' '.join(tokens)
            except Exception as e:
                print(f"Ошибка токенизации: {str(e)[:100]}")
                return text  # Возвращаем текст без токенизации

        return text

    except Exception as e:
        print(f"Ошибка в clean_text: {str(e)[:100]}")
        return None

def process_corpus(input_file='corpus.jsonl', output_file='cleaned_corpus.jsonl', to_lower=True, remove_stopwords=True):
    """
    Обработка корпуса из JSONL-файла.

    Args:
        input_file (str): Путь к входному файлу corpus.jsonl.
        output_file (str): Путь к выходному файлу с очищенным текстом.
        to_lower (bool): Приводить ли текст к нижнему регистру.
        remove_stopwords (bool): Удалять ли стоп-слова.

    Returns:
        tuple: (количество обработанных статей, количество ошибок, общее количество слов)
    """
    processed_count = 0
    error_count = 0
    total_words = 0

    with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            try:
                article = json.loads(line.strip())
                cleaned_text = clean_text(article['text'], to_lower=to_lower, remove_stopwords=remove_stopwords)
                if cleaned_text:
                    article['cleaned_text'] = cleaned_text
                    words = len(cleaned_text.split())
                    total_words += words
                    json.dump(article, f_out, ensure_ascii=False)
                    f_out.write('\n')
                    processed_count += 1
                else:
                    print(f"Пропущена статья: {article.get('url', 'N/A')} (title: {article.get('title', 'N/A')[:50]}...) - пустой очищенный текст")
                    error_count += 1
            except Exception as e:
                print(f"Ошибка обработки статьи: {article.get('url', 'N/A')} (title: {article.get('title', 'N/A')[:50]}...) - {str(e)[:100]}")
                error_count += 1
                continue

    return processed_count, error_count, total_words

def main():
    """Пример использования модуля."""
    input_file = 'corpus.jsonl'
    output_file = 'cleaned_corpus.jsonl'

    print("Начало обработки корпуса...")
    start_time = time.time()
    processed_count, error_count, total_words = process_corpus(input_file, output_file, to_lower=True, remove_stopwords=True)
    print(f"Обработка завершена за {time.time() - start_time:.2f} секунд")
    print(f"Обработано статей: {processed_count}")
    print(f"Ошибок: {error_count}")
    print(f"Общее слов после очистки: {total_words}")

    # Вывод примера первой очищенной статьи
    with open(output_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        if lines:
            print("Пример первой очищенной статьи:")
            first_article = json.loads(lines[0])
            print(f"Title: {first_article['title'][:60]}...")
            print(f"Cleaned text: {first_article['cleaned_text'][:200]}...")

if __name__ == '__main__':
    main()