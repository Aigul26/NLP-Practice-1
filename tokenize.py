import json
import time
import re
import csv
from collections import Counter
from nltk.tokenize import word_tokenize
import nltk
from razdel import tokenize as razdel_tokenize
import spacy
from nltk.stem import PorterStemmer, SnowballStemmer
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import subprocess

# Попытка установки модели spaCy
def ensure_spacy_model():
    try:
        spacy_nlp = spacy.load('ru_core_news_sm', disable=['parser', 'ner'])
        print("Модель ru_core_news_sm загружена")
        return spacy_nlp
    except OSError:
        print("Модель ru_core_news_sm не найдена, пытаемся установить...")
        try:
            subprocess.run(["python", "-m", "spacy", "download", "ru_core_news_sm"], check=True)
            spacy_nlp = spacy.load('ru_core_news_sm', disable=['parser', 'ner'])
            print("Модель ru_core_news_sm успешно установлена")
            return spacy_nlp
        except Exception as e:
            print(f"Не удалось установить ru_core_news_sm: {str(e)[:100]}")
            return None

# Попытка инициализации pymorphy2
def ensure_pymorphy():
    try:
        from pymorphy2 import MorphAnalyzer
        morph = MorphAnalyzer()
        print("pymorphy2 успешно инициализирован")
        return morph
    except Exception as e:
        print(f"Не удалось инициализировать pymorphy2: {str(e)[:100]}")
        return None

# Загрузка ресурсов NLTK
def ensure_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        print("Загружаем необходимые ресурсы NLTK...")
        nltk.download('punkt_tab')

ensure_nltk_resources()

# Инициализация инструментов
spacy_nlp = ensure_spacy_model()
morph = ensure_pymorphy()
snowball = SnowballStemmer('russian')
porter = PorterStemmer()
sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def naive_tokenize(text):
    return [t for t in text.split() if t.strip()]

def regex_tokenize(text):
    return [t for t in re.findall(r'\b\w+\b', text) if t.strip()]

def nltk_tokenize(text):
    try:
        return [t for t in word_tokenize(text, language='russian') if t.strip()]
    except:
        return []

def spacy_tokenize(text):
    if spacy_nlp is None:
        return []
    doc = spacy_nlp(text)
    return [token.text for token in doc if token.text.strip()]

def razdel_tokenize_text(text):
    return [t.text for t in razdel_tokenize(text) if t.text.strip()]

def porter_stem(tokens):
    return [porter.stem(token) for token in tokens]

#Стемминг с использованием SnowballStemmer
def snowball_stem(tokens):
    return [snowball.stem(token) for token in tokens]

def pymorphy_lemmatize(tokens):
    if morph is None:
        return tokens
    return [morph.parse(token)[0].normal_form for token in tokens]

def spacy_lemmatize(tokens):
    if spacy_nlp is None:
        return tokens
    doc = spacy_nlp(' '.join(tokens))
    return [token.lemma_ for token in doc]

def compute_oov(tokens_list, vocab):
    total_tokens = sum(len(tokens) for tokens in tokens_list)
    oov_tokens = sum(1 for tokens in tokens_list for token in tokens if token not in vocab)
    return oov_tokens / total_tokens * 100 if total_tokens > 0 else 0

#Вычисление косинусного сходства эмбеддингов
def compute_cosine_similarity(original_text, processed_tokens):
    processed_text = ' '.join(processed_tokens)
    if not processed_text or not original_text:
        return 0.0
    embeddings = sentence_model.encode([original_text, processed_text], convert_to_tensor=True)
    return util.cos_sim(embeddings[0], embeddings[1]).item()

#Чтение корпуса и извлечение текстов"""
def process_corpus(input_file='preprocessed_corpus.jsonl'):
    texts = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                article = json.loads(line.strip())
                text = article.get('preprocessed_text', article.get('cleaned_text', article.get('text', '')))
                if text:
                    texts.append(text)
            except:
                continue
    return texts

def run_experiment(texts, num_articles=123):
    methods = [
        ('naive', naive_tokenize, None),
        ('regex', regex_tokenize, None),
        ('nltk', nltk_tokenize, None),
        ('razdel', razdel_tokenize_text, None),
        ('nltk_porter', nltk_tokenize, porter_stem),
        ('nltk_snowball', nltk_tokenize, snowball_stem)
    ]
    if spacy_nlp:
        methods.extend([
            ('spacy', spacy_tokenize, None),
            ('spacy_lem', spacy_tokenize, spacy_lemmatize)
        ])
    if morph:
        methods.append(('nltk_pymorphy', nltk_tokenize, pymorphy_lemmatize))
    else:
        print("Пропущен метод nltk_pymorphy из-за проблем с pymorphy2")

    results = []
    vocab = set()

    for method_name, tokenize_fn, normalize_fn in methods:
        print(f"Обработка методом: {method_name}")
        start_time = time.time()
        tokens_list = []
        total_tokens = 0
        similarities = []

        for text in texts:
            tokens = tokenize_fn(text)
            if normalize_fn:
                tokens = normalize_fn(tokens)
            tokens_list.append(tokens)
            total_tokens += len(tokens)

            if len(tokens_list) <= 10:
                sim = compute_cosine_similarity(text, tokens)
                similarities.append(sim)

        vocab_size = len(set(token for tokens in tokens_list for token in tokens))
        vocab.update(token for tokens in tokens_list for token in tokens)

        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
        processing_time = time.time() - start_time
        time_per_1000 = (processing_time / num_articles) * 1000

        results.append({
            'method': method_name,
            'vocab_size': vocab_size,
            'total_tokens': total_tokens,
            'avg_similarity': avg_similarity,
            'time_per_1000_articles': time_per_1000
        })

    for result in results:
        method_name = result['method']
        tokens_list = [tokenize_fn(text) for text in texts]
        if method_name in ['nltk_porter', 'nltk_snowball', 'nltk_pymorphy', 'spacy_lem']:
            normalize_fn = next((m[2] for m in methods if m[0] == method_name), None)
            if normalize_fn:
                tokens_list = [normalize_fn(tokens) for tokens in tokens_list]
        result['oov_percentage'] = compute_oov(tokens_list, vocab)

    return results