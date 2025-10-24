import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from nltk.tokenize import word_tokenize
from razdel import tokenize as razdel_tokenize
import spacy
from nltk.stem import SnowballStemmer
import nltk
import subprocess
import os
import warnings
from datetime import datetime

# Игнорируем предупреждения
warnings.filterwarnings("ignore")

# Check if running in Streamlit context
def is_streamlit_running():
    return hasattr(st, 'runtime') and st.runtime.exists()

# Загрузка ресурсов NLTK
def ensure_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)

# Функции токенизации
def nltk_tokenize(text, language):
    try:
        lang = 'russian' if language == 'Русский' else 'english'
        return [t for t in word_tokenize(text, language=lang) if t.strip()]
    except:
        return []

def razdel_tokenize_func(text, language):
    if language == 'Русский':
        return [token.text for token in razdel_tokenize(text) if token.text.strip()]
    return text.split()

# Функции нормализации
def snowball_stem(tokens, language):
    lang = 'russian' if language == 'Русский' else 'english'
    stemmer = SnowballStemmer(lang)
    return [stemmer.stem(token) for token in tokens]

# Вычисление метрик
def compute_metrics(tokens_list, vocab, test_ratio=0.2):
    # Разделяем данные на "известные" и "неизвестные" токены
    all_tokens = [token for tokens in tokens_list for token in tokens]
    
    if not all_tokens:
        return {'oov_percentage': 0, 'vocab_size': 0, 'token_freq': {}, 'token_lengths': []}
    
    # Создаем эталонный словарь из части данных
    split_index = int(len(all_tokens) * (1 - test_ratio))
    train_tokens = all_tokens[:split_index]
    test_tokens = all_tokens[split_index:]
    
    reference_vocab = set(train_tokens)
    
    # OOV считаем на тестовой части
    oov_tokens = sum(1 for token in test_tokens if token not in reference_vocab)
    oov_percentage = (oov_tokens / len(test_tokens)) * 100 if test_tokens else 0
    
    token_freq = Counter(all_tokens)
    top_tokens = dict(token_freq.most_common(10))
    
    return {
        'token_lengths': [len(token) for token in all_tokens],
        'oov_percentage': oov_percentage,
        'token_freq': top_tokens,
        'vocab_size': len(set(all_tokens)),
        'oov_count': oov_tokens,
        'test_tokens_count': len(test_tokens)
    }

# Чтение корпуса
def read_corpus(file_path):
    texts = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    article = json.loads(line.strip())
                    text = article.get('preprocessed_text', article.get('cleaned_text', article.get('text', '')))
                    if text:
                        texts.append(text)
                except:
                    continue
    except Exception as e:
        st.error(f"Ошибка чтения файла: {str(e)[:100]}")
        return []
    return texts

# Генерация отчёта
def generate_report(metrics, method, language):
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Подготовка данных для таблицы
    total_tokens = sum(metrics['token_freq'].values())
    token_rows = ""
    for token, freq in metrics['token_freq'].items():
        percentage = (freq / total_tokens * 100) if total_tokens > 0 else 0
        token_rows += f"""
        <tr>
            <td><strong>{token}</strong></td>
            <td>{freq}</td>
            <td>{percentage:.2f}%</td>
        </tr>
        """
    
    html_content = f"""
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Анализ текстовых данных</title>
    <style>
        :root {{
            --primary: #6366f1;
            --primary-dark: #4f46e5;
            --primary-light: #c7d2fe;
            --secondary: #10b981;
            --accent: #f59e0b;
            --danger: #ef4444;
            --warning: #fbbf24;
            --dark: #1f2937;
            --darker: #111827;
            --light: #f8fafc;
            --lighter: #f9fafb;
            --gray: #6b7280;
            --gray-light: #9ca3af;
            --border: #e5e7eb;
        }}
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
            line-height: 1.6;
            color: var(--dark);
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .dashboard-container {{
            max-width: 1400px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.98);
            backdrop-filter: blur(20px);
            border-radius: 24px;
            box-shadow: 
                0 32px 64px rgba(0, 0, 0, 0.2),
                0 0 0 1px rgba(255, 255, 255, 0.3);
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.4);
        }}
        
        /* Header Styles */
        .dashboard-header {{
            background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
            color: white;
            padding: 60px 50px;
            position: relative;
            overflow: hidden;
        }}
        
        .header-background {{
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: 
                radial-gradient(circle at 20% 80%, rgba(255,255,255,0.1) 0%, transparent 50%),
                radial-gradient(circle at 80% 20%, rgba(255,255,255,0.1) 0%, transparent 50%);
        }}
        
        .header-content {{
            position: relative;
            z-index: 2;
            text-align: center;
        }}
        
        .header-title {{
            font-size: 3.2em;
            margin-bottom: 15px;
            font-weight: 800;
            letter-spacing: -0.5px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 15px;
        }}
        
        .header-subtitle {{
            font-size: 1.4em;
            opacity: 0.9;
            font-weight: 300;
            max-width: 600px;
            margin: 0 auto;
        }}
        
        /* Main Content Grid */
        .dashboard-content {{
            padding: 50px;
            display: grid;
            grid-template-columns: 1fr 400px;
            gap: 40px;
        }}
        
        .main-content {{
            display: flex;
            flex-direction: column;
            gap: 40px;
        }}
        
        .sidebar {{
            display: flex;
            flex-direction: column;
            gap: 30px;
        }}
        
        /* Card Styles */
        .card {{
            background: white;
            border-radius: 20px;
            padding: 35px;
            box-shadow: 
                0 4px 25px rgba(0, 0, 0, 0.08),
                0 0 0 1px rgba(0, 0, 0, 0.03);
            border: 1px solid var(--border);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }}
        
        .card:hover {{
            transform: translateY(-5px);
            box-shadow: 
                0 12px 40px rgba(0, 0, 0, 0.15),
                0 0 0 1px rgba(0, 0, 0, 0.05);
        }}
        
        .card-header {{
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 25px;
            padding-bottom: 15px;
            border-bottom: 2px solid var(--primary-light);
        }}
        
        .card-icon {{
            width: 48px;
            height: 48px;
            border-radius: 12px;
            background: linear-gradient(135deg, var(--primary), var(--primary-dark));
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5em;
            color: white;
        }}
        
        .card-title {{
            font-size: 1.5em;
            font-weight: 700;
            color: var(--dark);
        }}
        
        /* Metrics Grid */
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 25px;
            margin-bottom: 10px;
        }}
        
        .metric-item {{
            text-align: center;
            padding: 25px;
            background: var(--lighter);
            border-radius: 16px;
            border: 1px solid var(--border);
            transition: all 0.3s ease;
        }}
        
        .metric-item:hover {{
            background: white;
            border-color: var(--primary-light);
            transform: scale(1.05);
        }}
        
        .metric-value {{
            font-size: 2.5em;
            font-weight: 800;
            margin: 10px 0;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        
        .metric-label {{
            color: var(--gray);
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1.2px;
            font-weight: 600;
        }}
        
        /* Table Styles */
        .table-container {{
            overflow-x: auto;
            border-radius: 16px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
            background: white;
            border: 1px solid var(--border);
        }}
        
        .styled-table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
        }}
        
        .styled-table th {{
            background: linear-gradient(135deg, var(--primary), var(--primary-dark));
            color: white;
            padding: 20px 25px;
            text-align: left;
            font-weight: 600;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .styled-table th:first-child {{
            border-top-left-radius: 16px;
        }}
        
        .styled-table th:last-child {{
            border-top-right-radius: 16px;
        }}
        
        .styled-table td {{
            padding: 18px 25px;
            border-bottom: 1px solid var(--border);
            font-weight: 500;
            transition: all 0.2s ease;
        }}
        
        .styled-table tr:nth-child(even) {{
            background: rgba(99, 102, 241, 0.02);
        }}
        
        .styled-table tr:hover {{
            background: rgba(99, 102, 241, 0.08);
        }}
        
        .styled-table tr:hover td {{
            transform: translateX(5px);
        }}
        
        /* Badge Styles */
        .badges-container {{
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            margin-top: 20px;
        }}
        
        .badge {{
            display: inline-flex;
            align-items: center;
            gap: 10px;
            padding: 12px 24px;
            border-radius: 50px;
            font-weight: 600;
            font-size: 0.9em;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
        }}
        
        .badge:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
        }}
        
        .method-badge {{
            background: linear-gradient(135deg, var(--primary), var(--primary-dark));
            color: white;
        }}
        
        .language-badge {{
            background: linear-gradient(135deg, var(--secondary), #059669);
            color: white;
        }}
        
        .stats-badge {{
            background: linear-gradient(135deg, var(--accent), #d97706);
            color: white;
        }}
        
        /* Progress Bars */
        .progress-section {{
            margin-top: 25px;
        }}
        
        .progress-item {{
            margin-bottom: 20px;
        }}
        
        .progress-label {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
            font-weight: 600;
            color: var(--dark);
        }}
        
        .progress-bar {{
            height: 8px;
            background: var(--border);
            border-radius: 4px;
            overflow: hidden;
        }}
        
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, var(--primary), var(--secondary));
            border-radius: 4px;
            transition: width 0.8s ease;
        }}
        
        /* Stats Grid */
        .stats-grid {{
            display: grid;
            grid-template-columns: 1fr;
            gap: 15px;
        }}
        
        .stat-item {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 20px;
            background: var(--lighter);
            border-radius: 12px;
            border: 1px solid var(--border);
            transition: all 0.3s ease;
        }}
        
        .stat-item:hover {{
            background: white;
            border-color: var(--primary-light);
        }}
        
        .stat-label {{
            color: var(--gray);
            font-weight: 500;
        }}
        
        .stat-value {{
            font-size: 1.3em;
            font-weight: 700;
            color: var(--dark);
        }}
        
        /* Footer */
        .dashboard-footer {{
            text-align: center;
            padding: 35px;
            background: linear-gradient(135deg, var(--darker), #0f172a);
            color: white;
            margin-top: 20px;
        }}
        
        .footer-content {{
            opacity: 0.8;
            font-weight: 300;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
        }}
        
        /* Responsive Design */
        @media (max-width: 1200px) {{
            .dashboard-content {{
                grid-template-columns: 1fr;
                gap: 30px;
            }}
            
            .sidebar {{
                order: -1;
            }}
        }}
        
        @media (max-width: 768px) {{
            .dashboard-header {{
                padding: 40px 30px;
            }}
            
            .header-title {{
                font-size: 2.2em;
            }}
            
            .dashboard-content {{
                padding: 30px;
            }}
            
            .metrics-grid {{
                grid-template-columns: repeat(2, 1fr);
            }}
            
            .card {{
                padding: 25px;
            }}
        }}
        
        @media (max-width: 480px) {{
            body {{
                padding: 10px;
            }}
            
            .dashboard-header {{
                padding: 30px 20px;
            }}
            
            .header-title {{
                font-size: 1.8em;
            }}
            
            .metrics-grid {{
                grid-template-columns: 1fr;
            }}
            
            .dashboard-content {{
                padding: 20px;
            }}
        }}
        
        /* Animations */
        @keyframes fadeInUp {{
            from {{
                opacity: 0;
                transform: translateY(30px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}
        
        .card, .metric-item {{
            animation: fadeInUp 0.6s ease-out;
        }}
        
        /* Color coding for values */
        .value-high {{
            color: var(--secondary);
        }}
        
        .value-medium {{
            color: var(--accent);
        }}
        
        .value-low {{
            color: var(--danger);
        }}
    </style>
</head>
<body>
    <div class="dashboard-container">
        <!-- Header -->
        <div class="dashboard-header">
            <div class="header-background"></div>
            <div class="header-content">
                <h1 class="header-title">
                    <span>📊</span>
                    Анализ текстовых данных
                </h1>
                <div class="header-subtitle">
                    Комплексная аналитическая панель для обработки и визуализации текстового корпуса
                </div>
            </div>
        </div>
        
        <!-- Main Content -->
        <div class="dashboard-content">
            <!-- Main Column -->
            <div class="main-content">
                <!-- Key Metrics -->
                <div class="card">
                    <div class="card-header">
                        <div class="card-icon">📈</div>
                        <h2 class="card-title">Ключевые показатели</h2>
                    </div>
                    <div class="metrics-grid">
                        <div class="metric-item">
                            <div class="metric-label">Метод обработки</div>
                            <div class="metric-value method-badge">{method.upper()}</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-label">Язык анализа</div>
                            <div class="metric-value language-badge">{language.upper()}</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-label">Размер словаря</div>
                            <div class="metric-value value-high">{metrics['vocab_size']}</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-label">Доля OOV</div>
                            <div class="metric-value value-low">{metrics['oov_percentage']:.2f}%</div>
                        </div>
                    </div>
                </div>
                
                <!-- Frequency Table -->
                <div class="card">
                    <div class="card-header">
                        <div class="card-icon">🔤</div>
                        <h2 class="card-title">Частотность токенов (Топ-15)</h2>
                    </div>
                    <div class="table-container">
                        <table class="styled-table">
                            <thead>
                                <tr>
                                    <th>#</th>
                                    <th>Токен</th>
                                    <th>Частота</th>
                                    <th>Доля</th>
                                    <th>Статус</th>
                                </tr>
                            </thead>
                            <tbody>
                                {token_rows}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
            
            <!-- Sidebar -->
            <div class="sidebar">
                <!-- Processing Info -->
                <div class="card">
                    <div class="card-header">
                        <div class="card-icon">⚙️</div>
                        <h2 class="card-title">Информация о обработке</h2>
                    </div>
                    <div class="badges-container">
                        <div class="badge method-badge">
                            <span>🔧</span>
                            {method.upper()}
                        </div>
                        <div class="badge language-badge">
                            <span>🌐</span>
                            {language.upper()}
                        </div>
                        <div class="badge stats-badge">
                            <span>📚</span>
                            Словарь: {metrics['vocab_size']}
                        </div>
                    </div>
                    
                    <div class="progress-section">
                        <div class="progress-item">
                            <div class="progress-label">
                                <span>OOV Rate</span>
                                <span>{metrics['oov_percentage']:.2f}%</span>
                            </div>
                            <div class="progress-bar">
                                <div class="progress-fill" style="width: {metrics['oov_percentage']}%"></div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Detailed Stats -->
                <div class="card">
                    <div class="card-header">
                        <div class="card-icon">📋</div>
                        <h2 class="card-title">Детальная статистика</h2>
                    </div>
                    <div class="stats-grid">
                        <div class="stat-item">
                            <span class="stat-label">OOV Rate</span>
                            <span class="stat-value value-low">{metrics['oov_percentage']:.2f}%</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Vocabulary Size</span>
                            <span class="stat-value value-high">{metrics['vocab_size']}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Processing Method</span>
                            <span class="stat-value" style="color: var(--primary);">{method.upper()}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Language</span>
                            <span class="stat-value" style="color: var(--accent);">{language.upper()}</span>
                        </div>
                    </div>
                </div>
                
                <!-- Additional Info -->
                <div class="card">
                    <div class="card-header">
                        <div class="card-icon">ℹ️</div>
                        <h2 class="card-title">Справка</h2>
                    </div>
                    <div style="color: var(--gray); line-height: 1.6;">
                        <p><strong>OOV (Out-of-Vocabulary)</strong> - токены, отсутствующие в словаре</p>
                        <p><strong>Частотность</strong> - количество вхождений токена в тексте</p>
                        <p style="margin-top: 15px; font-size: 0.9em; opacity: 0.7;">
                            Анализ выполнен с использованием современных методов NLP
                        </p>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Footer -->
        <div class="dashboard-footer">
            <div class="footer-content">
                <span>🕒</span>
                <p>Сгенерировано автоматически • {timestamp}</p>
            </div>
        </div>
    </div>
</body>
</html>
"""
    
    return html_content

# Основной интерфейс Streamlit
def main():
    # Настройка страницы
    st.set_page_config(
        page_title="Text Processing Pro",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Кастомные стили CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 800;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        border-left: 5px solid #667eea;
        margin: 0.5rem 0;
    }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    
    .section-header {
        font-size: 1.5rem;
        color: #333;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #667eea;
    }
    
    .upload-box {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: #f8f9fa;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    if not is_streamlit_running():
        st.error("Пожалуйста, запустите приложение с помощью команды: `streamlit run text_processing_app.py`")
        return
    
    # Заголовок с градиентом
    st.markdown('<h1 class="main-header">📊 Text Processing Pro</h1>', unsafe_allow_html=True)
    
    # Боковая панель с настройками
    with st.sidebar:
        st.header("⚙️ Настройки обработки")
        
        # Выбор языка
        language = st.selectbox("🌐 Выберите язык", ["Русский", "Английский"])
        
        # Выбор метода обработки
        method = st.selectbox(
            "🔧 Метод обработки",
            ['nltk', 'razdel', 'nltk_snowball'],
            format_func=lambda x: {
                'nltk': 'NLTK Tokenizer',
                'razdel': 'Razdel Tokenizer', 
                'nltk_snowball': 'NLTK + Snowball Stemmer'
            }[x]
        )
        
        min_token_length = 2
        remove_stopwords = True
        lowercase = True
        
        # Информация о методах
        with st.expander("ℹ️ О методах обработки"):
            st.info("""
            **NLTK** - классическая токенизация с поддержкой пунктуации\n
            **Razdel** - эффективный токенизатор для русского языка\n
            **Snowball** - стемминг с приведением слов к основе
            """)
    
    # Основная область контента
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📁 Загрузка данных")
        
        # Карточка загрузки файла
        with st.container():
            use_default = st.checkbox("Использовать демо-датасет", value=True, 
                                    help="Предзагруженный корпус новостных текстов")
            
            if not use_default:
                uploaded_file = st.file_uploader(
                    "Загрузите JSONL файл", 
                    type=["jsonl", "json", "txt"],
                    help="Поддерживаются файлы в формате JSONL, JSON или текстовые файлы"
                )
                
                if uploaded_file:
                    with st.spinner("Сохраняем файл..."):
                        with open("uploaded_corpus.jsonl", "wb") as f:
                            f.write(uploaded_file.getbuffer())
                    st.success("Файл успешно загружен!")
            
            file_path = 'preprocessed_corpus.jsonl' if use_default else "uploaded_corpus.jsonl" if uploaded_file else None
    
    with col2:
        # Статистика и информация
        if file_path and os.path.exists(file_path):
            texts = read_corpus(file_path)
            if texts:
                st.metric("📊 Загружено текстов", len(texts))
                avg_length = sum(len(text.split()) for text in texts) / len(texts)
                st.metric("📝 Средняя длина", f"{avg_length:.1f} слов")
            else:
                st.warning("Файл пуст или поврежден")
        else:
            st.info("👆 Загрузите данные для начала анализа")
    
    # Обработка данных
    if file_path and os.path.exists(file_path):
        texts = read_corpus(file_path)
        if not texts:
            st.error("❌ Не удалось загрузить данные из файла!")
            return
        
        # Кнопка обработки с визуальным акцентом
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            process_btn = st.button(
                "🚀 Начать обработку", 
                type="primary", 
                use_container_width=True,
                help="Запуск анализа текстового корпуса"
            )
        
        if process_btn:
            # Прогресс-бар и статус
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Обработка корпуса
            tokens_list = []
            vocab = set()
            
            for i, text in enumerate(texts):
                status_text.text(f"Обработка текста {i+1}/{len(texts)}...")
                
                if method == 'nltk':
                    tokens = nltk_tokenize(text, language)
                elif method == 'razdel':
                    tokens = razdel_tokenize_func(text, language)
                elif method == 'nltk_snowball':
                    tokens = snowball_stem(nltk_tokenize(text, language), language)
                else:
                    tokens = []
                
                # Применение дополнительных фильтров
                if lowercase:
                    tokens = [token.lower() for token in tokens]
                if remove_stopwords:
                    tokens = [token for token in tokens if token not in get_stopwords(language)]
                tokens = [token for token in tokens if len(token) >= min_token_length]
                
                if tokens:
                    tokens_list.append(tokens)
                    vocab.update(tokens)
                
                progress_bar.progress((i + 1) / len(texts))
            
            if not tokens_list:
                st.error("❌ Ошибка обработки: токены не получены!")
                return
            
            # Вычисление метрик
            with st.spinner("📈 Вычисляем метрики..."):
                metrics = compute_metrics(tokens_list, vocab)
            
            st.success("✅ Обработка завершена!")
            
            # Визуализация результатов
            st.markdown('<div class="section-header">📊 Результаты анализа</div>', unsafe_allow_html=True)
            
            # Ключевые метрики в карточках
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>📚 Словарь</h3>
                    <h2>{metrics['vocab_size']}</h2>
                    <p>уникальных токенов</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>⚠️ OOV</h3>
                    <h2>{metrics['oov_percentage']:.2f}%</h2>
                    <p>вне словаря</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                avg_len = sum(metrics['token_lengths']) / len(metrics['token_lengths'])
                st.markdown(f"""
                <div class="metric-card">
                    <h3>📏 Длина</h3>
                    <h2>{avg_len:.1f}</h2>
                    <p>средняя длина токена</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                total_tokens = sum(len(tokens) for tokens in tokens_list)
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🔤 Токены</h3>
                    <h2>{total_tokens}</h2>
                    <p>всего токенов</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Визуализация в табах
            tab1, tab2, tab3, tab4 = st.tabs(["📈 Распределения", "🔤 Частотность", "📋 Детали", "💾 Экспорт"])
            
            with tab1:
                # Распределение длин токенов
                fig1 = px.histogram(metrics['token_lengths'], nbins=50, 
                                  title="Распределение длин токенов",
                                  labels={'value': 'Длина токена', 'count': 'Количество'},
                                  color_discrete_sequence=['#667eea'])
                fig1.update_layout(showlegend=False)
                st.plotly_chart(fig1, use_container_width=True)
            
            with tab2:
                # Частотность токенов
                token_freq_df = pd.DataFrame(list(metrics['token_freq'].items())[:15], 
                                           columns=['Токен', 'Частота'])
                fig2 = px.bar(token_freq_df, x='Токен', y='Частота', 
                            title="Топ-15 самых частых токенов",
                            color='Частота',
                            color_continuous_scale='Viridis')
                st.plotly_chart(fig2, use_container_width=True)
            
            with tab3:
                # Детальная таблица токенов
                st.subheader("📊 Детальная статистика токенов")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Топ токены
                    st.dataframe(
                        token_freq_df,
                        use_container_width=True,
                        hide_index=True
                    )
                
                with col2:
                    # Круговая диаграмма OOV
                    fig3 = px.pie(
                        values=[metrics['oov_percentage'], 100 - metrics['oov_percentage']],
                        names=['OOV токены', 'В словаре'],
                        title="Распределение OOV",
                        color_discrete_sequence=['#ef4444', '#10b981']
                    )
                    st.plotly_chart(fig3, use_container_width=True)
            
            with tab4:
                st.subheader("📤 Экспорт результатов")
                
                # Генерация отчетов
                report_html = generate_report(metrics, method, language)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.download_button(
                        "📊 Скачать HTML отчёт",
                        report_html,
                        file_name="text_analysis_report.html",
                        mime="text/html",
                        use_container_width=True,
                        help="Полный отчет с графиками в HTML формате"
                    )
                
                with col2:
                    # JSON экспорт
                    json_report = json.dumps({
                        'metrics': metrics,
                        'method': method,
                        'language': language,
                        'timestamp': str(datetime.now())
                    }, ensure_ascii=False, indent=2)
                    
                    st.download_button(
                        "📄 Скачать JSON данные",
                        json_report,
                        file_name="analysis_data.json",
                        mime="application/json",
                        use_container_width=True,
                        help="Скачать все данные анализа в JSON формате"
                    )
                
                with col3:
                    # CSV экспорт токенов
                    csv_data = token_freq_df.to_csv(index=False)
                    st.download_button(
                        "📋 Скачать CSV таблицу",
                        csv_data,
                        file_name="token_frequency.csv",
                        mime="text/csv",
                        use_container_width=True,
                        help="Таблица частотности токенов в CSV формате"
                    )
                
                # Предпросмотр отчета
                with st.expander("👁️ Предпросмотр HTML отчёта", expanded=False):
                    st.components.v1.html(report_html, height=600, scrolling=True)

def get_stopwords(language):
    """Получение стоп-слов для указанного языка"""
    # Заглушка - нужно реализовать получение стоп-слов
    stopwords = {
        'Русский': ['и', 'в', 'не', 'на', 'я', 'быть', 'с', 'что', 'а', 'по'],
        'Английский': ['the', 'a', 'an', 'in', 'on', 'at', 'for', 'to', 'of']
    }
    return stopwords.get(language, [])

if __name__ == '__main__':
    main()