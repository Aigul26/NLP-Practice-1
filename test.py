from transformers import AutoTokenizer

def test_models():
    models = [
        "SherifAnar/russian-bpe-16k",
        "SherifAnar/russian-unigram-20k"
    ]

    for model_name in models:
        print(f"\n🔍 Тестируем {model_name}:")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            text = "Привет, как дела?"
            tokens = tokenizer.tokenize(text)
            print(f"✅ Работает: '{text}' → {tokens}")
        except Exception as e:
            print(f"❌ Ошибка: {e}")

if __name__ == "__main__":
    test_models()