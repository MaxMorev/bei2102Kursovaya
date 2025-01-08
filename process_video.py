import whisper
import os
import subprocess
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
import re
from nltk.corpus import stopwords
import nltk
from natasha import NamesExtractor
import json
import requests


russian_stopwords = stopwords.words("russian")

# Функция извлечения аудио из видео
def extract_audio(video_path, output_audio_path):
    """
    Извлекает аудио из видеофайла с помощью команды ffmpeg.
    
    Аргументы:
    - video_path (str): Путь к видеофайлу.
    - output_audio_path (str): Путь для сохранения извлеченного аудио.
    
    Действие:
    - Использует ffmpeg для извлечения аудиодорожки и сохранения её в указанном формате.
    """
    command = [
        "ffmpeg", "-i", video_path, "-q:a", "0", "-map", "a", output_audio_path, "-y"
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

# Транскрипция аудио с помощью Whisper
def transcribe_audio(audio_path, model_name="base"):
    """
    Выполняет транскрипцию аудиофайла с помощью модели Whisper.
    
    Аргументы:
    - audio_path (str): Путь к аудиофайлу.
    - model_name (str): Название используемой модели Whisper (по умолчанию "large-v3-turbo").
    
    Возвращает:
    - str: Текстовая транскрипция аудио.
    """
    model = whisper.load_model(model_name)
    result = model.transcribe(audio_path)
    return result["text"]

# Очистка текста от шума
def clean_text(text):
    """
    Очищает текст от лишних символов, коротких слов и пробелов.
    
    Аргументы:
    - text (str): Исходный текст.
    
    Возвращает:
    - str: Очищенный текст.
    """
    text = re.sub(r"[^а-яА-ЯёЁa-zA-Z0-9\s.,!?]", "", text)  # Удаление спецсимволов
    text = re.sub(r"\b(?:\w{1,2})\b", "", text)  # Удаление коротких слов
    text = re.sub(r"\s+", " ", text)  # Удаление лишних пробелов
    return text.strip()

# Извлечение именованных сущностей
def extract_named_entities(text):
    """
    Извлекает именованные сущности (например, имена) из текста с использованием библиотеки Natasha.
    
    Аргументы:
    - text (str): Исходный текст.
    
    Возвращает:
    - list: Список сущностей в формате словарей.
    """
    extractor = NamesExtractor()
    matches = extractor(text)
    entities = [_.fact.as_dict for _ in matches]
    return entities

# Определение тем текста с использованием Eden AI
def detect_topics_with_eden_ai(full_text, api_key):
    """
    Использует API Eden AI для анализа текста и определения ключевых тем.
    
    Аргументы:
    - full_text (str): Текст для анализа.
    - api_key (str): API-ключ для доступа к Eden AI.
    
    Возвращает:
    - list: Список определенных тем с указанием их значимости.
    """
    url = "https://api.edenai.run/v2/text/topic_extraction"  # URL API для извлечения тем
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "providers": "openai",
        "language": "ru",
        "text": full_text
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        result = json.loads(response.text)
        
        # Извлечение тем из ответа API
        topics = []
        if result.get('openai/gpt-4o') and result['openai/gpt-4o'].get('items'):
            items = result['openai/gpt-4o']['items']
            for item in items:
                category = item.get('category')
                importance = item.get('importance')
                if category and importance:
                    topics.append(f"{category}: {importance}")
        
        return topics

    except requests.exceptions.RequestException as e:
        return f"Ошибка при обращении к API Eden AI: {e}"
    except Exception as e:
        return f"Ошибка обработки данных Eden AI: {e}"

# Основная функция обработки видео
def main(video_path):
    """
    Основная функция для обработки видеофайла: извлечение аудио, транскрипция и определение тем.
    
    Аргументы:
    - video_path (str): Путь к видеофайлу.
    
    Возвращает:
    - tuple: Транскрипция текста и список тем.
    """
    audio_path = "temp_audio.mp3"
    try:
        # Извлекаем аудио из видео
        extract_audio(video_path, audio_path)
        
        # Транскрибируем аудио в текст
        transcription_text = transcribe_audio(audio_path)
        
        # Определяем ключевые темы текста
        topics = detect_topics_with_eden_ai(
            transcription_text,
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiZWYwN2QwYmMtOTYzNy00ZWFhLWEyODktNWVlYzY1YzNjN2RiIiwidHlwZSI6ImFwaV90b2tlbiJ9.dyHwlPVn0d0Wkg5pdB2Em08xyOKfessOTJWN-nH-6po"
        )
        return transcription_text, topics
    finally:
        # Удаляем временный аудиофайл после обработки
        if os.path.exists(audio_path):
            os.remove(audio_path)
