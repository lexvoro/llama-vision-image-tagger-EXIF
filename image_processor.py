import ollama
from ollama import AsyncClient
from pathlib import Path
import logging
from typing import Dict, List, Optional
import json
from pydantic import BaseModel
from PIL import Image
import os
import asyncio

logger = logging.getLogger(__name__)

# Схемы данных для валидации JSON ответов от ИИ
class ImageDescription(BaseModel):
    description: str

class ImageTags(BaseModel):
    tags: List[str]

class ImageText(BaseModel):
    has_text: bool
    text_content: str

class ImageProcessor:
    def __init__(self, model_name: str = 'llama3.2-vision'):
        self.model_name = model_name
        self.temp_path = Path("temp_processing.jpg")
        self.client = AsyncClient()

    def _clean_text(self, text: str) -> str:
        """Очистка текста от спецсимволов, которые могут ломать JSON."""
        if not text:
            return ""
        return text.replace('\xad', '').strip()

    async def process_image(self, image_path: Path, tag_count: int = 10, languages: List[str] = ["en"]) -> Dict:
        """Основной цикл обработки: Описание -> Теги EN -> Перевод на RU -> Текст."""
        try:
            if not image_path.exists():
                return {"is_processed": False, "error": "File not found"}

            # Оптимизация изображения для GPU (RTX 5060 Ti)
            with Image.open(image_path) as img:
                img.thumbnail((1024, 1024))
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")
                img.save(self.temp_path, "JPEG", quality=85)

            image_path_str = str(self.temp_path.absolute())

            # 1. Получаем описание (всегда на англ для точности)
            description_res = await self._get_description(image_path_str)
            
            # 2. Получаем английские теги (базовая логика)
            tags_res = await self._get_tags(image_path_str, tag_count)
            en_tags = [self._clean_text(t).lower() for t in tags_res.tags]

            # 3. Если выбран русский — переводим список тегов
            ru_tags = []
            if "ru" in languages and en_tags:
                logger.info(f"Перевод тегов на русский для {image_path.name}...")
                ru_tags = await self._translate_tags(en_tags)

            # 4. Ищем текст на изображении
            text_res = await self._get_text_content(image_path_str)

            if self.temp_path.exists():
                os.remove(self.temp_path)

            return {
                "description": self._clean_text(description_res.description),
                "tags": en_tags,
                "tags_ru": ru_tags,
                "text_content": self._clean_text(text_res.text_content) if text_res.has_text else "",
                "is_processed": True
            }

        except Exception as e:
            logger.error(f"❌ Ошибка процессора: {str(e)}")
            if self.temp_path.exists():
                os.remove(self.temp_path)
            return {
                "description": "", "tags": [], "tags_ru": [], "text_content": "",
                "is_processed": False, "error": str(e)
            }

    async def _get_description(self, image_path: str) -> ImageDescription:
        prompt = "Describe this image in one short sentence in English."
        response = await self._query_ollama(prompt, image_path, ImageDescription.model_json_schema())
        return ImageDescription.model_validate_json(response)

    async def _get_tags(self, image_path: str, tag_count: int) -> ImageTags:
        prompt = f"List exactly {tag_count} unique one-word tags in English for this image. No repetitions. Output as JSON list."
        response = await self._query_ollama(prompt, image_path, ImageTags.model_json_schema())
        return ImageTags.model_validate_json(response)

    async def _translate_tags(self, tags: List[str]) -> List[str]:
        """Улучшенный перевод с жестким требованием кириллицы."""
        if not tags:
            return []
            
        tags_str = ", ".join(tags)
        # Добавляем явное требование использовать КИРИЛЛИЦУ
        prompt = (
            f"Translate the following English tags into Russian: {tags_str}. "
            f"CRITICAL: Use only Cyrillic characters (Russian alphabet). "
            f"Output ONLY a JSON list of strings."
        )
        
        try:
            response = await self.client.chat(
                model=self.model_name,
                messages=[
                    {'role': 'system', 'content': 'You are a translator. You always translate English to Russian. Your output is always a JSON list of Russian words in Cyrillic.'},
                    {'role': 'user', 'content': prompt}
                ],
                options={
                    'temperature': 0.1, # Минимальная температура для точности
                },
                format=ImageTags.model_json_schema()
            )
            
            content = response['message']['content']
            data = ImageTags.model_validate_json(content)
            
            # Дополнительная проверка: если модель вернула латиницу, помечаем для логов
            translated = [self._clean_text(t).lower() for t in data.tags]
            return translated
            
        except Exception as e:
            logger.warning(f"Translation failed: {e}")
            return []


    async def _get_text_content(self, image_path: str) -> ImageText:
        prompt = "Is there any text in this image? If yes, transcribe it to JSON."
        response = await self._query_ollama(prompt, image_path, ImageText.model_json_schema())
        return ImageText.model_validate_json(response)

    async def _query_ollama(self, prompt: str, image_path: str, format_schema: dict) -> str:
        """Запрос к Ollama с повторами при сбоях."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = await self.client.chat(
                    model=self.model_name,
                    messages=[
                        {'role': 'system', 'content': 'You are a helpful assistant that outputs only JSON.'},
                        {'role': 'user', 'content': prompt, 'images': [image_path]}
                    ],
                    options={
                        'temperature': 0.3,
                        'num_gpu': -1,
                        'repeat_penalty': 1.2,
                        'num_predict': 1024  # Запас по длине ответа
                    },
                    format=format_schema
                )
                return response['message']['content']
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                await asyncio.sleep(1)

def update_image_metadata(folder_path: Path, image_path: str, metadata: Dict) -> None:
    """Запись текущего прогресса в JSON-файл."""
    metadata_file = folder_path / "image_metadata.json"
    try:
        all_metadata = {}
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                try:
                    all_metadata = json.load(f)
                except:
                    all_metadata = {}
        
        all_metadata[image_path] = metadata
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(all_metadata, f, indent=4, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error saving JSON metadata: {str(e)}")
