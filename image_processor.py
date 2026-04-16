import ollama
from ollama import AsyncClient
from pathlib import Path
import logging
from typing import Dict, List, Optional
import json
from pydantic import BaseModel
from PIL import Image
Image.MAX_IMAGE_PIXELS = 150000000
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
        """Основной цикл обработки с ускоренным созданием превью."""
        try:
            await asyncio.sleep(0.4)
            if not image_path.exists():
                return {"is_processed": False, "error": "File not found"}

            # --- УСКОРЕННЫЙ РЕСАЙЗ ---
            with Image.open(image_path) as img:
                # 1. Draft позволяет декодировать JPEG сразу в нужном размере (в разы быстрее)
                if image_path.suffix.lower() in ['.jpg', '.jpeg']:
                    img.draft(None, (1024, 1024))
                
                # 2. Используем BILINEAR вместо LANCZOS — он быстрее, а нейронке качества хватит
                img.thumbnail((1024, 1024), resample=Image.BILINEAR)
                
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")
                
                # 3. Отключаем лишние оптимизации при сохранении временного файла
                img.save(self.temp_path, "JPEG", quality=75, optimize=False)
            # -------------------------

            image_path_str = str(self.temp_path.absolute())

            # 1. Получаем описание
            description_res = await self._get_description(image_path_str)
            
            # 2. Получаем английские теги
            tags_res = await self._get_tags(image_path_str, tag_count)
            en_tags = [self._clean_text(t).lower() for t in tags_res.tags]

            # 3. Перевод на русский
            ru_tags = []
            if "ru" in languages and en_tags:
                ru_tags = await self._translate_tags(en_tags)

            # 4. Ищем текст
            text_res = await self._get_text_content(image_path_str)

            if self.temp_path.exists():
                os.remove(self.temp_path)

            import gc
            gc.collect()

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
        max_retries = 2
        # Устанавливаем лимит ожидания в секундах
        TIMEOUT_SECONDS = 120 

        for attempt in range(max_retries):
            try:
                # Оборачиваем запрос в wait_for
                response = await asyncio.wait_for(
                    self.client.chat(
                        model=self.model_name,
                        messages=[
                            {'role': 'system', 'content': 'You are a helpful assistant that outputs only JSON.'},
                            {'role': 'user', 'content': prompt, 'images': [image_path]}
                        ],
                        options={
                            'temperature': 0.3,
                            'num_gpu': -1,
                            'num_predict': 4096, # Ограничиваем длину ответа (меньше слов = быстрее)
                            'repeat_penalty': 1.2,
                        },
                        keep_alive='10m',
                        format=format_schema
                    ),
                    timeout=TIMEOUT_SECONDS
                )
                return response['message']['content']
            
            except asyncio.TimeoutError:
                logger.warning(f"⏰ Тайм-аут запроса (попытка {attempt + 1})")
                if attempt == max_retries - 1:
                    raise Exception("Нейросеть отвечала слишком долго и была прервана.")
            
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
