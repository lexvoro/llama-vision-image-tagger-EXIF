import ollama
from pathlib import Path
import logging
from typing import Dict, List, Optional
import json
from pydantic import BaseModel
from PIL import Image
import os

logger = logging.getLogger(__name__)

# Схемы данных
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

    async def process_image(self, image_path: Path) -> Dict:
        """Обработка изображения с защитой от вылетов."""
        try:
            if not image_path.exists():
                return {"is_processed": False, "error": "File not found"}

            # Уменьшаем фото до 1024px для стабильности GPU
            with Image.open(image_path) as img:
                img.thumbnail((1024, 1024))
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")
                img.save(self.temp_path, "JPEG", quality=85)

            image_path_str = str(self.temp_path.absolute())

            # Запросы к нейросети
            logger.info(f"Analyzing image: {image_path.name}")
            description_res = await self._get_description(image_path_str)
            tags_res = await self._get_tags(image_path_str)
            text_res = await self._get_text_content(image_path_str)

            if self.temp_path.exists():
                os.remove(self.temp_path)

            return {
                "description": description_res.description,
                "tags": tags_res.tags,
                "text_content": text_res.text_content if text_res.has_text else "",
                "is_processed": True
            }

        except Exception as e:
            # Тихая обработка ошибки: не прерывает работу всей программы
            logger.warning(f"Skipping {image_path.name}: {str(e)}")
            if self.temp_path.exists():
                os.remove(self.temp_path)
            return {
                "description": "", "tags": [], "text_content": "",
                "is_processed": False, "error": str(e)
            }

    async def _get_description(self, image_path: str) -> ImageDescription:
        prompt = "Describe this image in one short sentence."
        response = await self._query_ollama(prompt, image_path, ImageDescription.model_json_schema())
        return ImageDescription.model_validate_json(response)

    async def _get_tags(self, image_path: str) -> ImageTags:
        # Ограничиваем количество, чтобы модель не зацикливалась
        prompt = "List 5-10 unique one-word tags. No repetitions."
        response = await self._query_ollama(prompt, image_path, ImageTags.model_json_schema())
        return ImageTags.model_validate_json(response)

    async def _get_text_content(self, image_path: str) -> ImageText:
        prompt = "Identify if there is text. Respond in JSON."
        response = await self._query_ollama(prompt, image_path, ImageText.model_json_schema())
        return ImageText.model_validate_json(response)

    async def _query_ollama(self, prompt: str, image_path: str, format_schema: dict) -> str:
        """Запрос к Ollama с 3 попытками в случае сбоя."""
        max_retries = 3
        last_error = None

        for attempt in range(max_retries):
            try:
                logger.info(f"Запрос к ИИ (попытка {attempt + 1}/{max_retries})...")
                
                response = ollama.chat(
                    model=self.model_name,
                    messages=[
                        {'role': 'system', 'content': 'Ты русский ассистент. Отвечай строго в формате JSON.'},
                        {'role': 'user', 'content': prompt, 'images': [image_path]}
                    ],
                    options={
                        'temperature': 0.1,    # Низкая температура для стабильности
                        'num_predict': 512,
                        'num_ctx': 4096,
                        'num_gpu': -1,         # Используем всю мощь RTX 5060 Ti
                        'repeat_penalty': 2.0  # Защита от повторов
                    },
                    format=format_schema
                )
                
                # Если ответ получен, выходим из цикла и возвращаем результат
                return response['message']['content']

            except Exception as e:
                last_error = e
                logger.warning(f"Попытка {attempt + 1} не удалась: {str(e)}")
                # Можно добавить микро-паузу перед следующей попыткой
                import asyncio
                await asyncio.sleep(1)

        # Если после 3 попыток ничего не вышло, выбрасываем ошибку
        logger.error(f"Все {max_retries} попытки завершились ошибкой.")
        raise last_error

def update_image_metadata(folder_path: Path, image_path: str, metadata: Dict) -> None:
    """Запись результатов в JSON-файл."""
    metadata_file = folder_path / "image_metadata.json"
    try:
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                all_metadata = json.load(f)
        else:
            all_metadata = {}
        
        all_metadata[image_path] = metadata
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(all_metadata, f, indent=4, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error saving metadata: {str(e)}")
