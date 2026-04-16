from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import json
import logging
import piexif
import io
from fastapi import Response
from pathlib import Path
from typing import List, Dict, Optional, Any
from PIL import Image

from image_processor import ImageProcessor, update_image_metadata

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FolderRequest(BaseModel):
    folder_path: str
    recursive: bool = False

class ProcessImageRequest(BaseModel):
    image_path: str
    tag_count: int = 10
    languages: List[str] = ["en"]


def write_metadata_to_file(file_path: Path, description: str, tags: List[str], tags_ru: List[str] = None):
    if file_path.suffix.lower() not in ['.jpg', '.jpeg']:
        return
    try:
        img = Image.open(file_path)
        exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
        if "exif" in img.info:
            try:
                exif_dict = piexif.load(img.info["exif"])
            except:
                pass

        exif_dict["0th"][piexif.ImageIFD.ImageDescription] = description.encode('utf-8')
        all_tags = tags + (tags_ru or [])
        exif_dict["0th"][0x9c9e] = ";".join(all_tags).encode('utf-16le')

        exif_bytes = piexif.dump(exif_dict)
        img.save(file_path, exif=exif_bytes, quality="keep")
        logger.info(f"💾 EXIF updated: {file_path.name}")
    except Exception as e:
        logger.error(f"❌ Failed to write EXIF: {e}")



def load_simple_metadata(folder_path: Path, recursive: bool = False) -> Dict[str, Dict]:
    metadata_file = folder_path / "image_metadata.json"
    supported_ext = {'.jpg', '.jpeg', '.png', '.webp'}

    # 1. Загружаем JSON
    metadata = {}
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        except Exception as e:
            logger.error(f"Error loading JSON: {e}")

    # 2. Поиск файлов (Универсальный способ)
    current_files = []
    
    # Если recursive=True, используем rglob('*'), если False — glob('*')
    files_iterator = folder_path.rglob('*') if recursive else folder_path.glob('*')
    
    for p in files_iterator:
        # Проверяем, что это файл и у него нужное расширение
        if p.is_file() and p.suffix.lower() in supported_ext:
            try:
                # Получаем путь относительно корня выбранной папки
                rel_path = str(p.relative_to(folder_path)).replace("\\", "/")
                current_files.append(rel_path)
            except Exception as e:
                continue

    logger.info(f"🔎 Найдено файлов: {len(current_files)} (Рекурсия: {recursive})")

    # 3. Синхронизируем с метаданными
    new_metadata = {}
    for filename in current_files:
        if filename in metadata:
            new_metadata[filename] = metadata[filename]
        else:
            new_metadata[filename] = {
                "description": "",
                "tags": [],
                "tags_ru": [],
                "is_processed": False
            }

    return new_metadata

@app.post("/images")
async def get_images(request: FolderRequest):
    # 1. Печатаем в консоль, что пришло с фронтенда
    print(f"\n--- НОВЫЙ ЗАПРОС ---")
    print(f"Путь из браузера: {request.folder_path}")
    print(f"Галочка рекурсии: {request.recursive}")

    folder_path = Path(request.folder_path)
    
    if not folder_path.exists():
        print(f"❌ ОШИБКА: Папка не найдена по пути {folder_path.absolute()}")
        raise HTTPException(status_code=404, detail=f"Folder not found: {request.folder_path}")

    app.current_folder = str(folder_path)
    
    # 2. Вызываем поиск
    metadata = load_simple_metadata(folder_path, recursive=request.recursive)
    print(f"✅ Итого найдено картинок: {len(metadata)}")

    images = []
    for path, data in metadata.items():
        images.append({
            "name": path.split('/')[-1], 
            "path": path,
            "description": data.get("description", ""),
            "tags": data.get("tags", []),
            "tags_ru": data.get("tags_ru", []),
            "is_processed": data.get("is_processed", False)
        })
    return {"images": images}


@app.post("/process-image")
async def process_image_endpoint(request: ProcessImageRequest):
    folder_path = Path(app.current_folder)

    # 1. Читаем данные ТОЛЬКО из JSON, не сканируя всю папку
    metadata_file = folder_path / "image_metadata.json"
    current_data = {}
    
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                all_meta = json.load(f)
                # Берем данные конкретной картинки
                current_data = all_meta.get(request.image_path, {})
        except Exception as e:
            logger.error(f"Error reading JSON in process: {e}")

    # 2. Теперь переменная current_data существует, и ошибки не будет
    old_tags = current_data.get("tags", [])
    old_tags_ru = current_data.get("tags_ru", [])
    old_description = current_data.get("description", "").strip()

    img_processor = ImageProcessor()
    try:
        # Запускаем обработку (теперь с поддержкой перевода)
        result = await img_processor.process_image(
            folder_path / request.image_path,
            tag_count=request.tag_count,
            languages=request.languages
        )
    except Exception as e:
        logger.error(f"Processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    if result["is_processed"]:
        # 1. Объединяем теги (старые + новые уникальные)
        ai_tags = result.get("tags", [])
        ai_tags_ru = result.get("tags_ru", [])
        result["tags"] = list(set(old_tags + ai_tags))
        result["tags_ru"] = list(set(old_tags_ru + ai_tags_ru))

        # 2. ЛОГИКА ОПИСАНИЯ: 
        # Если старого описания нет, используем описание от ИИ.
        # Если старое есть, оставляем его (чтобы не затереть ручные правки).
        if old_description:
            result["description"] = old_description
            logger.info(f"Keeping manual description for {request.image_path}")
        else:
            logger.info(f"Using AI description for {request.image_path}")

        # 3. ЗАПИСЬ В JSON (image_metadata.json)
        update_image_metadata(folder_path, request.image_path, result)
        
        # 4. ЗАПИСЬ В EXIF (В сам файл изображения)
        #write_metadata_to_file(
        #    folder_path / request.image_path,
        #    result["description"],
        #    result["tags"],
        #    result.get("tags_ru", [])
        #)

    return result


@app.post("/close-folder")
async def close_folder():
    try:
        if hasattr(app, 'current_folder') and app.current_folder:
            folder_path = Path(app.current_folder)
            metadata_file = folder_path / "image_metadata.json"
            
            if metadata_file.exists():
                os.remove(metadata_file)
                logger.info(f"🔥 Final cleanup: deleted {metadata_file.name}")

        app.current_folder = None
        return {"status": "success", "message": "Folder closed and metadata deleted"}
    except Exception as e:
        logger.error(f"Error closing folder: {e}")
        raise HTTPException(status_code=500, detail=str(e))



from fastapi import Request # Добавьте в импорты наверху

@app.post("/save-all-metadata")
@app.post("/update-metadata")
async def save_all_metadata(request: Request):
    try:
        # Получаем "сырые" данные, чтобы Pydantic не ругался раньше времени
        try:
            body = await request.json()
        except Exception as e:
            logger.error(f"Raw JSON error: {e}")
            return {"status": "error", "message": "Invalid JSON format"}

        # Извлекаем словарь метаданных
        data_to_save = body.get("metadata", body)
        
        if not isinstance(data_to_save, dict):
            # Если пришел список или строка, пытаемся работать с этим
            logger.warning(f"Data is not a dict, it is {type(data_to_save)}")
            return {"status": "error", "message": "Data must be a dictionary"}

        if not hasattr(app, 'current_folder') or not app.current_folder:
            logger.error("No folder selected in app.current_folder")
            return {"status": "error", "message": "No active folder"}

        folder_path = Path(app.current_folder)

        # Сохраняем EXIF
        counter = 0
        for rel_path, img_data in data_to_save.items():
            if isinstance(img_data, dict):
                full_path = folder_path / rel_path
                if full_path.exists():
                    # Чистим теги от возможных битых символов \xad и т.д.
                    clean_description = str(img_data.get("description", "")).replace('\xad', '')
                    clean_tags = [str(t).replace('\xad', '') for t in img_data.get("tags", [])]
                    clean_tags_ru = [str(t).replace('\xad', '') for t in img_data.get("tags_ru", [])]

                    write_metadata_to_file(
                        full_path,
                        clean_description,
                        clean_tags,
                        clean_tags_ru
                    )
                    counter += 1

        return {
            "status": "success", 
            "message": f"Saved {counter} images to EXIF",
            "folder": app.current_folder
        }

    except Exception as e:
        logger.error(f"🔥 Critical Save Error: {str(e)}")
        return {"status": "error", "message": str(e)}
@app.get("/")
async def read_root():
    return FileResponse("static/index.html")
@app.get("/thumbnail/{path:path}")
async def get_thumbnail(path: str):
    # Берем текущую папку из состояния приложения
    if not hasattr(app, 'current_folder') or not app.current_folder:
        raise HTTPException(status_code=400, detail="No folder selected")
        
    full_path = os.path.join(app.current_folder, path)
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404)
        
    try:
        with Image.open(full_path) as img:
            # Ускоряем чтение JPEG
            if path.lower().endswith(('.jpg', '.jpeg')):
                img.draft(None, (400, 400))
            
            # Делаем маленькое квадратное превью
            img.thumbnail((400, 400), resample=Image.BILINEAR)
            
            if img.mode != "RGB":
                img = img.convert("RGB")
                
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=60) # Качество 60 для веса в 5-10 Кб
            return Response(content=buf.getvalue(), media_type="image/jpeg")
    except Exception as e:
        logger.error(f"Thumbnail error: {e}")
        raise HTTPException(status_code=500)


@app.get("/image/{path:path}")
async def get_image(path: str):
    return FileResponse(os.path.join(app.current_folder, path))


@app.get("/check-init-status")
async def check_init():
    return {"needs_init": False}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)