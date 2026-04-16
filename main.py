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


def load_simple_metadata(folder_path: Path) -> Dict[str, Dict]:
    metadata_file = folder_path / "image_metadata.json"
    supported_ext = {'.jpg', '.jpeg', '.png', '.webp'}

    if metadata_file.exists():
        with open(metadata_file, 'r', encoding='utf-8') as f:
            try:
                metadata = json.load(f)
            except:
                metadata = {}
    else:
        metadata = {}

    current_files = [str(p.name) for p in folder_path.glob("*") if p.suffix.lower() in supported_ext]
    for filename in current_files:
        if filename not in metadata:
            metadata[filename] = {
                "description": "",
                "tags": [],
                "tags_ru": [],
                "is_processed": False
            }

    return {k: v for k, v in metadata.items() if k in current_files}


@app.get("/")
async def read_root():
    return FileResponse("static/index.html")


@app.post("/images")
async def get_images(request: FolderRequest):
    folder_path = Path(request.folder_path)
    if not folder_path.exists():
        raise HTTPException(status_code=404, detail="Folder not found")

    app.current_folder = str(folder_path)
    metadata = load_simple_metadata(folder_path)

    images = []
    for path, data in metadata.items():
        images.append({
            "name": path, "path": path,
            "description": data.get("description", ""),
            "tags": data.get("tags", []),
            "tags_ru": data.get("tags_ru", []),
            "is_processed": data.get("is_processed", False)
        })
    return {"images": images}


@app.post("/process-image")
async def process_image_endpoint(request: ProcessImageRequest):
    folder_path = Path(app.current_folder)

    metadata = load_simple_metadata(folder_path)
    current_data = metadata.get(request.image_path, {})
    
    # Сохраняем старые данные для объединения
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
        write_metadata_to_file(
            folder_path / request.image_path,
            result["description"],
            result["tags"],
            result.get("tags_ru", [])
        )

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