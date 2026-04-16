from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import json
import logging
import piexif
from pathlib import Path
from typing import List, Dict, Set, Optional, Any
from PIL import Image

# Импортируем процессор (убедитесь, что файл image_processor.py в той же папке)
from image_processor import ImageProcessor, update_image_metadata

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FolderRequest(BaseModel):
    folder_path: str

class ProcessImageRequest(BaseModel):
    image_path: str

# --- ЗАПИСЬ В EXIF (WINDOWS XPKeywords) ---
def write_metadata_to_file(file_path: Path, description: str, tags: List[str]):
    if file_path.suffix.lower() not in ['.jpg', '.jpeg']:
        return 
    try:
        img = Image.open(file_path)
        exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
        if "exif" in img.info:
            try:
                exif_dict = piexif.load(img.info["exif"])
            except: pass

        # Описание (UTF-8)
        exif_dict["0th"][piexif.ImageIFD.ImageDescription] = description.encode('utf-8')
        # Теги для Windows (UTF-16LE)
        exif_dict["0th"][0x9c9e] = ";".join(tags).encode('utf-16le')

        exif_bytes = piexif.dump(exif_dict)
        img.save(file_path, exif=exif_bytes, quality="keep")
        logger.info(f"💾 EXIF updated: {file_path.name}")
    except Exception as e:
        logger.error(f"❌ Failed to write EXIF: {e}")

# --- СКАНЕР ПАПКИ ---
def load_simple_metadata(folder_path: Path) -> Dict[str, Dict]:
    metadata_file = folder_path / "image_metadata.json"
    supported_ext = {'.jpg', '.jpeg', '.png', '.webp'}
    
    if metadata_file.exists():
        with open(metadata_file, 'r', encoding='utf-8') as f:
            try:
                metadata = json.load(f)
            except: metadata = {}
    else:
        metadata = {}

    current_files = [str(p.name) for p in folder_path.glob("*") if p.suffix.lower() in supported_ext]
    for filename in current_files:
        if filename not in metadata:
            metadata[filename] = {"description": "", "tags": [], "is_processed": False}
    
    return {k: v for k, v in metadata.items() if k in current_files}

# --- ЭНДПОИНТЫ ---

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
            "is_processed": data.get("is_processed", False)
        })
    return {"images": images}

@app.post("/process-image")
async def process_image_endpoint(request: ProcessImageRequest):
    folder_path = Path(app.current_folder)
    
    # 1. Загружаем текущие данные из JSON
    metadata = load_simple_metadata(folder_path)
    current_data = metadata.get(request.image_path, {})
    old_tags = current_data.get("tags", [])
    old_description = current_data.get("description", "").strip()

    # 2. Запускаем нейросеть
    img_processor = ImageProcessor()
    result = await img_processor.process_image(folder_path / request.image_path)
    
    if result["is_processed"]:
        # 3. ОБЪЕДИНЕНИЕ ТЕГОВ
        ai_tags = result.get("tags", [])
        result["tags"] = list(set(old_tags + ai_tags))

        # 4. УМНОЕ ОБНОВЛЕНИЕ ОПИСАНИЯ
        # Если вы уже ввели описание вручную, мы его НЕ затираем.
        # Если описания не было — берем то, что придумала нейросеть.
        if old_description:
            result["description"] = old_description
            logger.info(f"Keeping manual description for {request.image_path}")
        
        # 5. Сохраняем всё вместе
        update_image_metadata(folder_path, request.image_path, result)
        write_metadata_to_file(folder_path / request.image_path, result["description"], result["tags"])
        
    return result

@app.post("/save-all-metadata")
@app.post("/update-metadata")
async def save_all_metadata(request: Any):
    try:
        # 1. Получаем данные из запроса (поддержка разных форматов)
        if isinstance(request, dict):
            data_to_save = request.get("metadata", request)
        else:
            try:
                body = await request.json()
                data_to_save = body.get("metadata", body)
            except:
                data_to_save = request

        if not isinstance(data_to_save, dict):
            logger.error(f"Invalid data type: {type(data_to_save)}")
            return {"status": "error", "message": "Invalid data format"}

        folder_path = Path(app.current_folder)
        metadata_file = folder_path / "image_metadata.json"

        # 2. Массовая запись в EXIF каждого файла
        logger.info("Starting EXIF update for all images...")
        for rel_path, img_data in data_to_save.items():
            if isinstance(img_data, dict) and img_data.get("is_processed"):
                full_path = folder_path / rel_path
                # Записываем данные прямо в файл
                write_metadata_to_file(
                    full_path, 
                    img_data.get("description", ""), 
                    img_data.get("tags", [])
                )
        
        # 3. АВТО-УДАЛЕНИЕ JSON
        # Теперь, когда данные в файлах, временный JSON больше не нужен
        if metadata_file.exists():
            try:
                os.remove(metadata_file)
                logger.info(f"🔥 Временный файл {metadata_file.name} удален. Данные в EXIF.")
            except Exception as e:
                logger.warning(f"Не удалось удалить JSON (возможно, открыт): {e}")

        return {"status": "success", "message": "All metadata saved to files and JSON cleaned up"}

    except Exception as e:
        logger.error(f"Save All error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/image/{path:path}")
async def get_image(path: str):
    return FileResponse(os.path.join(app.current_folder, path))

@app.get("/check-init-status")
async def check_init():
    return {"needs_init": False}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
