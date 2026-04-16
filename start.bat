@echo off
chcp 65001 > nul
set PYTHONIOENCODING=utf-8

echo Проверка статуса Ollama...
ollama list >nul 2>&1
if %errorlevel% neq 0 (
    echo ОШИБКА: Ollama не запущена! Запустите приложение Ollama и попробуйте снова.
    pause
    exit
)

echo Запуск Llama Vision Tagger...
python main.py

if %errorlevel% neq 0 (
    echo Произошла ошибка при работе скрипта.
    pause
)
pause