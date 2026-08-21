@echo off
REM One-click launcher for the GRPO Math Reasoner demo.
REM Loads base Qwen2.5-1.5B-Instruct + your trained LoRA adapter, serves at http://127.0.0.1:7860
set BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct
set ADAPTER_ID=%~dp0adapter-1.5b
echo Starting GRPO Math Reasoner demo... open http://127.0.0.1:7860 once the model loads (~1 min)
"C:\Users\Mehul Mathodia\AppData\Local\Python\pythoncore-3.14-64\python.exe" "%~dp0demo\app.py"
