@echo off
setlocal
set PYTHONIOENCODING=utf-8
pushd "%~dp0"
python -c "import alphagenome, fastapi, uvicorn" >nul 2>&1
if errorlevel 1 (
  echo Missing dependencies. Run in this folder:
  echo   python -m pip install -r requirements.txt
  pause
  popd
  exit /b 1
)
echo Open http://127.0.0.1:8000 in your browser.
echo Configure your API Key in the page, not in this command window.
echo Press Ctrl+C to stop the local server.
python server.py --host 127.0.0.1 --port 8000
if errorlevel 1 pause
popd
endlocal
