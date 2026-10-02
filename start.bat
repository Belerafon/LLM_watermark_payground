@echo off
cd /d "%~dp0"
title Watermarking Playground
echo.
echo  LLM Watermarking Playground
echo  http://127.0.0.1:8765/
echo.
echo  Close this window to stop the server.
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1"
if errorlevel 1 (
  echo.
  echo  Server failed to start.
  pause
)
