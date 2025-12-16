@echo off
REM Batch script to activate Tgnn_gpu and run main.py

echo ============================================================
echo Activating Tgnn_gpu virtual environment...
echo ============================================================

REM Activate the virtual environment
call ..\Tgnn_gpu\Scripts\activate.bat

REM Verify GPU is available
echo.
echo ============================================================
echo Verifying GPU availability...
echo ============================================================
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU-ONLY')"

echo.
echo ============================================================
echo Running main.py with GPU acceleration...
echo ============================================================
python main.py

REM Keep window open
echo.
echo ============================================================
echo Script completed. Press any key to exit...
pause
