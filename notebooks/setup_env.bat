@echo off
REM Sets up (or reuses) ONE dedicated conda environment for trash_tracker training,
REM and registers it as a clearly-named Jupyter kernel so there's no ambiguity
REM about which environment/kernel this notebook uses.
REM Run this from an Anaconda Prompt, from inside this notebooks/ folder.

echo Checking for existing "trash_tracker" conda environment...
call conda env list | findstr /C:"trash_tracker" >nul
if errorlevel 1 (
    echo Creating conda environment "trash_tracker" ^(Python 3.10^)...
    call conda create -n trash_tracker python=3.10 -y
) else (
    echo Environment already exists - reusing it.
)

echo Activating environment...
call conda activate trash_tracker

echo Installing dependencies from requirements.txt...
call pip install -r requirements.txt

echo Registering this environment as a named Jupyter kernel...
call python -m ipykernel install --user --name trash_tracker --display-name "Python (trash_tracker)"

echo.
echo Done. This environment now shows up in Jupyter as "Python (trash_tracker)" -
echo pick that kernel every time, regardless of which other envs/kernels exist.
echo.
echo One-time secret setup:
echo   Copy .env.example to .env in this same folder, then edit .env and fill
echo   in your real ROBOFLOW_API_KEY. The notebook loads it automatically -
echo   you never need to "set" it in the terminal.
echo.
echo Next steps:
echo   1. Launch Jupyter:  jupyter notebook --notebook-dir="%cd%"
echo   2. Open train_trash_tracker.ipynb
echo   3. Kernel menu -^> Change Kernel -^> "Python (trash_tracker)"
echo   4. Run cells in order.
echo.
echo Next time, just run:
echo   conda activate trash_tracker
echo   jupyter notebook --notebook-dir="%cd%"
echo   (confirm the notebook's kernel is "Python (trash_tracker)" before running cells)
