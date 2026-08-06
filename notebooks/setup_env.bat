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
echo Next steps:
echo   1. In THIS SAME window, set your Roboflow key:  set ROBOFLOW_API_KEY=your_key_here
echo   2. Launch Jupyter:  jupyter notebook --notebook-dir="%cd%"
echo   3. Open train_trash_tracker.ipynb
echo   4. Kernel menu -^> Change Kernel -^> "Python (trash_tracker)"
echo   5. Run cells in order.
echo.
echo Next time, just run:
echo   conda activate trash_tracker
echo   set ROBOFLOW_API_KEY=your_key_here
echo   jupyter notebook --notebook-dir="%cd%"
echo   (confirm the notebook's kernel is "Python (trash_tracker)" before running cells)
