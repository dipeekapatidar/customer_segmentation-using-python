@echo off
REM Windows: double-click this file to install packages and start the dashboard
python -m pip install -r requirements.txt
python -m streamlit run app.py
pause
