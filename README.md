Export App

This folder contains a minimal Streamlit app to upload a CSV file, preview it, and export it as XLSX or CSV.

Setup (macOS, zsh):

1. Create and activate a virtual environment
   python3 -m venv .venv
   source .venv/bin/activate

2. Install dependencies
   pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt

3. Run the app
   streamlit run app_streamlit.py

Files:
- app_streamlit.py: main Streamlit app
- requirements.txt: Python dependencies

Notes:
- This is a minimal export app; adapt input validation and error handling as needed.
