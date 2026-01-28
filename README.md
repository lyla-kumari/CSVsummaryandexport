# Export App

## README

**Version:** 1.0  
**Audience:** End users with no coding background

---

## Overview

Export App is a simple data export and summary tool for CSV files.  
It allows users to upload multiple CSV files (or a ZIP of CSVs), preview their contents, export raw data into Excel, and calculate basic summary statistics (mean and standard deviation).

No programming knowledge is required.

---

## Features

- Upload multiple CSV files  
- Upload ZIP archives containing CSV files  
- Load CSV files from a server folder  
- Preview CSV data  
- Export raw CSVs into a single Excel workbook  
- Calculate summary statistics (mean, standard deviation)  
- Export summary statistics to Excel or CSV  

---

## Supported File Types

- `.csv` — Comma-Separated Values  
- `.zip` — ZIP archive containing CSV files  

---

## Filename Requirements (Important)

The app extracts metadata from filenames using the following format:

STUDY_INDIVIDUAL_TREATMENT_TIME.csv


**Example:**
A974_L01_IMV_45min.eit.csv


From this filename, the app extracts:
- **Study:** A974  
- **Individual:** L01  
- **Treatment:** IMV  
- **Time:** 45min  

If filenames do not follow this structure:
- Raw exports will still work  
- Grouping, time ordering, and summary structure may be incomplete  

---

## Getting Started

When the app opens, the screen is divided into:
- A **left sidebar** (upload and settings)
- A **main panel** (file selection, preview, and export)

---

## Step 1 — Load Data

Open the sidebar and locate **Upload / Source**.

### Option A: Multiple CSV Files
1. Select **Multiple CSV files**
2. Click **Choose one or more CSV files**
3. Select your files
4. Click **Load uploaded files** in the main panel

### Option B: ZIP Archive
1. Select **ZIP archive (upload)**
2. Upload a ZIP file containing CSV files
3. Click **Load ZIP and extract CSVs**

### Option C: Server Folder
1. Select **Folder (server)**
2. Enter the folder path
3. Tick **Scan folder now**
4. Click **Scan folder and load CSVs**

---

## Step 2 — Group Assignment (Optional)

Under **Summary options** in the sidebar:

### Default Grouping
- Individuals are automatically assigned to:
  - `IMV`, `NIV`, `NIVe`, `NIVf`
- Based on predefined individual ID lists

### Custom Grouping
1. Select **Custom**
2. Enter individual IDs separated by commas  
   Example:
L1, L2, L3

3. IDs not listed will remain unassigned

**Notes:**
- IDs are case-insensitive  
- Use formats such as `L01`, `L14`

---

## Step 3 — Review Files

After loading, the app displays:
- A list of detected files
- File selection controls
- Number of selected files

### Optional File Preview
1. Enable **Show file preview**
2. Select a file
3. View the first 10 rows and column names

---

## Step 4 — Export Raw Data

Scroll to **Export raw files**.

1. Enter an output filename
2. Click **Create Excel from selected files**
3. Click **Download XLSX**

**Result:**
- One Excel file
- One sheet per CSV file
- Sheet names derived from filenames
- Excel limits sheet names to 31 characters

---

## Step 5 — Calculate Summary Statistics

Scroll to **Summary statistics export**.

1. Select one or more columns to summarise  
- Column names must match exactly
2. Select files to include
3. Click **Calculate summary for selected files**

For each selected file and column, the app calculates:
- Mean  
- Standard deviation  

Only numeric values are included.

---

## Step 6 — Export Summary Results

Once calculated, summaries can be exported.

### Export to Excel
- Click **Export summary to Excel**
- Produces a formatted workbook:
- One sheet per variable
- Means at the top
- Standard deviations below
- Timepoints sorted logically

### Export to CSV
- Click **Export summary to CSV**
- Produces a flat summary table

---

## Clearing Data

To reset the app:
- Click **Clear loaded files** at the bottom of the page
- All loaded files and summaries are removed

---

## Troubleshooting

**“No valid CSV data available yet”**
- Files have not been loaded
- Upload files and click the Load button

**“Failed to parse file”**
- File may not be a valid CSV
- Try opening and re-saving in Excel

**“No columns selected for summary”**
- At least one column must be selected

**Blank group column**
- Individual ID not recognised
- Filename may not match required format

**Truncated Excel sheet names**
- Excel limits sheet names to 31 characters

---

## Common Workflows

### Raw Data Export
1. Upload files
2. Load files
3. Select files
4. Export Excel

### Summary Export
1. Upload files
2. Load files
3. Select files
4. Select columns
5. Calculate summary
6. Export Excel or CSV

---

## Run locally (privacy / offline use)

If you are concerned about privacy or prefer to keep all data on your machine, you can run the app locally. When run locally the app processes files only on your computer — no data is uploaded to any external server.

Recommended steps (macOS / zsh):

1. Create and activate a virtual environment

   zsh commands:
   python3 -m venv venv
   source venv/bin/activate

2. Install dependencies

   pip install -r requirements.txt

   Note: This requires internet access once to download packages. If you need fully offline installation, create a wheelhouse on another machine and install from that.

3. Run the Streamlit app

   streamlit run app_streamlit.py

   By default this opens the app in your browser at http://localhost:8501 and keeps all processing local.

4. Using a local folder as data source

   - In the app sidebar choose "Folder (server)" or the equivalent option.
   - Enter an absolute or relative path to the folder on your machine containing the CSV files.
   - Tick "Scan folder now" (or click the scan/load button) to load files. The app will read files from that path on your disk.

5. Tips for strict offline use

   - Install dependencies before disconnecting from the internet.
   - Do not enable any cloud integrations or remote paths in the app settings.
   - If you need to share the app with colleagues in a closed network, package the virtual environment or provide a requirements wheelhouse.

Quick commands (zsh):

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app_streamlit.py

---

## End of README