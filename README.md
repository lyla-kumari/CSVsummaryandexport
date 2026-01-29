# Export App

**User manual:** See `USER_MANUAL.md` in the project root. The app also displays this manual inside the UI under "User manual — click to view".

## README

**Version:** 1.1  
**Audience:** End users with no coding background

---

## Overview

Export App is a simple data export and summary tool for CSV files.  
It allows users to upload multiple CSV files (or a ZIP of CSVs), preview their contents, export raw data into Excel, and calculate basic summary statistics (mean and standard deviation).

---

## Features

- Upload multiple CSV files  
- Upload ZIP archives containing CSV files  
- Load CSV files from a server folder  
- Preview CSV data (first rows and columns) and view original → sanitized column name mapping
- Export raw CSVs into a single Excel workbook
- Define derived columns using simple expression syntax and preview them on a chosen file
- Calculate summary statistics (mean and standard deviation) using selected columns and/or derived columns
- Export summary statistics to Excel or CSV

---

## Derived columns (quick guide)

You can define temporary derived columns that are applied only when calculating summaries (they do not mutate your stored data). Enter one definition per line using the format:

NAME = op(col1,col2,...)

Supported operations:
- sum(A,B,...)  — elementwise sum across columns
- mean(A,B,...) | avg(A,B,...)  — elementwise mean (ignores NaNs)
- median(A,B,...)  — elementwise median
- ratio(num, den) or r(num, den)  — elementwise division (denominator zeros become NaN)
- wmean(A:weight, B:weight, ...)  — weighted mean; weights are numeric and optional (default weight 1)

Examples (use the UI example buttons to auto-insert these):
- TOTAL = sum(Flow,Volume)
- AVG = mean(Flow,Volume)
- W = wmean(Flow:0.3, Volume:0.7)
- R = ratio(sum(Flow,Volume), Volume)

Notes:
- Column names must match the sanitized names shown in the preview mapping (spaces and special characters are replaced by underscores on load).
- The parser will warn about non-numeric or negative weights and other parse issues.
- Use the "Preview derived columns on preview file" button to see the computed columns for a single file before calculating the full summary.

---

## Getting Started

When the app opens, the screen is divided into:
- A **left sidebar** (upload and settings)
- A **main panel** (file selection, preview, and export)

Follow the steps below to load data and run summaries.

### Step 1 — Load Data

Open the sidebar and locate **Upload / Source**.

Option A: Multiple CSV Files
1. Select **Multiple CSV files**
2. Click **Choose one or more CSV files**
3. Select your files
4. Click **Load uploaded files** in the main panel

Option B: ZIP Archive
1. Select **ZIP archive (upload)**
2. Upload a ZIP file containing CSV files
3. Click **Load ZIP and extract CSVs**

Option C: Server Folder
1. Select **Folder (server)**
2. Enter the folder path
3. Tick **Scan folder now**
4. Click **Scan folder and load CSVs**

After loading, the app captures original header names and shows the sanitized names used internally — check the preview pane for the mapping.

---

### Step 2 — Select Files and Columns

- Use the file list to choose which files to include in the summary.
- Use the **Select all files** / **Clear selection** buttons for convenience.
- Expand **Show file preview** to view first rows of a file and a per-file button **Select columns from <file>** to quickly populate the column selector.
- Use the column select controls to pick which numeric columns to summarize. The app also provides **Select all columns** / **Clear columns** buttons.

---

### Step 3 — Define Derived Columns (optional)

- Use the **Derived columns** text area to enter expressions (one per line).
- Use the auto-generated example buttons to insert templates based on detected column names.
- Review parser warnings shown beneath the text area and fix any issues.
- Use **Preview derived columns on preview file** to validate results for a single file.

---

### Step 4 — Calculate Summary Statistics

1. Select files and columns (and/or define derived columns).
2. Click **Calculate summary for selected files**.
3. The app computes mean and standard deviation per-file and produces a combined summary preview.

---

### Step 5 — Export Summary Results

Once calculated, summaries can be exported.

Export to Excel
- Click **Export summary to Excel** — produces a formatted workbook with one sheet per variable, group means on top and std devs below.

Export to CSV
- Click **Export summary to CSV** — produces a flat summary table.

---

### Filename parsing & export filename tokens

The app provides flexible filename parsing to extract study, individual, treatment and time information from file names. You can choose the parsing mode in the sidebar:

- Auto (underscore parts): splits the file stem on underscores and assigns the last three parts to individual, treatment and time (the remainder is treated as study).
- Custom pattern (tokens): use a token template like `{study}_{individual}_{treatment}_{time}` to map parts directly.
- Regex (named groups): provide a regular expression with named groups such as `(?P<study>...)`, `(?P<individual>...)`, `(?P<treatment>...)`, `(?P<time>...)` for full control over parsing.

When exporting files or summaries the app also supports a user-editable filename pattern with tokens. Supported tokens are:

- `{date}`      - current date YYYYMMDD
- `{datetime}`  - current datetime YYYYMMDD_HHMMSS
- `{prefix}`    - fallback prefix (usually default_export_name or default_summary)
- `{study}`     - parsed study from the first selected file name
- `{individual}`- parsed individual from the first selected file name
- `{treatment}` - parsed treatment from the first selected file name
- `{time}`      - parsed time from the first selected file name
- `{group}`     - derived group for the first selected file (if available)
- `{firstfile}` - stem (no ext) of the first selected file
- `{count}`     - number of selected files
- `{index}`     - numeric index (useful for multiple outputs)

The app sanitises and truncates generated names so they are safe for most filesystems.

### Custom individual allocation mapping

You can override the built-in allocation sets by choosing "Custom" under "Individual allocation mapping" in the sidebar. The UI now accepts arbitrary group names and their member IDs using a single free-form textarea. Enter one group per line with the format:

GROUP_NAME = ID1, ID2, ID3

Example:
GROUP_A = L1, L2, L3

Group names are free-form (not limited to IMV/NIV/NIVe/NIVf) and will be used as the `group` value in summaries and export filenames. Definitions are parsed and validated when you click "Save settings" and are stored in the session for subsequent summary calculations. If no custom groups are provided the app falls back to the built-in allocation sets.

### Filename regex validation and time cleaning

- When using "Regex (named groups)" mode for filename parsing, the app validates the supplied regular expression when you click "Save settings" and will warn if the regex is invalid.
- Parsed time tokens are cleaned to remove embedded extensions or trailing suffixes (for example `45min.eit` will be parsed as `45min`) so the `time` field contains only the timepoint part of the filename.

### Privacy mode and server folder access

By default the app runs in privacy mode (data stays local and the app will not access server folders). If you disable privacy mode you may choose the "Folder (server)" upload option to read CSVs from a path on the machine running the app. Folder access is powerful but potentially risky on public/shared servers — the app warns you and requires a confirmation checkbox before enabling server folder scanning.

---

## Additional notes & troubleshooting

- Column sanitization: headers are converted to letters, numbers and underscores only; see the mapping in the preview to find the correct names for expressions.
- Ratio division by zero yields NaN and is ignored in mean/std computations.
- If derived evaluation fails on a file, detailed errors are shown to help debugging (which derived name, which file, and a short message).
- Large files: the app will warn about files exceeding the configured MAX_FILE_MB and skip them to avoid memory issues.
- Session persistence: derived definitions and column selections persist in the session so you don't lose them on reruns.

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