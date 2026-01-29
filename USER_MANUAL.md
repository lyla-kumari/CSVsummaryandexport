CSV Summary & Export — User Manual

This short user manual explains how to use the Export App (Streamlit).

1) Opening the app

- Run locally: open a terminal in this folder and run:
  - streamlit run app_streamlit.py
- The app will open in your web browser. If the command fails, ask your administrator to install Streamlit (pip install streamlit).

2) Main steps (simple workflow)

- Upload files:
  - Use the sidebar to upload one or more CSV files, or upload a ZIP containing CSVs.
  - If your data is already on the same machine running the app, you can use the "Folder (server)" option (only if you understand the privacy warning).

- Load files:
  - After selecting files in the sidebar, click the "Load uploaded files" (or "Load ZIP and extract CSVs" / "Scan folder and load CSVs") button.
  - Wait for the success message that tells you how many files were loaded.

- Select which files to process:
  - In the main area use the "Select files to include" picker.
  - Use the "Select all files" / "Clear selection" buttons for convenience.

- (Optional) Preview a file:
  - Check "Show file preview" and choose a file to inspect its first rows and columns.

- Choose summary columns:
  - In "Summary statistics export" choose "Select columns manually" and then pick the exact column names you want summarized.
  - Alternatively, use "Derived columns" to create new variables computed from existing columns (see below).

- Calculate summary:
  - Click "Calculate summary for selected files". The app will process the selected files and show a preview of the summary table.

- Export results:
  - Export the raw selected files as an Excel workbook using "Create Excel from selected files".
  - Export the summary as Excel or CSV using the export buttons in the Summary preview area.

3) Filename parsing and groups (why this matters)

- The app extracts metadata (study, individual, treatment, time) from each filename so it can group data.
- There are three parsing modes in the sidebar:
  - Auto (underscore parts): expects filenames like study_individual_treatment_time.csv
  - Custom pattern (tokens): you specify a token pattern like {study}_{individual}_{treatment}_{time}
  - Regex (named groups): for advanced matching using a regular expression with named groups (?P<study>...)
- If you don't know which to choose, start with Auto and check the preview to see parsed values.

- Group assignment:
  - The app will try to assign each individual to a group automatically.
  - You can define your own groups in the sidebar (choose "Custom" under Individual allocation mapping) using lines like:
    GROUP = L1, L2, L3
  - Enter one group per line and click "Save settings".

4) Derived columns (no coding required)

- Derived columns let you compute new columns using simple formulas.
- In the "Derived columns" box enter one definition per line in the format:
  NAME = op(A,B,...)  
  Examples:
  - TOTAL = sum(A,B)
  - AVG_AB = mean(A,B)
  - W = wmean(A:0.2,B:0.8)
  - R = ratio(sum(A,B), C)
- Click the example buttons to add templates automatically, then click "Preview derived columns on preview file" to see results.

5) Column name mappings

- If your files use different names for the same measurement, use "Column name mappings" to normalise them.
- Enter mappings one per line like:
  CANONICAL = alias1, alias2
- After entering mappings, click "Apply column mappings to loaded data" then "Confirm and apply mappings".

6) Tips and troubleshooting

- If files fail to load, check they are valid CSV files and not too large.
- If the filename parsing looks wrong, try switching parsing modes or edit the custom pattern.
- When using Regex mode, an invalid regular expression will show a warning when you save settings.
- If derived columns produce errors, check that the column names referenced exist exactly as shown in the preview (use the preview to check names).

7) Privacy and server folder access

- By default, privacy mode is enabled and folder access is disabled. Do not enable folder access on a public/shared server unless you understand the risks.


Appendix: Quick examples

Example custom group definitions:

GROUP_A = L1, L2, L3
GROUP_B = L4, L5

Example derived column:

TOTAL_VOL = sum(VolumeA, VolumeB)


End of manual.
