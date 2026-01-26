import io
import re
import streamlit as st
import pandas as pd
import zipfile
from pathlib import Path
from typing import Dict, List

st.set_page_config(page_title="Export App", layout="centered")

st.title("Export App")

st.markdown("Upload a CSV file and preview it. You can then export to Excel or CSV.")

# Upload mode: multiple files, folder on the server, or a ZIP archive upload
upload_mode = st.radio("Upload mode", ["Multiple CSV files", "Folder (server)", "ZIP archive (upload)"], index=0)

uploaded = None
folder_path = None
scan_folder = False
if upload_mode == "Multiple CSV files":
    uploaded = st.file_uploader("Choose one or more CSV files", type=["csv"], accept_multiple_files=True)
elif upload_mode == "ZIP archive (upload)":
    uploaded = st.file_uploader("Upload a ZIP containing CSV files", type=["zip"], accept_multiple_files=False)
else:
    # Folder on the server: user supplies a path that the app can read (useful when running locally)
    folder_path = st.text_input("Enter folder path on server (absolute or relative to app)")
    scan_folder = st.button("Scan folder for CSV files")

MAX_FILE_MB = 100  # refuse files larger than this (adjust as needed)

def read_csv_fileobj(file_obj, name: str):
    try:
        file_obj.seek(0)
        return pd.read_csv(file_obj)
    except pd.errors.EmptyDataError:
        st.error(f"File '{name}' is empty.")
    except pd.errors.ParserError as e:
        st.error(f"Failed to parse '{name}': {e}")
    except Exception as e:
        st.error(f"Unexpected error reading '{name}': {e}")
    return None

def extract_csvs_from_zip(zip_bytes) -> Dict[str, pd.DataFrame]:
    results: Dict[str, pd.DataFrame] = {}
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
            for info in z.infolist():
                if info.is_dir():
                    continue
                if Path(info.filename).suffix.lower() != ".csv":
                    continue
                with z.open(info) as f:
                    try:
                        file_bytes = f.read()
                        df = read_csv_fileobj(io.BytesIO(file_bytes), info.filename)
                        if df is not None:
                            results[Path(info.filename).name] = df
                    except Exception as e:
                        st.error(f"Error extracting {info.filename} from ZIP: {e}")
    except zipfile.BadZipFile:
        st.error("Uploaded file is not a valid ZIP archive.")
    except Exception as e:
        st.error(f"Error processing ZIP: {e}")
    return results

# --- Helpers ported from lambspython.py to create the same summary output ---
IMV = {'L14','L16','L18','L33','L35','L39','L43','L47','L49','L55','L61','L63','L68','L69','L70'}
NIV = {'L1','L7','L9','L13','L20','L22','L30','L34','L44','L45','L59','L51','L57','L59','L65','L67'}
NIVe = {'L8','L12','L15','L23','L32','L36','L38','L40','L48','L52','L58','L60','L62','L66'}
NIVf = {'L2','L17','L19','L31','L46','L54','L56'}


def assign_group(ind):
    ind = str(ind).upper()
    if ind in IMV: return 'IMV'
    if ind in NIV: return 'NIV'
    if ind in NIVe: return 'NIVe'
    if ind in NIVf: return 'NIVf'
    return ''


def parse_filename(name):
    # try to replicate the original splitting logic: base parts separated by '_'
    base = Path(str(name)).stem
    parts = base.split('_')
    study = '_'.join(parts[:-3]) if len(parts) >= 4 else parts[0]
    individual = parts[-3] if len(parts) >= 4 else ''
    treatment = parts[-2] if len(parts) >= 4 else ''
    time = parts[-1] if len(parts) >= 4 else ''
    return {'study': study, 'individual': individual, 'treatment': treatment, 'time': time}


def norm(s):
    return ''.join(ch for ch in str(s).lower() if ch.isalnum())


def find_column(df, keywords):
    cols = [(c, norm(c)) for c in df.columns]
    for kw in keywords:
        nk = norm(kw)
        for orig, nc in cols:
            if nk in nc or nc in nk:
                return orig
    return None


def time_key(t):
    if not isinstance(t, str):
        return 10**9
    tl = t.lower()
    if 'birth' in tl:
        return -1
    m = re.search(r"(\d+)", tl)
    if m:
        return int(m.group(1))
    return 10**6


def process_files_simple(dfs_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    # Build records similar to process_folder_simple in lambspython.py
    records = []
    varmap = {
        'EELI': ('eeli',),
        'CoV(H)': ('cov(h)', 'covh', 'cov_h', 'cov h'),
        'CoV(V)': ('cov(v)', 'covv', 'cov_v', 'cov v'),
    }

    for name, df in dfs_dict.items():
        meta = parse_filename(name)
        group = assign_group(meta.get('individual','')) or meta.get('treatment','')
        # ensure df is a DataFrame
        try:
            if not isinstance(df, pd.DataFrame):
                continue
        except Exception:
            continue

        for label, keys in varmap.items():
            col = find_column(df, keys)
            if not col:
                continue
            ser = pd.to_numeric(df[col], errors='coerce')
            if ser.count() == 0:
                continue
            records.append({
                'study': meta['study'],
                'individual': meta['individual'],
                'treatment': meta['treatment'],
                'group': group,
                'time': meta['time'],
                'variable': label,
                'mean': float(ser.mean()),
                'std': float(ser.std()),
            })

    return pd.DataFrame.from_records(records)


def export_summary_excel(all_df: pd.DataFrame) -> bytes:
    # Export the summary workbook in memory, matching export_simple behavior
    vars = ['EELI','CoV(H)','CoV(V)']
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine='xlsxwriter') as writer:
        workbook = writer.book
        for v in vars:
            dfv = all_df[all_df['variable'] == v]
            sheet_name = v[:31]
            if dfv.empty:
                pd.DataFrame().to_excel(writer, sheet_name=sheet_name, index=False)
                continue

            mean_pivot = dfv.pivot_table(index='time', columns=['group','individual'], values='mean', aggfunc='mean')
            std_pivot = dfv.pivot_table(index='time', columns=['group','individual'], values='std', aggfunc='mean')

            try:
                mean_pivot = mean_pivot.reindex(sorted(mean_pivot.index, key=time_key))
            except Exception:
                pass
            try:
                std_pivot = std_pivot.reindex(sorted(std_pivot.index, key=time_key))
            except Exception:
                pass

            mean_df = mean_pivot.reset_index()
            std_df = std_pivot.reset_index()

            if mean_pivot.empty and std_pivot.empty:
                pd.DataFrame().to_excel(writer, sheet_name=sheet_name, index=False)
                continue

            groups = list(dict.fromkeys([g for g in mean_pivot.columns.get_level_values(0) if pd.notna(g)]))

            group_means = {}
            group_stds = {}
            for g in groups:
                try:
                    gm = mean_pivot.xs(g, axis=1, level=0)
                except Exception:
                    gm = pd.DataFrame()
                if isinstance(gm, pd.Series):
                    gm = gm.to_frame()
                gm = gm.reset_index()
                group_means[g] = gm

                try:
                    gs = std_pivot.xs(g, axis=1, level=0)
                except Exception:
                    gs = pd.DataFrame()
                if isinstance(gs, pd.Series):
                    gs = gs.to_frame()
                gs = gs.reset_index()
                group_stds[g] = gs

            mean_startrow = 1
            startcol = 0
            for g in groups:
                gm = group_means[g]
                gm.to_excel(writer, sheet_name=sheet_name, index=False, startrow=mean_startrow, startcol=startcol)
                worksheet = writer.sheets[sheet_name]
                bold = workbook.add_format({'bold': True})
                worksheet.write(0, startcol, str(g), bold)
                ncols = len(gm.columns) if not gm.empty else 1
                startcol += ncols

            nrows = max([len(df) for df in group_means.values()]) if group_means else 0
            std_startrow = mean_startrow + nrows + 3

            startcol = 0
            for g in groups:
                gs = group_stds[g]
                gs.to_excel(writer, sheet_name=sheet_name, index=False, startrow=std_startrow, startcol=startcol)
                worksheet = writer.sheets[sheet_name]
                bold = workbook.add_format({'bold': True})
                worksheet.write(std_startrow - 1, startcol, 'Std Dev - ' + str(g), bold)
                ncols = len(gs.columns) if not gs.empty else 1
                startcol += ncols

    out.seek(0)
    return out.getvalue()

# Collect DataFrames into a dict keyed by filename
# For simplicity load all uploaded CSVs fully into memory and auto-select them for export
dfs: Dict[str, pd.DataFrame] = {}

if upload_mode == "Multiple CSV files" and uploaded:
    files = uploaded if isinstance(uploaded, list) else [uploaded]
    total_mb = 0.0
    for f in files:
        name = getattr(f, 'name', f"file_{len(dfs)+1}.csv")
        try:
            f.seek(0)
            raw = f.read()
            size_mb = len(raw) / (1024 * 1024)
            total_mb += size_mb
            try:
                # parse full dataframe immediately
                df_full = pd.read_csv(io.BytesIO(raw))
                dfs[name] = df_full
            except Exception as e:
                st.warning(f"Failed to parse {name}: {e}")
        except Exception as e:
            st.warning(f"Failed to read uploaded file {name}: {e}")
    st.info(f"{len(dfs)} files loaded — total ~{total_mb:.1f} MB")

elif upload_mode == "ZIP archive (upload)" and uploaded:
    try:
        uploaded.seek(0)
        zip_bytes = uploaded.read()
        dfs = extract_csvs_from_zip(zip_bytes)
    except Exception as e:
        st.error(f"Failed to read uploaded ZIP: {e}")

elif upload_mode == "Folder (server)":
    # Scan the provided folder path and read CSVs from it when the user clicks the scan button
    if folder_path:
        if scan_folder:
            p = Path(folder_path)
            if not p.exists():
                st.error("Folder does not exist on the server. Check the path and try again.")
            elif not p.is_dir():
                st.error("Provided path is not a folder. Please provide a directory path.")
            else:
                csv_files = sorted([x for x in p.iterdir() if x.is_file() and x.suffix.lower() == '.csv'])
                if not csv_files:
                    st.info("No CSV files found in the specified folder.")
                else:
                    total_mb = 0.0
                    for fpath in csv_files:
                        try:
                            size_mb = fpath.stat().st_size / (1024 * 1024)
                            total_mb += size_mb
                            if size_mb > MAX_FILE_MB:
                                st.warning(f"Skipping {fpath.name} - too large ({size_mb:.1f} MB)")
                                continue
                            with open(fpath, 'r', encoding='utf-8') as fh:
                                df = read_csv_fileobj(fh, fpath.name)
                                if df is not None:
                                    dfs[fpath.name] = df
                        except PermissionError:
                            st.warning(f"Permission denied reading {fpath}")
                        except Exception as e:
                            st.error(f"Error reading {fpath.name}: {e}")
                    st.info(f"{len(dfs)} files loaded from folder — total ~{total_mb:.1f} MB")
        else:
            st.info("Enter a folder path and click 'Scan folder for CSV files' to load CSVs from the server.")
    else:
        st.info("Provide a folder path to read CSV files from the server where the app is running.")

if not dfs:
    st.info("No valid CSV data available yet. Upload CSV(s) or a folder containing CSV files.")
else:
    filenames = list(dfs.keys())
    # Collapse the detailed file listing by default
    with st.expander(f"Files found ({len(filenames)}) — click to expand", expanded=False):
        st.write("Detected files:")
        for name in filenames:
            st.write(f"- {name} ({len(dfs[name])} rows, {len(dfs[name].columns)} cols)")

    # Auto-select all imported files
    selected = filenames.copy()

    # Export only to Excel (XLSX)
    filename = st.text_input("Output filename (without extension)", "exported_data")
    safe_filename = re.sub(r"[^A-Za-z0-9_.-]", "_", (filename or "").strip())
    if not safe_filename:
        st.warning("Please provide a valid filename before exporting.")

    if st.button("Export to Excel"):
        if not safe_filename:
            st.error("Invalid filename. Use letters, numbers, dot, underscore or hyphen.")
        elif not selected:
            st.error("No files selected to export.")
        else:
            try:
                with st.spinner("Creating Excel workbook..."):
                    towrite = io.BytesIO()
                    with pd.ExcelWriter(towrite, engine="openpyxl") as writer:
                        for name in selected:
                            sheet_name = Path(name).stem[:31]
                            dfs[name].to_excel(writer, index=False, sheet_name=sheet_name)
                    towrite.seek(0)
                    export_bytes = towrite.getvalue()
                st.success("Export ready — click the download button below")
                st.download_button("Download XLSX", data=export_bytes, file_name=f"{safe_filename}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            except ImportError as e:
                st.exception(e)
                st.error("Missing dependency 'openpyxl'. Install it with: pip install openpyxl")
            except Exception as e:
                st.exception(e)
                st.error(f"Failed to create Excel workbook: {e}")

    # --- Summary export section ---
    st.subheader("Summary statistics export")

    if len(dfs) == 1:
        st.info("Summary statistics are not available for a single file. Please upload multiple files or a ZIP archive.")
    else:
        # Options: allocation mapping and column selection for summary
        st.write("Customize summary options:")
        alloc_choice = st.selectbox("Lamb allocation mapping", ["Default", "Custom"], index=0)
        custom_sets = None
        if alloc_choice == "Custom":
            st.info("Enter comma-separated individual IDs (e.g. L1,L2) for each group")
            imv_txt = st.text_area("IMV members (comma-separated)", value="")
            niv_txt = st.text_area("NIV members (comma-separated)", value="")
            nive_txt = st.text_area("NIVe members (comma-separated)", value="")
            nivf_txt = st.text_area("NIVf members (comma-separated)", value="")
            def parse_members(s):
                return {x.strip().upper() for x in re.split(r'[,\s]+', s) if x.strip()} if s else set()
            custom_sets = {
                'IMV': parse_members(imv_txt),
                'NIV': parse_members(niv_txt),
                'NIVe': parse_members(nive_txt),
                'NIVf': parse_members(nivf_txt),
            }

        # Column selection: either manual selection of columns or auto-detect variables
        cols_union = sorted({c for df in dfs.values() for c in df.columns}) if dfs else []
        col_mode = st.radio("Summary column mode", ["Auto-detect (EELI, CoV(H), CoV(V))", "Select columns manually"], index=0)
        manual_cols = []
        if col_mode.startswith("Select"):
            manual_cols = st.multiselect("Select columns to include in summary (exact column names)", cols_union, default=[])

        # Allow overriding auto keywords for default vars
        var_defaults = {
            'EELI': 'eeli',
            'CoV(H)': 'cov(h),covh,cov_h,cov h',
            'CoV(V)': 'cov(v),covv,cov_v,cov v',
        }
        var_choices = list(var_defaults.keys())
        selected_vars = var_choices.copy()
        if col_mode.startswith("Auto"):
            st.write("Auto-detected variable keywords (edit if needed):")
            for v in var_choices:
                txt = st.text_input(f"Keywords for {v}", value=var_defaults[v])
                var_defaults[v] = txt

        def get_group_for(individual):
            ind = str(individual).upper()
            if custom_sets:
                for gname, s in custom_sets.items():
                    if ind in s:
                        return gname
                return ''
            return assign_group(ind)

        def process_with_options(dfs_local: Dict[str, pd.DataFrame]):
            records = []
            if col_mode.startswith("Select") and manual_cols:
                for name, df in dfs_local.items():
                    meta = parse_filename(name)
                    group = get_group_for(meta.get('individual','')) or meta.get('treatment','')
                    for col in manual_cols:
                        if col not in df.columns:
                            continue
                        ser = pd.to_numeric(df[col], errors='coerce')
                        if ser.count() == 0:
                            continue
                        records.append({
                            'study': meta['study'],
                            'individual': meta['individual'],
                            'treatment': meta['treatment'],
                            'group': group,
                            'time': meta['time'],
                            'variable': col,
                            'mean': float(ser.mean()),
                            'std': float(ser.std()),
                        })
            else:
                # build varmap from var_defaults
                varmap = {}
                for v, kwtxt in var_defaults.items():
                    kws = [s.strip() for s in re.split(r'[,\s]+', kwtxt) if s.strip()]
                    if kws:
                        varmap[v] = tuple(kws)
                # process using keywords
                for name, df in dfs_local.items():
                    meta = parse_filename(name)
                    group = get_group_for(meta.get('individual','')) or meta.get('treatment','')
                    for label, keys in varmap.items():
                        col = find_column(df, keys)
                        if not col:
                            continue
                        ser = pd.to_numeric(df[col], errors='coerce')
                        if ser.count() == 0:
                            continue
                        records.append({
                            'study': meta['study'],
                            'individual': meta['individual'],
                            'treatment': meta['treatment'],
                            'group': group,
                            'time': meta['time'],
                            'variable': label,
                            'mean': float(ser.mean()),
                            'std': float(ser.std()),
                        })
            return pd.DataFrame.from_records(records)

        # Perform processing and store summary
        with st.spinner("Processing files for summary statistics..."):
            try:
                all_summary = process_with_options(dfs)
                st.session_state.all_summary = all_summary
                st.success("Summary statistics calculated")
            except Exception as e:
                st.exception(e)
                st.error(f"Error processing files: {e}")

        if 'all_summary' in st.session_state:
            summary_df = st.session_state.all_summary
            # Summary preview removed; export available below

            # Export summary statistics
            if st.button("Export summary to Excel"):
                try:
                    with st.spinner("Creating summary Excel workbook..."):
                        export_bytes = export_summary_excel(summary_df)
                    st.success("Summary export ready — click the download button below")
                    st.download_button("Download summary XLSX", data=export_bytes, file_name="summary_statistics.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                except Exception as e:
                    st.error(f"Failed to export summary statistics: {e}")
