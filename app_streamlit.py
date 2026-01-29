import io
import re
import streamlit as st
import pandas as pd
import numpy as np
import zipfile
from pathlib import Path
from typing import Dict, List
from datetime import datetime

st.set_page_config(page_title="Export App", layout="centered")

st.title("Export App")
st.markdown("Upload CSV files (or a ZIP) and export raw sheets or calculated summary statistics.")
# show user manual content from local file so the link works when app is run locally
manual_path = Path("USER_MANUAL.md")
if manual_path.exists():
    with st.expander("User manual — click to view", expanded=False):
        try:
            content = manual_path.read_text(encoding="utf-8")
            st.markdown(content)
        except Exception as e:
            st.error(f"Failed to load USER_MANUAL.md: {e}")
else:
    st.markdown("**User manual:** not found in app folder. Create `USER_MANUAL.md` in the same folder to show the guide.")

# --- Sidebar: upload and settings ---
with st.sidebar.form("upload_form"):
    st.header("Upload / Source")
    privacy_mode = st.checkbox(
        "Enable privacy mode (data stays local; disables server folder access)",
        value=True,
    )

    if privacy_mode:
        upload_mode = st.selectbox(
            "Upload mode",
            ["Multiple CSV files", "ZIP archive (upload)"],
            index=0,
        )
    else:
        upload_mode = st.selectbox(
            "Upload mode",
            ["Multiple CSV files", "Folder (server)", "ZIP archive (upload)"],
            index=0,
        )

    uploaded = None
    folder_path = None
    scan_folder = False

    if upload_mode == "Multiple CSV files":
        uploaded = st.file_uploader("Choose one or more CSV files", type=["csv"], accept_multiple_files=True)
    elif upload_mode == "ZIP archive (upload)":
        uploaded = st.file_uploader("Upload a ZIP containing CSV files", type=["zip"], accept_multiple_files=False)
    else:
        st.warning("Folder access will read files from the server running this app. Do not use this on a public/shared server with sensitive data.")
        confirm_folder = st.checkbox("I understand the privacy implications and want to load files from a server folder")
        if confirm_folder:
            folder_path = st.text_input("Enter folder path on server (absolute or relative to app)")
            scan_folder = st.checkbox("Scan folder now")
        else:
            st.info("Folder access is disabled until you confirm the privacy implications.")

    MAX_FILE_MB = 300

    st.markdown("---")
    st.header("Summary options")
    alloc_choice = st.selectbox("Individual allocation mapping", ["Default", "Custom"], index=0)
    custom_sets = None
    if alloc_choice == "Custom":
        st.info("Define groups and their member IDs (one group per line). Format: GROUP = ID1, ID2, ...")
        custom_sets_text = st.text_area("Custom group definitions (one per line)", value="")
        def parse_alloc(text: str):
            out = {}
            if not text:
                return out
            for lineno, line in enumerate(text.splitlines(), start=1):
                line = line.strip()
                if not line:
                    continue
                m = re.match(r'^\s*([^=]+?)\s*=\s*(.+)$', line)
                if not m:
                    continue
                gname = m.group(1).strip()
                rhs = m.group(2).strip()
                members = {x.strip().upper() for x in re.split(r'[,\s]+' , rhs) if x.strip()}
                if members:
                    out[gname] = members
            return out
        custom_sets = parse_alloc(custom_sets_text)
        if not custom_sets:
            st.warning("No valid custom groups parsed yet. Use the format: GROUP = L1,L2,...")

    col_mode = st.radio("Summary column mode", ["Select columns manually"], index=0)

    # Filename parsing options: allow user to select parsing mode and supply patterns/regex
    parse_modes = ["Auto (underscore parts)", "Custom pattern (tokens)", "Regex (named groups)"]
    if 'filename_parse_mode' not in st.session_state:
        st.session_state['filename_parse_mode'] = parse_modes[0]
    st.write("Filename parsing")
    st.caption("Choose how study/individual/treatment/time are extracted from filenames.")
    filename_parse_mode = st.selectbox("Filename parsing mode", parse_modes, index=parse_modes.index(st.session_state['filename_parse_mode']))
    st.session_state['filename_parse_mode'] = filename_parse_mode

    # Custom token pattern example: {study}_{individual}_{treatment}_{time}
    if filename_parse_mode.startswith("Custom"):
        if 'filename_parse_pattern' not in st.session_state:
            st.session_state['filename_parse_pattern'] = "{study}_{individual}_{treatment}_{time}"
        st.text_input("Custom filename token pattern", key='filename_parse_pattern')
        st.caption("Use tokens {study} {individual} {treatment} {time} and separators, e.g. {study}_{individual}_{treatment}_{time}. Use Regex mode if tokens may contain separators.")

    # Regex mode: allow named groups like (?P<study>.+?)_(?P<individual>[^_]+)
    if filename_parse_mode.startswith("Regex"):
        if 'filename_parse_regex' not in st.session_state:
            st.session_state['filename_parse_regex'] = r'(?P<study>.+?)_(?P<individual>[^_]+)_(?P<treatment>[^_]+)_(?P<time>[^_]+)'
        st.text_input("Filename regex (with named groups)", key='filename_parse_regex')
        st.caption("Provide a regular expression with named groups like (?P<study>...), (?P<individual>...), (?P<treatment>...), (?P<time>...).")

    saved = st.form_submit_button("Save settings")
    if saved:
        # validate regex early so users get feedback
        if st.session_state.get('filename_parse_mode','').startswith('Regex'):
            pat = st.session_state.get('filename_parse_regex','')
            if pat:
                try:
                    re.compile(pat)
                except re.error as e:
                    st.warning(f"Filename regex invalid: {e}")
        # ensure custom_sets is stored in session state so it's accessible elsewhere
        if alloc_choice == 'Custom':
            st.session_state['custom_sets_text'] = custom_sets_text
            st.session_state['custom_sets'] = custom_sets
        else:
            st.session_state['custom_sets_text'] = ''
            st.session_state['custom_sets'] = {}

# --- Helpers ---
IMV = {'L14','L16','L18','L33','L35','L39','L43','L47','L49','L55','L61','L63','L68','L69','L70'}
NIV = {'L1','L7','L9','L13','L20','L22','L30','L34','L44','L45','L51','L57','L59','L65','L67'}
NIVe = {'L8','L12','L15','L23','L32','L36','L38','L40','L48','L52','L58','L60','L62','L66'}
NIVf = {'L2','L17','L19','L31','L46','L54','L56'}


def assign_group(ind):
    ind = str(ind).upper()
    # prefer user-specified custom sets (allowing arbitrary group names)
    try:
        if custom_sets:
            for gname, s in custom_sets.items():
                if ind in s:
                    return gname
    except Exception:
        pass
    # legacy hard-coded sets as fallback
    if ind in IMV: return 'IMV'
    if ind in NIV: return 'NIV'
    if ind in NIVe: return 'NIVe'
    if ind in NIVf: return 'NIVf'
    return ''


def parse_filename(name):
    base = Path(str(name)).stem
    mode = st.session_state.get('filename_parse_mode', 'Auto (underscore parts)')

    def _clean_time(s: str) -> str:
        if not s:
            return ''
        s = str(s).strip()
        # remove anything after a dot (handles filenames like '..._45min.eit.csv')
        if '.' in s:
            s = s.split('.')[0]
        return s

    if mode.startswith('Regex'):
        pattern = st.session_state.get('filename_parse_regex', '')
        if pattern:
            try:
                m = re.match(pattern, base)
                if m:
                    gd = m.groupdict()
                    return {
                        'study': gd.get('study','') or '',
                        'individual': gd.get('individual','') or '',
                        'treatment': gd.get('treatment','') or '',
                        'time': _clean_time(gd.get('time','') or ''),
                    }
            except re.error:
                pass

    if mode.startswith('Custom'):
        pattern = st.session_state.get('filename_parse_pattern', '')
        if pattern and '{' in pattern:
            token_names = re.findall(r"\{(study|individual|treatment|time)\}", pattern)
            seps = re.split(r"\{(?:study|individual|treatment|time)\}", pattern)
            sep = next((s for s in seps if s), '_')
            parts = base.split(sep)
            if len(parts) >= len(token_names):
                mapping = {tok: (parts[i] if i < len(parts) else '') for i, tok in enumerate(token_names)}
                return {
                    'study': mapping.get('study','') or '',
                    'individual': mapping.get('individual','') or '',
                    'treatment': mapping.get('treatment','') or '',
                    'time': _clean_time(mapping.get('time','') or ''),
                }

    parts = base.split('_')
    study = '_'.join(parts[:-3]) if len(parts) >= 4 else parts[0]
    individual = parts[-3] if len(parts) >= 4 else ''
    treatment = parts[-2] if len(parts) >= 4 else ''
    time = parts[-1] if len(parts) >= 4 else ''
    time = _clean_time(time)
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


def process_files_simple(dfs_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    records = []
    varmap = {
        'EELI': ('eeli',),
        'CoV(H)': ('cov(h)', 'covh', 'cov_h', 'cov h'),
        'CoV(V)': ('cov(v)', 'covv', 'cov_v', 'cov v'),
    }

    for name, df in dfs_dict.items():
        meta = parse_filename(name)
        group = assign_group(meta.get('individual','')) or meta.get('treatment','')
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
    vars = [v for v in list(dict.fromkeys(all_df.get('variable', pd.Series()).tolist())) if v]
    if not vars:
        vars = ['EELI','CoV(H)','CoV(V)']

    out = io.BytesIO()
    with pd.ExcelWriter(out, engine='xlsxwriter') as writer:
        workbook = writer.book
        for v in vars:
            dfv = all_df[all_df['variable'] == v]
            sheet_name = str(v)[:31]
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

            if (mean_pivot.empty if hasattr(mean_pivot, 'empty') else True) and (std_pivot.empty if hasattr(std_pivot, 'empty') else True):
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


def parse_derived_definitions(text: str):
    defs = []
    warnings = []
    if not text:
        return defs, warnings

    def split_top_level_commas(s: str):
        parts = []
        cur = []
        depth = 0
        for ch in s:
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
            if ch == ',' and depth == 0:
                parts.append(''.join(cur).strip())
                cur = []
            else:
                cur.append(ch)
        if cur:
            parts.append(''.join(cur).strip())
        return parts

    def find_top_level_colon(s: str):
        depth = 0
        idx = -1
        for i, ch in enumerate(s):
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
            elif ch == ':' and depth == 0:
                idx = i
        return idx

    def parse_expr(s: str):
        s = s.strip()
        if re.fullmatch(r'[+-]?\d+(?:\.\d+)?', s):
            return {'type': 'const', 'value': float(s)}
        m = re.fullmatch(r'([A-Za-z_]\w*)\s*\((.*)\)', s)
        if m:
            fname = m.group(1).lower()
            inner = m.group(2)
            raw_args = split_top_level_commas(inner) if inner.strip() else []
            args = []
            for raw in raw_args:
                if not raw:
                    continue
                cidx = find_top_level_colon(raw)
                if cidx >= 0:
                    expr_part = raw[:cidx].strip()
                    weight_part = raw[cidx+1:].strip()
                    try:
                        weight_val = float(weight_part)
                        if weight_val < 0:
                            warnings.append(f"Negative weight {weight_val} in argument '{raw}'")
                    except Exception:
                        weight_val = None
                        warnings.append(f"Non-numeric weight in argument '{raw}'; treating as None")
                    node = parse_expr(expr_part)
                    args.append({'node': node, 'weight': weight_val})
                else:
                    node = parse_expr(raw)
                    args.append({'node': node, 'weight': None})
            if fname in ('wmean', 'weightedmean', 'wweightedmean'):
                return {'type': 'wmean', 'parts': args}
            return {'type': 'func', 'name': fname, 'args': [a['node'] for a in args], 'raw_args': args}
        return {'type': 'col', 'name': s}

    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        m = re.match(r'^\s*([^=]+?)\s*=\s*(.+)$', line)
        if not m:
            warnings.append(f"Line {lineno}: unable to parse (no '='): {line}")
            continue
        name = m.group(1).strip()
        rhs = m.group(2).strip()
        try:
            expr = parse_expr(rhs)
            defs.append({'name': name, 'expr': expr})
        except Exception as e:
            warnings.append(f"Line {lineno}: failed to parse expression for {name}: {e}")
            continue
    return defs, warnings


def evaluate_expr(node, df: pd.DataFrame):
    if node is None:
        return pd.Series([np.nan] * len(df), index=df.index)
    ntype = node.get('type')
    if ntype == 'const':
        return pd.Series([node['value']] * len(df), index=df.index)
    if ntype == 'col':
        col = node.get('name')
        if col in df.columns:
            return pd.to_numeric(df[col], errors='coerce')
        return pd.Series([np.nan] * len(df), index=df.index)
    if ntype == 'func':
        name = node.get('name')
        args = node.get('args', [])
        series_args = [evaluate_expr(a, df) for a in args]
        if name in ('sum',):
            if not series_args:
                return pd.Series([np.nan] * len(df), index=df.index)
            stacked = pd.concat(series_args, axis=1)
            return stacked.sum(axis=1, skipna=True)
        if name in ('mean', 'avg', 'average'):
            if not series_args:
                return pd.Series([np.nan] * len(df), index=df.index)
            stacked = pd.concat(series_args, axis=1)
            return stacked.mean(axis=1, skipna=True)
        if name in ('median',):
            if not series_args:
                return pd.Series([np.nan] * len(df), index=df.index)
            stacked = pd.concat(series_args, axis=1)
            return stacked.median(axis=1, skipna=True)
        if name in ('ratio', 'r'):
            if len(series_args) < 2:
                return pd.Series([np.nan] * len(df), index=df.index)
            num = series_args[0]
            den = series_args[1]
            den_safe = den.replace(0, np.nan)
            return num.divide(den_safe)
        return pd.Series([np.nan] * len(df), index=df.index)
    if ntype == 'wmean':
        parts = node.get('parts', [])
        if not parts:
            return pd.Series([np.nan] * len(df), index=df.index)
        numer = pd.Series([0.0] * len(df), index=df.index)
        denom = pd.Series([0.0] * len(df), index=df.index)
        for p in parts:
            subnode = p.get('node')
            weight = p.get('weight')
            if weight is None:
                weight = 1.0
            try:
                s = evaluate_expr(subnode, df)
            except Exception:
                s = pd.Series([np.nan] * len(df), index=df.index)
            s_num = s.fillna(0.0)
            valid_mask = ~s.isna()
            numer += s_num * float(weight)
            denom += valid_mask.astype(float) * float(weight)
        denom_safe = denom.replace(0, np.nan)
        return numer.divide(denom_safe)
    return pd.Series([np.nan] * len(df), index=df.index)


def expr_to_str(node):
    """Render expression AST back to a readable string."""
    if not node:
        return ''
    t = node.get('type')
    if t == 'const':
        return str(node.get('value'))
    if t == 'col':
        return node.get('name', '')
    if t == 'wmean':
        parts = []
        for p in node.get('parts', []):
            sub = expr_to_str(p.get('node'))
            w = p.get('weight')
            if w is not None:
                parts.append(f"{sub}:{w}")
            else:
                parts.append(sub)
        return f"wmean({', '.join(parts)})"
    if t == 'func':
        name = node.get('name', '')
        raw = node.get('raw_args')
        if raw:
            parts = []
            for a in raw:
                sub = expr_to_str(a.get('node'))
                w = a.get('weight')
                if w is not None:
                    parts.append(f"{sub}:{w}")
                else:
                    parts.append(sub)
            return f"{name}({', '.join(parts)})"
        else:
            parts = [expr_to_str(a) for a in node.get('args', [])]
            return f"{name}({', '.join(parts)})"
    return ''


def build_filename_from_pattern(pattern: str, selected_files: List[str], dfs_local: Dict[str, pd.DataFrame], default_prefix: str, index: int = 1) -> str:
    meta = {'study': '', 'individual': '', 'treatment': '', 'time': ''}
    group = ''
    firstfile = ''
    try:
        if selected_files:
            first = selected_files[0]
            firstfile = Path(first).stem
            meta = parse_filename(first) or meta
            try:
                group = get_group_for(meta.get('individual',''))
            except Exception:
                group = assign_group(meta.get('individual',''))
    except Exception:
        pass

    tokens = {
        'date': datetime.now().strftime('%Y%m%d'),
        'datetime': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'prefix': default_prefix or '',
        'study': meta.get('study','') or '',
        'individual': meta.get('individual','') or '',
        'treatment': meta.get('treatment','') or '',
        'time': meta.get('time','') or '',
        'group': group or '',
        'firstfile': firstfile or '',
        'count': str(len(selected_files) if selected_files else 0),
        'index': str(index),
    }

    result = str(pattern or '')
    for k, v in tokens.items():
        result = result.replace('{' + k + '}', str(v))

    result = re.sub(r'\s+', '_', result)
    safe = re.sub(r'[^A-Za-z0-9_.-]', '_', result).strip('_')
    return safe[:120]


# --- Data loading: explicit action ---
# store uploaded dfs in session to avoid re-loading on every interaction
if 'dfs' not in st.session_state:
    st.session_state.dfs = {}
# map: filename -> list of (original_col, sanitized_col)
if 'colname_map' not in st.session_state:
    st.session_state.colname_map = {}

load_trigger = False
if upload_mode == "Multiple CSV files" and uploaded:
    if uploaded:
        # user provided files in the sidebar; provide a button to load them
        if st.button("Load uploaded files"):
            st.session_state.dfs = {}
            st.session_state.colname_map = {}
            total_mb = 0.0
            files = uploaded if isinstance(uploaded, list) else [uploaded]
            for f in files:
                name = getattr(f, 'name', f"file_{len(st.session_state.dfs)+1}.csv")
                try:
                    f.seek(0)
                    raw = f.read()
                    size_mb = len(raw) / (1024 * 1024)
                    total_mb += size_mb
                    try:
                        # parse once to capture original columns, then sanitize
                        df_full = pd.read_csv(io.BytesIO(raw))
                        orig_cols = list(df_full.columns)
                        sanitized = [re.sub(r'[^A-Za-z0-9_]+', '_', c).strip('_') for c in orig_cols]
                        df_full.columns = sanitized
                        st.session_state.dfs[name] = df_full
                        st.session_state.colname_map[name] = list(zip(orig_cols, sanitized))
                    except Exception as e:
                        st.warning(f"Failed to parse {name}: {e}")
                except Exception as e:
                    st.warning(f"Failed to read uploaded file {name}: {e}")
            st.success(f"{len(st.session_state.dfs)} files loaded — total ~{total_mb:.1f} MB")
            load_trigger = True

elif upload_mode == "ZIP archive (upload)" and uploaded:
    if st.button("Load ZIP and extract CSVs"):
        try:
            uploaded.seek(0)
            zip_bytes = uploaded.read()
            # extract_csvs_from_zip will populate dfs and colname_map
            st.session_state.dfs = {}
            st.session_state.colname_map = {}
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
                for info in z.infolist():
                    if info.is_dir():
                        continue
                    if Path(info.filename).suffix.lower() != ".csv":
                        continue
                    with z.open(info) as f:
                        try:
                            file_bytes = f.read()
                            df_tmp = pd.read_csv(io.BytesIO(file_bytes))
                            orig_cols = list(df_tmp.columns)
                            sanitized = [re.sub(r'[^A-Za-z0-9_]+', '_', c).strip('_') for c in orig_cols]
                            df_tmp.columns = sanitized
                            st.session_state.dfs[Path(info.filename).name] = df_tmp
                            st.session_state.colname_map[Path(info.filename).name] = list(zip(orig_cols, sanitized))
                        except Exception as e:
                            st.error(f"Error extracting {info.filename} from ZIP: {e}")
            st.success(f"{len(st.session_state.dfs)} files loaded from ZIP")
            load_trigger = True
        except Exception as e:
            st.error(f"Failed to read uploaded ZIP: {e}")

elif upload_mode == "Folder (server)" and not privacy_mode:
    if folder_path and scan_folder:
        if st.button("Scan folder and load CSVs"):
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
                    st.session_state.dfs = {}
                    st.session_state.colname_map = {}
                    total_mb = 0.0
                    for fpath in csv_files:
                        try:
                            size_mb = fpath.stat().st_size / (1024 * 1024)
                            total_mb += size_mb
                            if size_mb > MAX_FILE_MB:
                                st.warning(f"Skipping {fpath.name} - too large ({size_mb:.1f} MB)")
                                continue
                            # read raw bytes so we capture original headers before sanitizing
                            raw = fpath.read_bytes()
                            df_tmp = pd.read_csv(io.BytesIO(raw))
                            orig_cols = list(df_tmp.columns)
                            sanitized = [re.sub(r'[^A-Za-z0-9_]+', '_', c).strip('_') for c in orig_cols]
                            df_tmp.columns = sanitized
                            st.session_state.dfs[fpath.name] = df_tmp
                            st.session_state.colname_map[fpath.name] = list(zip(orig_cols, sanitized))
                        except PermissionError:
                            st.warning(f"Permission denied reading {fpath}")
                        except Exception as e:
                            st.error(f"Error reading {fpath.name}: {e}")
                    st.success(f"{len(st.session_state.dfs)} files loaded from folder — total ~{total_mb:.1f} MB")
                    load_trigger = True

# If no files loaded, show message and return
dfs = st.session_state.get('dfs', {})

if not dfs:
    st.info("No valid CSV data available yet. Upload CSV(s) or choose a folder and load files.")
else:
    # File picker in main area for convenience
    st.subheader("Files")
    filenames = list(dfs.keys())

    # Quick-select controls for convenience (Select all / Clear selection)
    col_a, col_b = st.columns([1, 1])
    with col_a:
        if st.button("Select all files"):
            st.session_state['file_selector'] = filenames
    with col_b:
        if st.button("Clear selection"):
            st.session_state['file_selector'] = []

    # File picker in main area for convenience. Use a known session key so the select-all/clear buttons can update it.
    selected = st.multiselect("Select files to include (checked = included)", filenames, key='file_selector')

    # Show a compact summary of selection
    st.write(f"{len(selected)} file(s) selected")

    with st.expander(f"Detected files ({len(filenames)})", expanded=False):
        for name in filenames:
            st.write(f"- {name} ({len(dfs[name])} rows, {len(dfs[name].columns)} cols)")

    # Preview pane for a single selected file
    show_preview = st.checkbox("Show file preview", value=False)
    if show_preview:
        # default preview selection should be the first selected file if available
        preview_options = filenames
        default_index = 0
        if selected:
            try:
                default_index = preview_options.index(selected[0])
            except ValueError:
                default_index = 0
        preview_file = st.selectbox("Preview file (choose one)", options=preview_options, index=default_index)
        if preview_file:
            dfp = dfs.get(preview_file)
            if isinstance(dfp, pd.DataFrame):
                st.write(f"Preview of {preview_file} — first 10 rows")
                st.dataframe(dfp.head(10))
                st.write(f"Columns: {', '.join(map(str, dfp.columns))}")
                # show original -> sanitized mapping if available
                mapping = st.session_state.colname_map.get(preview_file)
                if mapping:
                    st.write("Column name mapping (original -> sanitized):")
                    for orig, san in mapping:
                        st.write(f"- {orig} -> {san}")
                # per-file column select convenience
                if st.button(f"Select columns from {preview_file}"):
                    st.session_state['col_selector'] = list(dfp.columns)

    # --- Export raw selected files ---
    st.markdown("---")
    st.subheader("Export raw files")
    # keep a date_suffix for summary/export defaults
    date_suffix = datetime.now().strftime("%Y%m%d")
    default_export_name = f"exported_data_{date_suffix}"
    filename = st.text_input("Output filename (without extension)", default_export_name, key='raw_filename')
    safe_filename = re.sub(r"[^A-Za-z0-9_.-]", "_", (filename or "").strip())

    if st.button("Create Excel from selected files"):
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
                            df_to_write = dfs[name]
                            try:
                                df_to_write.to_excel(writer, index=False, sheet_name=sheet_name)
                            except Exception:
                                # ensure any odd objects are converted to DataFrame
                                pd.DataFrame(df_to_write).to_excel(writer, index=False, sheet_name=sheet_name)
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

    # --- Summary calculation controls ---
    st.markdown("---")
    st.subheader("Summary statistics export")
    st.write("Choose columns and calculate summary for selected files.")

    cols_union = sorted({c for df in dfs.values() for c in df.columns}) if dfs else []
    manual_cols = []
    if col_mode.startswith("Select"):
        # allow quick select/clear for columns and store selection in session state
        if 'col_selector' not in st.session_state:
            st.session_state['col_selector'] = []
        ca, cb = st.columns([1, 1])
        with ca:
            if st.button("Select all columns"):
                st.session_state['col_selector'] = cols_union
        with cb:
            if st.button("Clear columns"):
                st.session_state['col_selector'] = []
        # use session state for the current selection (do not pass default when using the same key)
        manual_cols = st.multiselect("Select columns to include in summary (exact column names)", cols_union, key='col_selector')

    # Allow derived column definitions: user can define new columns computed from existing ones
    st.markdown("Derived columns (optional)")
    st.caption("Define one derived column per line using the format: NAME = op(col1,col2,...). Supported ops: sum, mean, median, ratio, wmean")

    # keep derived defs text in session so example buttons can append templates
    if 'derived_defs_text' not in st.session_state:
        st.session_state['derived_defs_text'] = ''

    ex1, ex2, ex3, ex4 = st.columns([1,1,1,1])
    with ex1:
        if st.button("Example: TOTAL = sum(A,B)"):
            cur = st.session_state.get('derived_defs_text','')
            add = "TOTAL = sum(A,B)"
            st.session_state['derived_defs_text'] = (cur + "\n" + add).strip()
    with ex2:
        if st.button("Example: AVG_AB = mean(A,B)"):
            cur = st.session_state.get('derived_defs_text','')
            add = "AVG_AB = mean(A,B)"
            st.session_state['derived_defs_text'] = (cur + "\n" + add).strip()
    with ex3:
        if st.button("Example: W = wmean(A:0.2,B:0.8)"):
            cur = st.session_state.get('derived_defs_text','')
            add = "W = wmean(A:0.2,B:0.8)"
            st.session_state['derived_defs_text'] = (cur + "\n" + add).strip()
    with ex4:
        if st.button("Example: R = ratio(sum(A,B), C)"):
            cur = st.session_state.get('derived_defs_text','')
            add = "R = ratio(sum(A,B), C)"
            st.session_state['derived_defs_text'] = (cur + "\n" + add).strip()

    # use session state for derived defs text (do not pass value when using the same key)
    derived_defs_text = st.text_area("Derived column definitions (one per line)", height=120, key='derived_defs_text')
    derived_defs, derived_warnings = parse_derived_definitions(derived_defs_text)
    if derived_warnings:
        for w in derived_warnings:
            st.warning(w)
    if derived_defs:
        st.write("Parsed derived columns:")
        for d in derived_defs:
            try:
                expr_str = expr_to_str(d.get('expr'))
            except Exception:
                expr_str = '<invalid>'
            st.write(f"- {d.get('name')} = {expr_str}")

    # add preview-evaluate button to show derived results for currently previewed file
    if show_preview and preview_file and derived_defs:
        if st.button("Preview derived columns on preview file"):
            dfp = dfs.get(preview_file)
            preview_cols = {}
            eval_errors = []
            if dfp is not None:
                for d in derived_defs:
                    try:
                        series = evaluate_expr(d.get('expr'), dfp)
                        if not isinstance(series, pd.Series):
                            series = pd.Series(series, index=dfp.index)
                        preview_cols[d['name']] = series
                    except Exception as e:
                        eval_errors.append(f"Failed to evaluate {d.get('name')} on {preview_file}: {e}")
                if preview_cols:
                    outdf = pd.DataFrame(preview_cols)
                    st.write(f"Preview of derived columns for {preview_file} (first 10 rows)")
                    st.dataframe(outdf.head(10))
                if eval_errors:
                    for err in eval_errors:
                        st.error(err)

    var_defaults = {
        'EELI': 'eeli',
        'CoV(H)': 'cov(h),covh,cov_h,cov h',
        'CoV(V)': 'cov(v),covv,cov_v,cov v',
    }
    var_choices = list(var_defaults.keys())
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

    def process_with_options(dfs_local: Dict[str, pd.DataFrame], chosen_files: List[str], manual_cols: List[str], derived_defs: List[Dict]):
        records = []
        if not manual_cols and not derived_defs:
            return pd.DataFrame.from_records(records)

        for name in chosen_files:
            df_orig = dfs_local.get(name)
            if df_orig is None:
                continue

            # Work on a copy so stored dfs are not mutated
            try:
                df = df_orig.copy()
            except Exception:
                df = pd.DataFrame(df_orig)

            # Apply derived column definitions (if any) by evaluating ASTs
            for d in derived_defs:
                try:
                    series = evaluate_expr(d.get('expr'), df)
                    # if the result is a scalar or list-like, ensure it's a Series aligned with df
                    if not isinstance(series, pd.Series):
                        series = pd.Series(series, index=df.index)
                    df[d['name']] = series
                except Exception as e:
                    st.warning(f"Failed to compute derived column {d.get('name')} for file {name}: {e}")
                    continue

            meta = parse_filename(name)
            group = get_group_for(meta.get('individual','')) or meta.get('treatment','')

            # Combine manual columns and derived column names for processing
            cols_to_process = list(dict.fromkeys(list(manual_cols) + [d['name'] for d in derived_defs]))

            for col in cols_to_process:
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

        return pd.DataFrame.from_records(records)

    # Summary is only recalculated when user requests it
    if 'all_summary' not in st.session_state:
        st.session_state.all_summary = pd.DataFrame()

    # Only show calculation button when prerequisites are met to reduce accidental clicks
    if not selected:
        st.info("Select one or more files to enable summary calculation.")
    elif not manual_cols and not derived_defs:
        st.info("No columns selected for summary. Choose columns in 'Select columns manually' mode or define derived columns.")
    else:
        if st.button("Calculate summary for selected files"):
            if not selected:
                st.error("No files selected for summary calculation.")
            elif not manual_cols and not derived_defs:
                st.error("No columns selected for summary. Choose columns in 'Select columns manually' mode or define derived columns.")
            else:
                with st.spinner("Processing files for summary statistics..."):
                    try:
                        all_summary = process_with_options(dfs, selected, manual_cols, derived_defs)
                        st.session_state.all_summary = all_summary
                        st.success("Summary statistics calculated")
                    except Exception as e:
                        st.exception(e)
                        st.error(f"Error processing files: {e}")

    if not st.session_state.all_summary.empty:
        st.markdown("### Summary preview")
        summary_df = st.session_state.all_summary
        st.dataframe(summary_df.head(200))

        # Export options for summary
        st.write("Export summary:")
        default_summary = f"summary_statistics_{date_suffix}"
        summary_name = st.text_input("Summary filename (without extension)", default_summary)
        safe_summary = re.sub(r"[^A-Za-z0-9_.-]", "_", (summary_name or "").strip())
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Export summary to Excel"):
                try:
                    with st.spinner("Creating summary Excel workbook..."):
                        export_bytes = export_summary_excel(summary_df)
                    st.success("Summary export ready — click the download button below")
                    st.download_button("Download summary XLSX", data=export_bytes, file_name=f"{safe_summary}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                except Exception as e:
                    st.error(f"Failed to export summary statistics: {e}")
        with col2:
            if st.button("Export summary to CSV"):
                try:
                    csv_bytes = summary_df.to_csv(index=False).encode('utf-8')
                    st.success("CSV ready — click the download button below")
                    st.download_button("Download summary CSV", data=csv_bytes, file_name=f"{safe_summary}.csv", mime="text/csv")
                except Exception as e:
                    st.error(f"Failed to export summary CSV: {e}")

    # Utility actions
    st.markdown("---")
    if st.button("Clear loaded files"):
        st.session_state.dfs = {}
        st.session_state.all_summary = pd.DataFrame()
        # Try to trigger a rerun; fall back gracefully if the Streamlit version lacks experimental_rerun
        try:
            st.experimental_rerun()
        except AttributeError:
            # Some Streamlit builds do not expose experimental_rerun. Toggle a session-state key
            # to force Streamlit to notice a state change which typically causes a rerun.
            try:
                st.session_state['_rerun_toggle'] = not st.session_state.get('_rerun_toggle', False)
            except Exception:
                # As a last resort, inform the user — do not raise an exception
                st.info("Files cleared. Please refresh the app to see the changes.")

    # add UI: column name mappings so users can map equivalent names across files
    st.markdown("Column name mappings (optional)")
    st.caption("Define mappings so different column names from different files are treated as the same. Use the format: CANONICAL = alias1, alias2, ...\nExample: EELI = eeli, EEL_I")
    if 'col_mappings_text' not in st.session_state:
        st.session_state['col_mappings_text'] = ''

    ex_map, ex_map2 = st.columns([1,3])
    with ex_map:
        if st.button("Example mapping: EELI = eeli, EEL_I"):
            cur = st.session_state.get('col_mappings_text','')
            add = "EELI = eeli, EEL_I"
            st.session_state['col_mappings_text'] = (cur + "\n" + add).strip()
    with ex_map2:
        st.write("")

    col_mappings_text = st.text_area("Column mappings (one per line)", key='col_mappings_text', height=100)

    def parse_col_mappings(text: str):
        out = {}
        if not text:
            return out
        for lineno, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            m = re.match(r'^\s*([^=]+?)\s*=\s*(.+)$', line)
            if not m:
                continue
            canon = m.group(1).strip()
            rhs = m.group(2).strip()
            aliases = [a.strip() for a in re.split(r'[,;]\\s*|[,\\s]+' , rhs) if a.strip()]
            aliases = {a for a in aliases}
            aliases.add(canon)
            out[canon] = aliases
        return out

    # parsed mappings and reverse lookup (alias_lower -> canonical)
    user_col_mappings = parse_col_mappings(st.session_state.get('col_mappings_text',''))
    _col_alias_to_canon = {}
    for canon, aliases in user_col_mappings.items():
        for a in aliases:
            _col_alias_to_canon[a.lower()] = canon

    # --- apply column mappings to loaded data ---
    if st.button("Apply column mappings to loaded data"):
        if not dfs:
            st.warning("No data loaded. Please load CSV files or a ZIP archive first.")
        else:
            # collect all columns needing mapping
            all_cols = set()
            for df in dfs.values():
                all_cols.update(df.columns)
            all_cols = sorted(list(all_cols))

            # preview: which columns will be affected?
            preview_map = {}
            for canon, aliases in user_col_mappings.items():
                matched = [a for a in aliases if a in all_cols]
                if matched:
                    preview_map[canon] = matched

            st.write("Columns affected by mappings:")
            if not preview_map:
                st.write("No columns will be affected by the current mappings.")
            else:
                for canon, matched in preview_map.items():
                    st.write(f"- {canon} = {', '.join(matched)}")

            # confirm and apply mappings
            if st.button("Confirm and apply mappings"):
                # remap columns in each DataFrame
                for name, df in dfs.items():
                    try:
                        # create a mapping from current column names to new names
                        # apply user mappings: for each canonical name, find aliases present in this df
                        for canon, aliases in user_col_mappings.items():
                            # sanitize canonical to match sanitized column style
                            san_canon = re.sub(r'[^A-Za-z0-9_]+', '_', canon).strip('_')
                            # ensure we don't inadvertently overwrite derived columns
                            for alias in sorted(list(aliases)):
                                # try to find a matching column in df for this alias
                                matched = find_column(df, [alias, canon])
                                if not matched:
                                    # also try case-insensitive exact match
                                    for c in df.columns:
                                        if c.lower() == alias.lower() or c.lower() == san_canon.lower():
                                            matched = c
                                            break
                                if not matched:
                                    continue
                                if matched == san_canon or matched == canon:
                                    # already has canonical column
                                    continue
                                # if canonical column not present, create it; otherwise merge
                                target = san_canon if san_canon in df.columns else canon if canon in df.columns else san_canon
                                # create/merge preserving existing values (prefer existing canonical non-null)
                                try:
                                    if target not in df.columns:
                                        df[target] = df[matched]
                                    else:
                                        df[target] = df[target].fillna(df[matched])
                                except Exception:
                                    # fallback: assign directly
                                    df[target] = df[matched]
                                # drop the alias column if it's different from the target
                                if matched != target and matched in df.columns:
                                    try:
                                        df = df.drop(columns=[matched])
                                    except Exception:
                                        pass
                                # update column name mapping if available
                                cmap = st.session_state.get('colname_map', {}).get(name, [])
                                new_cmap = []
                                for orig, san in cmap:
                                    if san == matched:
                                        new_cmap.append((orig, target))
                                    else:
                                        new_cmap.append((orig, san))
                                if name in st.session_state.get('colname_map', {}):
                                    st.session_state['colname_map'][name] = new_cmap
                        # ensure updated dataframe stored back
                        st.session_state['dfs'][name] = df
                    except Exception as e:
                        st.warning(f"Failed to apply mappings to {name}: {e}")
                st.success("Column mappings applied to loaded data")
                # refresh local dfs reference
                dfs = st.session_state.get('dfs', {})
