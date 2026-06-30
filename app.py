"""Mozi Wash Sentiment Analysis Dashboard."""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta, date
import os

st.set_page_config(page_title="Mozi Wash · Sentiment Analysis", layout="wide")

SCENT_NAMES = {
    'AW': 'Alpine Woods', 'CC': 'Central Coast', 'CZ': 'Signature Cozy',
    'DP': 'Desert Poppy', 'FC': 'Free & Clear', 'GH': 'Golden Hour',
    'HR': 'Hollywood Rouge', 'MM': 'Malibu Mornings', 'SD': 'Sugar Dew', 'VM': 'Vanilla Moon',
}
SCENT_ORDER = ['AW','CC','CZ','DP','FC','GH','HR','MM','SD','VM']
SCENT_COLORS = {
    'AW': '#4F7942', 'CC': '#4F86C6', 'CZ': '#C8A96E', 'DP': '#E8734A',
    'FC': '#88BBAA', 'GH': '#F2C94C', 'HR': '#9B2335', 'MM': '#6BAED6',
    'SD': '#F4A460', 'VM': '#9B59B6',
}

# ── Password gate ─────────────────────────────────────────────────────────────
def check_password():
    if st.session_state.get('authenticated'):
        return True
    st.markdown("""
    <style>
    .login-wrap {
        max-width: 380px; margin: 80px auto; padding: 40px;
        background: #fff; border-radius: 12px;
        box-shadow: 0 4px 24px rgba(0,0,0,.08);
        text-align: center;
    }
    .login-title { font-size: 22px; font-weight: 700; color: #1C1C2E; margin-bottom: 4px; }
    .login-sub   { font-size: 13px; color: #888; margin-bottom: 28px; }
    </style>
    <div class="login-wrap">
      <div class="login-title">Mozi Wash</div>
      <div class="login-sub">Sentiment Analysis Dashboard</div>
    </div>
    """, unsafe_allow_html=True)
    pwd = st.text_input("Password", type="password", key="login_pwd")
    if st.button("Login", type="primary", use_container_width=False):
        if pwd == st.secrets["APP_PASSWORD"]:
            st.session_state['authenticated'] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False

if not check_password():
    st.stop()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
* { box-sizing: border-box; }
.dash-header {
    background: #1C1C2E; color: #fff; padding: 20px 32px;
    display: flex; justify-content: space-between; align-items: center;
    margin: -1rem -1rem 1.5rem -1rem;
}
.dash-header h1 { font-size: 20px; font-weight: 700; margin: 0; }
.dash-header h1 span { color: #C8A96E; }
.dash-meta { font-size: 12px; color: #9999BB; }
.section-card {
    background: #fff; border: 1px solid #E8E4DF; border-radius: 10px;
    padding: 24px; margin-bottom: 20px;
}
.section-title { font-size: 17px; font-weight: 700; color: #1C1C2E; margin-bottom: 4px; }
.section-sub   { font-size: 13px; color: #888; margin-bottom: 16px; }
</style>
""", unsafe_allow_html=True)

run_date = datetime.now().strftime('%B %d, %Y')
st.markdown(f"""
<div class="dash-header">
  <h1>Mozi Wash · <span>Sentiment Analysis</span></h1>
  <div class="dash-meta">Updated {run_date}</div>
</div>
""", unsafe_allow_html=True)

# ── Data loaders ──────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

@st.cache_data(ttl=3600)
def load_okendo():
    path = os.path.join(DATA_DIR, 'okendo_reviews.csv')
    df = pd.read_csv(path, parse_dates=['date'])
    df['sku'] = df['sku'].str.strip().str.upper()
    df['scent_name'] = df['scent_name'].fillna(df['sku'].map(SCENT_NAMES))
    return df

@st.cache_data(ttl=3600)
def load_nps():
    path = os.path.join(DATA_DIR, 'nps_omniconvert.csv')
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df['sku'] = df['sku'].str.strip().str.upper()
    return df

@st.cache_data(ttl=3600)
def load_nps_individual():
    path = os.path.join(DATA_DIR, 'nps_individual.csv')
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=['date'])
    df['sku'] = df['sku'].str.strip().str.upper()
    df['survey_type'] = df['survey_type'].astype(str)
    return df

@st.cache_data(ttl=3600)
def load_nps_po_historical():
    path = os.path.join(DATA_DIR, 'nps_po_historical.csv')
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df['sku'] = df['sku'].str.strip().str.upper()
    df['survey_type_days'] = df['survey_type_days'].astype(str)
    return df

df_ok       = load_okendo()
df_nps      = load_nps()
df_indiv    = load_nps_individual()
df_hist_nps = load_nps_po_historical()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["By Scent", "By Formula (PO Batch)"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: BY SCENT
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">By Scent</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Star ratings and attribute scores by scent, filterable by time period. NPS from Omniconvert.</div>', unsafe_allow_html=True)

    # Time period filter
    period_opts = ['All Time','This Week','This Month','Last 30 Days','Last 90 Days','Custom Range']
    col_p, col_custom1, col_custom2 = st.columns([2, 1.5, 1.5])
    with col_p:
        period = st.selectbox("Time Period", period_opts, index=0, key='period')

    today = date.today()
    if period == 'All Time':
        date_from = date(2020, 1, 1)
        date_to   = today
    elif period == 'This Week':
        date_from = today - timedelta(days=today.weekday())
        date_to   = today
    elif period == 'This Month':
        date_from = today.replace(day=1)
        date_to   = today
    elif period == 'Last 30 Days':
        date_from = today - timedelta(days=30)
        date_to   = today
    elif period == 'Last 90 Days':
        date_from = today - timedelta(days=90)
        date_to   = today
    else:  # Custom Range
        with col_custom1:
            date_from = st.date_input("From", value=today - timedelta(days=30), key='d_from')
        with col_custom2:
            date_to = st.date_input("To", value=today, key='d_to')

    # Filter Okendo data
    mask = (df_ok['date'].dt.date >= date_from) & (df_ok['date'].dt.date <= date_to)
    df_f = df_ok[mask]

    # Build summary table
    summary_rows = []
    for sku in SCENT_ORDER:
        sub = df_f[df_f['sku'] == sku]
        n   = len(sub)
        avg_rating = sub['rating'].mean() if n else None
        avg_ss     = sub['scent_strength'].mean() if sub['scent_strength'].notna().any() else None
        avg_sa     = sub['scent_appeal'].mean()   if sub['scent_appeal'].notna().any()   else None
        avg_cp     = sub['cleaning_power'].mean() if sub['cleaning_power'].notna().any() else None
        avg_cap    = sub['cap_pouring'].mean()     if sub['cap_pouring'].notna().any()    else None

        # NPS — pick column based on period
        nps_10, nps_10n, nps_40, nps_40n = None, 0, None, 0
        if not df_nps.empty:
            for days, nps_ref, nps_ref_n in [(10, 'nps_10', 'nps_10n'), (40, 'nps_40', 'nps_40n')]:
                sub_nps = df_nps[(df_nps['sku'] == sku) & (df_nps['survey_type_days'].astype(str) == str(days))]
                if sub_nps.empty:
                    continue
                # Use most recent row
                latest = sub_nps.sort_values('week_end_sunday', ascending=False).iloc[0]
                if period in ('All Time',):
                    val = latest.get('ytd_nps')
                    cnt = latest.get('ytd_responses', 0)
                elif period in ('Last 30 Days', 'Last 90 Days'):
                    val = latest.get('last_5_weeks_nps')
                    cnt = latest.get('last_5_weeks_responses', 0)
                else:
                    val = latest.get('nps')
                    cnt = latest.get('responses_count', 0)
                try:
                    val = float(val) if val is not None and str(val) not in ('', 'nan') else None
                    cnt = int(float(cnt)) if cnt else 0
                except (ValueError, TypeError):
                    val, cnt = None, 0
                if days == 10:
                    nps_10, nps_10n = val, cnt
                else:
                    nps_40, nps_40n = val, cnt

        summary_rows.append({
            'SKU': sku,
            'Scent': SCENT_NAMES.get(sku, sku),
            '# Reviews': n,
            'Avg Rating': round(avg_rating, 2) if avg_rating else None,
            'Scent Strength': round(avg_ss, 2) if avg_ss is not None else None,
            'Scent Appeal': round(avg_sa, 2) if avg_sa is not None else None,
            'Cleaning Power': round(avg_cp, 2) if avg_cp is not None else None,
            'Cap/Pouring': round(avg_cap, 2) if avg_cap is not None else None,
            'NPS 10-Day': nps_10,
            'NPS 10-Day n': nps_10n,
            'NPS 40-Day': nps_40,
            'NPS 40-Day n': nps_40n,
        })

    df_sum = pd.DataFrame(summary_rows)

    def color_rating(val):
        if val is None or (isinstance(val, float) and val != val): return ''
        if val >= 4.0: return 'color: #27AE60; font-weight: 600'
        if val >= 3.0: return 'color: #E67E22; font-weight: 600'
        return 'color: #E74C3C; font-weight: 600'

    def color_ss(val):
        if val is None or (isinstance(val, float) and val != val): return ''
        av = abs(val)
        if av <= 0.25: return 'color: #27AE60; font-weight: 600'   # -0.24 to +0.25 → green
        if av <= 0.50: return 'color: #D4AC0D; font-weight: 600'   # ±0.25–0.50 → yellow
        return 'color: #E74C3C; font-weight: 600'                  # ±0.51–1.0 → red

    def color_attr(val):
        if val is None or (isinstance(val, float) and val != val): return ''
        if val >= 4.0: return 'color: #27AE60; font-weight: 600'
        if val >= 3.0: return 'color: #E67E22; font-weight: 600'
        return 'color: #E74C3C; font-weight: 600'

    def color_nps(val):
        if val is None or (isinstance(val, float) and val != val): return ''
        if val >= 40: return 'color: #27AE60; font-weight: 600'
        if val >= 0:  return 'color: #E67E22; font-weight: 600'
        return 'color: #E74C3C; font-weight: 600'

    def fmt_nps(row, col, n_col):
        val = row[col]
        n   = row[n_col]
        if val is None or (isinstance(val, float) and val != val): return '—'
        sign = '+' if val >= 0 else ''
        n_str = f' (n={int(n)})' if n else ''
        return f'{sign}{int(round(val))}{n_str}'

    # Display table with custom styling
    display_cols = ['Scent','# Reviews','Avg Rating','Scent Strength',
                    'Scent Appeal','Cleaning Power','Cap/Pouring','NPS 10-Day','NPS 40-Day']

    df_display = df_sum.copy()
    df_display['NPS 10-Day'] = df_sum.apply(lambda r: fmt_nps(r,'NPS 10-Day','NPS 10-Day n'), axis=1)
    df_display['NPS 40-Day'] = df_sum.apply(lambda r: fmt_nps(r,'NPS 40-Day','NPS 40-Day n'), axis=1)

    def style_table(df_s):
        styles = pd.DataFrame('', index=df_s.index, columns=df_s.columns)
        for i, row in df_s.iterrows():
            styles.at[i, 'Avg Rating']     = color_rating(row['Avg Rating'])
            styles.at[i, 'Scent Strength'] = color_ss(row['Scent Strength'])
            styles.at[i, 'Scent Appeal']   = color_attr(row['Scent Appeal'])
            styles.at[i, 'Cleaning Power'] = color_attr(row['Cleaning Power'])
            styles.at[i, 'Cap/Pouring']    = color_attr(row['Cap/Pouring'])
            # NPS styling — parse back from formatted string
            for col in ['NPS 10-Day','NPS 40-Day']:
                val_str = str(row[col])
                if val_str and val_str != '—':
                    try:
                        num = float(val_str.split('(')[0].replace('+',''))
                        styles.at[i, col] = color_nps(num)
                    except ValueError:
                        pass
        return styles

    styled = (
        df_display[display_cols]
        .style
        .apply(style_table, axis=None)
        .format({
            'Avg Rating':     lambda x: f'{x:.2f} ★' if x is not None and x == x else '—',
            'Scent Strength': lambda x: f'{x:+.2f}' if x is not None and x == x else '—',
            'Scent Appeal':   lambda x: f'{x:.2f}' if x is not None and x == x else '—',
            'Cleaning Power': lambda x: f'{x:.2f}' if x is not None and x == x else '—',
            'Cap/Pouring':    lambda x: f'{x:.2f}' if x is not None and x == x else '—',
        }, na_rep='—')
        .set_properties(**{'text-align': 'center'})
        .set_properties(subset=['Scent'], **{'text-align': 'left', 'font-weight': '600'})
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Charts
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Rating Trends Over Time</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Weekly average star rating per scent. Toggle SKUs using the legend.</div>', unsafe_allow_html=True)

    # Weekly aggregation
    df_ok_filt = df_ok[(df_ok['date'].dt.date >= date_from) & (df_ok['date'].dt.date <= date_to)].copy()
    df_ok_filt['week'] = df_ok_filt['date'].dt.to_period('W').apply(lambda p: p.start_time)
    weekly = (df_ok_filt.groupby(['week','sku'])
              .agg(avg_rating=('rating','mean'), n=('rating','count'))
              .reset_index())
    weekly = weekly[weekly['n'] >= 2]  # only show weeks with 2+ reviews
    weekly['scent'] = weekly['sku'].map(SCENT_NAMES)

    fig_rating = px.line(
        weekly, x='week', y='avg_rating', color='sku',
        color_discrete_map=SCENT_COLORS,
        labels={'week':'Week','avg_rating':'Avg Rating','sku':'SKU'},
        markers=True,
    )
    fig_rating.update_layout(
        yaxis=dict(range=[1,5.2], title='Avg Rating ★'),
        xaxis_title='', legend_title='SKU',
        plot_bgcolor='white', paper_bgcolor='white',
        hovermode='x unified',
        margin=dict(l=0,r=0,t=10,b=0),
    )
    fig_rating.update_traces(line=dict(width=2), marker=dict(size=5))
    st.plotly_chart(fig_rating, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Scent Strength trend
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Scent Strength Trends</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Weekly average Scent Strength per scent. 0 = perfect, negative = too weak, positive = too strong.</div>', unsafe_allow_html=True)

    weekly_ss = (df_ok_filt[df_ok_filt['scent_strength'].notna()]
                 .groupby(['week','sku'])
                 .agg(avg_ss=('scent_strength','mean'), n=('scent_strength','count'))
                 .reset_index())
    weekly_ss = weekly_ss[weekly_ss['n'] >= 2]

    fig_ss = px.line(
        weekly_ss, x='week', y='avg_ss', color='sku',
        color_discrete_map=SCENT_COLORS,
        labels={'week':'Week','avg_ss':'Scent Strength','sku':'SKU'},
        markers=True,
    )
    fig_ss.add_hline(y=0, line_dash='dash', line_color='#888', opacity=0.5, annotation_text='Perfect (0)')
    fig_ss.add_hrect(y0=-0.25, y1=0.25, fillcolor='#27AE60', opacity=0.05, line_width=0)
    fig_ss.update_layout(
        yaxis=dict(range=[-1.1,1.1], title='Scent Strength'),
        xaxis_title='', legend_title='SKU',
        plot_bgcolor='white', paper_bgcolor='white',
        hovermode='x unified',
        margin=dict(l=0,r=0,t=10,b=0),
    )
    fig_ss.update_traces(line=dict(width=2), marker=dict(size=5))
    st.plotly_chart(fig_ss, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: BY FORMULA
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">By Formula (PO Batch)</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Star ratings and NPS by production batch. NPS from Omniconvert (current batch) and Survicate (historical).</div>', unsafe_allow_html=True)

    # PO batch definitions
    PO_BATCHES = [
        ('Jan 18 Stock',                    ['AW','CC','CZ','DP','FC','GH','HR','MM','SD','VM']),
        ('March Order - 6 Degrees',         ['AW','CC','CZ','DP','GH','HR','SD','VM']),
        ('January Order - Product Society', ['DP','MM']),
    ]
    PO_DATE_RANGES = {
        'Jan 18 Stock':                    (date(2025,10,1), date(2026,3,14)),
        'March Order - 6 Degrees':         (date(2026,3,15), date(2099,1,1)),
        'January Order - Product Society': (date(2026,1,1), date(2026,3,14)),
    }

    # ── NPS lookups ───────────────────────────────────────────────────────────
    # Live: compute NPS from Omniconvert individual responses, assigning PO by approx order date
    def _assign_po(row):
        try:
            od = (pd.to_datetime(row['date']) - pd.Timedelta(days=int(row['survey_type']))).date()
        except Exception:
            return None
        for po_name, (d_from, d_to) in PO_DATE_RANGES.items():
            if d_from <= od <= d_to:
                return po_name
        return None

    nps_live = {}
    if not df_indiv.empty:
        dfi = df_indiv.copy()
        dfi['po_name'] = dfi.apply(_assign_po, axis=1)
        dfi = dfi.dropna(subset=['po_name'])
        for (po, sku, stype), grp in dfi.groupby(['po_name','sku','survey_type']):
            scores = grp['score'].dropna().astype(int).tolist()
            if not scores:
                continue
            promoters  = sum(1 for s in scores if s >= 9)
            detractors = sum(1 for s in scores if s <= 6)
            nps_live[(po, sku, str(stype))] = (
                round((promoters - detractors) / len(scores) * 100, 1),
                len(scores),
            )

    # Historical: pre-aggregated Survicate NPS from nps_po_historical.csv
    nps_hist = {}
    if not df_hist_nps.empty:
        for _, r in df_hist_nps.iterrows():
            nps_hist[(r['po_name'], r['sku'], str(r['survey_type_days']))] = (
                float(r['nps']), int(r['n_responses'])
            )

    def get_nps(po_name, sku, stype):
        """Return (nps_val, n) — live data takes priority over historical."""
        key = (po_name, sku, stype)
        if key in nps_live:
            return nps_live[key]
        if key in nps_hist:
            return nps_hist[key]
        return None, 0

    def fmt_nps(val, n):
        if val is None:
            return '—'
        return f'{int(val):+d} (n={n})'

    def color_nps_val(val):
        if val is None:
            return 'color: #BBB; font-style: italic'
        if val >= 40:
            return 'color: #27AE60; font-weight: 600'
        if val >= 0:
            return 'color: #E67E22; font-weight: 600'
        return 'color: #E74C3C; font-weight: 600'

    # ── Build table ───────────────────────────────────────────────────────────
    po_rows = []
    for po_name, skus in PO_BATCHES:
        d_from_po, d_to_po = PO_DATE_RANGES[po_name]
        for sku in skus:
            sub = df_ok[
                (df_ok['sku'] == sku) &
                (df_ok['date'].dt.date >= d_from_po) &
                (df_ok['date'].dt.date <= d_to_po)
            ]
            n_rev = len(sub)
            avg_r = round(sub['rating'].mean(), 2) if n_rev else None
            nps10_v, n10 = get_nps(po_name, sku, '10')
            nps40_v, n40 = get_nps(po_name, sku, '40')
            po_rows.append({
                'PO Batch':   po_name,
                'Scent':      SCENT_NAMES.get(sku, sku),
                '# Reviews':  n_rev,
                'Avg Rating': f'{avg_r:.2f} ★' if avg_r else '—',
                'NPS 10-Day': fmt_nps(nps10_v, n10),
                'NPS 40-Day': fmt_nps(nps40_v, n40),
                '_nps10':     nps10_v,
                '_nps40':     nps40_v,
            })

    df_po = pd.DataFrame(po_rows)
    display_cols = ['PO Batch','Scent','# Reviews','Avg Rating','NPS 10-Day','NPS 40-Day']

    def style_po(df_s):
        styles = pd.DataFrame('', index=df_s.index, columns=df_s.columns)
        for i in df_s.index:
            val_str = str(df_s.at[i, 'Avg Rating']).replace('★','').strip()
            try:
                styles.at[i, 'Avg Rating'] = color_rating(float(val_str))
            except ValueError:
                pass
            styles.at[i, 'NPS 10-Day'] = color_nps_val(df_po.at[i, '_nps10'])
            styles.at[i, 'NPS 40-Day'] = color_nps_val(df_po.at[i, '_nps40'])
        return styles

    st.dataframe(
        df_po[display_cols].style.apply(style_po, axis=None)
        .set_properties(**{'text-align': 'center'})
        .set_properties(subset=['PO Batch','Scent'], **{'text-align': 'left'}),
        use_container_width=True, hide_index=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)
