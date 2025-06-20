import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO

st.set_page_config(page_title="Дашборд форматов", layout="wide")

@st.cache_data
def load_df(file):
    """Load CSV or Excel and normalize column names."""
    name = file.name.lower()
    if name.endswith('.csv'):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file, header=0)
    df.columns = df.columns.str.strip().str.lower()
    col_map = {}
    for col in df.columns:
        if 'категор' in col:
            col_map[col] = 'Категория'
        elif 'недел' in col:
            col_map[col] = 'Неделя'
        elif 'день' in col or 'dayofweek' in col:
            col_map[col] = 'DayOfWeek'
        elif 'доля' in col:
            col_map[col] = 'Доля списаний и ЗЦ'
        elif 'выруч' in col:
            col_map[col] = 'Выручка'
    return df.rename(columns=col_map)

@st.cache_data
def prepare_data(files):
    df_share = df_rev = None
    for f in files:
        df = load_df(f)
        if 'Доля списаний и ЗЦ' in df.columns:
            df_share = df[['Категория','Неделя','DayOfWeek','Доля списаний и ЗЦ']].copy()
        elif 'Выручка' in df.columns:
            df_rev = df[['Категория','Неделя','DayOfWeek','Выручка']].copy()
    if df_share is None or df_rev is None:
        raise ValueError("Нужно два файла: один с процентами, другой с выручкой")
    df_share['Доля списаний и ЗЦ'] = pd.to_numeric(
        df_share['Доля списаний и ЗЦ']
          .astype(str).str.replace(',', '.').str.rstrip('%'),
        errors='coerce'
    ).fillna(0)
    df_rev['Выручка'] = pd.to_numeric(df_rev['Выручка'], errors='coerce').fillna(0)
    return pd.merge(df_share, df_rev, on=['Категория','Неделя','DayOfWeek'], how='inner')

def display_format_section(title, files):
    st.header(title)
    if len(files) != 2:
        st.info("Загрузите два файла: один с процентами и один с выручкой")
        return
    try:
        df = prepare_data(files)
    except Exception as e:
        st.error(f"Ошибка при подготовке: {e}")
        return

    df['avg_rev'] = df.groupby('Неделя')['Выручка'].transform('mean')
    df['rev_pct'] = df['Выручка'] / df['avg_rev'] * 100

    cats  = st.multiselect(f"Категории ({title})", sorted(df['Категория'].unique()), default=None)
    weeks = st.multiselect(f"Недели ({title})",    sorted(df['Неделя'].unique()),     default=None)
    if cats:  df = df[df['Категория'].isin(cats)]
    if weeks: df = df[df['Неделя'].isin(weeks)]

    share_thr = st.slider(f"Порог Доли списаний % ({title})", 0.0, 100.0, 20.0)
    rev_thr   = st.slider(f"Мин. % выручки от среднего ({title})", 0.0, 200.0, 80.0)

    # pivot and flatten
    pivot = df.pivot_table(
        index=['Неделя','DayOfWeek'], columns='Категория',
        values=['Доля списаний и ЗЦ','Выручка','rev_pct']
    )
    # Flatten column MultiIndex
    flat_cols = [f"{val}_{cat}" for val,cat in pivot.columns]
    pivot_flat = pivot.copy()
    pivot_flat.columns = flat_cols

    # Identify subsets for styling
    waste_cols   = [c for c in flat_cols if c.startswith('Доля списаний и ЗЦ_')]
    rev_pct_cols = [c for c in flat_cols if c.startswith('rev_pct_')]

    styled = pivot_flat.style.format('{:.1f}') \
        .applymap(lambda v: 'background-color: tomato', subset=waste_cols) \
        .applymap(lambda v: 'background-color: tomato', subset=rev_pct_cols)
    st.subheader("Таблица")
    st.dataframe(styled, use_container_width=True)

    st.subheader("График выручки")
    fig1 = px.line(df, x='DayOfWeek', y='Выручка', color='Категория',
                   line_group='Неделя', markers=True,
                   labels={'DayOfWeek':'День недели'})
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("График доли списаний")
    fig2 = px.line(df, x='DayOfWeek', y='Доля списаний и ЗЦ', color='Категория',
                   line_group='Неделя', markers=True,
                   labels={'DayOfWeek':'День недели'})
    st.plotly_chart(fig2, use_container_width=True)

    buf = BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name=title, index=False)
    buf.seek(0)
    st.download_button(f"Скачать Excel ({title})", buf,
                       f"{title.replace(' ','_')}.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

def main():
    st.title("Дашборд форматов: Жук и Тест")
    juk_files  = st.file_uploader("Жук: файлы процент и выручка", accept_multiple_files=True, type=['csv','xlsx'], key='juk')
    test_files = st.file_uploader("Тест: файлы процент и выручка", accept_multiple_files=True, type=['csv','xlsx'], key='test')

    display_format_section("Формат Жук", juk_files)
    st.markdown("---")
    display_format_section("Тестовый формат", test_files)

if __name__ == "__main__":
    main()
