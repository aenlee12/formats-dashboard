import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO

st.set_page_config(page_title="Дашборд форматов", layout="wide")

@st.cache_data
def load_df(file):
    """Load CSV or Excel, normalize column names."""
    name = file.name.lower()
    if name.endswith('.csv'):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file, header=0)  # first row is header
    # Standardize column names
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
            col_map[col] = 'Доля сысп'
        elif 'выруч' in col:
            col_map[col] = 'Выручка'
    return df.rename(columns=col_map)

@st.cache_data
def prepare_data(files):
    df_share = None
    df_rev = None
    for f in files:
        df = load_df(f)
        if 'Доля сысп' in df.columns:
            df_share = df[['Категория','Неделя','DayOfWeek','Доля сысп']].copy()
        elif 'Выручка' in df.columns:
            df_rev = df[['Категория','Неделя','DayOfWeek','Выручка']].copy()
    if df_share is None or df_rev is None:
        raise ValueError("Нужно два файла: один с процентами, другой с выручкой")
    # Clean share column: convert to numeric
    s = df_share['Доля сысп'].astype(str).str.replace(',', '.').str.rstrip('%')
    df_share['Доля списаний и ЗЦ'] = pd.to_numeric(s, errors='coerce').fillna(0)
    # Clean revenue
    df_rev['Выручка'] = pd.to_numeric(df_rev['Выручка'], errors='coerce').fillna(0)
    # Merge
    dfm = pd.merge(df_share.drop(columns=['Доля сысп']), df_rev,
                   on=['Категория','Неделя','DayOfWeek'], how='inner')
    return dfm

def display_format_section(title, files):
    st.header(title)
    if len(files) != 2:
        st.info("Загрузите два файла: процент и выручку")
        return
    try:
        df = prepare_data(files)
    except Exception as e:
        st.error(f"Ошибка при подготовке: {e}")
        return

    # Relative revenue
    df['avg_rev'] = df.groupby('Неделя')['Выручка'].transform('mean')
    df['rev_pct'] = df['Выручка'] / df['avg_rev'] * 100

    cats = st.multiselect(f"Категории ({title})", sorted(df['Категория'].unique()), default=None)
    weeks= st.multiselect(f"Недели ({title})", sorted(df['Неделя'].unique()), default=None)
    if cats:
        df = df[df['Категория'].isin(cats)]
    if weeks:
        df = df[df['Неделя'].isin(weeks)]

    share_thr = st.slider(f"Порог Доли списаний % ({title})", 0.0, 100.0, 20.0)
    rev_thr   = st.slider(f"Мин. % выручки ({title})", 0.0, 200.0, 80.0)

    pivot = df.pivot_table(index=['Неделя','DayOfWeek'], columns='Категория',
                           values=['Доля списаний и ЗЦ','Выручка','rev_pct'])
    styled = pivot.style.format('{:.1f}') \
        .applymap(lambda v: 'background-color: tomato',
                  subset=pd.IndexSlice[:, pivot['Доля списаний и ЗЦ']>=share_thr]) \
        .applymap(lambda v: 'background-color: tomato',
                  subset=pd.IndexSlice[:, pivot['rev_pct']<=rev_thr])
    st.subheader("Таблица")
    st.dataframe(styled, use_container_width=True)

    st.subheader("График выручки")
    fig1 = px.line(df, x='DayOfWeek', y='Выручка', color='Категория',
                   line_group='Неделя', markers=True)
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("График доли списаний")
    fig2 = px.line(df, x='DayOfWeek', y='Доля списаний и ЗЦ', color='Категория',
                   line_group='Неделя', markers=True)
    st.plotly_chart(fig2, use_container_width=True)

    # Export
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name=title, index=False)
    buf.seek(0)
    st.download_button(f"Скачать Excel ({title})", buf,
                       f"{title.replace(' ','_')}.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

def main():
    st.title("Дашборд форматов: Жук и Тест")
    juk_files = st.file_uploader("Жук: файлы процент и выручка", accept_multiple_files=True, type=['csv','xlsx'])
    test_files = st.file_uploader("Тест: файлы процент и выручка", accept_multiple_files=True, type=['csv','xlsx'])

    display_format_section("Формат Жук", juk_files)
    st.markdown("---")
    display_format_section("Тестовый формат", test_files)

if __name__ == "__main__":
    main()

