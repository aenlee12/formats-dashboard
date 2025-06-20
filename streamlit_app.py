import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO

st.set_page_config(page_title="Дашборд форматов", layout="wide")

@st.cache_data
def load_df(file):
    if file.name.lower().endswith('.csv'):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)

@st.cache_data
def prepare_data(files):
    df_share = None
    df_rev = None
    for f in files:
        df = load_df(f)
        cols = [c.lower() for c in df.columns]
        if any('доля' in c for c in cols):
            df_share = df.copy()
        elif any('выруч' in c for c in cols):
            df_rev = df.copy()
    if df_share is None or df_rev is None:
        raise ValueError("Нужны файлы: один с колонкой 'доля' и один с 'выручка'")
    df_share['Доля списаний и ЗЦ'] = (
        df_share[df_share.columns[df_share.columns.str.lower().str.contains('доля')][0]]
        .astype(str).str.replace(',', '.').str.rstrip('%').astype(float)
    )
    rev_col = df_rev.columns[df_rev.columns.str.lower().str.contains('выруч')][0]
    df_rev['Выручка'] = pd.to_numeric(df_rev[rev_col], errors='coerce').fillna(0)
    return pd.merge(df_share, df_rev, on=['Категория','Неделя','DayOfWeek'], how='inner')

def display_format_section(title, files):
    st.header(title)
    if len(files) != 2:
        st.info("Загрузите 2 файла: один с 'доля' и один с 'выручка'")
        return
    try:
        df = prepare_data(files)
    except Exception as e:
        st.error(f"Ошибка при подготовке данных: {e}")
        return

    avg_rev = df.groupby('Неделя')['Выручка'].transform('mean')
    df['rev_pct'] = df['Выручка'] / avg_rev * 100

    cats = st.multiselect(f"Категории {title}", sorted(df['Категория'].unique()), default=None, key=title+"cats")
    weeks= st.multiselect(f"Недели {title}",    sorted(df['Неделя'].unique()),    default=None, key=title+"weeks")
    if cats:  df = df[df['Категория'].isin(cats)]
    if weeks: df = df[df['Неделя'].isin(weeks)]

    share_thr = st.slider(f"Порог Доли списаний % ({title})", 0.0, 100.0, 20.0, key=title+"share_thr")
    rev_pct_thr= st.slider(f"Мин. % выручки от среднего ({title})", 0, 200, 80, key=title+"rev_pct")

    pivot = df.pivot_table(
        index=['Неделя','DayOfWeek'], columns='Категория',
        values=['Доля списаний и ЗЦ','Выручка','rev_pct']
    )
    styled = pivot.style.format('{:.1f}') \
        .applymap(lambda v: 'background-color: tomato',
                  subset=pd.IndexSlice[:, pivot['Доля списаний и ЗЦ']>=share_thr]) \
        .applymap(lambda v: 'background-color: tomato',
                  subset=pd.IndexSlice[:, pivot['rev_pct']<=rev_pct_thr])
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

    buf = BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name=title, index=False)
    buf.seek(0)
    st.download_button(f"Скачать Excel ({title})", buf,
                       f"{title.replace(' ','_')}.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

def main():
    st.title("Дашборд форматов Жук и Тест")
    st.sidebar.header("Загрузка для Формата Жук")
    juk_files = st.sidebar.file_uploader("Файлы Жук", type=['csv','xlsx'], accept_multiple_files=True, key='juk')
    st.sidebar.header("Загрузка для Тестового Формата")
    test_files = st.sidebar.file_uploader("Файлы Тест", type=['csv','xlsx'], accept_multiple_files=True, key='test')

    display_format_section("Формат Жук", juk_files)
    st.markdown("---")
    display_format_section("Тестовый формат", test_files)

if __name__ == "__main__":
    main()
