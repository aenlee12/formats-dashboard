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
    # Ожидаем два файла: Доля списаний и Выручка
    df_share, df_rev = None, None
    for f in files:
        df = load_df(f)
        if 'Доля списаний' in df.columns:
            df_share = df.copy()
        elif 'Выручка' in df.columns:
            df_rev = df.copy()
    df_share['Доля списаний и ЗЦ'] = (
        df_share['Доля списаний и ЗЦ']
        .astype(str).str.replace(',', '.').str.rstrip('%').astype(float)
    )
    df_rev['Выручка'] = pd.to_numeric(df_rev['Выручка'], errors='coerce').fillna(0)
    return pd.merge(df_share, df_rev, on=['Категория','Неделя','DayOfWeek'], how='inner')

def display_format_section(title, files):
    st.header(title)
    if len(files) != 2:
        st.info("Загрузите 2 файла: Доля списаний и Выручка")
        return
    df = prepare_data(files)
    # вычисляем среднее по неделе и %
    avg_rev = df.groupby('Неделя')['Выручка'].transform('mean')
    df['rev_pct'] = df['Выручка'] / avg_rev * 100
    # фильтры
    cats = st.multiselect(f"Категории {title}", sorted(df['Категория'].unique()), default=None, key=title+"cats")
    weeks= st.multiselect(f"Недели {title}",    sorted(df['Неделя'].unique()),    default=None, key=title+"weeks")
    if cats:  df = df[df['Категория'].isin(cats)]
    if weeks: df = df[df['Неделя'].isin(weeks)]
    # пороги
    share_thr = st.slider(f"Порог Доли списаний % ({title})", 0.0, 100.0, 20.0, key=title+"share_thr")
    rev_pct_thr= st.slider(f"Мин. % выручки от среднего ({title})", 0, 200, 80, key=title+"rev_pct")
    # таблица
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
    # графики
    st.subheader("График выручки")
    fig1 = px.line(df, x='DayOfWeek', y='Выручка', color='Категория',
                   line_group='Неделя', markers=True)
    st.plotly_chart(fig1, use_container_width=True)
    st.subheader("График доли списаний")
    fig2 = px.line(df, x='DayOfWeek', y='Доля списаний и ЗЦ', color='Категория',
                   line_group='Неделя', markers=True)
    st.plotly_chart(fig2, use_container_width=True)
    # экспорт
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
