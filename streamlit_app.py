import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO

st.set_page_config(page_title="Дашборд форматов", layout="wide")

@st.cache_data
def load_df(file):
    """Load CSV or Excel, normalize column names to expected Russian names."""
    name = file.name.lower()
    if name.endswith('.csv'):
        df = pd.read_csv(file)
    else:
        # Excel: skip first row of metadata
        df = pd.read_excel(file, header=1)
    # Normalize column names: strip, lower
    df.columns = df.columns.str.strip().str.lower()
    # Mapping to standard names
    col_map = {}
    for col in df.columns:
        c = col.replace('\n', ' ').strip()
        if 'категория' in c:
            col_map[col] = 'Категория'
        elif 'неделя' in c:
            col_map[col] = 'Неделя'
        elif 'dayofweek' in c or 'день' in c:
            col_map[col] = 'DayOfWeek'
        elif 'доля' in c:
            col_map[col] = 'Доля списаний и ЗЦ'
        elif 'выруч' in c:
            col_map[col] = 'Выручка'
    df = df.rename(columns=col_map)
    return df

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
        raise ValueError("Нужно два файла: один с % списаний, другой с выручкой")
    # Clean types
    df_share['Доля списаний и ЗЦ'] = (
        df_share['Доля списаний и ЗЦ']
        .astype(str).str.replace(',', '.').str.rstrip('%').astype(float)
    )
    df_rev['Выручка'] = pd.to_numeric(df_rev['Выручка'], errors='coerce').fillna(0)
    # Merge
    dfm = pd.merge(
        df_share, df_rev,
        on=['Категория','Неделя','DayOfWeek'],
        how='inner'
    )
    return dfm

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

    # Compute relative revenue
    df['avg_rev'] = df.groupby('Неделя')['Выручка'].transform('mean')
    df['rev_pct'] = df['Выручка'] / df['avg_rev'] * 100

    # Filters
    cats = st.multiselect(f"Категории {title}", sorted(df['Категория'].unique()), default=None, key=title+"cats")
    weeks= st.multiselect(f"Недели {title}",    sorted(df['Неделя'].unique()),    default=None, key=title+"weeks")
    if cats:  df = df[df['Категория'].isin(cats)]
    if weeks: df = df[df['Неделя'].isin(weeks)]

    # Thresholds
    share_thr = st.slider(f"Порог доли списаний % ({title})", 0.0, 100.0, 20.0, key=title+"share_thr")
    rev_pct_thr = st.slider(f"Мин. % выручки от среднего ({title})", 0.0, 200.0, 80.0, key=title+"rev_pct")

    # Pivot table
    pivot = df.pivot_table(
        index=['Неделя','DayOfWeek'], columns='Категория',
        values=['Доля списаний и ЗЦ','Выручка','rev_pct']
    )
    styled = pivot.style.format('{:.1f}') \
        .applymap(lambda v: 'background-color: tomato',
                  subset=pd.IndexSlice[:, pivot['Доля списаний и ЗЦ'] >= share_thr]) \
        .applymap(lambda v: 'background-color: tomato',
                  subset=pd.IndexSlice[:, pivot['rev_pct'] <= rev_pct_thr])
    st.subheader("Таблица")
    st.dataframe(styled, use_container_width=True)

    # Charts
    st.subheader("График выручки")
    fig1 = px.line(df, x='DayOfWeek', y='Выручка', color='Категория',
                   line_group='Неделя', markers=True, labels={'DayOfWeek':'День недели'})
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("График доли списаний")
    fig2 = px.line(df, x='DayOfWeek', y='Доля списаний и ЗЦ', color='Категория',
                   line_group='Неделя', markers=True, labels={'DayOfWeek':'День недели'})
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
    juk_files = st.file_uploader("Файлы Формата Жук (2 файла)", type=['csv','xlsx'], accept_multiple_files=True, key='juk')
    test_files = st.file_uploader("Файлы Тестового Формата (2 файла)", type=['csv','xlsx'], accept_multiple_files=True, key='test')

    display_format_section("Формат Жук", juk_files)
    st.markdown("---")
    display_format_section("Тестовый формат", test_files)

if __name__ == "__main__":
    main()

