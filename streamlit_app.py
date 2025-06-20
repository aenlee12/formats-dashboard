import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO

st.set_page_config(page_title="Дашборд форматов", layout="wide")

@st.cache_data
def load_df(file):
    """Загрузить CSV или Excel и привести названия столбцов к стандартным."""
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
    """Привести два файла (доля + выручка) в один DataFrame."""
    df_share = df_rev = None
    for f in files:
        df = load_df(f)
        if 'доля списаний и зц' in df.columns:
            df_share = df[['Категория','Неделя','DayOfWeek','Доля списаний и ЗЦ']].copy()
        elif 'выручка' in df.columns:
            df_rev = df[['Категория','Неделя','DayOfWeek','Выручка']].copy()
    if df_share is None or df_rev is None:
        raise ValueError("Нужно два файла: один с долей списаний, другой с выручкой")
    # Привести типы
    df_share['Доля списаний и ЗЦ'] = (
        df_share['Доля списаний и ЗЦ']
        .astype(str).str.replace(',', '.').str.rstrip('%')
    )
    df_share['Доля списаний и ЗЦ'] = pd.to_numeric(df_share['Доля списаний и ЗЦ'], errors='coerce').fillna(0)
    df_rev['Выручка'] = pd.to_numeric(df_rev['Выручка'], errors='coerce').fillna(0)
    # Объединить
    dfm = pd.merge(
        df_share, df_rev,
        on=['Категория','Неделя','DayOfWeek'],
        how='inner'
    )
    return dfm

def display_format_section(title, files):
    st.header(title)
    if len(files) != 2:
        st.info("Загрузите два файла: один с долей списаний и один с выручкой.")
        return

    try:
        df = prepare_data(files)
    except Exception as e:
        st.error(f"Ошибка при подготовке данных: {e}")
        return

    # Общие вычисления
    df['avg_rev_week'] = df.groupby('Неделя')['Выручка'].transform('mean')
    df['rev_pct']      = df['Выручка'] / df['avg_rev_week'] * 100

    # Фильтры
    cats  = st.multiselect(f"Категории ({title})", sorted(df['Категория'].unique()), default=None, key=title+"cats")
    weeks = st.multiselect(f"Недели ({title})",    sorted(df['Неделя'].unique()),     default=None, key=title+"weeks")
    if cats:
        df = df[df['Категория'].isin(cats)]
    if weeks:
        df = df[df['Неделя'].isin(weeks)]

    # Пороги подсветки
    share_thr = st.slider(f"Порог % списаний ({title})",      0.0, 100.0, 20.0, key=title+"share")
    rev_thr   = st.slider(f"Мин. % выручки от среднего ({title})", 0.0, 200.0, 80.0, key=title+"rev")

    # 1) Таблица по [Неделя × День]
    pivot = df.pivot_table(
        index=['Неделя','DayOfWeek'],
        columns='Категория',
        values=['Доля списаний и ЗЦ','rev_pct'],
        aggfunc='mean'
    )
    # Выравниваем колонки
    flat = [f"{val}_{cat}" for val,cat in pivot.columns]
    pivot.columns = flat
    styled = pivot.style.format("{:.1f}") \
        .applymap(lambda v: 'background-color: tomato' if v>=share_thr else '',
                  subset=[c for c in flat if c.startswith('Доля списаний и ЗЦ_')]) \
        .applymap(lambda v: 'background-color: tomato' if v<=rev_thr else '',
                  subset=[c for c in flat if c.startswith('rev_pct_')])
    st.subheader("Таблица: по неделям и дням")
    st.dataframe(styled, use_container_width=True)

    # 2) Тренды по неделям (агрегация)
    wk = df.groupby(['Неделя','Категория']).agg({
        'Выручка': 'sum',
        'Доля списаний и ЗЦ': lambda s: np.average(s, weights=df.loc[s.index,'Выручка']) if df.loc[s.index,'Выручка'].sum()>0 else s.mean()
    }).reset_index()
    st.subheader("Тренд выручки по неделям")
    fig_rev = px.line(
        wk, x='Неделя', y='Выручка', color='Категория',
        markers=True, title="Суммарная выручка"
    )
    st.plotly_chart(fig_rev, use_container_width=True)

    st.subheader("Тренд % списаний по неделям")
    fig_waste = px.line(
        wk, x='Неделя', y='Доля списаний и ЗЦ', color='Категория',
        markers=True, title="Средневзвешенная доля списаний"
    )
    st.plotly_chart(fig_waste, use_container_width=True)

    # 3) Тепловая карта по дням недели и категориям (выручка)
    heat = df.pivot_table(
        index='Категория', columns='DayOfWeek',
        values='Выручка', aggfunc='sum'
    )
    st.subheader("Heatmap выручки (категория vs день недели)")
    fig_heat = px.imshow(
        heat,
        labels=dict(x="День недели", y="Категория", color="Выручка"),
        aspect="auto", title="Выручка по дням недели"
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # Экспорт в Excel
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name=title, index=False)
        wk.to_excel(writer, sheet_name=f"{title}_по_неделям", index=False)
        heat.to_excel(writer, sheet_name=f"{title}_heatmap", index=True)
    buf.seek(0)
    st.download_button(f"Скачать Excel ({title})", buf,
                       f"{title.replace(' ','_')}.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


def main():
    st.title("Дашборд форматов: Жук и Тест")
    st.sidebar.header("Загрузка файлов")
    juk_files  = st.sidebar.file_uploader("Формат Жук: проценты и выручка",  type=['csv','xlsx'], accept_multiple_files=True, key='juk')
    test_files = st.sidebar.file_uploader("Тестовый формат: проценты и выручка", type=['csv','xlsx'], accept_multiple_files=True, key='test')

    display_format_section("Формат Жук", juk_files)
    st.markdown("---")
    display_format_section("Тестовый формат", test_files)


if __name__ == "__main__":
    main()
