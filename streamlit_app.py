import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO

st.set_page_config(page_title="Дашборд форматов", layout="wide")

@st.cache_data
def load_df(file):
    """Загрузить CSV/Excel и переименовать колонки в стандартизированные."""
    if file.name.lower().endswith('.csv'):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file, header=0)

    # Сразу в нижний регистр для упрощённого поиска
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
    """Из двух файлов (доля списаний и выручка) собрать один DataFrame."""
    df_share = df_rev = None

    for f in files:
        df = load_df(f)
        # Ищем уже стандартизированные имена
        if 'Доля списаний и ЗЦ' in df.columns:
            df_share = df[['Категория','Неделя','DayOfWeek','Доля списаний и ЗЦ']].copy()
        elif 'Выручка' in df.columns:
            df_rev = df[['Категория','Неделя','DayOfWeek','Выручка']].copy()

    if df_share is None or df_rev is None:
        raise ValueError(
            "Нужно загрузить ровно два файла:\n"
            "• один с колонкой «Доля списаний и ЗЦ»\n"
            "• второй с колонкой «Выручка»"
        )

    # Приводим % к числу
    s = df_share['Доля списаний и ЗЦ'].astype(str).str.replace(',', '.').str.rstrip('%')
    df_share['Доля списаний и ЗЦ'] = pd.to_numeric(s, errors='coerce').fillna(0)

    # Приводим выручку к числу
    df_rev['Выручка'] = pd.to_numeric(df_rev['Выручка'], errors='coerce').fillna(0)

    # Объединяем
    return pd.merge(
        df_share, df_rev,
        on=['Категория','Неделя','DayOfWeek'],
        how='inner'
    )

def main():
    st.title("📊 Дашборд форматов: Жук и Тест")

    st.sidebar.header("1. Загрузка файлов")
    files = st.sidebar.file_uploader(
        "Загрузите два файла:\n"
        "• «Доля списаний и ЗЦ»\n"
        "• «Выручка»",
        type=['csv','xlsx'], accept_multiple_files=True
    )
    if len(files) != 2:
        st.sidebar.info("Пожалуйста, выберите ровно два файла.")
        return

    try:
        df = prepare_data(files)
    except Exception as e:
        st.sidebar.error(str(e))
        return

    # Вычисляем относительную выручку
    df['avg_rev_week'] = df.groupby('Неделя')['Выручка'].transform('mean')
    df['rev_pct']      = df['Выручка'] / df['avg_rev_week'] * 100

    # Фильтры
    st.sidebar.header("2. Фильтры")
    cats  = st.sidebar.multiselect("Категории", sorted(df['Категория'].unique()), default=None)
    weeks = st.sidebar.multiselect("Недели",    sorted(df['Неделя'].unique()),     default=None)
    if cats:
        df = df[df['Категория'].isin(cats)]
    if weeks:
        df = df[df['Неделя'].isin(weeks)]

    # Пороги для подсветки
    st.sidebar.header("3. Пороги подсветки")
    share_thr = st.sidebar.slider("Порог % списаний", 0.0, 100.0, 20.0)
    rev_thr   = st.sidebar.slider("Мин. % выручки от среднего", 0.0, 200.0, 80.0)

    # Выбор недели для Heatmap
    st.sidebar.header("4. Heatmap неделя")
    last_week = int(df['Неделя'].max())
    sel_week  = st.sidebar.selectbox(
        "Выберите неделю для Heatmap",
        options=sorted(df['Неделя'].unique()),
        index=sorted(df['Неделя'].unique()).index(last_week)
    )

    # ----------------------------------------------------------------
    # 1) Таблица по [Неделя × ДеньOfWeek]
    st.subheader("📅 Таблица: Неделя × День недели")
    pivot = df.pivot_table(
        index=['Неделя','DayOfWeek'],
        columns='Категория',
        values=['Доля списаний и ЗЦ','rev_pct'],
        aggfunc='mean'
    )
    # распрямляем MultiIndex
    flat = [f"{metric}_{cat}" for metric,cat in pivot.columns]
    pivot.columns = flat

    waste_cols   = [c for c in flat if c.startswith('Доля списаний и ЗЦ_')]
    rev_pct_cols = [c for c in flat if c.startswith('rev_pct_')]

    styled = pivot.style.format("{:.1f}") \
        .applymap(lambda v: 'background-color: tomato' if v>=share_thr else '', subset=waste_cols) \
        .applymap(lambda v: 'background-color: tomato' if v<=rev_thr   else '', subset=rev_pct_cols)

    st.dataframe(styled, use_container_width=True)

    # ----------------------------------------------------------------
    # 2) Линейные тренды по неделям
    st.subheader("📈 Тренды по неделям")
    wk = df.groupby(['Неделя','Категория']).agg({
        'Выручка': 'sum',
        'Доля списаний и ЗЦ': lambda s: np.average(s, weights=df.loc[s.index,'Выручка'])
                                       if df.loc[s.index,'Выручка'].sum()>0 else s.mean()
    }).reset_index()

    fig_rev = px.line(wk, x='Неделя', y='Выручка', color='Категория',
                      markers=True, title="Суммарная выручка по неделям")
    fig_rev.update_layout(height=500)
    st.plotly_chart(fig_rev, use_container_width=True)

    fig_waste = px.line(wk, x='Неделя', y='Доля списаний и ЗЦ', color='Категория',
                        markers=True, title="Средневзвешенный % списаний по неделям")
    fig_waste.update_layout(height=500)
    st.plotly_chart(fig_waste, use_container_width=True)

    # ----------------------------------------------------------------
    # 3) Heatmap за выбранную неделю
    st.subheader(f"🗺 Heatmap выручки: Неделя {sel_week}")
    df_h = df[df['Неделя']==sel_week]
    heat = df_h.pivot_table(
        index='Категория',
        columns='DayOfWeek',
        values='Выручка',
        aggfunc='sum'
    )

    fig_heat = px.imshow(
        heat,
        labels=dict(x="День недели", y="Категория", color="Выручка"),
        aspect="auto",
        color_continuous_scale="Viridis"
    )
    fig_heat.update_traces(xgap=1, ygap=1)
    fig_heat.update_layout(height=700)
    st.plotly_chart(fig_heat, use_container_width=True)

    # ----------------------------------------------------------------
    # Экспорт
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='raw_data', index=False)
        wk.to_excel(writer, sheet_name='trend_by_week', index=False)
        heat.to_excel(writer, sheet_name=f'heatmap_week_{sel_week}', index=True)
    buf.seek(0)
    st.download_button("💾 Скачать отчёт (Excel)", buf,
                       "formats_dashboard.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

if __name__ == "__main__":
    main()
