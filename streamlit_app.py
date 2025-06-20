import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO

# Настройки страницы
st.set_page_config(page_title="Дашборд форматов", layout="wide")

@st.cache_data
def load_df(file):
    """Загрузить CSV или Excel и привести названия колонок к стандартным."""
    if file.name.lower().endswith('.csv'):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file, header=0)
    df.columns = df.columns.str.strip().str.lower()
    col_map = {}
    for c in df.columns:
        if 'категор' in c:        col_map[c] = 'Категория'
        elif 'недел' in c:        col_map[c] = 'Неделя'
        elif 'день' in c or 'dayofweek' in c: col_map[c] = 'DayOfWeek'
        elif 'доля' in c:         col_map[c] = 'Доля списаний и ЗЦ'
        elif 'выруч' in c:        col_map[c] = 'Выручка'
    return df.rename(columns=col_map)

@st.cache_data
def prepare_data(files):
    """Собрать один DataFrame из двух файлов."""
    df_share = df_rev = None
    info = []
    for f in files:
        df = load_df(f)
        info.append((f.name, df.columns.tolist()))
        if 'Доля списаний и ЗЦ' in df.columns:
            df_share = df[['Категория','Неделя','DayOfWeek','Доля списаний и ЗЦ']].copy()
        if 'Выручка' in df.columns:
            df_rev   = df[['Категория','Неделя','DayOfWeek','Выручка']].copy()
    if df_share is None or df_rev is None:
        details = "\n".join(f"{n}: {cols}" for n,cols in info)
        raise ValueError(f"Нужны колонки «Доля списаний и ЗЦ» и «Выручка».\n{details}")
    # Приводим проценты к числу
    s = df_share['Доля списаний и ЗЦ'].astype(str).str.replace(',', '.').str.rstrip('%')
    df_share['Доля списаний и ЗЦ'] = pd.to_numeric(s, errors='coerce').fillna(0)
    if df_share['Доля списаний и ЗЦ'].max() <= 1:
        df_share['Доля списаний и ЗЦ'] *= 100
    # Выручка в числа
    df_rev['Выручка'] = pd.to_numeric(df_rev['Выручка'], errors='coerce').fillna(0)
    # Мёржим
    return pd.merge(df_share, df_rev, on=['Категория','Неделя','DayOfWeek'], how='inner')

def main():
    st.title("📊 Дашборд форматов: анализ и тестирование")

    # загрузка
    st.sidebar.header("1. Загрузка данных")
    files = st.sidebar.file_uploader(
        "Загрузите два файла:\n• «Доля списаний и ЗЦ»\n• «Выручка»",
        type=['csv','xlsx'], accept_multiple_files=True)
    if len(files) != 2:
        st.sidebar.info("Нужно загрузить ровно два файла.")
        return
    try:
        df = prepare_data(files)
    except Exception as e:
        st.sidebar.error(e)
        return

    # вычисления
    df['avg_rev_week'] = df.groupby('Неделя')['Выручка'].transform('mean')
    df['rev_pct']      = df['Выручка'] / df['avg_rev_week'] * 100

    # фильтры
    st.sidebar.header("2. Фильтры")
    cats  = st.sidebar.multiselect("Категории", sorted(df['Категория'].unique()))
    weeks = st.sidebar.multiselect("Недели",    sorted(df['Неделя'].unique()))
    if cats:  df = df[df['Категория'].isin(cats)]
    if weeks: df = df[df['Неделя'].isin(weeks)]

    # heatmap неделя
    st.sidebar.header("3. Heatmap неделя")
    all_weeks = sorted(df['Неделя'].unique())
    sel_week = st.sidebar.selectbox("Неделя для Heatmap", all_weeks, index=len(all_weeks)-1)

    # Общие тренды по категориям
    st.subheader("📈 Общие тренды по категориям")
    weekly_cat = df.groupby(['Неделя','Категория']).agg(
        {'Выручка':'sum',
         'Доля списаний и ЗЦ': lambda s: np.average(s, weights=df.loc[s.index,'Выручка'])}
    ).reset_index().rename(columns={'Доля списаний и ЗЦ':'Среднее % списаний',
                                    'Выручка':'Сумма выручки'})

    fig_rev = px.line(weekly_cat, x='Неделя', y='Сумма выручки', color='Категория',
                      markers=True, title="Сумма выручки по неделям и категориям")
    fig_rev.update_layout(height=400)
    st.plotly_chart(fig_rev, use_container_width=True)

    fig_waste = px.line(weekly_cat, x='Неделя', y='Среднее % списаний', color='Категория',
                        markers=True, title="Средний % списаний по неделям и категориям")
    fig_waste.update_layout(height=400)
    st.plotly_chart(fig_waste, use_container_width=True)

    # heatmap всегда
    st.subheader(f"🗺 Heatmap выручки по дням недели (неделя {sel_week})")
    df_h = df[df['Неделя']==sel_week]
    heat = df_h.pivot_table(index='Категория', columns='DayOfWeek', values='Выручка', aggfunc='sum').fillna(0)
    # нормализация по каждой категории
    heat_norm = heat.div(heat.max(axis=1), axis=0).fillna(0)
    fig_heat = px.imshow(
        heat_norm,
        labels=dict(x="День недели", y="Категория", color="Нормал. выручка"),
        x=heat_norm.columns, y=heat_norm.index,
        color_continuous_scale=['red','white','green'],
        title="Heatmap выручки по дням недели (нормализовано по категории)"
    )
    fig_heat.update_traces(xgap=1, ygap=1)
    fig_heat.update_layout(height=600)
    st.plotly_chart(fig_heat, use_container_width=True)

    # анализ теста
    st.sidebar.header("4. Анализ тестового периода")
    test_mode = st.sidebar.checkbox("Включить анализ теста")
    if test_mode:
        days = sorted(df['DayOfWeek'].unique())
        day_map = {d:f"{d}-й" for d in days}
        test_week = st.sidebar.selectbox("Начальная неделя теста", all_weeks, index=len(all_weeks)-1)
        test_day  = st.sidebar.selectbox("Начальный день теста", days, format_func=lambda d: day_map[d])

        # полные недели
        complete = df.groupby('Неделя')['DayOfWeek'].nunique()==len(days)
        full_weeks = complete[complete].index

        # общие метрики до/после
        weekly_full = df[df['Неделя'].isin(full_weeks)].groupby('Неделя').apply(
            lambda g: pd.Series({
                'Сумма выручки': g['Выручка'].sum(),
                'Среднее % списаний': np.average(g['Доля списаний и ЗЦ'], weights=g['Выручка'])
            })
        ).reset_index()
        weekly_full['Чистая выручка'] = weekly_full['Сумма выручки'] * (1 - weekly_full['Среднее % списаний']/100)

        pre = weekly_full[weekly_full['Неделя']<test_week]
        post= weekly_full[weekly_full['Неделя']>=test_week]
        rev_pre = pre['Сумма выручки'].mean(); rev_post = post['Сумма выручки'].mean()
        waste_pre = pre['Среднее % списаний'].mean(); waste_post = post['Среднее % списаний'].mean()
        net_pre = pre['Чистая выручка'].mean(); net_post = post['Чистая выручка'].mean()

        st.subheader("📋 Сравнение до/после теста")
        st.markdown(f"""
- **Выручка**: {rev_pre:.0f} → {rev_post:.0f} ₽ ({(rev_post/rev_pre-1)*100:.1f}%)
- **% списаний**: {waste_pre:.1f}% → {waste_post:.1f}% ({waste_post-waste_pre:+.1f} п.п.)
- **Чистая выручка**: {net_pre:.0f} → {net_post:.0f} ₽ ({(net_post/net_pre-1)*100:.1f}%)
"""
        )

        # графики тестовых недель по категориям
        weekly_full_cat = df[df['Неделя'].isin(full_weeks)].groupby(['Неделя','Категория']).apply(
            lambda g: pd.Series({
                'Сумма выручки': g['Выручка'].sum(),
                'Среднее % списаний': np.average(g['Доля списаний и ЗЦ'], weights=g['Выручка'])
            })
        ).reset_index()

        fig_rev_cat = px.line(
            weekly_full_cat, x='Неделя', y='Сумма выручки', color='Категория',
            markers=True, title="Выручка тестовых недель по категориям"
        )
        fig_rev_cat.add_vline(x=test_week, line_color='red', line_dash='dash')
        fig_rev_cat.update_layout(height=400)
        st.plotly_chart(fig_rev_cat, use_container_width=True)

        fig_waste_cat = px.line(
            weekly_full_cat, x='Неделя', y='Среднее % списаний', color='Категория',
            markers=True, title="% списаний тестовых недель по категориям"
        )
        fig_waste_cat.add_vline(x=test_week, line_color='red', line_dash='dash')
        fig_waste_cat.update_layout(height=400)
        st.plotly_chart(fig_waste_cat, use_container_width=True)

    # экспорт
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='raw', index=False)
        weekly_cat.to_excel(writer, sheet_name='trend_by_cat', index=False)
        heat.to_excel(writer, sheet_name=f'heat_{sel_week}', index=True)
        if test_mode:
            weekly_full.to_excel(writer, sheet_name='trend_test', index=False)
            weekly_full_cat.to_excel(writer, sheet_name='trend_test_cat', index=False)
    buf.seek(0)
    st.download_button("💾 Скачать отчёт (Excel)", buf,
                       "dashboard.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

if __name__ == "__main__":
    main()
