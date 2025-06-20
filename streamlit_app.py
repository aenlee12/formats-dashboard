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
    for f in files:
        df = load_df(f)
        if 'Доля списаний и ЗЦ' in df.columns:
            df_share = df[['Категория','Неделя','DayOfWeek','Доля списаний и ЗЦ']].copy()
        if 'Выручка' in df.columns:
            df_rev   = df[['Категория','Неделя','DayOfWeek','Выручка']].copy()
    if df_share is None or df_rev is None:
        raise ValueError("Нужны колонки «Доля списаний и ЗЦ» и «Выручка».")
    s = df_share['Доля списаний и ЗЦ'].astype(str).str.replace(',', '.').str.rstrip('%')
    df_share['Доля списаний и ЗЦ'] = pd.to_numeric(s, errors='coerce').fillna(0)
    if df_share['Доля списаний и ЗЦ'].max() <= 1:
        df_share['Доля списаний и ЗЦ'] *= 100
    df_rev['Выручка'] = pd.to_numeric(df_rev['Выручка'], errors='coerce').fillna(0)
    return pd.merge(df_share, df_rev, on=['Категория','Неделя','DayOfWeek'], how='inner')


def main():
    st.title("📊 Дашборд форматов: анализ и тестирование")

    # Загрузка
    st.sidebar.header("1. Загрузка данных")
    files = st.sidebar.file_uploader("Загрузите два файла:\n• Доля списаний и ЗЦ\n• Выручка",
                                     type=['csv','xlsx'], accept_multiple_files=True)
    if len(files) != 2:
        st.sidebar.info("Нужно загрузить ровно два файла.")
        return
    try:
        df = prepare_data(files)
    except Exception as e:
        st.sidebar.error(e)
        return

    # Вычисления
    df['avg_rev_week'] = df.groupby('Неделя')['Выручка'].transform('mean')
    df['rev_pct']      = df['Выручка'] / df['avg_rev_week'] * 100

    # Фильтры
    st.sidebar.header("2. Фильтры")
    cats = st.sidebar.multiselect("Категории", sorted(df['Категория'].unique()))
    weeks = st.sidebar.multiselect("Недели", sorted(df['Неделя'].unique()))
    if cats: df = df[df['Категория'].isin(cats)]
    if weeks: df = df[df['Неделя'].isin(weeks)]

    # Heatmap
    st.sidebar.header("3. Heatmap неделя")
    all_weeks = sorted(df['Неделя'].unique())
    sel_week = st.sidebar.selectbox("Неделя для Heatmap", all_weeks)
    df_h = df[df['Неделя']==sel_week]
    heat = df_h.pivot_table(index='Категория', columns='DayOfWeek', values='Выручка', aggfunc='sum').fillna(0)
    heat_norm = heat.div(heat.max(axis=1), axis=0).fillna(0)

    st.subheader(f"🗺 Heatmap выручки по дням недели (неделя {sel_week})")
    fig_heat = px.imshow(heat_norm,
                        labels=dict(x="День недели", y="Категория"),
                        x=heat_norm.columns, y=heat_norm.index,
                        color_continuous_scale=['red','white','green'])
    fig_heat.data[0].text = heat.values.tolist()
    fig_heat.data[0].texttemplate = "%{text:.0f}"
    fig_heat.data[0].hovertemplate = 'Категория=%{y}<br>День=%{x}<br>Выручка=%{text:.0f}<extra></extra>'
    fig_heat.update_traces(xgap=1, ygap=1)
    fig_heat.update_layout(height=600)
    st.plotly_chart(fig_heat, use_container_width=True)

    # Общие тренды
    st.subheader("📈 Общие тренды по категориям")
    weekly_cat = df.groupby(['Неделя','Категория']).agg(
        Сумма_выручки=('Выручка','sum'),
        Среднее_списаний=('Доля списаний и ЗЦ', lambda s: np.average(s, weights=df.loc[s.index,'Выручка']))
    ).reset_index()

    fig_rev = px.line(weekly_cat, x='Неделя', y='Сумма_выручки', color='Категория', markers=True,
                      title="Сумма выручки по неделям и категориям")
    fig_rev.update_layout(height=400)

    fig_waste = px.line(weekly_cat, x='Неделя', y='Среднее_списаний', color='Категория', markers=True,
                        title="Средний % списаний по неделям и категориям")
    fig_waste.update_layout(height=400)

    # Анализ теста
    st.sidebar.header("4. Анализ тестового периода")
    test_mode = st.sidebar.checkbox("Включить анализ теста")
    if test_mode:
        days = sorted(df['DayOfWeek'].unique())
        day_map = {d: f"{d}-й" for d in days}
        test_week = st.sidebar.selectbox("Начальная неделя теста", all_weeks)
        test_day = st.sidebar.selectbox("Начальный день теста", days, format_func=lambda d: day_map[d])

        # Разделители на трендах
        fig_rev.add_vline(x=test_week, line_color='red', line_dash='dash')
        fig_waste.add_vline(x=test_week, line_color='red', line_dash='dash')

        # Полные недели
        complete = df.groupby('Неделя')['DayOfWeek'].nunique() == len(days)
        full_weeks = complete[complete].index
        weekly_full = df[df['Неделя'].isin(full_weeks)].groupby('Неделя').agg(
            Сумма_выручки=('Выручка','sum'),
            Среднее_списаний=('Доля списаний и ЗЦ', lambda s: np.average(s, weights=df.loc[s.index,'Выручка']))
        ).reset_index()
        weekly_full['Чистая_выручка'] = weekly_full['Сумма_выручки'] * (1 - weekly_full['Среднее_списаний']/100)

        pre = weekly_full[weekly_full['Неделя']<test_week]
        post = weekly_full[weekly_full['Неделя']>test_week]

        rev_pre = pre['Сумма_выручки'].mean()
        rev_post = post['Сумма_выручки'].mean()
        waste_pre = pre['Среднее_списаний'].mean()
        waste_post = post['Среднее_списаний'].mean()
        net_pre = pre['Чистая_выручка'].mean()
        net_post = post['Чистая_выручка'].mean()

        st.subheader("📋 Усреднённая аналитика до/во время теста (только полные недели)")
        st.markdown(f"""
- **Выручка (среднее)**: {rev_pre:.0f} → {rev_post:.0f} ₽ ({(rev_post/rev_pre-1)*100:.1f}% изменения)
- **% списаний (среднее)**: {waste_pre:.1f}% → {waste_post:.1f}% ({(waste_post-waste_pre):+.1f} п.п.)
- **Чистая выручка (среднее)**: {net_pre:.0f} → {net_post:.0f} ₽ ({(net_post/net_pre-1)*100:.1f}% изменения)
"""
        )

        # Общие графики теста
        fig_total_rev = px.line(weekly_full, x='Неделя', y='Сумма_выручки', markers=True,
                                title="Общая выручка по полным неделям")
        fig_total_rev.add_vline(x=test_week, line_color='red', line_dash='dash')
        fig_total_rev.update_layout(height=400)
        st.plotly_chart(fig_total_rev, use_container_width=True)

        fig_total_waste = px.line(weekly_full, x='Неделя', y='Среднее_списаний', markers=True,
                                  title="Общий % списаний по полным неделям")
        fig_total_waste.add_vline(x=test_week, line_color='red', line_dash='dash')
        fig_total_waste.update_layout(height=400)
        st.plotly_chart(fig_total_waste, use_container_width=True)

    # Вывод трендов
    st.plotly_chart(fig_rev, use_container_width=True)
    st.plotly_chart(fig_waste, use_container_width=True)

    # Экспорт
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='raw', index=False)
        weekly_cat.to_excel(writer, sheet_name='trend_by_cat', index=False)
        heat.to_excel(writer, sheet_name=f'heat_{sel_week}', index=True)
        if test_mode:
            weekly_full.to_excel(writer, sheet_name='trend_test', index=False)
    buf.seek(0)
    st.download_button("💾 Скачать отчёт (Excel)", buf,
                       "dashboard.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

if __name__ == "__main__":
    main()
