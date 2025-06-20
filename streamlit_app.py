import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO

st.set_page_config(page_title="Дашборд форматов", layout="wide")

@st.cache_data
def load_df(file):
    if file.name.lower().endswith('.csv'):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file, header=0)
    df.columns = df.columns.str.strip().str.lower()
    col_map = {}
    for c in df.columns:
        if 'категор' in c:        col_map[c] = 'Категория'
        elif 'недел' in c:        col_map[c] = 'Неделя'
        elif 'день' in c:         col_map[c] = 'DayOfWeek'
        elif 'доля' in c:         col_map[c] = 'Доля списаний и ЗЦ'
        elif 'выруч' in c:        col_map[c] = 'Выручка'
    return df.rename(columns=col_map)

@st.cache_data
def prepare_data(files):
    df_share = df_rev = None
    for f in files:
        df = load_df(f)
        if 'доля списаний и зц' in df.columns:
            df_share = df[['Категория','Неделя','DayOfWeek','Доля списаний и ЗЦ']].copy()
        elif 'выручка' in df.columns:
            df_rev = df[['Категория','Неделя','DayOfWeek','Выручка']].copy()
    if df_share is None or df_rev is None:
        raise ValueError("Нужно загрузить два файла: один с «Доля списаний и ЗЦ», второй с «Выручка»")
    s = df_share['Доля списаний и ЗЦ'].astype(str).str.replace(',', '.').str.rstrip('%')
    df_share['Доля списаний и ЗЦ'] = pd.to_numeric(s, errors='coerce').fillna(0)
    df_rev['Выручка'] = pd.to_numeric(df_rev['Выручка'], errors='coerce').fillna(0)
    return pd.merge(df_share, df_rev, on=['Категория','Неделя','DayOfWeek'], how='inner')

def main():
    st.title("📊 Дашборд форматов: общий анализ")
    files = st.sidebar.file_uploader("Загрузите два файла: % списаний и Выручка", type=['csv','xlsx'], accept_multiple_files=True)
    if len(files) != 2:
        st.sidebar.info("Нужно загрузить ровно два файла.")
        return
    try:
        df = prepare_data(files)
    except Exception as e:
        st.sidebar.error(e)
        return

    df['avg_rev_week'] = df.groupby('Неделя')['Выручка'].transform('mean')
    df['rev_pct'] = df['Выручка'] / df['avg_rev_week'] * 100

    st.sidebar.header("Фильтры")
    cats = st.sidebar.multiselect("Категории", sorted(df['Категория'].unique()), default=None)
    weeks = st.sidebar.multiselect("Недели", sorted(df['Неделя'].unique()), default=None)
    if cats: df = df[df['Категория'].isin(cats)]
    if weeks: df = df[df['Неделя'].isin(weeks)]

    st.sidebar.header("Настройки теста")
    test_week = st.sidebar.selectbox("Начальная неделя теста", sorted(df['Неделя'].unique()), index=len(df['Неделя'].unique())-1)
    test_day = st.sidebar.selectbox("Начальный день недели теста", sorted(df['DayOfWeek'].unique()), index=0)

    st.sidebar.header("Пороги подсветки")
    share_thr = st.sidebar.slider("Порог % списаний", 0.0, 100.0, 20.0)
    rev_thr = st.sidebar.slider("Мин. % выручки от среднего", 0.0, 200.0, 80.0)

    # Metrics comparison pre-test vs test
    weekly = df.groupby('Неделя').apply(lambda g: pd.Series({
        'revenue': g['Выручка'].sum(),
        'waste_avg': np.average(g['Доля списаний и ЗЦ'], weights=g['Выручка']) if g['Выручка'].sum()>0 else g['Доля списаний и ЗЦ'].mean()
    })).reset_index()
    pre = weekly[weekly['Неделя']<test_week]
    post = weekly[weekly['Неделя']>=test_week]
    rev_pre = pre['revenue'].mean() if not pre.empty else 0
    rev_post = post['revenue'].mean() if not post.empty else 0
    waste_pre = pre['waste_avg'].mean() if not pre.empty else 0
    waste_post = post['waste_avg'].mean() if not post.empty else 0
    net_pre = rev_pre*(1-waste_pre/100)
    net_post = rev_post*(1-waste_post/100)

    st.subheader("📋 Результаты тестового периода")
    st.markdown(f"""
- **Средняя выручка**: {rev_pre:.0f} → {rev_post:.0f} ₽ ({(rev_post/rev_pre-1)*100:.1f}%)
- **Средний % списаний**: {waste_pre:.1f}% → {waste_post:.1f}% ({waste_pre-waste_post:.1f} п.п.)
- **Средняя чистая выручка**: {net_pre:.0f} → {net_post:.0f} ₽ ({(net_post/net_pre-1)*100:.1f}%)
""")

    # Line charts with test start marker
    st.subheader("🚀 Тренды по неделям")
    fig_rev = px.line(weekly, x='Неделя', y='revenue', markers=True, title="Выручка по неделям")
    fig_rev.add_vline(x=test_week, line_color='red', line_dash='dash', annotation_text="Тест начался")
    fig_rev.update_layout(height=400)
    st.plotly_chart(fig_rev, use_container_width=True)

    fig_waste = px.line(weekly, x='Неделя', y='waste_avg', markers=True, title="% списаний по неделям")
    fig_waste.add_vline(x=test_week, line_color='red', line_dash='dash')
    fig_waste.update_layout(height=400)
    st.plotly_chart(fig_waste, use_container_width=True)

    # Heatmap for selected week
    st.subheader(f"🗺 Heatmap выручки: неделя {test_week}")
    df_h = df[df['Неделя']==test_week]
    heat = df_h.pivot_table(index='Категория', columns='DayOfWeek', values='Выручка', aggfunc='sum')
    fig_heat = px.imshow(heat, labels=dict(x="День недели", y="Категория", color="Выручка"), aspect="auto", color_continuous_scale="Viridis")
    fig_heat.update_traces(xgap=1, ygap=1)
    fig_heat.update_layout(height=500)
    st.plotly_chart(fig_heat, use_container_width=True)

    # Export
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='raw', index=False)
        weekly.to_excel(writer, sheet_name='weekly', index=False)
        heat.to_excel(writer, sheet_name=f'heat_{test_week}', index=True)
    buf.seek(0)
    st.download_button("💾 Скачать отчёт (Excel)", buf, "dashboard.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

if __name__ == "__main__":
    main()
