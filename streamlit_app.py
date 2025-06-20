import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO

st.set_page_config(page_title="Дашборд форматов", layout="wide")

@st.cache_data
def load_df(file):
    """Load CSV or Excel and normalize column names."""
    if file.name.lower().endswith('.csv'):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file, header=0)
    # all to lowercase for matching
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
    """Merge two files (share & revenue) into one DataFrame."""
    df_share = df_rev = None
    info = []
    for f in files:
        df = load_df(f)
        info.append((f.name, df.columns.tolist()))
        if 'Доля списаний и ЗЦ' in df.columns:
            df_share = df[['Категория','Неделя','DayOfWeek','Доля списаний и ЗЦ']].copy()
        if 'Выручка' in df.columns:
            df_rev = df[['Категория','Неделя','DayOfWeek','Выручка']].copy()
    if df_share is None or df_rev is None:
        msg = "Не найдены колонки в:\n" + "\n".join(f"{n}: {cols}" for n,cols in info)
        msg += "\nТребуются «Доля списаний и ЗЦ» и «Выручка»."
        raise ValueError(msg)

    # parse share as numeric
    s = df_share['Доля списаний и ЗЦ'].astype(str).str.replace(',', '.').str.rstrip('%')
    df_share['Доля списаний и ЗЦ'] = pd.to_numeric(s, errors='coerce').fillna(0)
    # if loaded as fraction (max ≤1), convert to percent
    if df_share['Доля списаний и ЗЦ'].max() <= 1:
        df_share['Доля списаний и ЗЦ'] *= 100

    # parse revenue
    df_rev['Выручка'] = pd.to_numeric(df_rev['Выручка'], errors='coerce').fillna(0)

    return pd.merge(df_share, df_rev, on=['Категория','Неделя','DayOfWeek'], how='inner')

def main():
    st.title("📊 Дашборд форматов: анализ и тестирование")

    # 1) загрузка
    st.sidebar.header("1. Загрузка данных")
    files = st.sidebar.file_uploader(
        "Загрузите ровно два файла:\n• «Доля списаний и ЗЦ»\n• «Выручка»",
        type=['csv','xlsx'], accept_multiple_files=True
    )
    if len(files) != 2:
        st.sidebar.info("Нужно два файла.")
        return
    try:
        df = prepare_data(files)
    except Exception as e:
        st.sidebar.error(e)
        return

    # 2) общие вычисления
    df['avg_rev_week'] = df.groupby('Неделя')['Выручка'].transform('mean')
    df['rev_pct']      = df['Выручка'] / df['avg_rev_week'] * 100

    # 3) фильтры
    st.sidebar.header("2. Фильтры")
    cats  = st.sidebar.multiselect("Категории", df['Категория'].unique(), default=None)
    weeks = st.sidebar.multiselect("Недели",    df['Неделя'].unique(),     default=None)
    if cats:  df = df[df['Категория'].isin(cats)]
    if weeks: df = df[df['Неделя'].isin(weeks)]

    # 4) пороги подсветки
    st.sidebar.header("3. Пороги подсветки")
    share_thr = st.sidebar.slider("Порог % списаний",      0.0, 100.0, 20.0)
    rev_thr   = st.sidebar.slider("Мин. % выручки от среднего", 0.0, 200.0, 80.0)

    # ——————————————— Default report ———————————————
    st.subheader("📈 Общие тренды (по всем неделям)")

    weekly = df.groupby('Неделя').apply(lambda g: pd.Series({
        'Выручка': g['Выручка'].sum(),
        '% списаний': np.average(g['Доля списаний и ЗЦ'], weights=g['Выручка'])
    })).reset_index()

    # revenue trend
    fig_rev = px.line(weekly, x='Неделя', y='Выручка',
                      markers=True, title="Выручка по неделям")
    fig_rev.update_layout(height=400)
    st.plotly_chart(fig_rev, use_container_width=True)

    # waste trend
    fig_waste = px.line(weekly, x='Неделя', y='% списаний',
                        markers=True, title="% списаний по неделям")
    fig_waste.update_layout(height=400)
    st.plotly_chart(fig_waste, use_container_width=True)

    # ————————————— Test Analysis —————————————
    st.sidebar.header("4. Анализ тестового периода")
    test_mode = st.sidebar.checkbox("Включить анализ теста")

    if test_mode:
        # selectors
        all_weeks = sorted(df['Неделя'].unique())
        test_week = st.sidebar.selectbox("Начальная неделя теста", all_weeks, index=len(all_weeks)-1)
        test_day  = st.sidebar.selectbox("Начальный день недели", sorted(df['DayOfWeek'].unique()))

        # prepare weekly series with net revenue
        weekly_full = df.groupby('Неделя').apply(lambda g: pd.Series({
            'revenue_sum': g['Выручка'].sum(),
            'waste_avg':   np.average(g['Доля списаний и ЗЦ'], weights=g['Выручка']) if g['Выручка'].sum()>0 else 0,
        })).reset_index()
        weekly_full['net_sum'] = weekly_full['revenue_sum'] * (1 - weekly_full['waste_avg']/100)

        pre  = weekly_full[weekly_full['Неделя'] < test_week]
        post = weekly_full[weekly_full['Неделя'] >= test_week]

        rev_pre, rev_post       = pre['revenue_sum'].mean(), post['revenue_sum'].mean()
        waste_pre, waste_post   = pre['waste_avg'].mean(),   post['waste_avg'].mean()
        net_pre, net_post       = pre['net_sum'].mean(),     post['net_sum'].mean()

        # metrics
        st.subheader("📋 Сравнение до/после теста")
        st.markdown(f"""
- **Выручка**: {rev_pre:.0f} → {rev_post:.0f} ₽ ({(rev_post/rev_pre-1)*100:.1f}%)
- **% списаний**: {waste_pre:.1f}% → {waste_post:.1f}% ({waste_post-waste_pre:+.1f} п.п.)
- **Чистая выручка**: {net_pre:.0f} → {net_post:.0f} ₽ ({(net_post/net_pre-1)*100:.1f}%)
""")

        # test trends with marker
        fig_rev_t = px.line(weekly_full, x='Неделя', y='revenue_sum', markers=True, title="Выручка по неделям")
        fig_rev_t.add_vline(x=test_week, line_color='red', line_dash='dash', annotation_text="Старт теста")
        fig_rev_t.update_layout(height=400)
        st.plotly_chart(fig_rev_t, use_container_width=True)

        fig_waste_t = px.line(weekly_full, x='Неделя', y='waste_avg', markers=True, title="% списаний по неделям")
        fig_waste_t.add_vline(x=test_week, line_color='red', line_dash='dash')
        fig_waste_t.update_layout(height=400)
        st.plotly_chart(fig_waste_t, use_container_width=True)

        # heatmap normalized per category
        st.subheader(f"🗺 Heatmap норм. выручки (неделя {test_week})")
        df_h = df[df['Неделя'] == test_week]
        heat = df_h.pivot(index='Категория', columns='DayOfWeek', values='Выручка').fillna(0)
        # normalize row-wise
        norm = heat.sub(heat.min(axis=1), axis=0).div(heat.max(axis=1)-heat.min(axis=1), axis=0).fillna(0.5)
        fig_heat = px.imshow(
            norm,
            labels=dict(x="День недели", y="Категория", color="Норм. выручка"),
            x=norm.columns, y=norm.index,
            color_continuous_scale=[(0, 'red'), (0.5, 'white'), (1, 'green')]
        )
        fig_heat.update_traces(xgap=1, ygap=1)
        fig_heat.update_layout(height=600)
        st.plotly_chart(fig_heat, use_container_width=True)

    # ———————————— Export ————————————
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='raw', index=False)
        weekly.to_excel(writer, sheet_name='trend_all', index=False)
        if test_mode:
            weekly_full.to_excel(writer, sheet_name='trend_test', index=False)
            norm.to_excel(writer, sheet_name=f'heat_norm_{test_week}', index=True)
    buf.seek(0)
    st.download_button("💾 Скачать полный отчёт", buf,
                       "dashboard.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


if __name__ == "__main__":
    main()
