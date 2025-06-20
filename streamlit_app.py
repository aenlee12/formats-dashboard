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
        elif 'день' in c or 'dayofweek' in c: col_map[c] = 'DayOfWeek'
        elif 'доля' in c:         col_map[c] = 'Доля списаний и ЗЦ'
        elif 'выруч' in c:        col_map[c] = 'Выручка'
    return df.rename(columns=col_map)

@st.cache_data
def prepare_data(files):
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
        details = "\n".join(f"{n}: {cols}" for n,cols in info)
        raise ValueError(f"Нужны колонки «Доля списаний и ЗЦ» и «Выручка».\n{details}")
    s = df_share['Доля списаний и ЗЦ'].astype(str).str.replace(',', '.').str.rstrip('%')
    df_share['Доля списаний и ЗЦ'] = pd.to_numeric(s, errors='coerce').fillna(0)
    if df_share['Доля списаний и ЗЦ'].max() <= 1:
        df_share['Доля списаний и ЗЦ'] *= 100
    df_rev['Выручка'] = pd.to_numeric(df_rev['Выручка'], errors='coerce').fillna(0)
    return pd.merge(df_share, df_rev, on=['Категория','Неделя','DayOfWeek'], how='inner')

def main():
    st.title("📊 Дашборд форматов: анализ и тестирование")

    st.sidebar.header("1. Загрузка данных")
    files = st.sidebar.file_uploader(
        "Загрузите два файла:\n• «Доля списаний и ЗЦ»\n• «Выручка»",
        type=['csv','xlsx'], accept_multiple_files=True
    )
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

    st.sidebar.header("2. Фильтры")
    cats = st.sidebar.multiselect("Категории", sorted(df['Категория'].unique()), default=None)
    weeks = st.sidebar.multiselect("Недели", sorted(df['Неделя'].unique()), default=None)
    if cats:
        df = df[df['Категория'].isin(cats)]
    if weeks:
        df = df[df['Неделя'].isin(weeks)]

    st.sidebar.header("3. Heatmap неделя")
    all_weeks = sorted(df['Неделя'].unique())
    sel_week = st.sidebar.selectbox("Выберите неделю для Heatmap", all_weeks, index=len(all_weeks)-1)

    # Общие тренды по категориям
    st.subheader("📈 Общие тренды по неделям и категориям")
    weekly_cat = df.groupby(['Неделя','Категория']).agg(
        Выручка=('Выручка','sum'),
        **{'% списаний': ( 'Доля списаний и ЗЦ',
                         lambda s: np.average(s, weights=df.loc[s.index,'Выручка']))}
    ).reset_index()

    fig_rev = px.line(weekly_cat, x='Неделя', y='Выручка', color='Категория',
                      markers=True, title="Выручка по неделям (категории)")
    fig_rev.update_layout(height=400)
    st.plotly_chart(fig_rev, use_container_width=True)

    fig_waste = px.line(weekly_cat, x='Неделя', y='% списаний', color='Категория',
                        markers=True, title="% списаний по неделям (категории)")
    fig_waste.update_layout(height=400)
    st.plotly_chart(fig_waste, use_container_width=True)

    # Heatmap всегда
    st.subheader(f"🗺 Heatmap выручки: неделя {sel_week}")
    df_h = df[df['Неделя']==sel_week]
    heat = df_h.pivot_table(index='Категория', columns='DayOfWeek', values='Выручка', aggfunc='sum').fillna(0)
    norm = heat.sub(heat.min(axis=1), axis=0).div(heat.max(axis=1)-heat.min(axis=1), axis=0).fillna(0.5)
    fig_heat = px.imshow(norm, labels=dict(x="День недели", y="Категория", color="Норм. выручка"),
                        x=norm.columns, y=norm.index,
                        color_continuous_scale=[(0,'red'),(0.5,'white'),(1,'green')])
    fig_heat.update_traces(xgap=1, ygap=1)
    fig_heat.update_layout(height=600)
    st.plotly_chart(fig_heat, use_container_width=True)

    # Анализ тестового периода
    st.sidebar.header("4. Анализ тестового периода")
    test_mode = st.sidebar.checkbox("Включить анализ теста")
    if test_mode:
        complete = df.groupby('Неделя')['DayOfWeek'].nunique()==7
        weekly_full = df[df['Неделя'].isin(complete[complete].index)].groupby('Неделя').apply(
            lambda g: pd.Series({'revenue_sum':g['Выручка'].sum(),
                                'waste_avg':np.average(g['Доля списаний и ЗЦ'], weights=g['Выручка'])})
        ).reset_index()
        weekly_full['net_sum'] = weekly_full['revenue_sum']*(1-weekly_full['waste_avg']/100)

        all_full = sorted(weekly_full['Неделя'])
        test_week = st.sidebar.selectbox("Начальная неделя теста", all_full, index=len(all_full)-1)

        pre = weekly_full[weekly_full['Неделя']<test_week]
        post= weekly_full[weekly_full['Неделя']>=test_week]
        rev_pre, rev_post = pre['revenue_sum'].mean(), post['revenue_sum'].mean()
        waste_pre, waste_post = pre['waste_avg'].mean(), post['waste_avg'].mean()
        net_pre, net_post = pre['net_sum'].mean(), post['net_sum'].mean()

        st.subheader("📋 Сравнение до/после теста")
        st.markdown(f"""
- **Выручка**: {rev_pre:.0f} → {rev_post:.0f} ₽ ({(rev_post/rev_pre-1)*100:.1f}%)
- **% списаний**: {waste_pre:.1f}% → {waste_post:.1f}% ({waste_post-waste_pre:+.1f} п.п.)
- **Чистая выручка**: {net_pre:.0f} → {net_post:.0f} ₽ ({(net_post/net_pre-1)*100:.1f}%)
""")

        fig1 = px.line(weekly_full, x='Неделя', y='revenue_sum', markers=True, title="Выручка по полным неделям")
        fig1.add_vline(x=test_week, line_color='red', line_dash='dash')
        fig1.update_layout(height=400)
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.line(weekly_full, x='Неделя', y='waste_avg', markers=True, title="% списаний по полным неделям")
        fig2.add_vline(x=test_week, line_color='red', line_dash='dash')
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)

    # Экспорт
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='raw', index=False)
        weekly_cat.to_excel(writer, sheet_name='trend_by_cat', index=False)
        norm.to_excel(writer, sheet_name=f'heat_{sel_week}', index=True)
        if test_mode:
            weekly_full.to_excel(writer, sheet_name='trend_test', index=False)
    buf.seek(0)
    st.download_button("💾 Скачать отчёт (Excel)", buf,
                       "dashboard.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

if __name__ == "__main__":
    main()
