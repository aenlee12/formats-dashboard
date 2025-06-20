import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO

st.set_page_config(page_title="Дашборд форматов", layout="wide")

def detect_columns(df):
    cols = list(df.columns)
    lower = [c.lower() for c in cols]
    # find by keywords
    def find(keywords):
        for i, c in enumerate(lower):
            for kw in keywords:
                if kw in c:
                    return cols[i]
        return None
    return {
        'category':  find(['категор']),
        'week':      find(['недел']),
        'day':       find(['день','dayofweek']),
        'share':     find(['доля']),
        'revenue':   find(['выруч'])
    }

@st.cache_data
def load_and_prepare(files):
    # load files as DataFrames
    dfs = [pd.read_csv(f) if f.name.lower().endswith('.csv') else pd.read_excel(f, header=0) 
           for f in files]
    # detect columns in each
    info = []
    share_df = rev_df = None
    for df in dfs:
        cols = detect_columns(df)
        info.append(cols)
        if cols['share'] and all(cols[k] for k in ('category','week','day')):
            share_df = df
            share_cols = cols
        if cols['revenue'] and all(cols[k] for k in ('category','week','day')):
            rev_df = df
            rev_cols = cols
    if share_df is None or rev_df is None:
        raise ValueError("Не удалось найти необходимые столбцы.\n" +
                         "\n".join(str(i) for i in info))
    # rename columns to standard
    def rename_df(df, cols):
        return df.rename(columns={
            cols['category']: 'Категория',
            cols['week']:     'Неделя',
            cols['day']:      'DayOfWeek'
        })
    df_s = rename_df(share_df, share_cols)
    df_r = rename_df(rev_df,   rev_cols)
    # convert share to numeric %
    scol = share_cols['share']
    df_s['Доля списаний и ЗЦ'] = pd.to_numeric(
        df_s[scol].astype(str).str.replace(',','.'), errors='coerce'
    ).fillna(0)
    # convert revenue
    rcol = rev_cols['revenue']
    df_r['Выручка'] = pd.to_numeric(df_r[rcol], errors='coerce').fillna(0)
    # keep only necessary columns
    df_s = df_s[['Категория','Неделя','DayOfWeek','Доля списаний и ЗЦ']]
    df_r = df_r[['Категория','Неделя','DayOfWeek','Выручка']]
    # merge
    dfm = pd.merge(df_s, df_r, on=['Категория','Неделя','DayOfWeek'], how='inner')
    return dfm

def main():
    st.title("📊 Дашборд форматов: анализ и тестирование")

    st.sidebar.header("1. Загрузка данных")
    files = st.sidebar.file_uploader(
        "Выберите два файла:\n• с % списаний\n• с выручкой",
        type=['csv','xlsx'], accept_multiple_files=True
    )
    if len(files)!=2:
        st.sidebar.info("Нужно загрузить ровно два файла.")
        return

    try:
        df = load_and_prepare(files)
    except Exception as e:
        st.sidebar.error(str(e))
        return

    # compute weekly averages
    df['avg_rev_week'] = df.groupby('Неделя')['Выручка'].transform('mean')
    df['rev_pct'] = df['Выручка']/df['avg_rev_week']*100

    # filters
    st.sidebar.header("2. Фильтры")
    cats  = st.sidebar.multiselect("Категории", df['Категория'].unique(), default=None)
    weeks = st.sidebar.multiselect("Недели",    df['Неделя'].unique(),     default=None)
    if cats:  df = df[df['Категория'].isin(cats)]
    if weeks:df = df[df['Неделя'].isin(weeks)]

    # test period inputs
    st.sidebar.header("3. Тестовый период")
    all_weeks = sorted(df['Неделя'].unique())
    test_week = st.sidebar.selectbox("Начальная неделя", all_weeks, index=len(all_weeks)-1)
    test_day  = st.sidebar.selectbox("Начальный день недели", sorted(df['DayOfWeek'].unique()))

    # thresholds
    st.sidebar.header("4. Пороги подсветки")
    share_thr = st.sidebar.slider("Порог % списаний", 0.0,100.0,20.0)
    rev_thr   = st.sidebar.slider("Мин. % выручки от среднего",0.0,200.0,80.0)

    # comparison table pre/post
    weekly = df.groupby('Неделя').apply(lambda g: pd.Series({
        'revenue_sum': g['Выручка'].sum(),
        'waste_avg':   np.average(g['Доля списаний и ЗЦ'], weights=g['Выручка'])
                         if g['Выручка'].sum()>0 else g['Доля списаний и ЗЦ'].mean(),
        'net_sum':     g['Выручка'].sum()*(1 - np.average(g['Доля списаний и ЗЦ'], 
                           weights=g['Выручка'])/100 if g['Выручка'].sum()>0 else 0)
    })).reset_index()

    pre  = weekly[weekly['Неделя']<test_week]
    post = weekly[weekly['Неделя']>=test_week]
    rev_pre, rev_post = pre['revenue_sum'].mean(), post['revenue_sum'].mean()
    waste_pre, waste_post = pre['waste_avg'].mean(), post['waste_avg'].mean()
    net_pre, net_post = pre['net_sum'].mean(), post['net_sum'].mean()

    st.subheader("📋 Сравнение до/после старта теста")
    st.markdown(f"""
- Выручка: **{rev_pre:.0f} → {rev_post:.0f} ₽** ({(rev_post/rev_pre-1)*100:.1f}%)
- % списаний: **{waste_pre:.1f}% → {waste_post:.1f}%** ({(waste_post-waste_pre):+.1f} п.п.)
- Чистая выручка: **{net_pre:.0f} → {net_post:.0f} ₽** ({(net_post/net_pre-1)*100:.1f}%)
""")

    # trends
    st.subheader("📈 Тренды по неделям")
    fig1 = px.line(weekly, x='Неделя', y='revenue_sum', markers=True, title="Выручка по неделям")
    fig1.add_vline(x=test_week, line_color='red', line_dash='dash')
    fig1.update_layout(height=450)
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.line(weekly, x='Неделя', y='waste_avg', markers=True, title="% списаний по неделям")
    fig2.add_vline(x=test_week, line_color='red', line_dash='dash')
    fig2.update_layout(height=450)
    st.plotly_chart(fig2, use_container_width=True)

    # heatmap normalized per category
    st.subheader(f"🗺 Heatmap нормированной выручки (неделя {test_week})")
    df_h = df[df['Неделя']==test_week]
    heat = df_h.pivot(index='Категория', columns='DayOfWeek', values='Выручка').fillna(0)
    # row normalization
    norm = heat.sub(heat.min(axis=1), axis=0).div(heat.max(axis=1)-heat.min(axis=1), axis=0).fillna(0.5)
    fig_heat = px.imshow(norm, 
                        labels=dict(x="День недели", y="Категория", color="Норм. выручка"),
                        x=norm.columns, y=norm.index,
                        color_continuous_scale=[(0,'red'),(0.5,'white'),(1,'green')])
    fig_heat.update_traces(xgap=1, ygap=1)
    fig_heat.update_layout(height=600)
    st.plotly_chart(fig_heat, use_container_width=True)

    # export
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='raw', index=False)
        weekly.to_excel(writer, sheet_name='weekly', index=False)
        norm.to_excel(writer, sheet_name=f'heat_norm_{test_week}', index=True)
    buf.seek(0)
    st.download_button("💾 Скачать отчёт (Excel)", buf,
                       "dashboard.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

if __name__ == "__main__":
    main()
