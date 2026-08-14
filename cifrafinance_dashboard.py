# ============================================
# ДАШБОРД ДЛЯ МОНИТОРИНГА ПОРТФЕЛЯ
# Банк "ЦифраФинанс"
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Настройка страницы
st.set_page_config(
    page_title="ЦифраФинанс - Мониторинг портфеля",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Заголовок
st.title("🏦 ЦифраФинанс - Еженедельный мониторинг кредитного портфеля")
st.markdown(f"**Дата обновления:** {datetime.now().strftime('%d.%m.%Y %H:%M')}")

# ============================================
# ЗАГРУЗКА ДАННЫХ
# ============================================

@st.cache_data
def load_data():
    """Загрузка данных из CSV"""
    # Загружаем данные из файла
    df = pd.read_csv('loan_portfolio.csv')
    
    # Преобразуем дату
    df['issue_date'] = pd.to_datetime(df['issue_date'])
    
    # Создаем квартал
    df['quarter'] = df['issue_date'].dt.to_period('Q').astype(str)
    
    # Создаем сегменты риска
    median_score = df['credit_score'].median()
    median_dti = df['dti_ratio'].median()
    df['score_above_median'] = df['credit_score'] > median_score
    df['dti_below_median'] = df['dti_ratio'] <= median_dti
    
    def assign_risk_segment(row):
        if row['score_above_median'] and row['dti_below_median']:
            return 'LOW'
        elif (not row['score_above_median']) and (not row['dti_below_median']):
            return 'HIGH'
        else:
            return 'MEDIUM'
    
    df['risk_segment'] = df.apply(assign_risk_segment, axis=1)
    
    # Удаляем отрицательные значения
    numeric_cols = ['monthly_income', 'loan_amount', 'monthly_payment']
    for col in numeric_cols:
        df = df[df[col] >= 0]
    
    # Расчет дохода
    df['income'] = 0
    df.loc[df['is_default'] == 0, 'income'] = (
        df.loc[df['is_default'] == 0, 'monthly_payment'] * 
        df.loc[df['is_default'] == 0, 'loan_term_months'] - 
        df.loc[df['is_default'] == 0, 'loan_amount']
    )
    df.loc[df['income'] < 0, 'income'] = 0
    
    # Расчет EL
    LGD = 0.45
    df['el'] = df['is_default'] * df['loan_amount'] * LGD
    
    return df

# Загружаем данные
try:
    df = load_data()
except FileNotFoundError:
    st.error("Файл loan_portfolio.csv не найден. Убедитесь, что он находится в той же папке.")
    st.stop()

# ============================================
# БОКОВАЯ ПАНЕЛЬ - ФИЛЬТРЫ
# ============================================

st.sidebar.header("🔍 Фильтры")

# Фильтр по региону
regions = ['Все'] + sorted(df['region'].unique().tolist())
selected_regions = st.sidebar.multiselect(
    "Регион",
    options=regions,
    default=['Все']
)

# Фильтр по каналу
channels = ['Все'] + sorted(df['channel'].unique().tolist())
selected_channels = st.sidebar.multiselect(
    "Канал",
    options=channels,
    default=['Все']
)

# Фильтр по продукту
products = ['Все'] + sorted(df['loan_product'].unique().tolist())
selected_products = st.sidebar.multiselect(
    "Продукт",
    options=products,
    default=['Все']
)

# Фильтр по периоду
min_date = df['issue_date'].min()
max_date = df['issue_date'].max()
date_range = st.sidebar.date_input(
    "Период",
    value=[min_date, max_date],
    min_value=min_date,
    max_value=max_date
)

# Применение фильтров
def apply_filters(data):
    filtered = data.copy()
    
    if 'Все' not in selected_regions:
        filtered = filtered[filtered['region'].isin(selected_regions)]
    
    if 'Все' not in selected_channels:
        filtered = filtered[filtered['channel'].isin(selected_channels)]
    
    if 'Все' not in selected_products:
        filtered = filtered[filtered['loan_product'].isin(selected_products)]
    
    if len(date_range) == 2:
        filtered = filtered[
            (filtered['issue_date'] >= pd.to_datetime(date_range[0])) &
            (filtered['issue_date'] <= pd.to_datetime(date_range[1]))
        ]
    
    return filtered

df_filtered = apply_filters(df)

# ============================================
# KPI
# ============================================

st.header("📊 Ключевые показатели")

# Расчет KPI
total_portfolio = df_filtered['loan_amount'].sum()
default_rate = df_filtered['is_default'].mean()
target_default_rate = 0.10
el_total = df_filtered['el'].sum()
avg_dti = df_filtered['dti_ratio'].mean()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Объем портфеля",
        value=f"{total_portfolio:,.0f} руб",
        delta=f"{len(df_filtered):,} кредитов"
    )

with col2:
    delta_color = "normal"
    if default_rate < target_default_rate:
        delta_color = "normal"
    else:
        delta_color = "inverse"
    st.metric(
        label="Default Rate",
        value=f"{default_rate*100:.1f}%",
        delta=f"{target_default_rate*100:.1f}% (цель)",
        delta_color=delta_color
    )

with col3:
    st.metric(
        label="Expected Loss (EL)",
        value=f"{el_total:,.0f} руб",
        delta=f"{(el_total/total_portfolio*100):.1f}% от портфеля"
    )

with col4:
    st.metric(
        label="Средний DTI",
        value=f"{avg_dti:.1f}%",
        delta=f"Медиана: {df_filtered['dti_ratio'].median():.1f}%"
    )

# ============================================
# ДИНАМИКА ВЫДАЧ И ДЕФОЛТОВ ПО КВАРТАЛАМ
# ============================================

st.header("📈 Динамика выдач и дефолтов по кварталам")

# Агрегация по кварталам
quarterly_data = df_filtered.groupby('quarter').agg({
    'loan_amount': 'sum',
    'is_default': 'sum',
    'loan_id': 'count'
}).reset_index()
quarterly_data.columns = ['quarter', 'loan_amount', 'n_defaults', 'n_loans']
quarterly_data['default_rate'] = quarterly_data['n_defaults'] / quarterly_data['n_loans'] * 100

# Создаем двухосевой график
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Столбцы - объем выдач
fig.add_trace(
    go.Bar(
        x=quarterly_data['quarter'],
        y=quarterly_data['loan_amount'] / 1e6,
        name="Объем выдач, млн руб",
        marker_color='#3498db'
    ),
    secondary_y=False
)

# Линия - уровень дефолтов
fig.add_trace(
    go.Scatter(
        x=quarterly_data['quarter'],
        y=quarterly_data['default_rate'],
        name="Default Rate, %",
        marker_color='#e74c3c',
        line=dict(width=3)
    ),
    secondary_y=True
)

fig.update_layout(
    title="Объем выдач и уровень дефолтов по кварталам",
    xaxis_title="Квартал",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=400
)

fig.update_yaxes(title_text="Объем выдач, млн руб", secondary_y=False)
fig.update_yaxes(title_text="Default Rate, %", secondary_y=True, range=[0, max(quarterly_data['default_rate']) * 1.2])

st.plotly_chart(fig, use_container_width=True)

# ============================================
# ТЕПЛОВАЯ КАРТА: КАНАЛ × ТИП ЗАНЯТОСТИ
# ============================================

st.header("🌡️ Тепловая карта Default Rate: Канал × Тип занятости")

# Создаем сводную таблицу
heatmap_data = df_filtered.pivot_table(
    values='is_default',
    index='channel',
    columns='employment_type',
    aggfunc='mean'
) * 100

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(
    heatmap_data,
    annot=True,
    fmt='.1f',
    cmap='RdYlGn_r',
    cbar_kws={'label': 'Default Rate, %'},
    ax=ax,
    vmin=0,
    vmax=30
)
ax.set_title('Default Rate по каналам и типам занятости', fontsize=14)
ax.set_xlabel('Тип занятости')
ax.set_ylabel('Канал')
plt.tight_layout()

st.pyplot(fig)

# ============================================
# SCATTER PLOT: CREDIT_SCORE / DTI
# ============================================

st.header("🎯 Scatter Plot: Credit Score vs DTI")

# Подготовка данных для scatter
scatter_data = df_filtered.copy()
scatter_data['color'] = scatter_data['is_default'].map({0: 'Не дефолт', 1: 'Дефолт'})

fig = px.scatter(
    scatter_data,
    x='credit_score',
    y='dti_ratio',
    color='color',
    color_discrete_map={'Не дефолт': '#2ecc71', 'Дефолт': '#e74c3c'},
    hover_data=['loan_id', 'region', 'channel', 'loan_amount'],
    title='Credit Score vs DTI (цвет = дефолт)',
    labels={'credit_score': 'Credit Score', 'dti_ratio': 'DTI, %'}
)

# Добавляем линии медиан
median_score = scatter_data['credit_score'].median()
median_dti = scatter_data['dti_ratio'].median()

fig.add_hline(y=median_dti, line_dash="dash", line_color="gray", 
              annotation_text=f"Медиана DTI: {median_dti:.1f}%")
fig.add_vline(x=median_score, line_dash="dash", line_color="gray",
              annotation_text=f"Медиана Score: {median_score:.0f}")

fig.update_layout(height=500)
st.plotly_chart(fig, use_container_width=True)

# ============================================
# ТАБЛИЦА ПЛАН-ФАКТ
# ============================================

st.header("📋 План-факт анализ по справочнику")

# Загружаем справочник
try:
    branch_ref = pd.read_csv('branch_reference.csv')
except FileNotFoundError:
    st.warning("Файл branch_reference.csv не найден. Используем встроенные данные.")
    branch_data = {
        'branch_id': ['B001', 'B002', 'B003', 'B004', 'B005', 'B006', 'B007'],
        'branch_name': ['Moscow', 'Moscow', 'SPb', 'SPb', 'Ural', 'Siberia', 'South'],
        'channel': ['branch', 'online', 'branch', 'online', 'branch', 'branch', 'partner'],
        'region': ['Moscow', 'Moscow', 'SPb', 'SPb', 'Ural', 'Siberia', 'South'],
        'target_default_rate': [0.09, 0.10, 0.10, 0.09, 0.09, 0.12, 0.16],
        'target_avg_credit_score': [623, 622, 612, 616, 601, 594, 577],
        'target_avg_dti': [47.5, 52.7, 50.4, 53.0, 48.8, 51.6, 54.7],
    }
    branch_ref = pd.DataFrame(branch_data)

# Расчет фактических показателей
actual_metrics = df_filtered.groupby(['region', 'channel']).agg({
    'is_default': 'mean',
    'credit_score': 'mean',
    'dti_ratio': 'mean',
    'loan_id': 'count'
}).reset_index()
actual_metrics.columns = ['region', 'channel', 'actual_default_rate', 'actual_avg_credit_score', 'actual_avg_dti', 'n_loans']

# Объединение с планом
plan_fact = actual_metrics.merge(
    branch_ref[['region', 'channel', 'target_default_rate', 'target_avg_credit_score', 'target_avg_dti']],
    on=['region', 'channel'],
    how='left'
)

# Расчет отклонений
plan_fact['has_plan'] = ~plan_fact['target_default_rate'].isna()
plan_fact['default_diff'] = plan_fact['actual_default_rate'] - plan_fact['target_default_rate']
plan_fact['score_diff'] = plan_fact['actual_avg_credit_score'] - plan_fact['target_avg_credit_score']
plan_fact['dti_diff'] = plan_fact['actual_avg_dti'] - plan_fact['target_avg_dti']

# Функция подсветки
def highlight_diff(val, threshold=0.01):
    if pd.isna(val):
        return ''
    if val > threshold:
        return 'background-color: #ffcccc'  # красный - плохо
    elif val < -threshold:
        return 'background-color: #ccffcc'  # зеленый - хорошо
    return ''

def highlight_diff_score(val, threshold=0):
    if pd.isna(val):
        return ''
    if val < threshold:
        return 'background-color: #ffcccc'
    elif val > threshold:
        return 'background-color: #ccffcc'
    return ''

# Форматирование для отображения
display_columns = ['region', 'channel', 'n_loans', 'actual_default_rate', 'target_default_rate', 
                   'default_diff', 'actual_avg_credit_score', 'target_avg_credit_score',
                   'actual_avg_dti', 'target_avg_dti']

styled_df = plan_fact[display_columns].copy()
styled_df['actual_default_rate'] = styled_df['actual_default_rate'] * 100
styled_df['target_default_rate'] = styled_df['target_default_rate'] * 100
styled_df['default_diff'] = styled_df['default_diff'] * 100

styled_df.columns = ['Регион', 'Канал', 'Кол-во', 'Факт DR, %', 'План DR, %', 
                     'Отклонение, п.п.', 'Факт Score', 'План Score', 'Факт DTI, %', 'План DTI, %']

# Отображаем таблицу
st.dataframe(
    styled_df,
    use_container_width=True,
    hide_index=True
)

# ============================================
# ДОПОЛНИТЕЛЬНЫЕ ГРАФИКИ
# ============================================

st.header("📊 Дополнительный анализ")

col1, col2 = st.columns(2)

with col1:
    # Распределение дефолтов по продуктам
    st.subheader("Default Rate по продуктам")
    product_default = df_filtered.groupby('loan_product')['is_default'].mean() * 100
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#e74c3c' if x > df_filtered['is_default'].mean()*100 else '#2ecc71' for x in product_default]
    product_default.plot(kind='bar', ax=ax, color=colors)
    ax.axhline(y=df_filtered['is_default'].mean()*100, color='red', linestyle='--', 
               label=f'Среднее: {df_filtered["is_default"].mean()*100:.1f}%')
    ax.set_title('Default Rate по продуктам')
    ax.set_ylabel('Default Rate, %')
    ax.set_xlabel('Продукт')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    # Распределение по регионам
    st.subheader("Default Rate по регионам")
    region_default = df_filtered.groupby('region')['is_default'].mean() * 100
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#e74c3c' if x > df_filtered['is_default'].mean()*100 else '#2ecc71' for x in region_default]
    region_default.plot(kind='bar', ax=ax, color=colors)
    ax.axhline(y=df_filtered['is_default'].mean()*100, color='red', linestyle='--',
               label=f'Среднее: {df_filtered["is_default"].mean()*100:.1f}%')
    ax.set_title('Default Rate по регионам')
    ax.set_ylabel('Default Rate, %')
    ax.set_xlabel('Регион')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

# ============================================
# РАСПРЕДЕЛЕНИЕ ПО СЕГМЕНТАМ РИСКА
# ============================================

st.subheader("📊 Распределение по сегментам риска")
segment_data = df_filtered.groupby('risk_segment').agg({
    'loan_id': 'count',
    'is_default': 'mean',
    'loan_amount': 'sum'
}).reset_index()
segment_data.columns = ['Сегмент', 'Количество', 'Default Rate', 'Объем кредитов']
segment_data['Default Rate'] = segment_data['Default Rate'] * 100

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {'LOW': '#2ecc71', 'MEDIUM': '#f39c12', 'HIGH': '#e74c3c'}
    bars = ax.bar(segment_data['Сегмент'], segment_data['Default Rate'], 
                  color=[colors[x] for x in segment_data['Сегмент']])
    ax.set_title('Default Rate по сегментам риска')
    ax.set_ylabel('Default Rate, %')
    ax.set_xlabel('Сегмент')
    ax.axhline(y=df_filtered['is_default'].mean()*100, color='red', linestyle='--',
               label=f'Среднее: {df_filtered["is_default"].mean()*100:.1f}%')
    ax.legend()
    ax.grid(True, alpha=0.3)
    for bar, val in zip(bars, segment_data['Default Rate']):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {'LOW': '#2ecc71', 'MEDIUM': '#f39c12', 'HIGH': '#e74c3c'}
    bars = ax.bar(segment_data['Сегмент'], segment_data['Объем кредитов'] / 1e6,
                  color=[colors[x] for x in segment_data['Сегмент']])
    ax.set_title('Объем кредитов по сегментам риска')
    ax.set_ylabel('Объем, млн руб')
    ax.set_xlabel('Сегмент')
    ax.grid(True, alpha=0.3)
    for bar, val in zip(bars, segment_data['Объем кредитов'] / 1e6):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{val:.1f} млн', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    st.pyplot(fig)

# ============================================
# ФУТТЕР
# ============================================

st.markdown("---")
st.markdown(f"*Дашборд создан для банка «ЦифраФинанс» | Данные обновлены: {datetime.now().strftime('%d.%m.%Y %H:%M')}*")