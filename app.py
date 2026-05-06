import os
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ---------- Page configuration ----------
st.set_page_config(
    page_title='Commodity Price Prediction Dashboard',
    page_icon='📈',
    layout='wide',
)

# ---------- Title and description ----------
st.title('📈 Commodity Price Prediction Dashboard')
st.markdown(
    'An interactive dashboard that loads a commodity price dataset, '
    'trains a Linear Regression model, and forecasts future prices.'
)

# ---------- Data loading ----------
@st.cache_data
def load_data():
    csv_path = None
    for dirname, _, filenames in os.walk('/kaggle/input'):
        for filename in filenames:
            if filename.lower().endswith('.csv'):
                csv_path = os.path.join(dirname, filename)
                break
        if csv_path:
            break
    if csv_path is None:
        # Local fallback
        for filename in os.listdir('.'):
            if filename.lower().endswith('.csv'):
                csv_path = filename
                break
    df = pd.read_csv(csv_path)

    date_col = None
    for col in df.columns:
        try:
            df[col] = pd.to_datetime(df[col], errors='raise')
            date_col = col
            break
        except Exception:
            continue
    if date_col is None:
        df['date'] = pd.to_datetime(pd.RangeIndex(len(df)), unit='D', origin='2020-01-01')
        date_col = 'date'

    df = df.sort_values(date_col).reset_index(drop=True)
    df['days'] = (df[date_col] - df[date_col].min()).dt.days
    target_col = [c for c in df.select_dtypes(include='number').columns if c != 'days'][0]
    return df, date_col, target_col

@st.cache_resource
def train_model(df, target_col):
    X = df[['days']]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, r2_score(y_test, y_pred), mean_squared_error(y_test, y_pred)

df, date_col, target_col = load_data()
model, r2, mse = train_model(df, target_col)

# ---------- Dataset preview ----------
st.subheader('📁 Dataset Preview')
st.dataframe(df.head(10), use_container_width=True)

# ---------- Model metrics ----------
st.subheader('📊 Model Performance')
m1, m2, m3 = st.columns(3)
m1.metric('R² Score', f'{r2:.4f}')
m2.metric('Mean Squared Error', f'{mse:.4f}')
m3.metric('Records', f'{len(df):,}')

# ---------- Interactive prediction controls ----------
st.subheader('🔮 Predict Future Price')
col_a, col_b = st.columns([2, 1])
with col_a:
    days_ahead = st.slider(
        'Days into the future',
        min_value=1, max_value=365, value=30, step=1,
    )
with col_b:
    predict_clicked = st.button('Predict', use_container_width=True)

future_days = int(df['days'].max()) + int(days_ahead)
future_date = df[date_col].max() + pd.Timedelta(days=int(days_ahead))
predicted_price = float(model.predict([[future_days]])[0])

if predict_clicked:
    st.success(
        f'Predicted {target_col} on {future_date.date()} '
        f'(day {future_days}): {predicted_price:.2f}'
    )
else:
    st.info('Adjust the slider and click **Predict** to forecast a future price.')

# ---------- Visualization ----------
st.subheader('📉 Price Trend and Forecast')
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(df[date_col], df[target_col], s=20, label='Historical price', color='#1f77b4')
ax.plot(df[date_col], model.predict(df[['days']]), color='red', label='Regression trend')
ax.scatter([future_date], [predicted_price], color='green', s=120, marker='*', label='Prediction')
ax.set_xlabel('Date')
ax.set_ylabel(target_col)
ax.set_title('Commodity Price Trend with Forecast')
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
st.pyplot(fig)

# ---------- AI-style analytical insights ----------
st.subheader('🧠 Analytical Insights')
latest_price = float(df[target_col].iloc[-1])
change_pct = (predicted_price - latest_price) / latest_price * 100 if latest_price else 0.0
slope = float(model.coef_[0])

if change_pct > 1:
    trend_label = 'upward trend 📈'
    remark = 'Prices are projected to rise. Consider monitoring supply-side factors.'
elif change_pct < -1:
    trend_label = 'downward trend 📉'
    remark = 'Prices are projected to decline. Demand pressure may be easing.'
else:
    trend_label = 'stable movement ➡️'
    remark = 'Prices appear stable with no significant short-term change.'

st.markdown(
    f"- **Latest observed price:** {latest_price:.2f}\n"
    f"- **Predicted price ({days_ahead} days ahead):** {predicted_price:.2f}\n"
    f"- **Expected change:** {change_pct:+.2f}%\n"
    f"- **Model daily slope:** {slope:+.4f} per day\n"
    f"- **Trend assessment:** {trend_label}\n\n"
    f"_{remark}_"
)

# ---------- Sidebar and footer refinements ----------
with st.sidebar:
    st.header('ℹ️ About')
    st.write('Commodity Price Prediction Dashboard')
    st.write('Built with Streamlit, pandas, scikit-learn and matplotlib.')
    st.caption(f'Dataset rows: {len(df):,}')
    st.caption(f'Target column: {target_col}')

st.markdown('---')
st.caption('© 2026 Commodity Price Prediction Demo — Streamlit ML Dashboard')
