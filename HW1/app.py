import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Прогнозирование стоимости автомобилей",
    layout="wide"
)

st.title("Прогнозирование стоимости автомобилей")
st.markdown("---")

@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv'
    df = pd.read_csv(url)
    return df

df = load_data()

st.header("Exploratory Data Analysis (EDA)")

chart_type = st.selectbox(
    "Выберите тип графика для анализа:",
    ["Гистограмма распределения цен", "Корреляционная матрица", 
     "Boxplot цен по типу топлива", "Зависимость цены от года выпуска"]
)

fig, ax = plt.subplots(figsize=(10, 6))

if chart_type == "Гистограмма распределения цен":
    ax.hist(df['selling_price'], bins=50, edgecolor='black', alpha=0.7, color='skyblue')
    ax.set_title('Распределение цен на автомобили')
    ax.set_xlabel('Цена')
    ax.set_ylabel('Количество')
    ax.grid(True, alpha=0.3)
    
elif chart_type == "Корреляционная матрица":
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
    ax.set_title('Корреляционная матрица числовых признаков')
    
elif chart_type == "Boxplot цен по типу топлива":
    if 'fuel' in df.columns:
        df.boxplot(column='selling_price', by='fuel', ax=ax)
        ax.set_title('Распределение цен по типу топлива')
        ax.set_xlabel('Тип топлива')
        ax.set_ylabel('Цена')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    else:
        st.warning("Столбец 'fuel' не найден в данных")
        
elif chart_type == "Зависимость цены от года выпуска":
    if 'year' in df.columns:
        ax.scatter(df['year'], df['selling_price'], alpha=0.5)
        ax.set_title('Зависимость цены от года выпуска')
        ax.set_xlabel('Год выпуска')
        ax.set_ylabel('Цена')
        ax.grid(True, alpha=0.3)

st.pyplot(fig)

st.markdown("---")
st.header("Подготовка данных для модели")

df_processed = df.copy()

if 'mileage' in df_processed.columns and df_processed['mileage'].dtype == object:
    df_processed['mileage'] = df_processed['mileage'].str.extract(r'(\d+\.?\d*)').astype(float)
    
if 'engine' in df_processed.columns and df_processed['engine'].dtype == object:
    df_processed['engine'] = df_processed['engine'].str.extract(r'(\d+\.?\d*)').astype(float)

if 'max_power' in df_processed.columns and df_processed['max_power'].dtype == object:
    df_processed['max_power'] = df_processed['max_power'].str.extract(r'(\d+\.?\d*)').astype(float)

numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
df_processed[numeric_cols] = df_processed[numeric_cols].fillna(df_processed[numeric_cols].median())

selected_features = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']
st.subheader(f"Используемые признаки для модели ({len(selected_features)}):")
st.write(selected_features)

X = df_processed[selected_features]
y = df_processed['selling_price']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LinearRegression()
model.fit(X_scaled, y)

st.success(f"Модель обучена! R² score: {model.score(X_scaled, y):.3f}")

st.header("Прогнозирование стоимости автомобиля")

input_method = st.radio(
    "Выберите способ ввода данных:",
    ["Ввести вручную", "Загрузить CSV файл"]
)

if input_method == "Ввести вручную":
    st.subheader("Введите характеристики автомобиля:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        year = st.number_input("Год выпуска", min_value=1990, max_value=2024, value=2018, step=1)
        km_driven = st.number_input("Пробег (км)", min_value=0, value=50000, step=1000)
        engine = st.number_input("Объем двигателя (cc)", min_value=500, max_value=5000, value=1498, step=100)
    
    with col2:
        max_power = st.number_input("Максимальная мощность (bhp)", min_value=30.0, max_value=500.0, value=103.5, step=5.0)
        mileage = st.number_input("Пробег на топливе (kmpl)", min_value=5.0, max_value=50.0, value=19.3, step=0.1)
        seats = st.number_input("Количество мест", min_value=2, max_value=10, value=5, step=1)
    
    if st.button("Предсказать цену", type="primary"):
        input_data = pd.DataFrame({
            'year': [year],
            'km_driven': [km_driven],
            'mileage': [mileage],
            'engine': [engine],
            'max_power': [max_power],
            'seats': [seats]
        })
        
        input_data = input_data[selected_features]
        
        try:
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            
            st.success(f"Предсказанная стоимость: {prediction:,.0f}")
            
            with st.expander("Показать введенные данные"):
                st.table(input_data)
        except Exception as e:
            st.error(f"Ошибка при предсказании: {e}")

else:
    st.subheader("Загрузите CSV файл с данными автомобилей")
    
    uploaded_file = st.file_uploader("Выберите CSV файл", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df_uploaded = pd.read_csv(uploaded_file)
            
            missing_cols = [col for col in selected_features if col not in df_uploaded.columns]
            
            if missing_cols:
                st.error(f"В файле отсутствуют необходимые столбцы: {missing_cols}")
            else:
                X_uploaded = df_uploaded[selected_features].copy()
                
                for col in ['mileage', 'engine', 'max_power']:
                    if col in X_uploaded.columns and X_uploaded[col].dtype == object:
                        X_uploaded[col] = X_uploaded[col].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)
                
                X_uploaded = X_uploaded.fillna(X.median())
                
                X_uploaded_scaled = scaler.transform(X_uploaded)
                predictions = model.predict(X_uploaded_scaled)
               
                result_df = df_uploaded.copy()
                result_df['predicted_price'] = predictions
                
                st.success(f"Обработано {len(result_df)} автомобилей")
                
                st.subheader("Результаты предсказаний:")
                st.dataframe(result_df[selected_features + ['predicted_price']].head(10))
                
                csv = result_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Скачать результаты в CSV",
                    data=csv,
                    file_name='car_price_predictions.csv',
                    mime='text/csv',
                )
                
        except Exception as e:
            st.error(f"Ошибка при обработке файла: {e}")

st.header("Визуализация весов обученной модели")

if hasattr(model, 'coef_'):
    weights_df = pd.DataFrame({
        'Признак': selected_features,
        'Вес (коэффициент)': model.coef_,
        'Абсолютное значение': np.abs(model.coef_)
    }).sort_values('Абсолютное значение', ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Веса признаков:")
        st.dataframe(weights_df)
    
    with col2:
        st.subheader("Визуализация весов")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['green' if w > 0 else 'red' for w in weights_df['Вес (коэффициент)']]
        
        bars = ax.barh(weights_df['Признак'], weights_df['Вес (коэффициент)'], color=colors)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Значение веса')
        ax.set_title('Веса признаков в линейной регрессии')
        ax.grid(True, alpha=0.3, axis='x')
        
        for bar in bars:
            width = bar.get_width()
            ax.text(width + (0.01 * max(weights_df['Вес (коэффициент)'])), 
                   bar.get_y() + bar.get_height()/2,
                   f'{width:.2f}', va='center')
        
        st.pyplot(fig)
    
    st.subheader("Интерпретация весов:")
    st.write("""
    Положительные веса (зеленые) означают, что увеличение значения признака ведет к увеличению цены.
    
    Отрицательные веса (красные) означают, что увеличение значения признака ведет к уменьшению цены.
    
    Пример:
    - Если вес 'year' положительный - более новые автомобили дороже
    - Если вес 'km_driven' отрицательный - автомобили с большим пробегом дешевле
    """)
else:
    st.warning("Модель не имеет атрибута coef_")

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Приложение для предсказания цен на автомобили</p>
        <p>Используется линейная регрессия на 6 основных признаках</p>
    </div>
    """,
    unsafe_allow_html=True
)