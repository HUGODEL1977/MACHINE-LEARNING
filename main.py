import pandas as pd
from pandas import json_normalize
import folium
from folium.plugins import MarkerCluster
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LinearRegression
from pydantic import BaseModel
from textblob import TextBlob
import nltk



# Especifica la ruta del archivo Parquet
ruta_parquet = "restaurants_limpio.parquet"

# Lee el archivo Parquet en un DataFrame
df = pd.read_parquet(ruta_parquet)

# Define una función para evaluar la cadena y manejar casos nulos
def safe_eval(attr):
    try:
        if pd.notna(attr) and isinstance(attr, str):
            return eval(attr)
        else:
            return {}
    except:
        return {}

# Normaliza 'attributes' usando la función safe_eval

df_attributes = df['attributes'].apply(lambda x: json_normalize(safe_eval(x)))


# Configura pandas para mostrar todas las columnas
pd.set_option('display.max_columns', None) # Convierte temporalmente todas las columnas a cadena para la visualización
df_str = df.applymap(lambda x: str(x))

df = df.drop(['address', 'postal_code'], axis=1)
# Elimina filas con valores NaN
df = df.dropna()

# Restaura la configuración original de pandas después de imprimir
pd.reset_option('display.max_columns')

# Muestra todas las filas y columnas del DataFrame como una cadena
print(df_str.head(1).to_string())


# Muestra las primeras filas del DataFrame después de las correcciones
print(df.head(3))

########################################

# columna 'city' en tu DataFrame
#  Se Ajusta esta línea de acuerdo a los datos reales
city_input = input("Ingrese la ciudad: ")

# Se Convierte la entrada de la ciudad a minúsculas para hacer la coincidencia insensible a mayúsculas y minúsculas
city_input = city_input.lower()

# Filtrar el DataFrame por la ciudad proporcionada
city_df = df[df['city'].str.lower() == city_input]

# Mostrar todos los restaurantes con sus estrellas y tipo de comida para la ciudad dada
if not city_df.empty:
    print(f"Recomendación para la ciudad {city_input}:")
    for _, row in city_df.iterrows():
        print(f"Ciudad: {row['city']}, Restaurante: {row['name']}, Número de estrellas: {row['stars']}, Tipo actividad y comida: {row['categories']}")
else:
    print(f"No hay datos disponibles para la ciudad {city_input}.")


######################################

# Cargar datos de restaurantes
ruta_parquet = "restaurants_limpio.parquet"
df = pd.read_parquet(ruta_parquet)

# Ingeniería de Características
df['stars_review_interaction'] = df['stars'] * df['review_count']

# Filtra solo los restaurantes abiertos
df_abiertos = df[df['is_open'] == 1].copy()

# Selecciona características
features = df_abiertos[['latitude', 'longitude', 'stars', 'review_count', 'stars_review_interaction', 'state', 'categories']].copy()
target_variable = df_abiertos['review_count']

# Codificar variables categóricas sin generar advertencias
label_encoder = LabelEncoder()
features['state_encoded'] = label_encoder.fit_transform(features['state'])

# Normalización y Estandarización
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features[['latitude', 'longitude', 'stars', 'review_count', 'stars_review_interaction', 'state_encoded']])

# Dividir el conjunto de datos
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target_variable, test_size=0.2, random_state=42)

# Construir y entrenar el modelo Random Forest Regressor para predicción basada en estrellas
param_grid = {'n_estimators': [50, 100, 150], 'max_depth': [None, 10, 20]}
grid_search_rf_stars = GridSearchCV(RandomForestRegressor(), param_grid, cv=3)
grid_search_rf_stars.fit(X_train, y_train)

# Obtener las predicciones de crecimiento para cada estado
df_abiertos['predicted_growth'] = grid_search_rf_stars.predict(features_scaled)

# Calcular el porcentaje de crecimiento para cada estado
df_abiertos['growth_percentage'] = ((df_abiertos['predicted_growth'] - df_abiertos['review_count']) / df_abiertos['review_count']) * 100

# Obtener los dos estados que más crecen y los dos estados que más decrecen
top_growth_states = df_abiertos.groupby('state')['growth_percentage'].mean().nlargest(2).reset_index()
top_decrease_states = df_abiertos.groupby('state')['growth_percentage'].mean().nsmallest(2).reset_index()

# Obtener el tipo de comida predominante en los estados que más crecen
top_growth_states_categories = df_abiertos[df_abiertos['state'].isin(top_growth_states['state'])].groupby('state')['categories'].apply(lambda x: x.mode().iloc[0]).reset_index()

# Obtener el tipo de comida predominante en los estados que más decrecen
top_decrease_states_categories = df_abiertos[df_abiertos['state'].isin(top_decrease_states['state'])].groupby('state')['categories'].apply(lambda x: x.mode().iloc[0]).reset_index()

# Convertir los resultados de tasas de crecimiento a porcentajes
top_growth_states['growth_percentage'] *= 100
top_decrease_states['growth_percentage'] *= 100

# Imprimir los resultados
print(f'Dos estados que más crecen:')
print(top_growth_states)
print(f'Tipo de comida predominante en estos estados:')
print(top_growth_states_categories)

print(f'\nDos estados que más decrecen:')
print(top_decrease_states)
print(f'Tipo de comida predominante en estos estados:')
print(top_decrease_states_categories)

#####################################################
#Calcular el crecimiento de un sector de restaurantes en una región o país como Estados Unidos puede ser un desafío y 
# generalmente se hace utilizando una combinación de datos y métricas específicas. Algunas formas comunes de evaluar 
# el crecimiento en este sector incluyen:

# Número de Establecimientos: Observar la cantidad total de restaurantes y cómo ha variado con el tiempo. 
# Este puede ser un indicador de crecimiento si hay un aumento neto en la cantidad de establecimientos.

# Empleo en el Sector: Analizar la cantidad de empleos generados por la industria de restaurantes y cómo ha evolucionado con el tiempo. 
# Un aumento en el empleo podría indicar un crecimiento en el sector.

# Ingresos y Ventas: Examinar los ingresos totales y las ventas generadas por los restaurantes. 
# Un aumento en estos valores podría indicar un crecimiento económico en el sector.

# Reseñas y Puntuaciones: Utilizar datos de reseñas y puntuaciones de restaurantes para evaluar su popularidad y la percepción del cliente. 
# Un aumento en las reseñas positivas podría indicar un crecimiento.

#Tendencias en la Demanda: Observar las tendencias en la demanda de diferentes tipos de alimentos y restaurantes. 
# Por ejemplo, un aumento en la demanda de alimentos saludables podría indicar un crecimiento en ese segmento.

# Datos Geoespaciales: Utilizar datos geoespaciales para evaluar la apertura y cierre de restaurantes en áreas específicas.

# Datos de Reservas y Tráfico de Clientes: Analizar datos de reservas y patrones de tráfico de clientes para entender 
# cómo la afluencia de personas a los restaurantes ha cambiado con el tiempo.


# Cargar datos de restaurantes
ruta_parquet = "restaurants_limpio.parquet"
df = pd.read_parquet(ruta_parquet)

# Ingeniería de Características
df['stars_review_interaction'] = df['stars'] * df['review_count']

# Filtra solo los restaurantes abiertos
df_abiertos = df[df['is_open'] == 1].copy()

# Selecciona características
features = df_abiertos[['latitude', 'longitude', 'stars', 'review_count', 'stars_review_interaction', 'state', 'categories']].copy()
target_variable = df_abiertos['review_count']

# Codificar variables categóricas sin generar advertencias
label_encoder = LabelEncoder()
features['state_encoded'] = label_encoder.fit_transform(features['state'])

# Normalización y Estandarización
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features[['latitude', 'longitude', 'stars', 'review_count', 'stars_review_interaction', 'state_encoded']])

# Dividir el conjunto de datos
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target_variable, test_size=0.2, random_state=42)

# Construir y entrenar el modelo Random Forest Regressor para predicción basada en estrellas
param_grid = {'n_estimators': [50, 100, 150], 'max_depth': [None, 10, 20]}
grid_search_rf_stars = GridSearchCV(RandomForestRegressor(), param_grid, cv=3)
grid_search_rf_stars.fit(X_train, y_train)

# Obtener las predicciones de crecimiento para cada estado
df_abiertos['predicted_growth'] = grid_search_rf_stars.predict(features_scaled)

# Calcular el porcentaje de crecimiento para cada estado
df_abiertos['growth_percentage'] = ((df_abiertos['predicted_growth'] - df_abiertos['review_count']) / df_abiertos['review_count']) * 100

# Obtener los dos estados que más crecen y los dos estados que más decrecen
top_growth_states = df_abiertos.groupby('state')['growth_percentage'].mean().nlargest(2).reset_index()
top_decrease_states = df_abiertos.groupby('state')['growth_percentage'].mean().nsmallest(2).reset_index()

# Obtener el tipo de comida predominante en los estados que más crecen
top_growth_states_categories = df_abiertos[df_abiertos['state'].isin(top_growth_states['state'])].groupby('state')['categories'].apply(lambda x: x.mode().iloc[0]).reset_index()

# Obtener el tipo de comida predominante en los estados que más decrecen
top_decrease_states_categories = df_abiertos[df_abiertos['state'].isin(top_decrease_states['state'])].groupby('state')['categories'].apply(lambda x: x.mode().iloc[0]).reset_index()

# Obtener la preferencia del usuario
preference = input("Ingrese 'crecimiento' o 'decrecimiento': ").lower()

# Filtrar los resultados según la preferencia del usuario
if preference == 'crecimiento':
    top_states = top_growth_states
    top_states_categories = top_growth_states_categories
elif preference == 'decrecimiento':
    top_states = top_decrease_states
    top_states_categories = top_decrease_states_categories
else:
    print("Opción no válida. Por favor, ingrese 'crecimiento' o 'decrecimiento'.")
    exit()

# Convertir los resultados de tasas de crecimiento a porcentajes
top_states['growth_percentage'] *= 100

# Imprimir los resultados
print(f'Dos estados con más {preference}:')
print(top_states)
print(f'Tipo de comida predominante en estos estados:')
print(top_states_categories)

####################################################

app = FastAPI()

class CiudadInput(BaseModel):
    ciudad: str

class PreferenciaInput(BaseModel):
    preferencia: str



@app.post("/recomendacion_ciudad/")
def recomendacion_ciudad(ciudad_input: CiudadInput):
    ciudad = ciudad_input.ciudad.lower()
    
    # Filtrar el DataFrame por la ciudad proporcionada
    city_df = df[df['city'].str.lower() == ciudad]

    # Mostrar todas las recomendaciones para la ciudad dada
    if not city_df.empty:
        recomendaciones = []
        for _, row in city_df.iterrows():
            recomendaciones.append({
                "Ciudad": row['city'],
                "Restaurante": row['name'],
                "Número de estrellas": row['stars'],
                "Tipo actividad y comida": row['categories']
            })
        
        # Devolver las recomendaciones
        return {"Recomendaciones": recomendaciones}
    else:
        raise HTTPException(status_code=404, detail=f"No hay datos disponibles para la ciudad {ciudad}.")


@app.post("/estados_destacados/")
def estados_destacados(preferencia_input: PreferenciaInput):
    preferencia = preferencia_input.preferencia.lower()

    # Filtrar los resultados según la preferencia del usuario
    if preferencia == 'crecimiento':
        top_states = top_growth_states
        top_states_categories = top_growth_states_categories
    elif preferencia == 'decrecimiento':
        top_states = top_decrease_states
        top_states_categories = top_decrease_states_categories
    else:
        raise HTTPException(status_code=400, detail="Opción no válida. Por favor, ingrese 'crecimiento' o 'decrecimiento'.")

    # Convertir los resultados de tasas de crecimiento a porcentajes
    top_states['growth_percentage'] *= 100

    # Devolver los resultados
    return {
        "Dos estados con más {preferencia}": top_states.to_dict(orient='records'),
        "Tipo de comida predominante en estos estados": top_states_categories.to_dict(orient='records')
    }

