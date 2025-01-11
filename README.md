# BOTCHARopena.PY
CODIGOS DEL PROYECTO final
import pandas as pd
import numpy as np
import openai
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder

#  en esta parte de los codigos se procede a confiConfigurar la clave de API de OpenAI
openai.api_key = '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'


datos = {
    'Fecha': pd.date_range(start='2023-01-01', periods=365),
    'Producto': np.random.choice(['A', 'B', 'C', 'D'], size=365),
    'Precio': np.random.uniform(10, 100, size=365),
    'Promoción': np.random.choice([0, 1], size=365),
    'Región': np.random.choice(['Norte', 'Sur', 'Este', 'Oeste'], size=365),
    'Ventas': np.random.poisson(lam=20, size=365)
}
df = pd.DataFrame(datos)

# en esta parte se Preprocesan los datos
ohe = OneHotEncoder()
caracteristicas_categoricas = ['Producto', 'Región']
ohe_df = pd.DataFrame(ohe.fit_transform(df[caracteristicas_categoricas]).toarray(),
                       columns=ohe.get_feature_names_out(caracteristicas_categoricas))

df = pd.concat([df, ohe_df], axis=1).drop(columns=caracteristicas_categoricas + ['Fecha'])

X = df.drop(columns=['Ventas'])
y = df['Ventas']

escalador = StandardScaler()
X_escalado = escalador.fit_transform(X)

X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X_escalado, y, test_size=0.3, random_state=42)

# estos son los Modelos de predicción
regresion_lineal = LinearRegression()
regresion_lineal.fit(X_entrenamiento, y_entrenamiento)

arbol_decision = DecisionTreeRegressor(random_state=42)
arbol_decision.fit(X_entrenamiento, y_entrenamiento)

# esta es la Función para predecir y generar explicaciones usando OpenAI
def generar_explicacion_openai(prediccion, modelo):
    prompt = f"Explica de manera detallada por qué el modelo {modelo} ha hecho una predicción de {prediccion} para las ventas de un producto."
    
    try:
        respuesta = openai.Completion.create(
            engine="text-davinci-003",  # Usamos el modelo Davinci de OpenAI
            prompt=prompt,
            max_tokens=150
        )
        return respuesta.choices[0].text.strip()
    except Exception as e:
        return f"Error al generar explicación: {str(e)}"

#  esta es la Función para predecir ventas
def predecir_ventas(datos_usuario):
    """
    Función para realizar predicciones basadas en datos proporcionados por el usuario.
    :param datos_usuario: Diccionario con las características del producto.
    :return: Diccionario con las predicciones de los modelos y sus explicaciones.
    """
    try:
        
        df_usuario = pd.DataFrame([datos_usuario])
        df_usuario_ohe = pd.DataFrame(ohe.transform(df_usuario[caracteristicas_categoricas]).toarray(),
                                      columns=ohe.get_feature_names_out(caracteristicas_categoricas))
        df_usuario = pd.concat([df_usuario, df_usuario_ohe], axis=1).drop(columns=caracteristicas_categoricas)
        df_usuario = df_usuario.reindex(columns=X.columns, fill_value=0)

        datos_escalados = escalador.transform(df_usuario)

        prediccion_rl = regresion_lineal.predict(datos_escalados)[0]
        prediccion_ad = arbol_decision.predict(datos_escalados)[0]

        explicacion_rl = generar_explicacion_openai(prediccion_rl, "Regresión Lineal")
        explicacion_ad = generar_explicacion_openai(prediccion_ad, "Árbol de Decisión")

        return {
            "Predicción Regresión Lineal": prediccion_rl,
            "Explicación Regresión Lineal": explicacion_rl,
            "Predicción Árbol de Decisión": prediccion_ad,
            "Explicación Árbol de Decisión": explicacion_ad
        }
    except Exception as e:
        return {"error": str(e)}

# Interacción con el usuario con el BOt
if __name__ == '__main__':
    print("Bienvenido al bot de predicción de ventas.")
    while True:
        try:
            producto = input("Ingrese el producto (A, B, C, D): ")
            precio = float(input("Ingrese el precio: "))
            promocion = int(input("¿Está en promoción? (1 para Sí, 0 para No): "))
            region = input("Ingrese la región (Norte, Sur, Este, Oeste): ")

            datos_usuario = {
                'Producto': producto,
                'Precio': precio,
                'Promoción': promocion,
                'Región': region
            }

            resultados = predecir_ventas(datos_usuario)
            print("Resultados de la predicción:")
            print(f"Predicción Regresión Lineal: {resultados['Predicción Regresión Lineal']}")
            print(f"Explicación Regresión Lineal: {resultados['Explicación Regresión Lineal']}")
            print(f"Predicción Árbol de Decisión: {resultados['Predicción Árbol de Decisión']}")
            print(f"Explicación Árbol de Decisión: {resultados['Explicación Árbol de Decisión']}")
        except Exception as e:
            print("Error:", e)

        continuar = input("¿Desea realizar otra predicción? (sí/no): ").strip().lower()
        if continuar != 'sí':
            print("Gracias por utilizar el bot de predicción de ventas. ¡Hasta luego!")
            break
