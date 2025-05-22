import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 1. Prepara tus datos
# Suponiendo que tienes un archivo CSV con los datos
# df = pd.read_csv('tus_datos.csv')

# Para fines del ejemplo, creamos un DataFrame de prueba:
data = {
    'script_size': [300, 400, 500, 600],
    'amount': [0.5, 1.0, 0.2, 1.5],
    'block_height': [67890, 67900, 68000, 68100],
    'is_opreturn': [0, 1, 0, 0],
    'spent': [1, 0, 0, 1]  # 1 = Gastado, 0 = No gastado
}

df = pd.DataFrame(data)

# 2. Divide los datos en características (X) y etiquetas (y)
X = df.drop('spent', axis=1)  # Características (sin la columna 'spent')
y = df['spent']  # Etiquetas (si se gastó o no)

# 3. Divide los datos en entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Crea y entrena el modelo
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# 5. Haz predicciones sobre el conjunto de prueba
y_pred = model.predict(X_test)

# 6. Evalúa el modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión del modelo: {accuracy * 100:.2f}%')
