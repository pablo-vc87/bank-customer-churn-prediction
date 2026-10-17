# Predicción de Fuga de Clientes Bancarios

> English version: [README.md](README.md)

Proyecto de clasificación binaria que predice **qué clientes de Beta Bank tienen probabilidad de dejar el banco**, para que el banco intente retenerlos antes de que se vayan.

El proyecto compara **Regresión Logística** y **Random Forest** con distintas estrategias para el desbalance de clases (umbrales de decisión, pesos de clase y sobremuestreo) y ajusta los hiperparámetros de Random Forest. El modelo elegido, **Random Forest (120 árboles, profundidad máxima 13) entrenado con sobremuestreo de la clase minoritaria**, obtuvo un **F1 de 0.6076 en validación** y **F1 de 0.6108 con ROC-AUC de 0.8655 en el conjunto de prueba independiente**, por encima del requisito de **F1 ≥ 0.59**.

---

# Contexto de Negocio

Beta Bank pierde clientes mes con mes, poco a poco pero de forma constante. El banco encontró que **conservar a un cliente existente es más barato que atraer a uno nuevo**, así que quiere identificar a los clientes que están por irse y ofrecerles beneficios de retención a tiempo. Un modelo confiable ayuda a:

* Enfocar las ofertas de retención en los clientes con mayor riesgo.
* No gastar en clientes que se quedarían de todos modos.
* Medir cuántos de los clientes que se van se pueden detectar con anticipación.

El modelo debe predecir una variable objetivo:

| Objetivo | Descripción |
| --- | --- |
| `Exited` | 1 si el cliente dejó el banco, 0 si se quedó |

---

# Conceptos de Modelado de Fuga

| Término | Significado |
| --- | --- |
| Fuga (churn) | Un cliente que deja el banco |
| Desbalance de clases | Solo cerca del 20% de los clientes se fue, así que un modelo puede parecer preciso prediciendo "se queda" para todos |
| Umbral de decisión | La probabilidad a partir de la cual el modelo predice fuga (0.5 por defecto) |
| `class_weight='balanced'` | Da más peso a la clase minoritaria durante el entrenamiento |
| Sobremuestreo | Repetir las filas de la clase minoritaria en el entrenamiento para balancear las clases |
| División estratificada | Mantener la misma proporción de fuga en train, validación y test |

## Qué Significa Cada Error para el Banco

| Resultado | Significado | Impacto |
| --- | --- | --- |
| Verdadero positivo | El cliente se va y el modelo lo detecta | El banco puede actuar para retenerlo |
| Verdadero negativo | El cliente se queda y el modelo lo predice | No se gastan recursos |
| Falso positivo | El modelo predice fuga, pero el cliente se queda | Se gasta una oferta de retención sin necesidad |
| Falso negativo | El modelo predice que se queda, pero se va | **Se pierde al cliente sin intentar retenerlo** |

Los falsos negativos son el error más costoso, así que además del F1 el proyecto pone especial atención al **Recall** de la clase de fuga.

---

# Métrica de Evaluación

La métrica principal es el **F1** de la clase de fuga, la media armónica de Precision y Recall:

$$
F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

El F1 solo es alto cuando el modelo **encuentra a la mayoría de los clientes que se van** (Recall) y **acierta cuando los marca** (Precision). El Accuracy no sirve aquí: un modelo que predice "se queda" para todos llegaría a cerca de 80% de Accuracy sin detectar ninguna fuga.

El **ROC-AUC** se registró como métrica secundaria, porque mide qué tan bien el modelo ordena a los clientes por riesgo sin depender del umbral.

El requisito del proyecto era **F1 ≥ 0.59** en el conjunto de prueba.

---

# Dataset

| Archivo | Filas | Columnas | Descripción |
| --- | ---: | ---: | --- |
| `data/Churn.csv` | 10,000 | 14 | Una fila por cliente: perfil, información de la cuenta y si se fue |

| Columna | Descripción |
| --- | --- |
| `RowNumber`, `CustomerId`, `Surname` | Índice de la fila, ID único del cliente y apellido |
| `CreditScore` | Puntaje de crédito |
| `Geography` | País de residencia (Francia, Alemania o España) |
| `Gender` | Género |
| `Age` | Edad |
| `Tenure` | Años que el cliente lleva con el banco (0–10) |
| `Balance` | Saldo de la cuenta |
| `NumOfProducts` | Número de productos del banco que usa |
| `HasCrCard` | 1 si el cliente tiene tarjeta de crédito |
| `IsActiveMember` | 1 si el cliente está activo |
| `EstimatedSalary` | Salario estimado |
| `Exited` | Objetivo: 1 si el cliente se fue |

* **2,037 clientes (20.37%)** dejaron el banco y **7,963 (79.63%)** se quedaron.
* Los clientes son de Francia (5,014), Alemania (2,509) y España (2,477).
* `Tenure` tiene **909 valores faltantes**; es la única columna con datos faltantes.

**Fuente de los datos.** El dataset fue proporcionado por **TripleTen** como parte de su Bootcamp de Data Science y se incluye en este repositorio con fines educativos y de reproducibilidad.

---

# Flujo del Proyecto

```text
Datos de clientes (10,000 filas)
            │
            ▼
1. Preparación de los datos
   ├── Tenure faltante → 0
   ├── Gender → binaria, Geography → One-Hot Encoding
   └── Eliminar identificadores (RowNumber, CustomerId, Surname)
            │
            ▼
2. División estratificada: 70% train / 15% validación / 15% test
            │
            ▼
3. Regresión Logística
   ├── Modelo base
   ├── Umbrales de decisión manuales
   ├── class_weight='balanced'
   └── Sobremuestreo (repeat=3)
            │
            ▼
4. Random Forest
   ├── Búsqueda de hiperparámetros (n_estimators, max_depth)
   ├── Búsqueda fina alrededor de los mejores valores
   ├── Sobremuestreo (repeat=3)
   └── class_weight='balanced'
            │
            ▼
5. Elegir el mejor modelo en validación y evaluarlo una sola vez en test
```

---

# 1. Preparación de los Datos

* **Valores faltantes:** los 909 valores faltantes de `Tenure` se reemplazaron por **0**.
* **Codificación:**
  * `Gender` se convirtió a binaria (`Male` = 1, `Female` = 0).
  * `Geography` se codificó con One-Hot Encoding y `drop_first=True`, lo que creó `Geography_Germany` y `Geography_Spain` (Francia es la referencia).
* **Identificadores eliminados:**
  * `RowNumber` y `CustomerId` casi no tienen correlación con `Exited` (−0.017 y −0.006).
  * `Surname` tiene 2,932 valores distintos, la mayoría compartidos por muy pocos clientes, y 1,181 apellidos aparecen en más de un país. No aporta una señal útil de fuga y sobre todo refleja el país del cliente, que ya está en `Geography`.

El conjunto final tiene **11 variables**:

```text
CreditScore, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard,
IsActiveMember, EstimatedSalary, Geography_Germany, Geography_Spain
```

---

# 2. Análisis de los Datos

## Desbalance de Clases

| Clase | Clientes | Proporción |
| --- | ---: | ---: |
| Se quedó (0) | 7,963 | 79.63% |
| Se fue (1) | 2,037 | 20.37% |

Con solo un cliente que se va por cada cuatro que se quedan, un modelo sin balancear tiende a predecir "se queda" casi siempre. Por eso se compararon varias estrategias de balanceo.

## División de los Datos

Los datos se dividieron con `stratify` para que cada conjunto conserve la tasa de fuga de 20.4%:

```text
100% de los datos (10,000 clientes)
│
├── 70% TRAIN ─────────→ 7,004 clientes (1,426 se fueron) → entrenamiento
├── 15% VALIDACIÓN ────→ 1,500 clientes   (306 se fueron) → selección del modelo
└── 15% TEST ──────────→ 1,496 clientes   (305 se fueron) → evaluación final (se usa una sola vez)
```

Se eligió la división 70/15/15 para tener la mayor cantidad posible de datos de entrenamiento y aun así conservar unos 300 clientes que se fueron en cada conjunto de evaluación.

## Sobremuestreo

Para el sobremuestreo, las filas de fuga del entrenamiento se repitieron **3 veces** (5,578 // 1,426 = 3): pasaron de 1,426 a **4,278 clientes que se fueron** contra 5,578 que se quedaron (43.4% de fuga). Validación y test nunca se modificaron.

---

# 3. Modelado

## Regresión Logística

| Estrategia | F1 | Recall | Precision | ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| Modelo base (umbral 0.5) | 0.1070 | 0.06 | 0.39 | 0.6740 |
| Modelo base, mejor umbral manual (0.20) | 0.4085 | 0.66 | 0.30 | 0.6740 |
| `class_weight='balanced'` | **0.4513** | 0.67 | 0.34 | **0.7237** |
| Sobremuestreo (repeat=3) | 0.4370 | 0.53 | 0.37 | 0.7036 |

*Resultados en validación.*

* El modelo base detectó solo al **6% de los clientes que se fueron**.
* Bajar el umbral aumentó el Recall (0.91 con umbral de 0.10), pero redujo la Precision, así que el F1 nunca pasó de 0.41.
* Balancear las clases ayudó, pero **ninguna Regresión Logística alcanzó el requisito de F1 ≥ 0.59**. Las variables no se escalaron y el modelo con sobremuestreo mostró una advertencia de convergencia, así que probablemente el modelo lineal no se aprovechó al máximo.

## Random Forest: Búsqueda de Hiperparámetros

Se entrenó y evaluó en validación cada combinación de `n_estimators` y `max_depth`:

| Búsqueda | `n_estimators` | `max_depth` | Mejor combinación | Mejor F1 |
| --- | --- | --- | --- | ---: |
| 1 | 50–200, de 25 en 25 | 3–15 | 100 árboles, profundidad 13 | 0.5912 |
| 2 (fina) | 80–120, de 10 en 10 | 11–16 | **120 árboles, profundidad 13** | **0.5941** |

Los árboles poco profundos (profundidad 3–4) tuvieron F1 menor a 0.36; el F1 subió de forma constante hasta una profundidad de alrededor de 13.

## Random Forest: Estrategias de Balanceo

Los tres modelos usan 120 árboles y profundidad máxima de 13:

| Estrategia | F1 | Recall | Precision | ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| Sin balanceo | 0.5941 | 0.46 | **0.83** | 0.8375 |
| `class_weight='balanced'` | 0.5939 | 0.51 | 0.72 | 0.8388 |
| **Sobremuestreo (repeat=3)** | **0.6076** | **0.57** | 0.65 | **0.8420** |

*Resultados en validación.*

* Sin balanceo, Random Forest ya cumplía el requisito de F1, pero **no detectaba a más de la mitad de los clientes que se fueron** (Recall 0.46).
* **El sobremuestreo dio el mejor F1 y el mayor Recall**: detectó al 57% de los clientes que se fueron en lugar del 46%, a cambio de una Precision menor.
* Los pesos de clase casi no cambiaron el F1 y mejoraron el Recall solo un poco.

## Resultado Final

El Random Forest con sobremuestreo se evaluó **una sola vez** en el conjunto de prueba:

```text
Modelo:          RandomForestClassifier(n_estimators=120, max_depth=13) + sobremuestreo (repeat=3)
F1 validación:   0.6076
F1 test:         0.6108
ROC-AUC test:    0.8655
Recall test:     0.61
Precision test:  0.61
Accuracy test:   0.84
Requisito:       F1 ≥ 0.59   ✓
```

| Clase | Precision | Recall | F1 | Soporte |
| --- | ---: | ---: | ---: | ---: |
| Se quedó (0) | 0.90 | 0.90 | 0.90 | 1,191 |
| Se fue (1) | 0.61 | 0.61 | 0.61 | 305 |

Los resultados de test son consistentes con los de validación (F1 0.6108 frente a 0.6076), así que el modelo generaliza bien a clientes nuevos.

---

# Conclusiones

1. **El desbalance de clases fue el principal reto.** Sin balanceo, la Regresión Logística detectó solo al 6% de los clientes que se fueron.

2. **La Regresión Logística no alcanzó el requisito** con ninguna estrategia (mejor F1 = 0.45 con pesos de clase).

3. **Random Forest superó claramente a la Regresión Logística** (ROC-AUC cercano a 0.84 frente a 0.72), lo que sugiere relaciones no lineales entre las variables y la fuga.

4. **El sobremuestreo fue la mejor estrategia de balanceo para Random Forest**: mejoró el F1 y el Recall sin cambiar los hiperparámetros.

5. **El modelo final cumple el requisito con datos nuevos**: F1 = 0.6108 y ROC-AUC = 0.8655 en el conjunto de prueba.

## Interpretación de Negocio

El modelo detecta cerca del **61% de los clientes que se van a ir**, y cerca del 61% de los clientes que marca efectivamente se van. El banco puede usar la probabilidad predicha para ordenar a los clientes por riesgo y enfocar las ofertas de retención, como cashback o tarjetas de regalo, en el grupo de mayor riesgo. Con el umbral actual todavía se escaparía cerca del 39% de los clientes que se van, así que el modelo debe apoyar las decisiones de retención, no reemplazarlas.

## Siguientes Pasos

* Escalar las variables numéricas (`StandardScaler`) antes de entrenar la Regresión Logística y volver a compararla.
* Ajustar el umbral de decisión **en validación** según el costo de una oferta de retención frente al costo de perder a un cliente.
* Usar validación cruzada estratificada en la búsqueda de hiperparámetros en lugar de un solo conjunto de validación.
* Analizar la importancia de las variables para explicar qué características del cliente influyen en la fuga.
* Probar modelos de gradient boosting (por ejemplo, LightGBM).

---

# Estructura del Proyecto

```text
bank-customer-churn-prediction/
│
├── data/
│   └── Churn.csv
│
├── notebooks/
│   └── customer_churn_prediction.ipynb
│
├── src/
│   └── funciones_personales.py
│
├── .gitignore
├── environment.yml
├── requirements.txt
├── LEEME.md
└── README.md
```

`src/funciones_personales.py` contiene las funciones auxiliares que usa el notebook: `roc_plot_enhanced` (reporte de clasificación, F1, ROC-AUC, curvas ROC y precision-recall, y tabla de umbrales), `upsample` (sobremuestreo) y `forest_classif_comb_f1` (búsqueda de hiperparámetros de Random Forest).

---

# Instalación

Clona el repositorio:

```bash
git clone https://github.com/pablo-vc87/bank-customer-churn-prediction.git

cd bank-customer-churn-prediction
```

## Opción 1: Conda

```bash
conda env create -f environment.yml
conda activate bank_customer_churn
jupyter notebook
```

## Opción 2: pip

Crea y activa un entorno virtual y después:

```bash
pip install -r requirements.txt
jupyter notebook
```

Abre `notebooks/customer_churn_prediction.ipynb` y ejecuta todas las celdas.

---

# Tecnologías

* Python
* Pandas
* NumPy
* Matplotlib
* scikit-learn (Regresión Logística, Random Forest)
* Jupyter Notebook

---

# Habilidades Demostradas

* Clasificación binaria con desbalance de clases
* Análisis de umbrales de decisión
* Pesos de clase y sobremuestreo
* Ajuste de hiperparámetros de Random Forest
* Metodología train / validación / test con divisiones estratificadas
* Evaluación con F1, Precision, Recall y ROC-AUC
* Curvas ROC y precision-recall
* Traducción de los errores de clasificación a impacto de negocio

---

## Autor

**Pablo Andrés Vázquez Calva**

Data Scientist

GitHub: https://github.com/pablo-vc87

LinkedIn: https://www.linkedin.com/in/pablo-avc/
