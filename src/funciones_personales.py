import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, classification_report, roc_curve, precision_recall_curve, precision_recall_fscore_support
#=============================================================
def roc_plot_enhanced(model, features_valid, target_valid, model_name="Modelo"):
    '''
    Función que evalúa un modelo y muestra métricas completas junto con gráficas ROC.
    
    Parámetros:
    - model: modelo entrenado (LogisticRegression, RandomForest, etc.)
    - features_valid: características del conjunto de validación
    - target_valid: etiquetas verdaderas del conjunto de validación  
    - model_name: nombre del modelo para personalizar los títulos
    '''
    
    # Hacer predicciones
    valid_preds = model.predict(features_valid)
    probabilities_one_valid = model.predict_proba(features_valid)[:, 1]
    
    # Reporte de clasificación
    print(f"Reporte de clasificación en validación ({model_name}):")
    print(classification_report(target_valid, valid_preds))
    
    # Calcular métricas
    f1_score_val = f1_score(target_valid, valid_preds)
    auc_score = roc_auc_score(target_valid, probabilities_one_valid)
    
    print(f"F1 Score ({model_name}): {f1_score_val:.4f}")
    print(f"AUC-ROC ({model_name}): {auc_score:.4f}")
    
    # Calcular curvas para las gráficas
    fpr, tpr, thresholds_roc = roc_curve(target_valid, probabilities_one_valid)
    precision, recall, thresholds_pr = precision_recall_curve(target_valid, probabilities_one_valid)

    # Gráfica de recall vs precisión
    plt.figure(figsize=(6, 6))
    plt.step(recall, precision, where='post')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f'Precision-Recall Curve ({model_name})')
    plt.show()

    #Prueba con diferentes umbrales
    thresholds_test = np.arange(0.0, 0.5, 0.05)
    results = []
    for threshold in thresholds_test:
        # Hacer predicciones con el nuevo umbral
        preds_threshold = (probabilities_one_valid >= threshold).astype(int)
        
        # Calcular métricas
        precision, recall, f1, _ = precision_recall_fscore_support(
            target_valid, preds_threshold, average='binary'
        )
        
        results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
    # Mostrar resultados
    results_df = pd.DataFrame(results)
    print(results_df)

    # Gráfica ROC
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('Tasa de falsos positivos')
    plt.ylabel('Tasa de verdaderos positivos')
    plt.title(f'Curva ROC ({model_name})')
    plt.show()
#============================================
def upsample(features, target, repeat):
    features_zeros = features[target == 0]
    features_ones = features[target == 1]
    target_zeros = target[target == 0]
    target_ones = target[target == 1]

    features_upsampled = pd.concat([features_zeros] + [features_ones] * repeat)
    target_upsampled = pd.concat([target_zeros] + [target_ones] * repeat)

    features_upsampled, target_upsampled = shuffle(
        features_upsampled, target_upsampled, random_state=54321
    )

    return features_upsampled, target_upsampled
#===============================================

def forest_classif_comb_f1(features_train, target_train, features_valid, target_valid,
                          min_depth=3, max_depth=15, min_est=50, max_est=200, step_est=25):
    '''
    Encuentra los mejores hiperparámetros para RandomForestClassifier optimizando F1-score.
    
    Esta función realiza una búsqueda exhaustiva (grid search) sobre los hiperparámetros
    n_estimators y max_depth para encontrar la combinación que maximice el F1-score
    en el conjunto de validación.
    
    Parámetros:
    -----------
    features_train : array-like
        Características del conjunto de entrenamiento
    target_train : array-like
        Variable objetivo del conjunto de entrenamiento
    features_valid : array-like
        Características del conjunto de validación
    target_valid : array-like
        Variable objetivo del conjunto de validación
    min_depth : int, default=3
        Profundidad mínima de los árboles a probar
    max_depth : int, default=15
        Profundidad máxima de los árboles a probar
    min_est : int, default=50
        Número mínimo de estimadores a probar
    max_est : int, default=200
        Número máximo de estimadores a probar
    step_est : int, default=25
        Paso para el rango de estimadores
    
    Retorna:
    --------
    tuple
        (best_model, best_est, best_depth, best_f1)
        - best_model: El modelo RandomForest con mejores hiperparámetros
        - best_est: Mejor número de estimadores encontrado
        - best_depth: Mejor profundidad máxima encontrada
        - best_f1: Mejor F1-score alcanzado
    
    Ejemplo:
    --------
    >>> model, n_est, depth, f1 = forest_classif_comb_f1_optimized(
    ...     X_train, y_train, X_valid, y_valid)
    >>> print(f"Mejor F1: {f1:.4f} con {n_est} estimadores y profundidad {depth}")
    '''
    best_f1 = 0
    best_est = 0
    best_depth = 0
    best_model = None
    
    for est in range(min_est, max_est + 1, step_est):
        for depth in range(min_depth, max_depth + 1):
            model = RandomForestClassifier(random_state=54321, n_estimators=est, max_depth=depth)
            model.fit(features_train, target_train)
            
            # Usar F1 en lugar de accuracy para datos desequilibrados
            predictions = model.predict(features_valid)
            f1 = f1_score(target_valid, predictions)
            
            if f1 > best_f1:
                best_f1 = f1
                best_est = est
                best_depth = depth
                best_model = model
                print(f"n_estimators = {est}, max_depth = {depth}, F1 = {f1:.4f}")
    
    print(f"Mejor modelo: n_estimators = {best_est}, max_depth = {best_depth}, F1 = {best_f1:.4f}")
    return best_model, best_est, best_depth, best_f1
