import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

def analyze_merge_bottleneck():
    # 1. Cargar Datos
    try:
        df = pd.read_csv("benchmark_granular.csv")
    except FileNotFoundError:
        print("Error: No se encuentra el archivo.")
        return

    # Filtrar solo SampleMerge
    df = df[df['Algorithm'] == 'SampleMerge'].copy()
    
    # 2. DEFINIR MODELO TEÓRICO CORRECTO
    # Hipótesis Usuario: Costo Merge ~ N * log2(Threads) + C
    # El término dominante es el movimiento de N datos, modulado por la lógica de K particiones
    df['Complexity_Merge'] = df['InputSize'] * np.log2(df['Threads'])
    
    # Vamos a probar la regresión con esta nueva variable
    X = df['Complexity_Merge']
    y = df['MergePhase']
    
    model = sm.OLS(y, sm.add_constant(X)).fit()
    
    print("\n=== Regresión Fase Merge con Modelo N * log2(Threads) ===")
    print(f"R-squared: {model.rsquared:.4f}")
    print(model.summary())

    # 3. VISUALIZACIÓN DE SATURACIÓN (ANCHO DE BANDA)
    # Calculamos cuántos Gigabytes por segundo estamos procesando
    # Asumimos int = 4 bytes. 
    # GB/s = (N * 4 bytes) / (Tiempo_ms * 1e-3) / 1e9
    df['Effective_Bandwidth_GBs'] = (df['InputSize'] * 4) / (df['MergePhase'] * 1e-3) / 1e9
    
    # Filtrar solo tamaños grandes donde la caché L3 ya no ayuda (los últimos dos que mencionaste)
    # Usamos el 10% de tamaños más grandes como muestra
    large_n_threshold = df['InputSize'].max() / 2
    df_large = df[df['InputSize'] >= large_n_threshold].copy()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # GRÁFICO 1: Tiempo vs Threads (Para N grande)
    # Aquí veremos claramente si sube al agregar threads
    sns.lineplot(data=df_large, x='Threads', y='MergePhase', hue='InputSize', 
                 marker='o', palette='Reds', ax=axes[0])
    axes[0].set_title('Tiempo Fase Merge vs Threads (N Grande)')
    axes[0].set_ylabel('Tiempo (ms)')
    axes[0].set_xlabel('Cantidad de Threads (K)')
    axes[0].grid(True)
    
    # GRÁFICO 2: Ancho de Banda Efectivo
    # Esto demostrará el límite físico de tu RAM
    sns.barplot(data=df_large, x='Threads', y='Effective_Bandwidth_GBs', hue='InputSize', 
                palette='viridis', ax=axes[1])
    
    # Linea promedio de ancho de banda máximo detectado
    max_bw = df_large['Effective_Bandwidth_GBs'].max()
    axes[1].axhline(max_bw, color='red', linestyle='--', label=f'Max Peak: {max_bw:.2f} GB/s')
    
    axes[1].set_title('Ancho de Banda Efectivo (Memory Wall)')
    axes[1].set_ylabel('Throughput (GB/s)')
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()
    
    # Análisis de Residuos del nuevo modelo
    fig_res, ax_res = plt.subplots(1, 2, figsize=(12, 5))
    
    # Scatter: Predicción vs Real
    y_pred = model.predict(sm.add_constant(X))
    ax_res[0].scatter(y_pred, y, alpha=0.5)
    ax_res[0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    ax_res[0].set_title('Predicted vs Actual (Model: N log K)')
    ax_res[0].set_xlabel('Predicted Time')
    ax_res[0].set_ylabel('Actual Time')
    
    # Residuos
    sns.histplot(model.resid, kde=True, ax=ax_res[1])
    ax_res[1].set_title('Distribución de Residuos')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    analyze_merge_bottleneck()