import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def analyze_bandwidth_bottleneck():
    try:
        df = pd.read_csv("benchmark_granular.csv")
    except FileNotFoundError:
        return

    # Filtrar datos
    df = df[df['Algorithm'] == 'SampleMerge'].copy()
    
    # Calcular métrica clave: Nanosegundos por Elemento
    # Si es constante, es escalable. Si sube, es saturación.
    df['ns_per_elem_sort'] = (df['SortPhase'] * 1e6) / (df['InputSize'] * np.log2(df['InputSize'])) # Normalizado por N log N
    df['ns_per_elem_merge'] = (df['MergePhase'] * 1e6) / df['InputSize'] # Normalizado por N
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # --- GRÁFICO 1: Eficiencia Fase Sort ---
    sns.lineplot(data=df, x='InputSize', y='ns_per_elem_sort', hue='Threads', marker='o', palette='viridis', ax=axes[0])
    axes[0].set_xscale('log')
    axes[0].set_ylabel('ns / (N log N)')
    axes[0].set_title('Fase Sort: Costo normalizado (Menos es mejor)')
    axes[0].grid(True, which="both", ls="-", alpha=0.2)
    
    # Interpretación visual:
    # La bajada inicial es la amortización del overhead.
    # Si luego se aplana, es perfecto.
    
    # --- GRÁFICO 2: Eficiencia Fase Merge (Bandwidth) ---
    sns.lineplot(data=df, x='InputSize', y='ns_per_elem_merge', hue='Threads', marker='o', palette='magma', ax=axes[1])
    axes[1].set_xscale('log')
    axes[1].set_ylabel('ns / N')
    axes[1].set_title('Fase Merge: Nanosegundos por Elemento (Detección de Saturación)')
    axes[1].grid(True, which="both", ls="-", alpha=0.2)
    
    # DIBUJAR LÍNEA DE CACHÉ L3 (Ejemplo aprox para un CPU moderno común, ej 20MB)
    # 20MB / 4 bytes (int) = 5,000,000 elementos
    L3_approx_elements = 5_000_000 
    axes[1].axvline(L3_approx_elements, color='red', linestyle='--', label='Límite Aprox L3 Cache')
    axes[1].legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    analyze_bandwidth_bottleneck()