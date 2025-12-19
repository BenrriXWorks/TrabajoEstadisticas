import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_benchmark():
    # 1. Cargar datos
    try:
        df = pd.read_csv("benchmark_results.csv")
    except FileNotFoundError:
        print("Error: No se encuentra 'benchmark_results.csv'. Ejecuta el C++ primero.")
        return

    # Configuración de estilo
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Análisis de Rendimiento: SampleMerge vs std::sort', fontsize=16)

    # ---------------------------------------------------------
    # Gráfico 1: Tiempo vs Tamaño de Entrada (Log-Log) - Uniforme
    # ---------------------------------------------------------
    ax1 = axes[0, 0]
    data_uniform = df[df['Distribution'] == 'Uniform']
    
    # Graficar std::sort
    std_data = data_uniform[data_uniform['Algorithm'] == 'std::sort']
    ax1.plot(std_data['InputSize'], std_data['TimeSeconds'], 
             label='std::sort (par)', marker='o', linestyle='--', color='black')

    # Graficar SampleMerge para cada configuración de threads
    sm_data = data_uniform[data_uniform['Algorithm'] == 'SampleMerge']
    for threads in sorted(sm_data['Threads'].unique()):
        subset = sm_data[sm_data['Threads'] == threads]
        ax1.plot(subset['InputSize'], subset['TimeSeconds'], 
                 label=f'SampleMerge ({threads} th)', marker='.')

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Tamaño de Entrada (Elementos)')
    ax1.set_ylabel('Tiempo (Segundos)')
    ax1.set_title('Escalabilidad por Tamaño (Uniforme)')
    ax1.legend()
    ax1.grid(True, which="both", ls="-", alpha=0.2)

    # ---------------------------------------------------------
    # Gráfico 2: Speedup vs std::sort (Último tamaño disponible)
    # ---------------------------------------------------------
    ax2 = axes[0, 1]
    max_size = df['InputSize'].max()
    df_max = df[df['InputSize'] == max_size].copy()
    
    # Obtener tiempo base de std::sort para ese tamaño
    base_time_uni = df_max[(df_max['Algorithm'] == 'std::sort') & (df_max['Distribution'] == 'Uniform')]['TimeSeconds'].values[0]
    
    # Calcular Speedup
    df_max['Speedup'] = base_time_uni / df_max['TimeSeconds']
    
    sns.barplot(data=df_max, x='Threads', y='Speedup', hue='Algorithm', ax=ax2, palette='viridis')
    ax2.axhline(1.0, color='red', linestyle='--', label='Baseline (std::sort)')
    ax2.set_title(f'Speedup Relativo a std::sort (N={max_size})')
    ax2.set_ylabel('Speedup (x veces más rápido)')

    # ---------------------------------------------------------
    # Gráfico 3: Eficiencia de Threads (Strong Scaling)
    # ---------------------------------------------------------
    ax3 = axes[1, 0]
    # Filtramos solo SampleMerge y Uniforme, tamaño máximo
    sm_scaling = df[(df['Algorithm'] == 'SampleMerge') & 
                    (df['InputSize'] == max_size) & 
                    (df['Distribution'] == 'Uniform')].sort_values('Threads')
    
    if not sm_scaling.empty:
        t1_time = sm_scaling[sm_scaling['Threads'] == sm_scaling['Threads'].min()]['TimeSeconds'].values[0]
        min_threads = sm_scaling['Threads'].min()
        
        # Eficiencia ideal
        ideal_x = sm_scaling['Threads']
        ideal_y = t1_time * (min_threads / ideal_x)
        
        ax3.plot(sm_scaling['Threads'], sm_scaling['TimeSeconds'], marker='o', label='Real')
        ax3.plot(ideal_x, ideal_y, linestyle='--', color='gray', label='Ideal (Lineal)')
        
        ax3.set_xlabel('Número de Threads')
        ax3.set_ylabel('Tiempo (Segundos)')
        ax3.set_title(f'Escalabilidad de Threads (N={max_size})')
        ax3.legend()

    # ---------------------------------------------------------
    # Gráfico 4: Impacto de la Distribución (Uniforme vs Normal)
    # ---------------------------------------------------------
    ax4 = axes[1, 1]
    # Comparamos el mejor caso de SampleMerge (max threads) vs std::sort
    max_threads = df['Threads'].max()
    df_dist = df[((df['Algorithm'] == 'SampleMerge') & (df['Threads'] == max_threads)) | 
                 (df['Algorithm'] == 'std::sort')]
    df_dist = df_dist[df_dist['InputSize'] == max_size]

    sns.barplot(data=df_dist, x='Algorithm', y='TimeSeconds', hue='Distribution', ax=ax4)
    ax4.set_title(f'Uniforme vs Normal (N={max_size})')
    ax4.set_ylabel('Tiempo (Segundos)')

    plt.tight_layout()
    plt.savefig('benchmark_plot.png')
    print("Gráfico generado: benchmark_plot.png")
    plt.show()

if __name__ == "__main__":
    plot_benchmark()