import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from matplotlib.ticker import ScalarFormatter

# Configuración de estilo
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 10})

def advanced_benchmark_analysis():
    # 1. Cargar Datos
    try:
        df = pd.read_csv("benchmark_granular.csv")
    except FileNotFoundError:
        print("Error: No se encuentra 'benchmark_granular.csv'")
        return

    # Convertir tamaños a formato legible para gráficos categóricos
    df['SizeLabel'] = df['InputSize'].apply(lambda x: f"{x:.0e}")

    # ==============================================================================
    # PARTE 1: DESGLOSE DE FASES (STACKED BAR CHART)
    # Objetivo: Ver qué fase se vuelve cuello de botella al aumentar threads
    # ==============================================================================
    
    max_size = df['InputSize'].max()
    df_max_uni = df[(df['InputSize'] == max_size) & 
                    (df['Distribution'] == 'Uniform') & 
                    (df['Algorithm'] == 'SampleMerge')].copy()
    
    if not df_max_uni.empty:
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        
        # Preparamos datos para stack
        phases = ['SortPhase', 'PivotPhase', 'SubpartPhase', 'OffsetPhase', 'MergePhase']
        df_max_uni.set_index('Threads')[phases].plot(kind='bar', stacked=True, ax=ax1, colormap='viridis')
        
        ax1.set_title(f'Composición del Tiempo por Fase (Input Size: {max_size:.0e}, Uniforme)')
        ax1.set_ylabel('Tiempo (ms)')
        ax1.set_xlabel('Número de Threads')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.show()

    # ==============================================================================
    # PARTE 2: ESCALABILIDAD (SPEEDUP) - SORT VS MERGE
    # Objetivo: Ver si la fase de Merge escala peor que la de Sort
    # ==============================================================================
    
    if not df_max_uni.empty:
        # Tomar base T=1 (o el mínimo disponible)
        min_thread = df_max_uni['Threads'].min()
        base_row = df_max_uni[df_max_uni['Threads'] == min_thread].iloc[0]
        
        df_max_uni['Speedup_Total'] = base_row['TotalTime'] / df_max_uni['TotalTime']
        df_max_uni['Speedup_Sort'] = base_row['SortPhase'] / df_max_uni['SortPhase']
        df_max_uni['Speedup_Merge'] = base_row['MergePhase'] / df_max_uni['MergePhase']
        
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        
        ax2.plot(df_max_uni['Threads'], df_max_uni['Speedup_Total'], marker='o', label='Total Speedup', linewidth=2)
        ax2.plot(df_max_uni['Threads'], df_max_uni['Speedup_Sort'], marker='s', linestyle='--', label='Sort Phase Only')
        ax2.plot(df_max_uni['Threads'], df_max_uni['Speedup_Merge'], marker='^', linestyle='--', label='Merge Phase Only')
        
        # Línea ideal
        ax2.plot([min_thread, df_max_uni['Threads'].max()], 
                 [1, df_max_uni['Threads'].max()/min_thread], 
                 color='gray', alpha=0.5, linestyle=':', label='Ideal (Lineal)')
        
        ax2.set_title(f'Speedup Analysis (N={max_size:.0e})')
        ax2.set_xlabel('Threads')
        ax2.set_ylabel('Speedup (x)')
        ax2.legend()
        plt.tight_layout()
        plt.show()

    # ==============================================================================
    # PARTE 3: ANÁLISIS DE REGRESIÓN MEJORADO (RESIDUOS VS PREDICTED)
    # Objetivo: Detectar patrones en el error (Heterocedasticidad)
    # ==============================================================================
    
    # Elegimos el thread count máximo para ver el peor caso de contención de memoria
    target_threads = df['Threads'].max()
    df_reg = df[(df['Algorithm'] == 'SampleMerge') & 
                (df['Distribution'] == 'Uniform') & 
                (df['Threads'] == target_threads)].copy()

    # Variables
    X_sort = df_reg['InputSize'] * np.log2(df_reg['InputSize'])
    Y_sort = df_reg['SortPhase']
    
    X_merge = df_reg['InputSize']
    Y_merge = df_reg['MergePhase'] # Hipótesis Lineal
    
    # Modelos
    model_sort = sm.OLS(Y_sort, sm.add_constant(X_sort)).fit()
    model_merge = sm.OLS(Y_merge, sm.add_constant(X_merge)).fit()
    
    # Gráficos de Diagnóstico
    fig3, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig3.suptitle(f'Diagnóstico de Regresión (Threads={target_threads})', fontsize=16)
    
    # --- A. Sort Phase: Ajuste ---
    ax_a = axes[0, 0]
    ax_a.scatter(df_reg['InputSize'], Y_sort, alpha=0.6, label='Datos')
    # Para visualizar la linea teórica necesitamos reordenar X
    sort_pred = model_sort.predict(sm.add_constant(X_sort))
    ax_a.plot(df_reg['InputSize'], sort_pred, color='red', label='Fit $N \log N$')
    ax_a.set_xscale('log') # Log scale para ver mejor los órdenes de magnitud
    ax_a.set_yscale('log')
    ax_a.set_title('Sort Phase (Log-Log Scale)')
    ax_a.set_xlabel('Input Size (N)')
    ax_a.set_ylabel('Time (ms)')
    ax_a.legend()
    
    # --- B. Sort Phase: Residuos vs Valores Ajustados ---
    # Este gráfico revela si el error crece con el tamaño (forma de cono)
    ax_b = axes[0, 1]
    ax_b.scatter(sort_pred, model_sort.resid, alpha=0.5)
    ax_b.axhline(0, color='black', linestyle='--')
    ax_b.set_title('Sort Phase: Residuals vs Fitted')
    ax_b.set_xlabel('Fitted Values (Time)')
    ax_b.set_ylabel('Residuals')

    # --- C. Merge Phase: Ajuste ---
    ax_c = axes[1, 0]
    ax_c.scatter(df_reg['InputSize'], Y_merge, alpha=0.6, color='green', label='Datos')
    ax_c.plot(df_reg['InputSize'], model_merge.predict(sm.add_constant(X_merge)), color='orange', label='Fit Linear $N$')
    ax_c.set_title('Merge Phase (Linear Scale)')
    ax_c.set_xlabel('Input Size (N)')
    ax_c.set_ylabel('Time (ms)')
    ax_c.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax_c.legend()

    # --- D. Merge Phase: Residuos vs Valores Ajustados ---
    # Si ves una "U" aquí, significa que no es lineal (probablemente cuadrático o N log N oculto)
    ax_d = axes[1, 1]
    ax_d.scatter(model_merge.predict(sm.add_constant(X_merge)), model_merge.resid, alpha=0.5, color='green')
    ax_d.axhline(0, color='black', linestyle='--')
    ax_d.set_title('Merge Phase: Residuals vs Fitted')
    ax_d.set_xlabel('Fitted Values (Time)')
    ax_d.set_ylabel('Residuals')
    
    plt.tight_layout()
    plt.show()
    
    # ==============================================================================
    # PARTE 4: UNIFORME VS NORMAL
    # ==============================================================================
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    
    # Filtramos para el mayor thread count
    df_dist = df[(df['Algorithm'] == 'SampleMerge') & 
                 (df['Threads'] == target_threads)].copy()
    
    sns.lineplot(data=df_dist, x='InputSize', y='TotalTime', hue='Distribution', marker='o', ax=ax4)
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.set_title(f'Impacto de la Distribución (Threads={target_threads})')
    ax4.set_ylabel('Total Time (ms)')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    advanced_benchmark_analysis()