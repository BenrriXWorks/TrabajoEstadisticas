import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

def analyze_cache_vs_ram():
    try:
        df = pd.read_csv("benchmark_granular.csv")
    except FileNotFoundError:
        return

    # Filtramos solo SampleMerge y Threads <= 4 (Tu región escalable)
    df = df[(df['Algorithm'] == 'SampleMerge') & (df['Threads'] <= 4)].copy()
    
    # === DEFINIR EL LÍMITE FÍSICO ===
    # Asumimos una L3 Cache de aprox 20MB. 
    # int = 4 bytes. 20MB / 4 = 5,000,000 elementos.
    CACHE_THRESHOLD = 5_000_000
    
    df['Regime'] = np.where(df['InputSize'] < CACHE_THRESHOLD, 'In-Cache (L3)', 'Out-of-Cache (RAM)')
    
    # Preparamos la variable X teórica
    df['Complexity'] = df['InputSize'] * np.log2(df['Threads'])

    # Graficar para visualizar el quiebre
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    
    # Scatter plot log-log para ver el cambio de pendiente
    sns.scatterplot(data=df, x='Complexity', y='MergePhase', hue='Regime', style='Threads', ax=ax[0], palette='bright')
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].set_title('Visualización del "Quiebre" de Memoria')
    ax[0].set_ylabel('Tiempo (ms)')
    
    # === REGRESIÓN SEGMENTADA ===
    print("\n" + "="*40)
    print("   ANÁLISIS SEGMENTADO (Caché vs RAM)")
    print("="*40)
    
    colors = {'In-Cache (L3)': 'blue', 'Out-of-Cache (RAM)': 'red'}
    
    for regime in ['In-Cache (L3)', 'Out-of-Cache (RAM)']:
        subset = df[df['Regime'] == regime].copy()
        
        if subset.empty: continue
            
        X = subset['Complexity']
        y = subset['MergePhase']
        
        # Ajuste OLS
        model = sm.OLS(y, sm.add_constant(X)).fit()
        
        print(f"\n>>> Régimen: {regime}")
        print(f"    R-squared: {model.rsquared:.4f}")
        print(f"    Coeficiente (Pendiente): {model.params['Complexity']:.2e}")
        print(f"    Cond. No.: {model.condition_number:.2e}")

        print(model.summary())
        
        # Graficar la línea de tendencia en el subplot derecho (Escala Lineal)
        ax[1].scatter(X, y, color=colors[regime], alpha=0.3, label=f'{regime} Data')
        
        # Crear linea de predicción
        x_pred = np.linspace(X.min(), X.max(), 100)
        y_pred = model.params['const'] + model.params['Complexity'] * x_pred
        ax[1].plot(x_pred, y_pred, color=colors[regime], linewidth=2, label=f'{regime} Fit ($R^2={model.rsquared:.2f}$)')

    ax[1].set_title('Regresiones Separadas por Régimen Físico')
    ax[1].set_xlabel('$N \cdot \log_2(Threads)$')
    ax[1].set_ylabel('Tiempo (ms)')
    ax[1].legend()
    ax[1].grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    analyze_cache_vs_ram()