"""Fanta-Advisor - Generazione Grafici"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import r2_score

plt.rcParams['figure.figsize'] = (10, 6)
OUTPUT_DIR = Path("grafici")
OUTPUT_DIR.mkdir(exist_ok=True)

def load_data():
    df = pd.read_csv("data/fanta_predictions.csv", sep=';')
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = df[col].str.replace(',', '.').astype(float)
            except:
                pass
    return df

def plot_qt_distribution(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    roles_order = ['P', 'D', 'C', 'A']
    colors = ['#3498db', '#2ecc71', '#f1c40f', '#e74c3c']
    bp = ax.boxplot([df[df['R']==r]['Qt.A'].values for r in roles_order],
                    labels=['Portieri', 'Difensori', 'Centrocampisti', 'Attaccanti'],
                    patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('Quotazione Asta (Qt.A)')
    ax.set_title('Distribuzione Quotazioni per Ruolo')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '1_distribuzione_quotazioni.png', dpi=150)
    plt.close()
    print("[OK] 1_distribuzione_quotazioni.png")

def plot_score_convenienza(df):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    roles = ['P', 'D', 'C', 'A']
    titles = ['Portieri', 'Difensori', 'Centrocampisti', 'Attaccanti']
    colors = ['#3498db', '#2ecc71', '#f1c40f', '#e74c3c']
    for ax, role, title, color in zip(axes.flat, roles, titles, colors):
        data = df[df['R']==role]['Score_Convenienza']
        ax.hist(data, bins=20, color=color, alpha=0.7, edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax.axvline(x=data.mean(), color='blue', linestyle='-', linewidth=2)
        ax.set_xlabel('Score Convenienza')
        ax.set_ylabel('Frequenza')
        ax.set_title(f'{title} (n={len(data)})')
    plt.suptitle('Score_Convenienza per Ruolo (>0 = sottovalutato)', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '2_score_convenienza.png', dpi=150)
    plt.close()
    print("[OK] 2_score_convenienza.png")

def plot_top_sottovalutati(df):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    roles = ['P', 'D', 'C', 'A']
    titles = ['Portieri', 'Difensori', 'Centrocampisti', 'Attaccanti']
    colors = ['#3498db', '#2ecc71', '#f1c40f', '#e74c3c']
    for ax, role, title, color in zip(axes.flat, roles, titles, colors):
        top = df[df['R']==role].nlargest(10, 'Score_Convenienza')[['Nome', 'Score_Convenienza', 'Qt.A']]
        y_pos = np.arange(len(top))
        ax.barh(y_pos, top['Score_Convenienza'].values, color=color, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{n} ({int(q)})" for n, q in zip(top['Nome'], top['Qt.A'])])
        ax.set_xlabel('Score Convenienza')
        ax.set_title(f'Top 10 {title} Sottovalutati')
        ax.axvline(x=0, color='gray', linestyle='--')
    plt.suptitle('Top 10 Giocatori Sottovalutati per Ruolo', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '4_top_sottovalutati.png', dpi=150)
    plt.close()
    print("[OK] 4_top_sottovalutati.png")

def plot_actual_vs_predicted(df):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    roles = ['P', 'D', 'C', 'A']
    titles = ['Portieri', 'Difensori', 'Centrocampisti', 'Attaccanti']
    colors = ['#3498db', '#2ecc71', '#f1c40f', '#e74c3c']
    for ax, role, title, color in zip(axes.flat, roles, titles, colors):
        data = df[df['R']==role]
        ax.scatter(data['Qt.A'], data['Qt.A_predicted'], c=color, alpha=0.6, s=50)
        max_val = max(data['Qt.A'].max(), data['Qt.A_predicted'].max())
        ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2)
        ax.set_xlabel('Qt.A Reale')
        ax.set_ylabel('Qt.A Predetto')
        ax.set_title(f'{title} (n={len(data)})')
    plt.suptitle('Quotazione Reale vs Predetta (sopra linea = sottovalutati)', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '5_actual_vs_predicted.png', dpi=150)
    plt.close()
    print("[OK] 5_actual_vs_predicted.png")

def plot_model_comparison(df):
    roles = ['P', 'D', 'C', 'A']
    r2_scores = []
    for r in roles:
        data_role = df[df['R'] == r]
        if len(data_role) > 1:
            score = r2_score(data_role['Qt.A'], data_role['Qt.A_predicted'])
            r2_scores.append(score)
        else:
            r2_scores.append(0)
            
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#3498db', '#2ecc71', '#f1c40f', '#e74c3c']
    ax.bar(['Portieri', 'Difensori', 'Centrocampisti', 'Attaccanti'], r2_scores, color=colors, alpha=0.8)
    ax.set_xlabel('Ruolo')
    ax.set_ylabel('R2 Score')
    ax.set_title('Confronto Performance Modelli ML per Ruolo')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '3_confronto_modelli.png', dpi=150)
    plt.close()
    print("[OK] 3_confronto_modelli.png")

def plot_feature_importance(df):
    features = ['Pv', 'Gf', 'Ass', 'Indice_Rigorista']
    correlations = df[features + ['Qt.A']].corr()['Qt.A'].drop('Qt.A').abs().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(correlations)))
    ax.barh(correlations.index, correlations.values, color=colors)
    ax.set_xlabel('Importanza Relativa')
    ax.set_title('Feature Importance (Correlazione Pearson)')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '6_feature_importance.png', dpi=150)
    plt.close()
    print("[OK] 6_feature_importance.png")

def main():
    print("Generazione Grafici...")
    df = load_data()
    print(f"Dati caricati: {len(df)} giocatori")
    
    plot_qt_distribution(df)
    plot_score_convenienza(df)
    plot_model_comparison(df)
    plot_top_sottovalutati(df)
    plot_actual_vs_predicted(df)
    plot_feature_importance(df)
    
    print(f"\nGrafici salvati in: grafici/")

if __name__ == "__main__":
    main()