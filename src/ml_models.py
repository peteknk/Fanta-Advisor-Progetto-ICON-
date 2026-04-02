"""
Fanta-Advisor - Modulo 2: Machine Learning Models
==================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
N_FOLDS = 10


def load_and_prepare_data(filepath: str = "data/dataset_clean.csv"):
    df = pd.read_csv(filepath, sep=';')
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = df[col].str.replace(',', '.').astype(float)
            except:
                pass
    return df


def get_features_for_role(df_role: pd.DataFrame) -> list:
    exclude_cols = ['Id', 'Nome', 'Squadra', 'R', 'Rm', 'RM', 'Qt.A', 'Mv', 'Fm']
    features = [c for c in df_role.columns if c not in exclude_cols]
    valid_features = [f for f in features if df_role[f].std() > 0]
    return valid_features


def train_models_for_role(df_role: pd.DataFrame, role: str):
    features = get_features_for_role(df_role)
    X = df_role[features].values
    y = np.log1p(df_role['Qt.A'].values)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    models = {
        'Ridge': Ridge(alpha=1.0, random_state=RANDOM_STATE),
        'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=RANDOM_STATE),
        'MLP': MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=RANDOM_STATE, early_stopping=True)
    }
    
    kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    print(f"\n  [Ruolo {role}] {len(df_role)} giocatori, {len(features)} features")
    
    for name, model in models.items():
        try:
            scores = cross_val_score(model, X_scaled, y, cv=kfold, scoring='r2')
            results[name] = {'mean_r2': scores.mean(), 'std_r2': scores.std(), 'model': model}
            print(f"    {name}: R2 = {scores.mean():.4f} (+/- {scores.std():.4f})")
        except Exception as e:
            print(f"    {name}: ERRORE")
    
    best_name = max(results, key=lambda k: results[k]['mean_r2'])
    best_model = results[best_name]['model']
    best_model.fit(X_scaled, y)
    print(f"    >>> Migliore: {best_name} (R2={results[best_name]['mean_r2']:.4f})")
    
    return {'model': best_model, 'scaler': scaler, 'features': features, 'name': best_name, 'r2': results[best_name]['mean_r2']}


def compute_score_convenienza(df: pd.DataFrame, role_models: dict) -> pd.DataFrame:
    df = df.copy()
    df['Qt.A_predicted'] = 0.0
    df['Score_Convenienza'] = 0.0
    df['Best_Model'] = ''
    
    for role, model_info in role_models.items():
        mask = df['R'] == role
        df_role = df[mask]
        if len(df_role) == 0:
            continue
        
        X = df_role[model_info['features']].values
        X_scaled = model_info['scaler'].transform(X)
        y_pred_log = model_info['model'].predict(X_scaled)
        y_pred = np.expm1(y_pred_log)
        
        qt_a = df_role['Qt.A'].values
        score_conv = (y_pred - qt_a) / np.maximum(qt_a, 1)
        
        df.loc[mask, 'Qt.A_predicted'] = y_pred
        df.loc[mask, 'Score_Convenienza'] = score_conv
        df.loc[mask, 'Best_Model'] = model_info['name']
    
    return df


def create_affidabilita_classe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    pv = pd.to_numeric(df['Pv'], errors='coerce').fillna(0)
    mv = pd.to_numeric(df['Mv'], errors='coerce').fillna(0)
    df['Affidabilita_Classe'] = ((pv >= 20) & (mv >= 6.0)).astype(int)
    return df


def export_for_prolog(df: pd.DataFrame, output_path: str = "data/giocatori.pl"):
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("% Fatti Prolog generati da ml_models.py\n")
        f.write("% giocatore(Id, Nome, Ruolo, Prezzo, ScoreConvenienza, Affidabilita, Squadra).\n\n")
        for _, row in df.iterrows():
            nome = str(row['Nome']).replace("'", "\\'")
            squadra = str(row['Squadra']).replace("'", "\\'")
            f.write(f"giocatore({int(row['Id'])}, '{nome}', '{row['R']}', "
                   f"{int(row['Qt.A'])}, {row['Score_Convenienza']:.4f}, "
                   f"{int(row['Affidabilita_Classe'])}, '{squadra}').\n")
    print(f"[OK] Fatti Prolog salvati: {output_path}")


def main():
    print("=" * 60)
    print("FANTA-ADVISOR - Machine Learning Pipeline")
    print("=" * 60)
    
    print("\n[STEP 1] Caricamento dataset...")
    df = load_and_prepare_data("data/dataset_clean.csv")
    print(f"[OK] Caricati {len(df)} giocatori")
    
    print("\n[STEP 2] Addestramento modelli per ruolo (K-Fold CV, k=10)...")
    roles = ['P', 'D', 'C', 'A']
    role_models = {}
    
    for role in roles:
        df_role = df[df['R'] == role].copy()
        if len(df_role) >= 10:
            role_models[role] = train_models_for_role(df_role, role)
    
    print("\n[STEP 3] Calcolo Score_Convenienza...")
    df = compute_score_convenienza(df, role_models)
    df = create_affidabilita_classe(df)
    
    print("\n[RISULTATI] Score_Convenienza per ruolo:")
    for role in roles:
        df_role = df[df['R'] == role]
        print(f"  {role}: mean={df_role['Score_Convenienza'].mean():.3f}, "
              f"min={df_role['Score_Convenienza'].min():.3f}, max={df_role['Score_Convenienza'].max():.3f}")
    
    print("\n[TOP 5 SOTTOVALUTATI PER RUOLO]")
    for role in roles:
        df_role = df[df['R'] == role].nlargest(5, 'Score_Convenienza')
        print(f"\n  {role}:")
        for _, row in df_role.iterrows():
            print(f"    {row['Nome']:20} | Qt.A={int(row['Qt.A']):3} | Pred={row['Qt.A_predicted']:.1f} | Score={row['Score_Convenienza']:.3f}")
    
    print("\n[STEP 4] Salvataggio risultati...")
    output_cols = ['Id', 'Nome', 'Squadra', 'R', 'Rm', 'Qt.A', 'Qt.A_predicted', 
                   'Score_Convenienza', 'Affidabilita_Classe', 'Best_Model', 'Pv', 'Gf', 'Ass', 'Indice_Rigorista']
    df_out = df[[c for c in output_cols if c in df.columns]]
    df_out.to_csv("data/fanta_predictions.csv", index=False, sep=';')
    print(f"[OK] Predizioni salvate: data/fanta_predictions.csv")
    
    export_for_prolog(df, "data/giocatori.pl")
    
    print("\n" + "=" * 60)
    print("Pipeline ML completata!")
    print("=" * 60)
    return df

if __name__ == "__main__":
    main()
