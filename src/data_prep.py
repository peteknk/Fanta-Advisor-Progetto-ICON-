"""
Fanta-Advisor - Modulo 1: Data Ingestion e Feature Engineering
================================================================

Questo script implementa le fasi 1.2 e 1.3 dell'architettura:
- Caricamento e merge dei CSV (statistiche + quotazioni)
- Filtro giocatori con Pv >= 3
- Gestione NaN con mediana per-ruolo
- Feature engineering secondo le specifiche del documento
- Salvataggio del dataset pulito per l'addestramento ML

Autore: Fanta-Advisor Team
Progetto ICON - Universita di Bari
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# === COSTANTI ===
MIN_PRESENZE = 3
GIORNATE_TOTALI = 38


class DataPreparation:
    """
    Classe per la preparazione dei dati del Fantacalcio.
    Implementa il pipeline di data ingestion e feature engineering.
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.df_stats: Optional[pd.DataFrame] = None
        self.df_quotes: Optional[pd.DataFrame] = None
        self.df_merged: Optional[pd.DataFrame] = None
        self.df_clean: Optional[pd.DataFrame] = None
        
    def _convert_decimal_separator(self, df: pd.DataFrame) -> pd.DataFrame:
        """Converte i separatori decimali da virgola italiana a punto."""
        df_converted = df.copy()
        for col in df_converted.columns:
            if df_converted[col].dtype == 'object':
                try:
                    df_converted[col] = df_converted[col].str.replace(',', '.').astype(float)
                except (ValueError, AttributeError):
                    pass
        return df_converted
    
    def load_csv(self, stats_file: str = "statistiche.csv", 
                 quotes_file: str = "quotazioni.csv") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Carica i file CSV delle statistiche e delle quotazioni."""
        stats_path = self.data_dir / stats_file
        quotes_path = self.data_dir / quotes_file
        
        if not stats_path.exists():
            raise FileNotFoundError(f"File statistiche non trovato: {stats_path}")
        if not quotes_path.exists():
            raise FileNotFoundError(f"File quotazioni non trovato: {quotes_path}")
        
        # I CSV hanno una riga di titolo prima dell'header, quindi skiprows=1
        self.df_stats = pd.read_csv(stats_path, sep=';', encoding='utf-8', skiprows=1)
        self.df_quotes = pd.read_csv(quotes_path, sep=';', encoding='utf-8', skiprows=1)
        
        self.df_stats = self._convert_decimal_separator(self.df_stats)
        self.df_quotes = self._convert_decimal_separator(self.df_quotes)
        
        print(f"[OK] Statistiche caricate: {len(self.df_stats)} giocatori")
        print(f"[OK] Quotazioni caricate: {len(self.df_quotes)} giocatori")
        print(f"    Colonne stats: {list(self.df_stats.columns)}")
        print(f"    Colonne quotes: {list(self.df_quotes.columns)}")
        
        return self.df_stats, self.df_quotes
    
    def merge_datasets(self) -> pd.DataFrame:
        """Effettua il merge dei dataset su Id."""
        if self.df_stats is None or self.df_quotes is None:
            raise ValueError("Caricare prima i CSV con load_csv()")
        
        quote_cols = ['Id', 'Qt.A']
        self.df_merged = pd.merge(
            self.df_stats,
            self.df_quotes[quote_cols],
            on='Id',
            how='inner'
        )
        
        print(f"[OK] Merge completato: {len(self.df_merged)} giocatori matchati")
        return self.df_merged
    
    def filter_by_presenze(self, min_pv: int = MIN_PRESENZE) -> pd.DataFrame:
        """Filtra i giocatori con presenze inferiori alla soglia."""
        if self.df_merged is None:
            raise ValueError("Eseguire prima merge_datasets()")
        
        n_before = len(self.df_merged)
        self.df_merged = self.df_merged[self.df_merged['Pv'] >= min_pv].copy()
        n_after = len(self.df_merged)
        
        print(f"[OK] Filtro Pv >= {min_pv}: {n_before} -> {n_after} giocatori ({n_before - n_after} rimossi)")
        return self.df_merged
    
    def handle_missing_values(self) -> pd.DataFrame:
        """Gestisce i valori mancanti con imputazione mediana per-ruolo."""
        if self.df_merged is None:
            raise ValueError("Eseguire prima merge_datasets() e filter_by_presenze()")
        
        numeric_cols = self.df_merged.select_dtypes(include=[np.number]).columns.tolist()
        nan_counts = self.df_merged[numeric_cols].isna().sum()
        total_nan = nan_counts.sum()
        
        if total_nan > 0:
            print(f"[!] Trovati {total_nan} valori NaN:")
            for col, count in nan_counts[nan_counts > 0].items():
                print(f"    - {col}: {count} NaN")
            
            for col in numeric_cols:
                if self.df_merged[col].isna().any():
                    self.df_merged[col] = self.df_merged.groupby('R')[col].transform(
                        lambda x: x.fillna(x.median())
                    )
            
            self.df_merged[numeric_cols] = self.df_merged[numeric_cols].fillna(
                self.df_merged[numeric_cols].median()
            )
            print("[OK] NaN imputati con mediana per-ruolo")
        else:
            print("[OK] Nessun valore NaN trovato")
        
        return self.df_merged
    
    def _compute_indice_rigorista(self, row: pd.Series) -> float:
        """
        Calcola l'Indice Rigorista per un singolo giocatore.
        
        PORTIERI: Indice = Rp * 3.0 (rigori parati)
        MOVIMENTO: Indice = 1.5 * I(Rc > 0) + 3.0 * (R+ - R-)
        """
        ruolo = row['R']
        
        if ruolo == 'P':
            return float(row.get('Rp', 0)) * 3.0
        else:
            rc = float(row.get('Rc', 0))
            r_plus = float(row.get('R+', 0))
            r_minus = float(row.get('R-', 0))
            is_rigorista = 1.0 if rc > 0 else 0.0
            return 1.5 * is_rigorista + 3.0 * (r_plus - r_minus)
    
    def engineer_features(self) -> pd.DataFrame:
        """
        Crea le feature ingegnerizzate:
        - Affidabilita: Pv / 38
        - Gf_per_90: Gf / max(Pv, 1)
        - Ass_per_90: Ass / max(Pv, 1)
        - Gs_per_90: Gs / max(Pv, 1)
        - Disciplina: -(Amm * 0.5 + Esp * 3) / max(Pv, 1)
        - Au_per_90: Au / max(Pv, 1)
        - Indice_Rigorista: formula domain-specific
        """
        if self.df_merged is None:
            raise ValueError("Eseguire prima la pipeline di pulizia")
        
        df = self.df_merged.copy()
        pv_safe = df['Pv'].clip(lower=1)
        
        df['Affidabilita'] = df['Pv'] / GIORNATE_TOTALI
        df['Gf_per_90'] = df['Gf'] / pv_safe
        df['Ass_per_90'] = df['Ass'] / pv_safe
        df['Gs_per_90'] = df['Gs'] / pv_safe
        df['Disciplina'] = -(df['Amm'] * 0.5 + df['Esp'] * 3) / pv_safe
        df['Au_per_90'] = df['Au'] / pv_safe
        df['Indice_Rigorista'] = df.apply(self._compute_indice_rigorista, axis=1)
        
        print("[OK] Feature ingegnerizzate create:")
        new_features = ['Affidabilita', 'Gf_per_90', 'Ass_per_90', 'Gs_per_90', 
                       'Disciplina', 'Au_per_90', 'Indice_Rigorista']
        for feat in new_features:
            print(f"    - {feat}: min={df[feat].min():.3f}, max={df[feat].max():.3f}, mean={df[feat].mean():.3f}")
        
        self.df_clean = df
        return df
    
    def get_feature_columns(self) -> dict:
        """Restituisce le liste delle colonne per feature e target."""
        features_base = ['Mv', 'Fm', 'Pv', 'Gf', 'Gs', 'Ass', 'Amm', 'Esp', 'Au', 'Rp', 'Rc', 'R+', 'R-']
        features_engineered = ['Affidabilita', 'Gf_per_90', 'Ass_per_90', 'Gs_per_90', 
                              'Disciplina', 'Au_per_90', 'Indice_Rigorista']
        
        return {
            'features': features_base + features_engineered,
            'target': 'Qt.A',
            'metadata': ['Id', 'Nome', 'Squadra', 'R', 'Rm']
        }
    
    def save_clean_dataset(self, output_file: str = "dataset_clean.csv") -> Path:
        """Salva il dataset pulito e pronto per l'addestramento."""
        if self.df_clean is None:
            raise ValueError("Eseguire prima engineer_features()")
        
        output_path = self.data_dir / output_file
        self.df_clean.to_csv(output_path, index=False, sep=';')
        
        print(f"\n[OK] Dataset salvato: {output_path}")
        print(f"    - Dimensioni: {self.df_clean.shape[0]} righe x {self.df_clean.shape[1]} colonne")
        print(f"    - Ruoli: {self.df_clean['R'].value_counts().to_dict()}")
        
        return output_path
    
    def get_summary_stats(self) -> pd.DataFrame:
        """Genera statistiche descrittive del dataset per ruolo."""
        if self.df_clean is None:
            raise ValueError("Eseguire prima engineer_features()")
        
        numeric_cols = self.df_clean.select_dtypes(include=[np.number]).columns
        return self.df_clean.groupby('R')[numeric_cols].agg(['mean', 'std', 'min', 'max'])
    
    def run_pipeline(self, stats_file: str = "statistiche.csv",
                    quotes_file: str = "quotazioni.csv",
                    output_file: str = "dataset_clean.csv") -> pd.DataFrame:
        """Esegue l'intero pipeline di data preparation."""
        print("=" * 60)
        print("FANTA-ADVISOR - Data Preparation Pipeline")
        print("=" * 60)
        print()
        
        print("[STEP 1] Caricamento CSV...")
        self.load_csv(stats_file, quotes_file)
        print()
        
        print("[STEP 2] Merge datasets...")
        self.merge_datasets()
        print()
        
        print("[STEP 3] Filtro Pv >= 3...")
        self.filter_by_presenze()
        print()
        
        print("[STEP 4] Gestione valori mancanti...")
        self.handle_missing_values()
        print()
        
        print("[STEP 5] Feature Engineering...")
        self.engineer_features()
        print()
        
        print("[STEP 6] Salvataggio dataset...")
        self.save_clean_dataset(output_file)
        
        print()
        print("=" * 60)
        print("Pipeline completata con successo!")
        print("=" * 60)
        
        return self.df_clean


def main():
    """Punto di ingresso principale."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Fanta-Advisor Data Preparation - Modulo 1.2 e 1.3',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Directory contenente i file CSV')
    parser.add_argument('--stats-file', type=str, default='statistiche.csv',
                       help='Nome del file CSV delle statistiche')
    parser.add_argument('--quotes-file', type=str, default='quotazioni.csv',
                       help='Nome del file CSV delle quotazioni')
    parser.add_argument('--output-file', type=str, default='dataset_clean.csv',
                       help='Nome del file CSV di output')
    
    args = parser.parse_args()
    
    prep = DataPreparation(data_dir=args.data_dir)
    df = prep.run_pipeline(
        stats_file=args.stats_file,
        quotes_file=args.quotes_file,
        output_file=args.output_file
    )
    
    print("\n[ANTEPRIMA] Prime 5 righe del dataset pulito:")
    cols_preview = ['Nome', 'Squadra', 'R', 'Qt.A', 'Fm', 'Gf_per_90', 'Indice_Rigorista']
    print(df[cols_preview].head(10).to_string())
    
    print("\n[COLONNE DISPONIBILI]")
    cols = prep.get_feature_columns()
    print(f"  Features ({len(cols['features'])}): {cols['features']}")
    print(f"  Target: {cols['target']}")
    print(f"  Metadata: {cols['metadata']}")


if __name__ == "__main__":
    main()
