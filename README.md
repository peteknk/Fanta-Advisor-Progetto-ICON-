# Fanta-Advisor: Sistema Intelligente per l'Asta del Fantacalcio

Repository per il progetto di Ingegneria della Conoscenza (ICON) 2024-2025.

**Realizzato da:**
* Pietro Conca (mat. 697870)

---

## Installazione

1. **Installare SWI Prolog** (necessario per eventuali future interrogazioni della base di conoscenza generata):
   https://www.swi-prolog.org/download/stable?show=all

2. **Posizionarsi all'interno della root principale del progetto:**
   ```bash
   cd "Progetto icon 2.0"
Creare e attivare l'ambiente virtuale (consigliato):

```bash
python -m venv venv
```
# Su Linux/Mac:
```
source venv/bin/activate
```
# Su Windows:
```
venv\Scripts\activate
```
Installare le dipendenze Python:

```
pip install pandas numpy scikit-learn matplotlib
```
Guida all'utilizzo
Il sistema è strutturato come una pipeline di Machine Learning che culmina con la generazione di una Knowledge Base logica formattata per Prolog.

1. Preparazione dei Dati (Data Ingestion)
Il programma necessita prima di tutto di pulire i CSV di partenza e ingegnerizzare le feature (come l'Indice Rigorista). Eseguire da terminale:

```
python src/data_prep.py --data-dir data
```
Questo genererà il file dataset_clean.csv contenente i giocatori con statistiche rilevanti.

2. Modelli di Machine Learning
Una volta puliti i dati, il sistema addestra e confronta diversi modelli di regressione (Ridge, Random Forest, Gradient Boosting, MLP) con K-Fold CV per calcolare lo Score Convenienza di ogni giocatore:

```
python src/ml_models.py
```
Questo step genererà le predizioni nel file fanta_predictions.csv e formatterà l'output esportando la Knowledge Base nel file data/giocatori.pl.

3. Visualizzazione
Per generare i grafici relativi alle performance dei modelli, alle distribuzioni delle quotazioni e all'importanza delle feature, eseguire:

```
python src/visualizations.py
```
I grafici verranno salvati nella cartella /grafici.

Integrazione Prolog (Knowledge Base)
Il modulo di Machine Learning funge da "ponte" verso la programmazione logica. L'output finale della pipeline Python è il file data/giocatori.pl.

Questo file contiene i fatti Prolog generati dinamicamente (nel formato giocatore(Id, Nome, Ruolo, Prezzo, ScoreConvenienza, Affidabilita, Squadra)) e rappresenta la base di conoscenza (Knowledge Base) pronta per essere importata e interrogata da un Constraint Satisfaction and Optimization Problem (CSOP) resolver per la generazione automatica della rosa ottimale.

Esempi concreti di utilizzo delle opzioni
Ricerca giocatori sottovalutati: Eseguendo il modulo ML, il sistema stampa a schermo la "Top 5" dei giocatori più sottovalutati per ogni ruolo, permettendo all'utente di identificare immediatamente i migliori "affari" per l'asta.

Analisi di convenienza generale: Consultando il file fanta_predictions.csv generato, è possibile ordinare tutti i giocatori della Serie A in base al loro residuo normalizzato (Score Convenienza) per capire rapidamente quali calciatori ignorare perché troppo costosi rispetto alle loro reali prestazioni statistiche.
