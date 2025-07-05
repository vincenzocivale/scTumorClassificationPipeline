from src.embedding import process_gene_expression
from src.telegram import TelegramNotifier
import os

# Crea la directory 'results' se non esiste

output_dir = '/equilibrium/datasets/TCGA-histological-data/lung_embeddings_lechunck'

os.makedirs(output_dir, exist_ok=True)

# Percorso al tuo file H5AD pre-processato
input_h5ad_file = '/equilibrium/datasets/TCGA-histological-data/lung_processed_copy.h5ad' 
# Assicurati che questo file esista! Potresti doverlo creare con il tuo preprocess.py script.

# Se il modello non è in './models/models.ckpt', specifica il percorso esatto
model_ckpt_path = '/equilibrium/datasets/TCGA-histological-data/models.ckpt'

print(f"Inizio generazione embedding per {input_h5ad_file}")

TELEGRAM_BOT_TOKEN = "8145248176:AAHjCprIvnz7AB4-oS8ycvdNlRBw519z-rg"
TELEGRAM_CHAT_ID = "564539816"

notifier = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)

try:
    process_gene_expression(
        data_path=input_h5ad_file,
        ckpt_path=model_ckpt_path,
        save_path=output_dir,
        task_name='batched_analysis',
        target_high_resolution='t4',
        pool_type='all',
        batch_size=1, 
        seed=42,
        start_offset_cells=800000
    )
    print(f"\nProcesso completato. Embedding salvati in: {output_dir}")
    notifier.send_message(f"\nProcesso completato. Embedding salvati in: {output_dir}")

except Exception as e:
    print(f"\nSi è verificato un errore durante la generazione degli embedding: {e}")
    notifier.send_message(f"\nSi è verificato un errore durante la generazione degli embedding: {e} LECHUNCK")
