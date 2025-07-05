import scanpy as sc
import numpy as np
import argparse
import scipy.sparse as sp
import pandas as pd
import scipy

def is_normalized(X):
    """Check se i dati sembrano normalizzati."""
    # Per matrici sparse converto in array per il check, ma solo piccola parte (sample)
    if sp.issparse(X):
        sample = X[:1000,:].toarray()  # campioniamo max 1000 righe per velocità
    else:
        sample = X[:1000,:]
    mean = sample.mean()
    min_val = sample.min()
    max_val = sample.max()
    return (0.5 < mean < 2) and (min_val >= 0) and (max_val < 50)


def main_gene_selection_sparse(matrix, genes_in_data, gene_list):
    """
    Seleziona e ordina colonne di matrix secondo gene_list,
    aggiungendo colonne zero per geni mancanti, senza DataFrame.

    matrix: scipy.sparse or np.ndarray (cells x genes)
    genes_in_data: list di geni corrispondenti a colonne di matrix
    gene_list: lista target di geni

    Ritorna:
    matrix_new: matrice cells x genes target (ordinata e padded)
    to_fill_columns: lista geni aggiunti con padding zero
    """

    genes_set = set(genes_in_data)
    gene_list_set = set(gene_list)
    to_fill_columns = list(gene_list_set - genes_set)
    common_genes = [g for g in gene_list if g in genes_set]

    # Indici colonne comuni
    common_indices = [genes_in_data.index(g) for g in common_genes]

    # Slicing matrice colonne comuni
    matrix_common = matrix[:, common_indices]

    # Crea matrice colonne zero per padding
    n_cells = matrix.shape[0]
    n_pad = len(to_fill_columns)
    if sp.issparse(matrix):
        zero_pad = sp.csr_matrix((n_cells, n_pad))
        matrix_new = sp.hstack([matrix_common, zero_pad], format='csr')
    else:
        zero_pad = np.zeros((n_cells, n_pad), dtype=matrix.dtype)
        matrix_new = np.hstack([matrix_common, zero_pad])


    return matrix_new, to_fill_columns


def preprocess_h5ad(input_path, gene_list_path, output_path, output_format='npz', demo=False, chunk_size=10_000):
    import scanpy as sc
    import pandas as pd
    import numpy as np
    import scipy.sparse as sp
    from tqdm import tqdm

    print(f"📥 Caricamento file (in modalità 'backed') da: {input_path}")
    adata_backed = sc.read_h5ad(input_path, backed='r')

    if 'cell_type' not in adata_backed.obs.columns:
        raise ValueError("Colonna 'cell_type' non trovata in adata.obs.")

    # Filtra solo gli indici delle cellule valide
    print("🔍 Indicizzazione delle celle con cell_type valido...")
    valid_idx = adata_backed.obs.index[adata_backed.obs['cell_type'] != 'unknown'].tolist()

    if demo:
        print("⚠️ Modalità demo: uso solo 1000 cellule.")
        valid_idx = valid_idx[:1000]

    print(f"🧪 Celle totali da processare: {len(valid_idx):,}")

    # Carica lista target geni
    gene_list_df = pd.read_csv(gene_list_path, sep='\t')
    target_gene_list = gene_list_df['gene_name'].tolist()
    print(f"📄 Geni target: {len(target_gene_list)}")

    # Carica nomi dei geni dall'h5ad
    if 'Gene' in adata_backed.var.columns:
        gene_names = adata_backed.var['Gene'].tolist()
    elif 'original_gene_symbols' in adata_backed.var.columns:
        gene_names = adata_backed.var['original_gene_symbols'].tolist()
    else:
        raise ValueError("Colonna dei geni non trovata. Attese: 'Gene' o 'original_gene_symbols'.")

    print(f"🧬 Geni disponibili nel dataset: {len(gene_names)}")

    # Controlla intersezione tra geni
    common_genes = set(gene_names).intersection(set(target_gene_list))
    print(f"✅ Geni comuni: {len(common_genes)}")

    # Inizializza lista dei dati
    processed_chunks = []
    processed_obs = []

    # Processa in chunk
    print(f"🔄 Elaborazione in chunk da {chunk_size}...")
    for start in tqdm(range(0, len(valid_idx), chunk_size)):
        end = min(start + chunk_size, len(valid_idx))
        idx_chunk = valid_idx[start:end]
        adata_chunk = sc.read_h5ad(input_path)[idx_chunk].copy()

        # Normalizzazione
        if adata_chunk.raw is not None:
            adata_chunk.X = adata_chunk.raw.X.copy()
            sc.pp.normalize_total(adata_chunk, target_sum=1e4)
            sc.pp.log1p(adata_chunk)
        else:
            if not is_normalized(adata_chunk.X):
                sc.pp.normalize_total(adata_chunk, target_sum=1e4)
                sc.pp.log1p(adata_chunk)

        matrix = adata_chunk.X
        matrix_proc, _ = main_gene_selection_sparse(matrix, gene_names, target_gene_list)

        processed_chunks.append(sp.csr_matrix(matrix_proc))
        processed_obs.append(adata_chunk.obs.copy())

    # Combina tutti i chunk
    print("🧩 Concatenazione finale...")
    full_matrix = sp.vstack(processed_chunks)
    full_obs = pd.concat(processed_obs)

    print(f"🧮 Forma finale: {full_matrix.shape}")

    # Salvataggio
    if output_format == 'npz':
        print(f"💾 Salvataggio in formato NPZ: {output_path}")
        sp.save_npz(output_path, full_matrix)
    else:
        raise ValueError("❌ Salvataggio in formato 'h5ad' non supportato in modalità chunk. Usa 'npz'.")

    print("✅ Completato!")
