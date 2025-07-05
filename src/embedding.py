import random
import os
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
import scanpy as sc
from scipy.sparse import issparse
from src.load import load_model_frommmf, gatherData

def process_gene_expression(
    data_path: str,
    ckpt_path: str = './models/models.ckpt',
    save_path: str = './output',
    task_name: str = 'embedding_task',
    ckpt_name: str = 'model',
    input_type: str = 'gene_expression',
    output_type: str = 'embedding',
    target_high_resolution: str = 'R1',
    pool_type: str = 'all',
    batch_size: int = 1000,
    seed: int = 0,
    use_fp16: bool = True,
    backup_interval_cells: int = 100000,
    resume_from_backup: bool = True,
    start_offset_cells: int = 0,
    end_offset_cells: int = -1
):
    # Imposta seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Caricamento dati
    try:
        gexpr_feature = sc.read_h5ad(data_path, backed='r')
        print(f"Loaded data with shape {gexpr_feature.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    total_cells = gexpr_feature.shape[0]
    actual_start_idx = max(0, start_offset_cells)
    actual_end_idx = total_cells if end_offset_cells == -1 else min(total_cells, end_offset_cells)

    if actual_start_idx >= actual_end_idx:
        print(f"Invalid range: {actual_start_idx} >= {actual_end_idx}")
        return

    num_cells = actual_end_idx - actual_start_idx
    specific_save_path = os.path.join(save_path, f"part_{actual_start_idx}_to_{actual_end_idx - 1}")
    os.makedirs(specific_save_path, exist_ok=True)

    print(f"Processing {num_cells} cells, saving to {specific_save_path}")

    # Carica modello
    try:
        pretrainmodel, pretrainconfig = load_model_frommmf(ckpt_path, 'cell')
        pretrainmodel.to(device).eval()
        if use_fp16 and device.type == 'cuda':
            pretrainmodel.half()
            print("Using float16")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Backup resume
    all_embeddings = []
    all_ids = []
    progress_idx = 0
    cells_since_last_backup = 0


    if resume_from_backup:
        backup_files = sorted([f for f in os.listdir(specific_save_path)
                               if f.startswith(f"{task_name}_backup_") and f.endswith(".h5ad")])
        if backup_files:
            last_backup = backup_files[-1]
            progress_idx = int(last_backup.split('_')[-1].replace('.h5ad', ''))
            print(f"Resuming from {last_backup} at {progress_idx} cells")
            backup_adata = sc.read_h5ad(os.path.join(specific_save_path, last_backup))
            all_embeddings.append(backup_adata.X)
            all_ids.extend(backup_adata.obs_names.tolist())
        else:
            print("No backup found, starting fresh")
    else:
        print("Backup resume disabled, starting fresh")

    # Prepara nomi finali
    npy_path = os.path.join(specific_save_path, f"{task_name}_{ckpt_name}_{input_type}_{output_type}_{target_high_resolution}_part_{actual_start_idx}_to_{actual_end_idx-1}.npy")
    h5ad_path = os.path.join(specific_save_path, f"{task_name}_{ckpt_name}_{input_type}_{output_type}_{target_high_resolution}_part_{actual_start_idx}_to_{actual_end_idx-1}.h5ad")

    try:
        resolution_value = float(target_high_resolution[1:])
    except:
        resolution_value = 0.0

    # Inference
    cell_ids = gexpr_feature.obs_names.tolist()
    batch_embeddings = []
    batch_ids = []

    for i in tqdm(range(actual_start_idx + progress_idx, actual_end_idx, batch_size),
                  desc="Processing batches"):
        end_idx = min(i + batch_size, actual_end_idx)
        batch = gexpr_feature[i:end_idx, :]
        batch_cell_ids = cell_ids[i:end_idx]

        cells_since_last_backup += len(batch_cell_ids)



        gene_x_list = []
        for row in batch.X:
            totalcount = row.sum()
            logcount = np.log10(totalcount) if totalcount > 0 else 0
            vals = row.toarray().flatten() if issparse(row) else row.flatten()
            gene_x_list.append(np.concatenate([vals, [resolution_value, logcount]]))

        gene_x_array = np.array(gene_x_list)
        tensor_batch = torch.tensor(gene_x_array).to(device)
        tensor_batch = tensor_batch.half() if use_fp16 and device.type == 'cuda' else tensor_batch.float()

        with torch.no_grad():
            data_gene_ids = torch.arange(19266, device=device).repeat(tensor_batch.shape[0], 1)
            val_labels = tensor_batch > 0
            x, x_pad = gatherData(tensor_batch, val_labels, pretrainconfig['pad_token_id'])
            pos_ids, _ = gatherData(data_gene_ids, val_labels, pretrainconfig['pad_token_id'])

            x = pretrainmodel.token_emb(x.unsqueeze(2), output_weight=0)
            x += pretrainmodel.pos_emb(pos_ids)
            geneemb = pretrainmodel.encoder(x, x_pad)

            e1 = geneemb[:, -1, :]
            e2 = geneemb[:, -2, :]
            e3, _ = torch.max(geneemb[:, :-2, :], dim=1)
            e4 = torch.mean(geneemb[:, :-2, :], dim=1)

            if pool_type == 'all':
                emb = torch.cat([e1, e2, e3, e4], dim=1)
            elif pool_type == 'max':
                emb, _ = torch.max(geneemb, dim=1)
            else:
                raise ValueError("pool_type must be 'all' or 'max'")

            batch_embeddings.append(emb.cpu().numpy())
            batch_ids.extend(batch_cell_ids)

        # Backup save
        if cells_since_last_backup >= backup_interval_cells or end_idx == actual_end_idx:

            cells_since_last_backup = 0

            combined = np.concatenate(batch_embeddings, axis=0)
            all_embeddings.append(combined)
            all_ids.extend(batch_ids)

            obs_df = pd.DataFrame(index=all_ids)
            backup_adata = sc.AnnData(X=np.concatenate(all_embeddings, axis=0), obs=obs_df)
            backup_name = f"{task_name}_backup_{actual_start_idx}_{actual_end_idx-1}_{len(all_ids)}.h5ad"
            backup_path = os.path.join(specific_save_path, backup_name)
            backup_adata.write(backup_path)
            print(f"Saved backup: {backup_path}")

            batch_embeddings = []
            batch_ids = []
            torch.cuda.empty_cache()

    # Final save
    if batch_embeddings:
        combined = np.concatenate(batch_embeddings, axis=0)
        all_embeddings.append(combined)
        all_ids.extend(batch_ids)

    final_emb = np.concatenate(all_embeddings, axis=0)
    np.save(npy_path, final_emb)
    obs_df = pd.DataFrame(index=all_ids)
    adata = sc.AnnData(X=final_emb, obs=obs_df)
    adata.write(h5ad_path)

    print(f"Saved final embeddings to {npy_path} and {h5ad_path}")
