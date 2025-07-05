from src.preprocess import preprocess_h5ad

try:
    input_path = "/equilibrium/datasets/TCGA-histological-data/nervous.h5ad"
    gene_list_path = "./OS_scRNA_gene_index.19264.tsv"
    output_path = "/equilibrium/datasets/TCGA-histological-data/nervous_processed.h5ad"
    output_format='h5ad'
    preprocess_h5ad(input_path, gene_list_path, output_path, output_format)
except Exception as e:
    print(f"Error during preprocessing: {e}")
    raise e