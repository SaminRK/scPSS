import scanpy as sc
from anndata import AnnData
import scanpy.external as sce
    
def integrate(adata: AnnData, integration_key: str = 'dataset') -> AnnData:
    """
    Ensures that PCA (X_pca) exists in the AnnData object and performs integration using Harmony.

    Parameters:
    adata (AnnData): The annotated data matrix.
    integration_key (str): The key in `adata.obs` to use for integration. Default is 'dataset'.

    Returns:
    AnnData: The updated AnnData object with X_pca and integration applied.
    """
    # Check if 'X_pca' already exists in .obsm
    if 'X_pca' not in adata.obsm:
        print("X_pca not found. Performing PCA...")
        
        # Check if the data matrix has been normalized and log-transformed
        if not adata.raw:
            print("Warning: The AnnData object does not have a raw layer. Ensure data is normalized/log-transformed.")
        
        # Perform PCA using Scanpy
        sc.tl.pca(adata)
    else:
        print("X_pca already exists in the AnnData object.")

    # Perform integration using Harmony
    print(f"Performing Harmony integration using key: {integration_key}...")
    sce.pp.harmony_integrate(adata, key=integration_key)

    return adata

# Example usage:
# adata = sc.datasets.pbmc3k()  # Load an example dataset
# adata.obs['dataset'] = ['batch1'] * len(adata)  # Example integration key
# adata = integrate(adata, integration_key='dataset')
