from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering, HDBSCAN
from sklearn.mixture import GaussianMixture,BayesianGaussianMixture
from sklearn.metrics import silhouette_score, pairwise_distances
import umap
import torch
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import matplotlib._color_data as mcd
import pdb
# import pycuda.gpuarray as gpuarray
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from scipy.spatial.distance import cdist
import matplotlib.colors as mcolors
import math


colors = [name for name in mcd.XKCD_COLORS]
random_state=42

def reduce_dimension(model, data_loader,device,pca=None):
    """
    Use the trained model to obtain low
    dimensional data for clustering
    """
    ### Obtain the latent embeddings for validation set
    embeddings = []
    labels = []
    xData = []
    emb_z = []
    eventId = []
    model.to(device)
    for x,xIp, eIdx in data_loader: #### Fetch validation samples

        xData.append(xIp)
        eventId.append(eIdx)
        if pca is not None:
            embeddings.append(torch.Tensor(pca.fit_transform(x.view(x.shape[0],x.shape[-1]))))
        else:

            x, xIp = x.to(device), xIp.to(device)
            xHat,z = model(x) ##########
            embeddings.append(z.detach().cpu())

    xData = torch.cat(xData,dim=0).squeeze().numpy()
    eventId = torch.cat(eventId,dim=0).squeeze().numpy()
    embeddings = torch.cat(embeddings,dim=0)
    embeddings = embeddings.view(-1,embeddings.shape[-1])
    # print(embeddings.shape)
    # embeddings = MinMaxScaler().fit_transform(embeddings)
    # embeddings = torch.Tensor(embeddings)

    return embeddings, xData, eventId

def reduce_dimension_VAE(model, data_loader,device,pca=None):
    """
    Use the trained model to obtain low
    dimensional data for clustering
    """
    ### Obtain the latent embeddings for validation set
    embeddings = []
    labels = []
    xData = []
    emb_z = []
    eventId = []
    model.to(device)
    for x,xIp, eIdx in data_loader: #### Fetch validation samples

        xData.append(xIp)
        eventId.append(eIdx)
        if pca is not None:
            embeddings.append(torch.Tensor(pca.fit_transform(x.view(x.shape[0],x.shape[-1]))))
        else:

            x, xIp = x.to(device), xIp.to(device)
            xHat,z,_ = model(x) ##########
            embeddings.append(z.detach().cpu())

    xData = torch.cat(xData,dim=0).squeeze().numpy()
    eventId = torch.cat(eventId,dim=0).squeeze().numpy()
    embeddings = torch.cat(embeddings,dim=0)
    embeddings = embeddings.view(-1,embeddings.shape[-1])
    # embeddings = MinMaxScaler().fit_transform(embeddings)
    # embeddings = torch.Tensor(embeddings)
    # print(embeddings.shape)
    return embeddings, xData, eventId

def draw_clustering_multi(xData, embeddings, cls_labels_list, cluster_name=['GMM', 'DPGMM', 'HDBSCAN'], method='pca', calc_silhouette=False, draw=False, filename=None):
    """
    Visualize clustering results using multiple methods such as PCA, t-SNE, or UMAP.

    Parameters:
        xData (numpy.ndarray): The original data.
        embeddings (numpy.ndarray): The latent space embeddings of the data.
        cls_labels_list (list of numpy.ndarray): List of cluster labels for each clustering method.
        cluster_name (list): Names of clustering methods used, e.g., ['GMM', 'DPGMM', 'HDBSCAN'].
        method (str): Dimensionality reduction method ('pca', 'tsne', 'umap').
        calc_silhouette (bool): Whether to calculate silhouette scores. Defaults to False.
        draw (bool): Whether to draw the plot or not. Defaults to False.
        filename (str): If provided, the figure will be saved to this file.

    Returns:
        silhouette_scores (list): List of silhouette scores for each clustering method (if calc_silhouette is True).
    """
    color_map = list(mcolors.TABLEAU_COLORS.values())
    silhouette_scores = []

    if draw:
        nSamp = 1000  # Number of samples for visualization
        
        # Perform dimensionality reduction on the embeddings
        if embeddings.shape[-1] > 2:
            if method == 'pca':
                print('Using PCA for final visualization')
                reducer = PCA(n_components=2)
                reduced_embeddings = reducer.fit_transform(embeddings)
            elif method == 'tsne':
                print('Using t-SNE for final visualization')
                reducer = TSNE(n_components=2, perplexity=30, n_iter=1000)
                reduced_embeddings = reducer.fit_transform(embeddings)
            elif method == 'umap':
                print('Using UMAP for final visualization')
                reducer = umap.UMAP(n_components=2)
                reduced_embeddings = reducer.fit_transform(embeddings)
            else:
                raise ValueError(f"Unknown method: {method}. Choose 'pca', 'tsne', or 'umap'.")
        else:
            reduced_embeddings = embeddings  # If already 2D, no need for reduction

        # Plot layout: one plot for event means, followed by clustering result visualizations
        num_clustering_methods = len(cls_labels_list)
        fig = plt.figure(figsize=(10, 8))  # Adjust the figure size dynamically
        gs = gridspec.GridSpec(num_clustering_methods, num_clustering_methods + 1, width_ratios=[1.2] + [1] * num_clustering_methods)

        # Calculate silhouette scores if required
        if calc_silhouette:
            for i, cls_labels in enumerate(cls_labels_list):
                score = silhouette_score(reduced_embeddings, cls_labels)
                silhouette_scores.append(score)
                print(f'Silhouette Score for Clustering {cluster_name[i]}: {score:.4f}')

        # Plot event means (separate rows for each clustering result)
        for row, cls_labels in enumerate(cls_labels_list):
            ax_event_means = plt.subplot(gs[row, 0])
            unique_clusters = np.unique(cls_labels)
            for k, cluster in enumerate(unique_clusters):
                xClus = xData[cls_labels == cluster]
                eMean = xClus.mean(axis=0)
                xRange = np.linspace(-8, 12, xClus.shape[-1])
                ax_event_means.plot(xRange, eMean.T, color=color_map[k % len(color_map)], label=f'Cluster {k + 1}')
            ax_event_means.set_title(f'Average Signal - {cluster_name[row]}')
            ax_event_means.set_xlabel('Time (ms)')

        # Plot reduced embeddings for each clustering result
        for i, cls_labels in enumerate(cls_labels_list):
            ax = plt.subplot(gs[:, i + 1])
            unique_clusters = np.unique(cls_labels)
            for k, cluster in enumerate(unique_clusters):
                emb_cluster = reduced_embeddings[cls_labels == cluster]
                ax.scatter(emb_cluster[:, 0], emb_cluster[:, 1], color=color_map[k % len(color_map)], label=f'Cluster {k + 1}', alpha=0.7, s=10)
            ax.set_title(f'{cluster_name[i]}')
            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout()

        # Save the figure if a filename is provided
        if filename is not None:
            plt.savefig(f'{filename}.pdf', dpi=150)
        else:
            plt.show()

    return silhouette_scores if calc_silhouette else None
    
    
def draw_clustering_final(xData, embeddings, cls_labels_list, cluster_name=['GMM', 'DPGMM', 'HDBSCAN'], calc_silhouette=False, draw=False, filename=None):
    """
    Visualize clustering results using multiple methods such as PCA, t-SNE, and UMAP.

    Parameters:
        xData (numpy.ndarray): The original data.
        embeddings (numpy.ndarray): The latent space embeddings of the data.
        cls_labels_list (list of numpy.ndarray): List of cluster labels for each clustering method.
        cluster_name (list): Names of clustering methods used, e.g., ['GMM', 'DPGMM', 'HDBSCAN'].
        calc_silhouette (bool): Whether to calculate silhouette scores. Defaults to False.
        draw (bool): Whether to draw the plot or not. Defaults to False.
        filename (str): If provided, the figure will be saved to this file.

    Returns:
        silhouette_scores (list): List of silhouette scores for each clustering method (if calc_silhouette is True).
    """
    color_map = list(mcolors.TABLEAU_COLORS.values())
    silhouette_scores = []

    if draw:
        nSamp = 1000  # Number of samples for visualization
        
        # Perform dimensionality reduction on the embeddings using all three methods
        print('Using PCA, t-SNE, and UMAP for final visualization')
        pca_reducer = PCA(n_components=2)
        pca_embeddings = pca_reducer.fit_transform(embeddings)

        tsne_reducer = TSNE(n_components=2, perplexity=30, n_iter=1000)
        tsne_embeddings = tsne_reducer.fit_transform(embeddings)

        umap_reducer = umap.UMAP(n_components=2)
        umap_embeddings = umap_reducer.fit_transform(embeddings)

        reduced_embeddings_list = [pca_embeddings, tsne_embeddings, umap_embeddings]
        method_names = ['PCA', 't-SNE', 'UMAP']

        # Plot layout: event means on the first row, followed by clustering result visualizations
        num_clustering_methods = len(cls_labels_list)
        fig = plt.figure(figsize=(18, 24))  # Adjust the figure size dynamically
        gs = gridspec.GridSpec(4, 3, height_ratios=[1, 1, 1, 1])

        # Calculate silhouette scores if required
        if calc_silhouette:
            for i, cls_labels in enumerate(cls_labels_list):
                score = silhouette_score(pca_embeddings, cls_labels)
                silhouette_scores.append(score)
                print(f'Silhouette Score for Clustering {cluster_name[i]}: {score:.4f}')

        # Plot event means (first row)
        for row, cls_labels in enumerate(cls_labels_list):
            ax_event_means = plt.subplot(gs[0, row])
            unique_clusters = np.unique(cls_labels)
            for k, cluster in enumerate(unique_clusters):
                xClus = xData[cls_labels == cluster]
                eMean = xClus.mean(axis=0)
                xRange = np.linspace(-8, 12, xClus.shape[-1])
                ax_event_means.plot(xRange, eMean.T, color=color_map[k % len(color_map)], label=f'Cluster {k + 1}')
            ax_event_means.set_title(f'Average Signal - {cluster_name[row]}')
            ax_event_means.set_xlabel('Time (ms)')
            if row == 0:
                ax_event_means.legend(loc='upper right')

        # Plot reduced embeddings for each clustering result with each method
        for method_idx, (reduced_embeddings, method_name) in enumerate(zip(reduced_embeddings_list, method_names)):
            for i, cls_labels in enumerate(cls_labels_list):
                ax = plt.subplot(gs[method_idx + 1, i])
                unique_clusters = np.unique(cls_labels)
                for k, cluster in enumerate(unique_clusters):
                    emb_cluster = reduced_embeddings[cls_labels == cluster]
                    ax.scatter(emb_cluster[:, 0], emb_cluster[:, 1], color=color_map[k % len(color_map)], label=f'Cluster {k + 1}', alpha=0.7, s=10)
                ax.set_title(f'{method_name} - Clustering {cluster_name[i]}')
                ax.set_xticks([])
                ax.set_yticks([])

        plt.tight_layout()

        # Save the figure if a filename is provided
        if filename is not None:
            plt.savefig(f'{filename}.pdf', dpi=150)
        else:
            plt.show()

    return silhouette_scores if calc_silhouette else None


def draw_clustering(xData, embeddings, cls_labels, method='pca', draw=False, filename=None):
    """
    Visualize clustering results using PCA, t-SNE, or UMAP.
    
    Parameters:
        xData (numpy.ndarray): The original data.
        embeddings (numpy.ndarray): The latent space embeddings of the data.
        cls_labels (numpy.ndarray): Cluster labels for the data.
        method (str): The dimensionality reduction method ('pca', 'tsne', 'umap'). Defaults to 'pca'.
        draw (bool): Whether to draw the plot or not. Defaults to False.
        filename (str): If provided, the figure will be saved to this file.
        
    Returns:
        cls_labels (numpy.ndarray): Cluster labels.
        clusters (numpy.ndarray): Unique cluster IDs.
        counts (numpy.ndarray): Number of samples per cluster.
        ordClsLabels (numpy.ndarray): Ordered cluster labels.
        embeddings (numpy.ndarray): Reduced embeddings (2D).
    """
    
    nClus = len(np.unique(cls_labels))
    clusters = np.unique(cls_labels)

    # Get total count per cluster
    counts = [(cls_labels == i).sum() for i in clusters]
    counts = np.array(counts)
    clusters = clusters[np.argsort(counts)[::-1]]
    counts = np.sort(counts)[::-1]

    ordClsLabels = np.zeros(len(cls_labels))

    if draw:
        idx = 0
        # N = len(embeddings)
        nSamp = 1000
        # Select dimensionality reduction method based on the provided argument
        if embeddings.shape[-1] > 2:
            if method == 'pca':
                print('Using PCA for final visualization')
                reducer = PCA(n_components=2)
                reducer.fit(embeddings[np.random.permutation(embeddings.shape[0])[:round(0.1*len(embeddings))]])
                embeddings = reducer.transform(embeddings)
            elif method == 'tsne':
                print('Using t-SNE for final visualization')
                reducer = TSNE(n_components=2, perplexity=30, n_iter=1000)
                embeddings = reducer.fit_transform(embeddings)
            elif method == 'umap':
                print('Using UMAP for final visualization')
                reducer = umap.UMAP(n_components=2)
                reducer.fit(embeddings[np.random.permutation(embeddings.shape[0])[:round(0.1*len(embeddings))]])
                embeddings = reducer.transform(embeddings)  
            
            else:
                raise ValueError(f"Unknown method: {method}. Choose 'pca', 'tsne', or 'umap'.")

        plt.clf()
        fig = plt.figure(figsize=(15, 6))  # Adjusted figure size to make the main plot wider
        
        # Define grid layout
        col_main = 2  # Increased to two columns for the main latent space plot to make it wider
        # Determine optimal layout for the cluster plots to leave the least blank space
        col_clusters = math.ceil(math.sqrt(nClus))
        row_clusters = math.ceil(nClus / col_clusters)
        total_cols = col_main + col_clusters  # Total columns including the main plot
        gs = gridspec.GridSpec(row_clusters, total_cols, wspace=0.2, hspace=0.2)  # Adjusted spacing between subplots
        
        # Main plot for latent space on the far left, fixed size
        ax0 = plt.subplot(gs[:, :col_main])
        for k in range(nClus):
            ordClsLabels[cls_labels == clusters[k]] = k + 1
            emb = embeddings[cls_labels == clusters[k]]
            if emb.shape[0] > nSamp:
                emb = emb[np.random.permutation(emb.shape[0])[:nSamp]]
            ax0.scatter(emb[:, 0], emb[:, 1], c='None', s=10, edgecolors=f'C{k}', alpha=0.75)
            eMean = embeddings[cls_labels == clusters[k]].mean(0)
            ax0.plot(eMean[0], eMean[1], '+', color=f'C{k}')
            
        ax0.set_title(f'Latent Space, N={len(embeddings)}, K={nClus}')
        ax0.grid(True)
        
        # Plot individual cluster data in a matrix layout on the right
        for k in range(nClus):
            row_idx = (k // col_clusters)
            col_idx = (k % col_clusters) + col_main  # Shift by col_main to account for the main plot
            ax1 = plt.subplot(gs[row_idx, col_idx])
            
            xClus = xData[cls_labels == clusters[k]]
            eMean = embeddings[cls_labels == clusters[k]].mean(0)
            eIdx = np.argsort(np.linalg.norm((embeddings[cls_labels == clusters[k]] - eMean), axis=1))
            xMean = xClus[eIdx[:5]].mean(0)
            
            xRange = np.linspace(0, 2.25, xClus.shape[-1])
            if xClus.shape[0] > nSamp:
                xClus = xClus[np.random.permutation(xClus.shape[0])[:nSamp]]
            ax1.plot(xRange, xClus.T, color=f'C{k}', alpha=0.05, linewidth=0.1)
            ax1.plot(xRange, xMean.T, color=f'C{k}', linewidth=1.5)
            
            string = f'K={k + 1}, N={counts[k]}'
            print(string)
            ax1.set_title(string)
            ax1.grid(True)
            ax1.set_ylim([-0.1, 1.1])
            # ax1.set_xlabel('t(ms)')

        plt.tight_layout()


    return cls_labels, clusters, counts, ordClsLabels, embeddings


    
def cluster_with_gmm(embeddings, max_clusters=7,method = 'bic'):
    lowest_ic = np.infty
    best_gmm = None
    ics = []
    n_components_range = range(1, max_clusters + 1)

    for n_components in n_components_range:
        # Fit a Gaussian mixture with EM
        gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=random_state)
        gmm.fit(embeddings)
        if method == 'bic':
            ic = gmm.bic(embeddings)
        elif method == 'aic':
            ic = gmm.aic(embeddings)
        ics.append(ic)

        if ics[-1] < lowest_ic:
            lowest_ic = ics[-1]
            best_gmm = gmm

    return best_gmm

def calculate_aic_bic_kmeans(kmeans, X):
    """Calculate AIC and BIC for the given KMeans clustering."""
    m = kmeans.n_clusters
    n = len(X)
    d = X.shape[1]
    p = m * (d + 1)
    log_likelihood = np.sum(np.log(np.min(cdist(X, kmeans.cluster_centers_, 'euclidean'), axis=1)))
    aic = 2 * p - 2 * log_likelihood
    bic = np.log(n) * p - 2 * log_likelihood
    return aic, bic

def cluster_with_kmeans(embeddings, max_clusters=7,method = 'bic'):
    lowest_ic = np.infty
    best = None
    ics = []
    n_components_range = range(1, max_clusters + 1)

    for n_components in n_components_range:
        kmeans = KMeans(n_clusters=n_components, random_state=random_state)
        kmeans.fit(embeddings)
        aic, bic = calculate_aic_bic_kmeans(kmeans, embeddings)
        if method == 'bic':
            ics.append(bic)
        elif method == 'aic':
            ics.append(aic)

        if ics[-1] < lowest_ic:
            lowest_ic = ics[-1]
            best =  kmeans

    return best

def compute_dispersion(X, labels):
    return np.sum([pairwise_distances(X[labels == k]).sum() for k in np.unique(labels)])

def gap_statistic(X, n_clusters, B=10):
    gaps = np.zeros(len(n_clusters))
    dispersions = np.zeros(len(n_clusters))
    reference_dispersions = np.zeros((len(n_clusters), B))
    
    for i, k in enumerate(n_clusters):
        # Fit GMM and compute dispersion for actual data
        gmm = GaussianMixture(n_components=k, random_state=random_state)
        gmm.fit(X)
        labels = gmm.predict(X)
        dispersions[i] = compute_dispersion(X, labels)
        
        # Generate reference datasets and compute dispersions
        for b in range(B):
            reference = np.random.uniform(0, 1, X.shape)
            gmm.fit(reference)
            ref_labels = gmm.predict(reference)
            reference_dispersions[i, b] = compute_dispersion(reference, ref_labels)
        
        # Compute GAP statistic
        gaps[i] = np.mean(np.log(reference_dispersions[i])) - np.log(dispersions[i])
    
    # Determine the optimal number of clusters
    optimal_k = n_clusters[np.argmax(gaps)]
    
    return optimal_k, gaps


def make_clusters(embeddings,nClus=0,method=None,automated = None):
    embeddings = StandardScaler().fit_transform(embeddings)
    embeddings = torch.tensor(embeddings)
    N = len(embeddings)
    idx = np.random.permutation(N)[:int(round(0.1*N))]
    #[:int(round(0.1*N))]
    if nClus > 0:
        if automated is None:
            if method == 'kmeans': 
                print("Using kmeans clustering...")
                alg = KMeans(n_clusters=nClus,
                                random_state=random_state)
            elif method == 'gmm':
        #    else nClus >0:
                print('Using GMM!')
                alg = GaussianMixture(n_components=nClus,
                                    random_state=random_state)
            elif method == 'hdbscan':
                print("Using HDBSCAN clustering...")
                alg = HDBSCAN(min_cluster_size=300)
            elif method == 'dpgmm':
                print('Using DPGMM!')
                alg = BayesianGaussianMixture(n_components=nClus, covariance_type='full',n_init = 10,weight_concentration_prior=0.001, random_state=random_state)
        elif automated == 'silhouette':
            print('Using silhouette score to determine number of clusters')
            best_n_clusters = None
            best_silhouette = -1
            # Convert to NumPy array if not already
            for n_clusters in range(2, nClus):  # Example: try from 2 to 9 clusters
                if method == 'kmeans':
                    alg = KMeans(n_clusters=n_clusters, random_state=random_state)
                else:
                    alg = GaussianMixture(n_components=n_clusters, random_state=random_state)
                
                cluster_labels = alg.fit_predict(embeddings)+1
                silhouette_avg = silhouette_score(embeddings, cluster_labels)
                
                if silhouette_avg > best_silhouette:
                    best_silhouette = silhouette_avg
                    best_n_clusters = n_clusters
            
            # Run the clustering with the best number of clusters
            if method == 'kmeans':
                alg = KMeans(n_clusters=best_n_clusters, random_state=random_state)
            else:
                print('Using GMM! Silhouette score: ', best_silhouette)
                alg = GaussianMixture(n_components=best_n_clusters, random_state=random_state)
        elif automated == 'bic' or automated == 'aic':
            if method == 'kmeans':
                print('Using KMeans with BIC/AIC!')
                alg = cluster_with_kmeans(embeddings[idx], max_clusters=nClus,method = automated)
            elif method == 'gmm':
                print('Using GMM with BIC/AIC!')
                alg = cluster_with_gmm(embeddings[idx], max_clusters=nClus,method = automated)
                
        elif automated == 'gap':
            if method == 'gmm':
                print('Using GAP statistic to determine number of clusters')
                n_clusters = range(2, nClus + 1)
                optimal_k, gaps = gap_statistic(embeddings, n_clusters)
                print('Optimal number of clusters by GAP statistic:', optimal_k)
                alg = GaussianMixture(n_components=optimal_k, random_state=random_state)

    alg.fit(embeddings[idx])
    cls_labels = alg.fit_predict(embeddings)+1
    nClus = len(np.unique(cls_labels))
    print('Found %d clusters'%nClus)
    clusters = np.unique(cls_labels)
    counts = [(cls_labels == i).sum() for i in clusters]
    counts = np.array(counts)
    clusters = clusters[np.argsort(counts)[::-1]]
    # cls_labels, clusters, counts, ordClsLabels, _ = draw_clustering(xData,embeddings,cls_labels,draw=False)

    return cls_labels, clusters, counts
