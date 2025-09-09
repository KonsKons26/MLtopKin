import pandas as pd

import numpy as np

from scipy.stats import pearsonr, spearmanr, kendalltau

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import umap

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

import seaborn as sns

import plotly.graph_objects as go


def get_correlations(
        df: pd.DataFrame,
        target: pd.Series,
        sort_by_abs: bool = True
    ) -> pd.DataFrame:
    """Calculate the spearman rho, kendall tau, and point biserial correlation
    coefficients of the given dataframe against the given target data.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the features to calculate the correlation
        coefficients. The columns of the dataframe are the features.
    target : pd.Series
        The target data to calculate the correlation coefficients against.
    sort_by_abs : bool, default=True
        If True, return the absolute value of the correlation coefficients,
        sorted by the absolute value. Otherwise, return the unsorted absolute
        value of the correlation coefficients. Sorting is based on the sum of
        the absolute values of the correlation coefficients for each feature.

    Returns
    -------
    pd.DataFrame
        A dataframe containing the correlation coefficients of the features
        against the target data. The columns of the dataframe are the features
        and the index is the correlation coefficient method.
    """

    corr_coeffs = pd.DataFrame(
        columns=df.columns,
        index=["pearson", "spearman", "kendall"]
    )

    for col in df.columns:
        corr_coeffs.loc["pearson", col] = abs(pearsonr(
            target,
            df[col]
        )[0])
        corr_coeffs.loc["spearman", col] = abs(spearmanr(
            target,
            df[col]
        )[0])
        corr_coeffs.loc["kendall", col] = abs(kendalltau(
            target,
            df[col]
        )[0])

    if sort_by_abs:
        corr_coeffs = corr_coeffs.reindex(
            corr_coeffs.abs().sum(axis=0).sort_values(ascending=False).index,
            axis=1
        )

    return corr_coeffs


def get_correlations_pair_matrix(
        df: pd.DataFrame
    ):
    """Calculate the point pearson correlation coefficients for all pairs of
    columns in the given dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the features to calculate the correlation
        coefficients. The columns of the dataframe are the features.

    Returns
    -------
    np.ndarray
        A 2D numpy array containing the point biserial correlation coefficients
        for all pairs of columns in the dataframe. The shape of the array is
        (n_features, n_features), where n_features is the number of columns in
        the dataframe.
    """

    corr_matrix = np.zeros((df.shape[1], df.shape[1]))

    for i in range(df.shape[1]):
        for j in range(i + 1, df.shape[1]):
            corr_matrix[i, j] = abs(pearsonr(
                df.iloc[:, i],
                df.iloc[:, j]
            )[0])
            corr_matrix[j, i] = corr_matrix[i, j]
    
    return corr_matrix


def dim_red(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Reduce the dimensions of a dataset with either PCA, t_SNE, or UMAP

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to reduce.
    **kwargs : keyword arguments
        Additional arguments to pass to the dimensionality reduction method.
        - method : str
            The method to use for dimensionality reduction. Must be one of
            'pca', 'tsne', or 'umap'.
        - n_components : int, optional
            The number of components to keep. Default is 3 for t-SNE and 5
            for UMAP. For PCA, it can be specified as a percentage of the
            explained variance.
        - percent : float, optional
            The percentage of explained variance to keep. Must be used with
            PCA. Default is None.

    Returns
    -------
    pd.DataFrame
        The reduced DataFrame. The columns are named according to the method
        used for dimensionality reduction.
    """

    if "method" not in kwargs:
        raise ValueError("Method must be specified in kwargs")

    if kwargs["method"] == "pca":
        if "n_components" in kwargs:
            n_components = kwargs["n_components"]
            pca = PCA()
            reduced = pca.fit_transform(df)
            reduced = reduced[:, :n_components]

        elif "percent" in kwargs:
            pca = PCA()
            reduced = pca.fit_transform(df)
            explained_variance = pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance)
            n_components = np.argmax(
                cumulative_variance >= kwargs["percent"]
            ) + 1
            reduced = reduced[:, :n_components]

        else:
            raise ValueError("Either n_components or percent must be specified")

        return pd.DataFrame(
            reduced,
            columns=[f"PC{i}" for i in range(1, n_components + 1)]
        )

    elif kwargs["method"] == "tsne":
        if "n_components" in kwargs:
            if kwargs["n_components"] > 3:
                raise ValueError("n_components must be <= 3 for t-SNE")
            n_components = kwargs["n_components"]

        else:
            n_components = 3

        tsne = TSNE(n_components)
        reduced = tsne.fit_transform(df)
    
        return pd.DataFrame(
            reduced,
            columns=[f"t-SNE{i}" for i in range(1, n_components + 1)]
        )

    elif kwargs["method"] == "umap":
        if "n_components" in kwargs:
            n_components = kwargs["n_components"]

        else:
            n_components = 5

        umap_model = umap.UMAP(n_components=n_components)
        reduced = umap_model.fit_transform(df)
    
        return pd.DataFrame(
            reduced,
            columns=[f"UMAP{i}" for i in range(1, n_components + 1)]
        )

    else:
        raise ValueError("Method must be one of 'pca', 'tsne', or 'umap'")


def plot_correlation_coefficients(
        corr_df: pd.DataFrame,
        title: str = "Correlation Coefficients",
    ) -> None:
    """
    Plot the correlation coefficients of the features against the target.

    Parameters
    ----------
    corr_df : pandas.DataFrame
        The DataFrame containing the correlation coefficients. The columns of
        the DataFrame are the features and the index is the correlation
        coefficient method.
    title : str, default "Correlation Coefficients"
        The title of the plot.

    Returns
    -------
    None
        Displays the plot of the correlation coefficients.
    """

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=corr_df.columns,
        y=corr_df.loc["pearson"],
        mode="markers",
        marker=dict(
            size=12,
            opacity=0.75
        ),
        name="Pearson"
    ))

    fig.add_trace(go.Scatter(
        x=corr_df.columns,
        y=corr_df.loc["spearman"],
        mode="markers",
        marker=dict(
            size=12,
            opacity=0.75
        ),
        name="Spearman"
    ))

    fig.add_trace(go.Scatter(
        x=corr_df.columns,
        y=corr_df.loc["kendall"],
        mode="markers",
        marker=dict(
            size=12,
            opacity=0.75
        ),
        name="Kendall"
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Features",
        yaxis_title="Correlation Coefficient",
        template="plotly_white",
        height=700
    )

    fig.show()


def heatmap_correlations(
        matrix: pd.DataFrame,
        labels: list[str],
        title: str = "Pearson correlation Heatmap",
        figsize: tuple[int, int] = (16, 16),
        cmap: str = "crest"
    ) -> None:
    """
    Plot a heatmap of the correlations between the features in the given
    dataframe.

    Parameters
    ----------
    matrix : pd.DataFrame
        The dataframe containing the features to plot the correlations for. The
        columns of the dataframe are the features.
    labels : list of str
        The labels for the features to plot the correlations for.
    figsize : tuple of int, default (20, 20)
        The size of the figure to create for the heatmap.
    title : str, default "Correlation Heatmap"
        The title of the heatmap.
    cmap : str, default "crest"
        The colormap to use for the heatmap.

    Returns
    -------
    None
        Displays the heatmap of the correlations between the features.
    """

    plt.figure(figsize=figsize)

    sns.heatmap(
        matrix,
        annot=False,
        fmt=".2f",
        cmap=cmap,
        cbar_kws={"shrink": .8},
        linewidths=0.5,
        linecolor="black",
        square=True
    )
    plt.xticks(
        ticks=np.arange(len(labels)) + 0.5,
        labels=labels,
        rotation=45,
        ha="right"
    )
    plt.yticks(
        ticks=np.arange(len(labels)) + 0.5,
        labels=labels,
        rotation=0,
        va="center"
    )
    plt.title(title, fontsize=20)
    plt.show()


def pairplot(
        data: pd.DataFrame,
        title: str,
        kde_color: str = "#421f6e",
        scatter_color: str = "#7a4db0",
        hue: pd.Series = None,
        cmap: str = "cool"
    ):
    """
    Plot a pairplot of the given data.

    Parameters
    ----------
    data : pd.DataFrame
        The data to plot the pairplot for. The columns of the dataframe are the
        features.
    title : str
        The title of the pairplot.
    kde_color : str
        The color to use for the KDE plots.
    scatter_color : str
        The color to use for the scatter plots.
    hue : pd.Series
        The hue variable to use for the pairplot. If None, the pairplot will
        not be colored by hue.
    cmap : str
        The colormap to use for the pairplot. If hue is None, this will be
        ignored.

    Returns
    -------
    None
        Displays the pairplot of the given data.
    """

    def annotate_correlations(x, y, **kwargs):
        """Annotate the correlation coefficients on the pairplot."""

        pearson_coef, _ = pearsonr(x, y)
        spearman_coef, _ = spearmanr(x, y)
        kendall_coef, _ = kendalltau(x, y)

        text = "".join([
            f"Pearson: {pearson_coef:.2f}\n",
            f"Spearman: {spearman_coef:.2f}\n",
            f"Kendall: {kendall_coef:.2f}"
        ])
        
        plt.annotate(
            text,
            xy=(0.95, 0.95),
            xycoords="axes fraction",
            ha="right",
            va="top",
            bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.75)
        )

    def plot_mean_median(x, **kwargs):
        """Plot the mean and median of the data on the diagonal of the
        pairplot."""

        mean_val = np.mean(x)
        plt.axvline(
            mean_val,
            color="blue",
            linestyle="-",
            label=f"Mean: {mean_val:.2f}"
        )

        median_val = np.median(x)
        plt.axvline(
            median_val,
            color="red",
            linestyle="--",
            label=f"Median {median_val:.2f}"
        )
        plt.legend()


    if hue is not None:
        plot_kws = {"hue": hue, "palette": cmap}
    else:
        plot_kws = {"color": scatter_color}

    g = sns.pairplot(
        data,
        diag_kind="kde",
        plot_kws=plot_kws,
        diag_kws={"color": kde_color}
    )

    g.map_upper(annotate_correlations)
    g.map_lower(sns.kdeplot, levels=4, color="black")
    g.map_diag(plot_mean_median)

    if hue is not None:
        # Create a normalized scalar mappable for the color bar
        norm = Normalize(vmin=hue.min(), vmax=hue.max())
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])

        g.figure.subplots_adjust(right=0.85)
        cbar_ax = g.figure.add_axes([0.88, 0.15, 0.02, 0.7])
        g.figure.colorbar(sm, cax=cbar_ax, label=hue.name)

    plt.suptitle(title, y=1.02)
    plt.show()
