import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE

np.random.seed(1000)
pd.options.display.float_format = '{:,.3f}'.format

if __name__ == '__main__':
    sales_filepath = "datasets\\Sales_Transactions_Dataset_Weekly.csv"

    print("\nReading data from file paths ...")
    sales_data = pd.read_table(
            sales_filepath,
            encoding='utf-8',
            sep=',',  # comma separated values
            skipinitialspace=True,
            index_col=0,
            header=0
        )

    print("\nPre-processing Sales Transactions dataset ...")
    # Drop MIN, MAX and Normalized columns
    normalised_cols = ['MIN', 'MAX']
    for col in sales_data.columns:
        if str(col).startswith('Normalize'):
            normalised_cols.append(str(col))
    sales_data = sales_data.drop(normalised_cols, axis=1)

    # Perform the TSNE non-linear dimensionality reduction
    tsne = TSNE(n_components=2, perplexity=20, random_state=1000)
    data_tsne = tsne.fit_transform(sales_data)

    df_tsne = pd.DataFrame(data_tsne, columns=['x', 'y'], index=sales_data.index)
    dff = pd.concat([sales_data, df_tsne], axis=1)

    # Show the dataset
    sns.set()

    fig, ax = plt.subplots(figsize=(18, 11))

    with sns.plotting_context("notebook", font_scale=1.5):
        sns.scatterplot(x='x',
                        y='y',
                        size=0,
                        sizes=(120, 120),
                        data=dff,
                        legend=False,
                        ax=ax)

    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')

    plt.show()

