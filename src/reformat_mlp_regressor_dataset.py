"""
    Reformat heart disease csv
"""


def reformat_mlp_regressor_dataset(dataset):
    """
        Dataset reformat function
    """
    reformatted_data = dataset.dropna()
    return reformatted_data
