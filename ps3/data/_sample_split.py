import hashlib
import numpy as np
import pandas as pd

# TODO: Write a function which creates a sample split based in some id_column and training_frac.
# Optional: If the dtype of id_column is a string, we can use hashlib to get an integer representation.
def create_sample_split(df: pd.DataFrame, id_column: str, training_frac: float = 0.8) -> pd.DataFrame:
    """Create sample split based on ID column.

    Parameters
    ----------
    df : pd.DataFrame
        Training data
    id_column : str
        Name of ID column
    training_frac : float, optional
        Fraction to use for training, by default 0.8

    Returns
    -------
    pd.DataFrame
        Training data with sample column containing train/test split based on IDs.
    """
    # Crate safe copy of dataframe so original is untouched
    df = df.copy()

    # Check training fraction input is correct
    # Convert fraction to number between 0 and 100
    if training_frac != 0.8 and not (0 <= training_frac <= 1):
        print('Please input a training fraction between 0 and 1')
        return
    else:
        training_frac = training_frac
    # Define cut of dataframe based on its length and the training_frac
    cut = int(len(df)*training_frac)
    
    # Function to hash a value and convert to an integer
    def hash_to_int(value):
        hash_obj = hashlib.sha256(str(value).encode('utf-8')).hexdigest()
        h_int = int(hash_obj, 16)
        # ensure the hash follows a uniform distribution [0,1]
        return h_int/(2**256)

    # Create hash column in dataframe
    df['hash'] = df[id_column].apply(hash_to_int)
    df = df.sort_values('hash')
    
    # Create sample column and apply train/test split
    df['sample'] = 'test'
    df.iloc[:cut, df.columns.get_loc('sample')] = 'train'

    # Resort dataframe back to original
    df = df.sort_index()

    return df
