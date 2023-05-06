import pandas as pd


def read_csv(filepath, sep='\t', header='infer', compression='gzip', encoding='utf-8', col_types=None):
    assert(isinstance(filepath, str))
    return pd.read_csv(filepath,
                       sep=sep,
                       encoding=encoding,
                       compression=compression,
                       dtype=col_types,
                       header=header)
