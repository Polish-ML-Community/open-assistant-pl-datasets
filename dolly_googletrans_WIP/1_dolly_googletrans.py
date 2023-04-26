"""
# Tłumaczenie Dolly dataset z uzyciem Googletrans

Skrypt pozwala na przetłumaczenie na PL wybranej kolumny ze zbioru Dolly. 

"""

# %%time
import time
import ast
import os
import pandas as pd
import numpy as np
from pathlib import Path
from googletrans import Translator
from tqdm import tqdm


# Załaduj dolly dataset
dolly_dset = pd.read_json(
    path_or_buf='https://raw.githubusercontent.com/databrickslabs/dolly/master/data/databricks-dolly-15k.jsonl', 
    lines=True
    )

translator = Translator()

def to_pl(x):
    """Przetłumacz string na PL za pomocą googletrans
    """

    # Minimalizuje ryzyko blokady IP przez Google
    time.sleep(np.random.randint(1,3))

    if (x is None) or (x == ''):
        return False, x

    try:
        t = translator.translate(str(x).strip(), dest='pl')
        return True, t.text
    except Exception as e:
        return False, x


def translate_col(dset, col, setdim):
    """Tłumaczy daną kolumnę dzieląc ją na fragmenty i później łącząc w dataframe. 
    # UWAGA: tłumaczenie instrukcji zajęło ~6h w GoogleColab!
    :param dset: pd.DataFrame
    :param col: str
    :param setdim: int
    :return: pd.DataFrame
    """

    out_pth = f'./dolly_ggltrans/{col}'

    # jesli nie ma folderu to go stworz
    if not os.path.exists(out_pth):
        os.mkdir(out_pth)

    num_rows_per_subset = setdim
    num_subsets = dset.shape[0] // num_rows_per_subset  # Calculate the number of subsets

    i = 0

    for subset in tqdm(range(0, num_subsets)):
        sub = dset.iloc[i:i+10].copy()
        sub[f'{col}_trans'] = sub[col].apply(lambda x: to_pl(x))
        sub.to_csv(out_pth + f'/{col}_to_pl_{i}.csv', index=False)
        i += setdim
    
    return True


# Policz znaki i koszt tłumaczenia całego zbioru za pomocą Google API
rename_cols = {
    'instruction':'instruction_en', 
    'context':'context_en', 
    'response':'response_en'
    }
dolly_dset = dolly_dset.rename(columns=rename_cols)
dolly_dset = (
    dolly_dset
    .replace('',np.nan)
    .fillna('Null')
    .assign(instr_chars = dolly_dset['instruction_en'].apply(lambda x: len(x)))
    .assign(contx_chars = dolly_dset['context_en'].apply(lambda x: len(x)))
    .assign(resp_chars = dolly_dset['response_en'].apply(lambda x: len(x)))
)

number_of_chars = dolly_dset[['instr_chars','contx_chars','resp_chars']].sum().sum()
cost = (number_of_chars / 1_000_000) * 20


# %%time
## Przetłumacz #################################################################
col_to_trans = 'instruction_en'
dolly_dset = dolly_dset.sample(100) # To comment out when running on full dataset
assert translate_col(
    dolly_dset,
    col_to_trans, 
    10
    )

################################################################################

# %%time

# read all csvs
trans_pth = Path(f'./dolly_ggltrans/{col_to_trans}')
dolly_trans_df = pd.concat([pd.read_csv(x) for x in trans_pth.glob('*.csv')])

# Rozdziel bool, str w kolumnie z tłumaczeniem
trans_col = (
    dolly_trans_df
    [f'{col_to_trans}_trans']
    .apply(ast.literal_eval) # Convert string to tuple
    )
dolly_trans_df[f'{col_to_trans}_success'], dolly_trans_df[f'{col_to_trans}_pl'] = zip(*trans_col)
dolly_trans_col_df = dolly_trans_df.drop(columns=[f'{col_to_trans}_trans'])

# Wylicz success rate
success_rate = dolly_trans_col_df[f'{col_to_trans}_success'].sum() / dolly_trans_col_df.shape[0]

# Sprawdź ilość tłumaczeń
misses = dolly_trans_col_df[dolly_trans_col_df[f'{col_to_trans}_success'] == False].shape[0]
hits = dolly_trans_col_df.shape[0] - misses

# Sprawdź losowe tłumaczenia
dolly_trans_col_df[[f'{col_to_trans}', f'{col_to_trans}_pl']].sample(10).values

# Sprawdź losowe tłumaczenia
sample_trans = dolly_trans_col_df[['instruction_en', 'instruction_en_pl']].sample(10).values
print(sample_trans)

# Zapisz do pliku
dolly_trans_col_df.to_json(
    f'./dolly_ggltrans/dolly_{col_to_trans}_translated.json', 
    orient='records',
    index=False
    )
