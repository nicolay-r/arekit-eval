import pandas as pd

from arekit.common.labels.base import NoLabel
from arekit.contrib.utils.data.readers.csv_pd import PandasCsvReader
from arekit.contrib.utils.data.storages.pandas_based import PandasBasedRowsStorage

from arekit_eval.core.evaluators.cmp_table import DocumentCompareTable
from arekit_eval.core.evaluators.utils import label_to_str
from arekit_eval.core.result import BaseEvalResult
from arekit_eval.core.utils import progress_bar_defined


# Corresponds to fields with attitude ends. (indices, INT)
S_IND = 's_ind'
T_IND = 't_ind'
# Provide sentence index.
SENT_IND = 'sent_ind'
# Row names by default.
DOC_ID = 'doc_id'
TEXT_A = 'text_a'
ENTITIES = 'entities'
ENTITY_VALUES = 'entity_values'
ENTITY_TYPES = 'entity_types'


def __get_entity_inds(sample_row):
    return [int(v) for v in sample_row[ENTITIES].split(',')]


def __format_entity(text_terms, index, entity_types, entity_inds):
    assert(isinstance(text_terms, list))
    return text_terms[index].upper() + "-[{}]".format(
        entity_types[entity_inds.index(index)].replace(' ', '-'))


def __get_text_terms(sample_row, do_lowercase=False):
    text = sample_row[TEXT_A]
    if do_lowercase:
        text = text.lower()
    return text.split(' ')


def __post_text_processing(sample_row, source_ind, target_ind):
    """ Document post-processing for better understanding of potential
        problems may occurred in texts.
    """
    assert(TEXT_A in sample_row)
    assert(ENTITIES in sample_row)
    assert(ENTITY_VALUES in sample_row)
    assert(ENTITY_TYPES in sample_row)

    text_terms = __get_text_terms(sample_row, do_lowercase=True)
    entity_inds = __get_entity_inds(sample_row)
    entity_values = sample_row[ENTITY_VALUES].split(',')
    entity_types = sample_row[ENTITY_TYPES].split(',')

    # заменить на значения сущностей.
    for i, e_ind in enumerate(entity_inds):
        text_terms[e_ind] = entity_values[i].replace(' ', '-')

    text_terms[source_ind] = __format_entity(text_terms,
                                             index=source_ind,
                                             entity_types=entity_types,
                                             entity_inds=entity_inds)
    text_terms[target_ind] = __format_entity(text_terms,
                                             index=target_ind,
                                             entity_types=entity_types,
                                             entity_inds=entity_inds)

    return text_terms


def __crop_text_terms(source_ind, target_ind, text_terms, window_crop=10):
    """ Boundaries cropping for better visualization.
    """
    left_participant = min(source_ind, target_ind)
    right_participant = max(source_ind, target_ind)
    crop_left = max(0, left_participant - window_crop)
    crop_right = min(right_participant + window_crop, len(text_terms))

    return " ".join(text_terms[crop_left:crop_right])


def __extract_df(storage, doc_id, ctx_id, source_ind, target_ind):
    assert(isinstance(storage, PandasBasedRowsStorage))
    df = storage.DataFrame
    return df[(df[DOC_ID] == doc_id) &
              (df[SENT_IND] == ctx_id) &
              (df[S_IND] == source_ind) &
              (df[T_IND] == target_ind)]


def extract_errors(eval_result, test_samples_filepath, etalon_samples_filepath, no_label=NoLabel()):
    """ Display the difference between automatically annotated (test) and ground truth (etalon) samples data.

        eval_result: BaseEvalResult
        test_samples_filepath: str
            filepath to annotation performed automatically
        etalon_samples_filepath: str
            filepath to expected annotation
    """
    assert(isinstance(eval_result, BaseEvalResult))
    assert(isinstance(test_samples_filepath, str))
    assert(isinstance(etalon_samples_filepath, str))

    dataframes = []
    for doc_id, doc_cmp_table in eval_result.iter_dataframe_cmp_tables():
        assert(isinstance(doc_cmp_table, DocumentCompareTable))
        df = doc_cmp_table.DataframeTable
        df.insert(2, DOC_ID, [doc_id] * len(df), allow_duplicates=True)
        dataframes.append(df[(df[DocumentCompareTable.C_CMP] == False) &
                             ~((df[DocumentCompareTable.C_ID_ORIG].isnull()) &
                               (df[DocumentCompareTable.C_RES] == label_to_str(no_label)))])

    eval_errors_df = pd.concat(dataframes, axis=0)
    eval_errors_df.reset_index(inplace=True)

    columns_to_copy = [ENTITY_TYPES, TEXT_A, T_IND, S_IND, SENT_IND]

    last_column_index = len(eval_errors_df.columns)
    for sample_col in columns_to_copy:
        eval_errors_df.insert(last_column_index, sample_col, [""] * len(eval_errors_df), allow_duplicates=True)

    # Дополняем содержимым из samples строки с неверно размеченными оценками.
    reader = PandasCsvReader()
    etalon_samples = reader.read(target=etalon_samples_filepath)
    test_samples = reader.read(target=test_samples_filepath)

    eval_rows_it = progress_bar_defined(iterable=eval_errors_df.iterrows(),
                                        total=len(eval_errors_df),
                                        desc="Analyze",
                                        unit="row")

    for row_id, eval_row in eval_rows_it:

        doc_id, ctx_id, source_ind, target_ind = [
            int(v) for v in eval_row[DocumentCompareTable.C_ID_ORIG].split("_")]

        sample_rows = __extract_df(storage=etalon_samples, doc_id=doc_id, ctx_id=ctx_id,
                                   source_ind=source_ind, target_ind=target_ind)

        if len(sample_rows) == 0:
            sample_rows = __extract_df(storage=test_samples, doc_id=doc_id, ctx_id=ctx_id,
                                       source_ind=source_ind, target_ind=target_ind)

        # Извлекаем первый ряд.
        sample_row = sample_rows.iloc[0]

        for sample_col in columns_to_copy:
            eval_errors_df.at[row_id, sample_col] = sample_row[sample_col]

        text_terms = __post_text_processing(sample_row=sample_row, source_ind=source_ind, target_ind=target_ind)
        cropped_text = __crop_text_terms(source_ind=source_ind, target_ind=target_ind, text_terms=text_terms)

        eval_errors_df.at[row_id, TEXT_A] = cropped_text

        # Replace source and target the values instead of indices.
        eval_errors_df.at[row_id, S_IND] = text_terms[source_ind]
        eval_errors_df.at[row_id, T_IND] = text_terms[target_ind]

    return eval_errors_df
