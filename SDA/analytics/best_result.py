import pandas

def best_result_impl(result: pandas.DataFrame, key: str, n_stages_min: int, n_stages_max: int, min_stage_length) -> dict:
    try:
        n_st_mask = (result['N_stages'] >= n_stages_min) & (result['N_stages'] <= n_stages_max)
        st_len_min_mask = result['St_len_min'] >= min_stage_length
        ok_rows = result[n_st_mask & st_len_min_mask].reset_index(drop = True)
        return ok_rows.iloc[ok_rows[key].idxmax()].to_dict()
    except Exception as e:
        return best_result_impl(result, key, n_stages_min - 1, n_stages_max + 1, min_stage_length)

def best_result(result: pandas.DataFrame, key: str, n_stages: int, min_stage_length: int = 0) -> dict:
    return best_result_impl(result, key, n_stages, n_stages, min_stage_length)