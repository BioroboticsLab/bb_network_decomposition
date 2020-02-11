import datetime
import h5py
import numpy as np
import pandas as pd
import tqdm

import bb_utils
import bb_utils.meta
import bb_utils.ids

def load_networks_h5(path, first_day, last_day):
    with h5py.File(path, 'r') as f:
        labels = list(f['labels'])
        interactions = f['interactions'][first_day:last_day]
        bee_ids = f['bee_ids'][:]
        bee_ages = f['bee_ages'][:]

        return interactions, labels, bee_ids, bee_ages


def load_alive_data(path, bee_ids):
    alive_df = pd.read_csv(path, index_col=0, parse_dates=['annotated_tagged_date', 'inferred_death_date'])
    alive_df = pd.concat([alive_df[alive_df.bee_id == bee_id] for bee_id in bee_ids])

    return alive_df


def load_location_data(path, keepcols=['bee_id', 'date', 'age', 'brood_area', 'dance_floor', 'honey_storage', 'near_exit']):
    loc_df = pd.read_pickle(path)[keepcols]
    loc_df = loc_df[np.logical_not(loc_df[['brood_area']].isna().max(axis=1))]
    loc_df['date'] = pd.to_datetime(loc_df.date)

    return loc_df


def get_daily_alive_matrices(alive_df, num_days, num_entities, from_date):
    is_alive = np.ones((num_days, num_entities, num_entities), dtype=np.bool)

    for day in tqdm.trange(num_days):
        date = from_date + datetime.timedelta(days=day)

        daily_dead = ((alive_df.annotated_tagged_date > date) | (alive_df.inferred_death_date <= date)).values

        is_alive[day, daily_dead, :] = 0
        is_alive[day, :, daily_dead] = 0

    return is_alive


def get_factor_dataframe(daily_factors, from_date, alive_df, bee_ids):
    num_days = len(daily_factors)
    num_factors = daily_factors[0].shape[-1]
    m = bb_utils.meta.BeeMetaInfo()

    dfs = []
    for day in tqdm.trange(num_days):
        date = from_date + datetime.timedelta(days=day)

        alive_ids = sorted(alive_df[(alive_df.annotated_tagged_date <= date) & (alive_df.inferred_death_date > date)].bee_id.values)

        bbids = [bb_utils.ids.BeesbookID.from_ferwar(bid) for bid in alive_ids]
        ages = [m.get_age(bid, date).days for bid in bbids]

        columns = ['day', 'date', 'bee_id', 'age'] + ['f_{}'.format(f) for f in range(num_factors)]
        factor_df = pd.DataFrame(np.concatenate((
            np.array([day for _ in range(len(ages))])[:, None], np.array([date for _ in range(len(ages))])[:, None], np.array(alive_ids)[:, None], np.array(ages)[:, None], daily_factors[day][np.array([(bee_id in alive_ids) for bee_id in bee_ids])]), axis=-1),
                                columns=columns)
        dfs.append(factor_df)

    factor_df = pd.concat(dfs)
    factor_df = factor_df[factor_df.age >= 0]

    return factor_df


def factors_from_dataframe(factor_df):
    num_factors = len([c for c in factor_df.columns if c.startswith('f_')])
    return factor_df[['f_{}'.format(f) for f in range(num_factors)]].values.astype(np.float32)


def merge_location_data(factor_df, location_df):
    print('Locations dataframe shape :', location_df.shape)
    print('Factors dataframe shape :', factor_df.shape)
    merged_df = pd.merge(location_df, factor_df, how='inner', on=['bee_id', 'age', 'date'])
    print('Merged dataframe shape: ', merged_df.shape)

    return merged_df
