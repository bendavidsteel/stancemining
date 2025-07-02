import datetime

import polars as pl
from sklearn.preprocessing import scale
import torch

from stancemining.estimate import _calculate_trends_for_filtered_df_with_batching, _get_classifier_profiles, get_stance_normal_for_all_targets

def test_batch_train():

    time_column = 'time'
    target_name = 'target'
    unique_values = list(range(10))
    filter_type = 'user'
    data = []
    for user in unique_values:
        train_x = torch.arange(1, torch.randint(low=10, high=50, size=()).item(), dtype=torch.float32)  # 1 to 30
        a = torch.rand(()) * 0.5 + 0.5
        b = torch.rand(()) * 0.5 + 0.5
        train_y = torch.round(torch.sin(a * train_x + b) + torch.normal(torch.zeros_like(train_x), 0.1 * torch.ones_like(train_x)))
        for x, y in zip(train_x, train_y):
            data.append({
                time_column: datetime.datetime(2025, 1, 1) + datetime.timedelta(days=int(x.item())),
                'Target': target_name,
                'Stance': y.item(),
                filter_type: user
            })


    target_df = pl.DataFrame(data)

    classifier_profiles = _get_classifier_profiles()

    lengthscale_loc = 2.0
    lengthscale_scale = 0.5
    sigma_loc = 1.0
    sigma_scale = 0.5
    
    
    time_scale = '1w'

    verbose = True

    trend_df, gp_params = _calculate_trends_for_filtered_df_with_batching(
        target_df, 
        target_name,
        filter_type,
        unique_values,
        classifier_profiles,
        lengthscale_loc,
        lengthscale_scale,
        sigma_loc,
        sigma_scale,
        time_column,
        time_scale,
        verbose=verbose
    )

def test_get_stance_normal():
    target_name = 'target'
    unique_values = list(range(10))
    filter_type = 'user'
    data = []
    for user in unique_values:
        loc = torch.rand(()) - 0.5
        scale = torch.rand(())
        stances = torch.clip(torch.round(torch.normal(loc, scale, size=(10,))), -1, 1)
        for stance in stances:
            data.append({
                'Targets': [target_name],
                'Stances': [stance.item()],
                filter_type: user
            })


    doc_target_df = pl.DataFrame(data)

    stance_mean_df = get_stance_normal_for_all_targets(doc_target_df, filter_cols=[filter_type])

if __name__ == "__main__":
    test_get_stance_normal()