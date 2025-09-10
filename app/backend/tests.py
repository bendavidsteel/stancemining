
import os
import glob

import polars as pl

from main import compute_pca_streamplot_data

def test_flow_field():
    DATA_DIR_PATH = os.environ['DATA_DIR_PATH']

    target_trends_dir_path = os.path.join(DATA_DIR_PATH, 'target_trends')
    if not os.path.exists(target_trends_dir_path):
        raise FileNotFoundError(f"Target trends directory must be accessible at {target_trends_dir_path}")
    trend_file_paths = glob.glob(os.path.join(target_trends_dir_path, '*.parquet.zstd'))
    if len(trend_file_paths) == 0:
        raise FileNotFoundError(f"No trend files found in {target_trends_dir_path}, some must be present for the app to work")
    all_trends_df = None
    for trend_file_path in trend_file_paths[:10]:
        try:
            file_df = pl.read_parquet(trend_file_path)
            if 'trend_mean' not in file_df.columns:
                # dataframe is interpolator output, skip
                continue

            if all_trends_df is None:
                all_trends_df = file_df
            else:
                all_trends_df = pl.concat([all_trends_df, file_df])
        except Exception as e:
            pass

    df = all_trends_df.filter(pl.col('filter_type') == 'PlatformHandleID')

    compute_pca_streamplot_data(df)

if __name__ == "__main__":
    test_flow_field()