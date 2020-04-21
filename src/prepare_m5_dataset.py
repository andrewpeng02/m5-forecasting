import os

import pandas as pd
import numpy as np


def main():
    project_dir = os.getcwd()[:-3]

    sales_df = pd.read_csv(project_dir + '/data/m5/sales_train_validation.csv')
    calendar_df = pd.read_csv(project_dir + '/data/m5/calendar.csv')

    prepared = np.expand_dims(sales_df.iloc[:, 6:].values, 2)
    prepared = np.concatenate((prepared, create_day_matrix(calendar_df)), axis=2)
    prepared = np.concatenate((prepared, create_month_matrix(calendar_df)), axis=2)
    prepared = np.concatenate((prepared, create_holiday_matrix(calendar_df)), axis=2)

    train = prepared[:, :-28, :]

    np.save(project_dir + '/data/out/train.npy', train)
    np.save(project_dir + '/data/out/entire_data.npy', prepared)


def create_day_matrix(calendar_df):
    days = calendar_df.loc[:, 'wday'].to_numpy()[:1913]
    day_mat = np.tile(days, (30490, 1))

    return np.expand_dims(day_mat, 2)


def create_month_matrix(calendar_df):
    months = calendar_df.loc[:, 'month'].to_numpy()[:1913]
    months_mat = np.tile(months, (30490, 1))

    return np.expand_dims(months_mat, 2)


def create_holiday_matrix(calendar_df):
    holidays_1 = np.where(calendar_df['event_name_1'].isnull(), 0, 1)[:1913]
    holidays_2 = np.where(calendar_df['event_name_2'].isnull(), 0, 1)[:1913]
    holidays = holidays_1 + holidays_2

    holidays_mat = np.tile(holidays, (30490, 1))

    return np.expand_dims(holidays_mat, 2)


if __name__ == "__main__":
    main()
