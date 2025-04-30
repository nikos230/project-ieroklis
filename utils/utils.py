import os
import glob


def get_files():
    return 0




def fix_lists(pre_train_years, pre_train_months, pre_train_exlude_days):
    pre_train_years = [pre_train_years]
    pre_train_years = pre_train_years[0].split(', ')

    pre_train_months = [pre_train_months]
    pre_train_months = pre_train_months[0].split(', ')

    if pre_train_exlude_days == "None":
        pre_train_exlude_days = -1
    else:
        pre_train_exlude_days = [pre_train_exlude_days]
        pre_train_exlude_days = pre_train_exlude_days[0].split(', ')   

    return pre_train_years, pre_train_months, pre_train_exlude_days