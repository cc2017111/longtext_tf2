import csv
import pandas as pd


def read_file(filename, names, delimiter='b'):
    if delimiter == 't':
        sep = '\t'
    elif delimiter == 'b':
        sep = ' '
    else:
        sep = delimiter
    return pd.read_csv(filename, sep=sep, quoting=csv.QUOTE_NONE, skip_blank_lines=False, header=None, names=names, encoding='utf-8')


def read_category():
    categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    category_to_id = dict(zip(categories, range(len(categories))))
    return categories, category_to_id


