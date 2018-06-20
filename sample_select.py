# -*- coding: utf-8 -*-

##
##python sample_select.py --dir src_dir
##

from nearst_neighbor import neighbor_sort
from multiSearch import retrieval
import os
import shutil
import pandas as pd
import argparse
import sys

reload(sys)
sys.setdefaultencoding("utf-8")

def integrate(source, classic, selet_num, true_out, false_out):
    """ pickup positive sample and negtive sample

    :param source:      dataframe of source files
    :param classic:     dataframe of classic files
    :param selet_num:   number of picture to pickup
    :param true_out:    path of positive sample
    :param false_out:   path of negtive sample
    :return: dataframe of scores
    """
    print "\source = \n", source
    print "\classic = \n", classic

    source=source.append(classic)
    source = source.drop_duplicates(['name'])
    neighbor = neighbor_sort(source, classic)
    #neighbor = pd.DataFrame.from_csv("neighbor.csv");
    #neighbor.to_csv("neighbor.csv")

    if not os.path.exists(true_out):
        os.makedirs(true_out)
    if not os.path.exists(false_out):
        os.makedirs(false_out)

    num = len(neighbor)
    sorted_neighbor = neighbor.sort_values(by='distance')
    knn_csv_path = os.path.split(true_out)[0] + "/knn_report.csv"
    try:
        os.remove(knn_csv_path)
    except OSError:
        pass
    sorted_neighbor.to_csv(path_or_buf=knn_csv_path)

    print "\rKNN put into CSV. sorted neighbor: ", sorted_neighbor

    integrated = sorted_neighbor
    del integrated['distance']
    integrated.insert(2, 'classification', -2)
    print "\rintegrated: ", integrated
    res, lowf, lowc = retrieval(source, classic, 0.2, 0.05, 0.15, true_out, False)
    print "\rboW put into CSV. num = ", num
    for i in range(0, num):
        if (sorted_neighbor.iat[i, 1]) in classic:
            integrated.iat[i, 2] = 1
        elif i < int(num * 0.3):
            if sorted_neighbor.index.tolist()[i] not in lowc:
                if sorted_neighbor.index.tolist()[i] not in lowf:
                    integrated.iat[i, 2] = 1
        elif int(num * 0.8) <= i < num:
            integrated.iat[i, 2] = -1
        else:
            integrated.iat[i, 2] = 1 if sorted_neighbor.index.tolist()[i] in res else 0
    print "integrated: ", integrated
    int_csv_path = os.path.split(true_out)[0] + "/integrated_report.csv"
    try:
        os.remove(int_csv_path)
    except OSError:
        pass
    integrated.to_csv(path_or_buf=int_csv_path)

    if not os.path.exists(true_out):
        os.makedirs(true_out)
    if not os.path.exists(false_out):
        os.makedirs(false_out)

    for index, row in integrated.iterrows():
        if row['classification'] == 1:
            shutil.copy(row['path'], true_out)
        if row['classification'] == -1:
            shutil.copy(row['path'], false_out)


if __name__ == '__main__':
    dir = "/home/xufeng02/develop/venv/develop/test/second"
    out_dir = "./out_dir"
    selet_num = 100
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir",    help="source to test")
    parser.add_argument("--selet_num",    help="src dir to copy")
    args = parser.parse_args()

    if args.dir:
        dir = args.dir
    if args.selet_num:
        selet_num = int(args.selet_num)

    source_path = os.path.join(dir, "source")
    filelist = os.listdir(source_path)
    df=pd.DataFrame(columns=['name','path'])
    for file in filelist:
        path=os.path.join(source_path,file)
        df=df.append([{'name':file, 'path':path}],ignore_index=True)

    bm_path = os.path.join(dir,"classic")
    bm_list = os.listdir(bm_path)
    bm = pd.DataFrame(columns=['name', 'path'])
    for file in bm_list:
        path=os.path.join(bm_path,file)
        #shutil.copy(os.path.join(bm_path, file), source_path)
        bm = bm.append([{'name':file, 'path':path}],ignore_index=True)

    df.drop_duplicates()

    true_dir = os.path.join(dir, "true")
    false_dir = os.path.join(dir, "false")

    integrate(df, bm, 100, true_dir, false_dir)




