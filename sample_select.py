from nearst_neighbor import neighbor_sort
import os
import shutil
import pandas as pd
import argparse


##
##python sample_select.py --dir src_dir
##

def cluster(source, classic, selet_num, true_out, false_out):
    """ pickup positive sample and negtive sample

    :param source:      dataframe of source files
    :param classic:     dataframe of classic files
    :param selet_num:   number of picture to pickup
    :param true_out:    path of positive sample
    :param false_out:   path of negtive sample
    :return: dataframe of scores
    """
    neighbor = neighbor_sort(source, classic)

    if not os.path.exists(true_out):
        os.mkdir(true_out)
    if not os.path.exists(false_out):
        os.mkdir(false_out)

    num = neighbor.size()
    sorted_neighbor = neighbor.sort_values(by='distance')

    if selet_num > neighbor.size()/3:
        selet_num = neighbor.size()/3
    for i in range(num):
        distance = sorted_neighbor.at[i, 'distance']
        path = sorted_neighbor[i,'path']

        if i < selet_num:
            cp_path = os.path.join(true_out, os.path.basename(path))
            shutil.copyfile(path, cp_path)
        elif i > num-selet_num:
            cp_path = os.path.join(false_out, os.path.basename(path))
            shutil.copyfile(path, cp_path)
        print(path, cp_path)


if __name__ == '__main__':
    dir = "./test/second"
    out_dir = "./out_dir"
    selet_num = 0
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

    bm_path =  os.path.join(dir,"classic")
    bm_list = os.listdir(source_path)
    bm = pd.DataFrame(columns=['name', 'path'])
    for file in bm_list:
        path=os.path.join(source_path,file)
        bm=df.append([{'name':file, 'path':path}],ignore_index=True)

    true_dir = os.path.join(dir, "true")
    false_dir = os.path.join(dir, "false")

    cluster(df, bm, 100, true_dir, false_dir)



