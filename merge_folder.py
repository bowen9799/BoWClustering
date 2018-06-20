# -*- coding: utf-8 -*-
import shutil
import os
import sys
from glob import glob
from multiprocessing import Pool
from os import path
import importlib
import click
#importlib.reload(sys)

#soure_dir = '//172.17.100.240/deleteSame/source'
#soure_dir ='./17_fengweixia'
#dest_dir = './17_fengweixia'
#soure_dir ='./tomato'

def traverse(source):
    # folders = glob(path.join(source, '*/'))
    # folders.sort()
    alllist = os.listdir(source)
    print("glob all floders",alllist)
    for folder in alllist:
        file_path = os.path.join(source,folder)
        if os.path.isdir(file_path): #判断是否是文件夹，不是文件夹才打开
            for root, dirs, files in os.walk(source+"/"+folder, topdown=False):
                    print("root is",root)
                    for filename in files:
                        print("file is",filename )
                        strSub = os.path.basename(root)
                        os.rename(os.path.join(root,filename), os.path.join(root, strSub+"__"+filename))
                        mymovefile(root + "/" + strSub+"__"+filename, source)
                    for name in dirs:
                        print("dir is ",os.path.join(root, name))
            os.rmdir(source+"/"+folder)


def mymovefile(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        print "%s not exist!"%(srcfile)
    else:
        fpath,fname=os.path.split(dstfile)    #分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)                #创建路径
        shutil.move(srcfile,dstfile)          #移动文件
        print "move %s -> %s"%( srcfile,dstfile)

def copyfile(file,dest):
    print(file,dest)
    if not os.path.exists(dest):
        os.mkdir(dest)
    shutil.copy(file, dest)

@click.command()
@click.option('--input_dir', default='./tomato', help='root path of image folder')
def main(input_dir):
    traverse(input_dir)

if __name__ == '__main__':
    main()
