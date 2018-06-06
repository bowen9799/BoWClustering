import os
import argparse as ap
parser = ap.ArgumentParser()
parser.add_argument("--dir", required=True)
args = vars(parser.parse_args())
dir = args["dir"]
list_pic = os.listdir(dir)
for pic in list_pic:
    p = dir + "\\" + pic
    print p
    os.rename(p, dir + "\\" + "true_" + pic)