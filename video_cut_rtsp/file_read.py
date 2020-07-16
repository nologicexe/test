import os
import glob

path_to_data='images'
direct_0=os.listdir(path_to_data)
for d_0 in direct_0:
    print(d_0)
    link_0=os.path.join(path_to_data, d_0)
    direct_1=os.listdir(link_0)
    for d_1 in direct_1:
        print(d_1)
        link_1=os.path.join(link_0,d_1)
        print(link_1)
        for p in glob.glob(link_1+'/*.jpg'):
            print(p)
