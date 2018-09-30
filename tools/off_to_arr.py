import numpy as np
import os
import config 
import linecache
import random

def off_to_arr(f):
    
    point = np.array([],dtype = np.float32)
    r = linecache.getline(f , 2)#"point flat edge"
    data = r.split(" ")
    point_num = data[0]
    sampling = random.sample(range(0,int(point_num)),config.sample_point_num)
    
    for i in sampling:
        r = linecache.getline(f,i)
        data = r.split(" ")
        point = np.append(point , np.float32(data).copy() )
    point = point.reshape(-1,3)
    return point
    
def get_data(train_or_test , class_name):
    file_name = config.DIR_MODELNET40+class_name+"/"+train_or_test
    for name in os.listdir(file_name):
        print("into file "+name)
        yield off_to_arr(file_name+"/"+name)
