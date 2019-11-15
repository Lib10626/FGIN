import csv
import scipy.io as sio
import numpy as np

DIR = 'data/CUB2011/CUB_Porter_7551D_TFIDF_new.mat'
text_feature = sio.loadmat(DIR)['PredicateMatrix']
reader = csv.reader(open('similarity.csv'))
N = 3

results = []
datas = []
temp = []

sum = 0
for line in reader:
    temp.append(line)
    sum += 1

for i in range(sum):
    if(i!=0):
        datas.append([float(x) for x in temp[i][1:]])

for i in range(sum-1):
    # top = sorted(datas[i],reverse=True)[:N]
    top = sorted(datas[i],reverse=True)[0]
    down = sorted(datas[i])[0]
    tindex = datas[i].index(top)
    dindex = datas[i].index(down)
    # tindex = list(map(datas[i].index, top))
    # dindex = list(map(datas.index, down))


    # results.append(text_feature[tindex[0]]*top[0]+text_feature[tindex[1]]*top[1]+text_feature[tindex[2]]*top[2])
    results.append(text_feature[tindex]*top+text_feature[dindex]*down)

sio.savemat('./data/back/CUB_Porter_7551D_TFIDF_new.mat', {'PredicateMatrix':results})



    