import glob

from myconfig import *

subsets = ['trainForTest','train', 'valForTest', 'val', 'test']
id = 0
for sub in subsets:
    if 'ForTest' in sub:
        names = glob.glob(DATA_ROOT + '/' + sub[:-7]+ '/A-*-0x0')
    else:
        names = glob.glob(DATA_ROOT + '/' + sub + '/A-*')
    f = open(DATA_ROOT + '/' + sub + '0.csv', 'w')
    f.write('id,subj_folder\n')
    for i, name in enumerate(names):
        f.write('%d,%s\n'%(id+i,name))
    f.close()
    id += len(names)




