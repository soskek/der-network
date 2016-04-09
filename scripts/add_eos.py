# coding: utf-8
import sys
from multiprocessing import Pool
from glob import glob
from datetime import datetime
from time import time
from itertools import izip
import os

n_pool = 4
depvocabS = set([])

end_pt1 = [".","!","?"]
end_pt1 = set(end_pt1)

end_pt2_s = [(e,"'") for e in [".","!","?"]] + [(e,"''") for e in [".","!","?"]]
end_pt2_s+= [("'",e) for e in [".","!","?"]] + [("''",e) for e in [".","!","?"]]
end_pt2_d = [(e,'"') for e in [".","!","?"]]
end_pt2_d+= [('"',e) for e in [".","!","?"]]
end_pt2_s = set(end_pt2_s)
end_pt2_d = set(end_pt2_d)


def add_eos(cont):
    sentL = []
    start = 0
    skip = -1
    aps_d = False
    sp = cont.split()
    for i, adj in enumerate( izip(sp[:-1],sp[1:]) ):
        if i <= skip:
            continue

        if aps_d and adj in end_pt2_d:
            sentL.append( " ".join(sp[start:i+2]) )
            start = i+2
            skip = i+1
            aps_d = False
            continue

        elif adj[0] == '"':
            aps_d = not aps_d
        elif adj[0] in end_pt1:
            if aps_d:
                continue
            sentL.append( " ".join(sp[start:i+1]) )
            start = i+1
            aps_d = False
    if start < len(sp):
        sentL.append( " ".join(sp[start:]) )
    sentL.append( "" )
    return " <eos> ".join( sentL )
            
def trans(dataname):
    cont = ""
    for i,l in enumerate( open(dataname) ):
        if i == 2:
            cont += add_eos( l.strip() ) + "\n"
        else:
            cont += l            
    return cont, dataname

def main(datadir):
    print "############################"
    print "#### counting"
    st = time()

    dir_list = [datadir.rstrip("/") + "/" + ty + "/*" for ty in ["test","training","validation"]]
    print dir_list
    for base_dir in set([dir_[:-2].replace("/questions/","/eos_questions/") for dir_ in dir_list]):
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
            print "mkdir", base_dir
        else:
            print base_dir, "already exists"

    file_list = reduce(lambda x,y:x+y, [glob(dir_) for dir_ in dir_list])
    n_file = len(file_list)
    print "# of files:", n_file

    pool = Pool(n_pool)
    i = 0
    for (cont, data_name) in pool.imap_unordered(trans, file_list):
        print >>open(data_name.replace("/questions/","/eos_questions/"), "w"), cont
        i += 1
        if i % 10000 == 0:
            print i,"\t",data_name.split("/")[-1],"\t",datetime.today().strftime("%m/%d %H:%M:%S"),"..."

    print "TIME:",time()-st

if __name__ == "__main__":
    main(sys.argv[1])
