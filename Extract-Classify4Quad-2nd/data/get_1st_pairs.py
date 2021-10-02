#coding=utf-8

import codecs as cs

for domian_type in ['laptop', 'rest16']:
    f = cs.open('/home/hjcai/8RTX/BERT/pytorch_pretrained_BERT/output/Extract-Classify4Quad/'+domian_type+'/pred4pipeline.txt', 'r').readlines()
    wf = cs.open('/home/hjcai/8RTX/BERT/pytorch_pretrained_BERT/Extract-Classify4Quad-2nd/data/'+domian_type+'_test_pair_1st.tsv', 'w')

    for line in f:
        asp = []; opi = []
        line = line.strip().split('\t')
        if len(line) <= 1:
            continue
        text = line[0]
        af = 0
        of = 0
        for ele in line[1:]:
            if ele.startswith('a'):
                asp.append(ele[2:])
                af = 1
            else:
                opi.append(ele[2:])
                of = 1
        if af == 0:
            asp.append('-1,-1')
        if of == 0:
            opi.append('-1,-1')
        if len(asp)>0 and len(opi)>0:
            pred = []

            for pa in asp:
                ast, aed = int(pa.split(',')[0]), int(pa.split(',')[1])
                for po in opi:
                    ost, oed = int(po.split(',')[0]), int(po.split(',')[1])
                    pred.append([pa, po])
            for ele in pred:  
                wf.write(text+'####'+ele[0]+' '+ele[1]+'\n')
