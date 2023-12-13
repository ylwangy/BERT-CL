import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import benepar, spacy
from trees import load_trees, tree_from_str, InternalTreebankNode, LeafTreebankNode
import re
import argparse

parsing_model = "../benepar_en3"

nlp = spacy.load("en_core_web_sm")

if spacy.__version__.startswith('2'):
    nlp.add_pipe(benepar.BeneparComponent(parsing_model))
else:
    nlp.add_pipe("benepar", config={"model": parsing_model})


from wordfreq import word_frequency,top_n_list

s=[]

def getwords(w,topn):
    if w.lower() in top_n_list('en',topn):
        return True
    else:
        return False


def getsents(tree_,topn):

    if isinstance(tree_, InternalTreebankNode):
        mask=True
        for child in tree_.children:
            for _ in child.leaves():
                # print(_.word)
                if getwords(_.word,topn):
                    mask=False
                    break

            if not mask:
                break

        if mask:
            s.append(tree_.label)
            
        else:
            for child in tree_.children: 
                getsents(child,topn)

    else:
        if getwords(tree_.word,topn):
            s.append(tree_.word)
        else:
            s.append(tree_.tag)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--topn', type=int, default=500)
    parser.add_argument('--inputs', type=str, required=True)
    args = parser.parse_args()

    done_line=0

    if os.path.exists(args.inputs + '.'+ str(args.topn)):

        with open(args.inputs + '.'+ str(args.topn) ,'r') as f:
            done_line = len(f.readlines())
    else:
        done_line = 0

    l=0    
    with open(args.inputs + '.'+ str(args.topn) ,'a+') as ff:
        if done_line>0:
            ff.write('\n')
        with open(args.inputs ,'r') as fin:
            for line in fin:

                l+=1
                if l > done_line:
                    input_sentences=' '.join(line.strip().split())
                    try:
                        doc = nlp(input_sentences)
                    except:
                        continue
                    sents = list(doc.sents)
                    s_=[]
                    for i in range(len(sents)):
                        s=[]
                        try:
                            example_tree = tree_from_str(sents[i]._.parse_string) #InternalTreebankNode

                            getsents(example_tree,args.topn)
                            s_.append(' '.join(s))
                        except:
                            print(sents[i])
                    ff.write(' '.join(s_))
                    ff.write('\n')
