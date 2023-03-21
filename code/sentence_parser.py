from stanfordcorenlp import StanfordCoreNLP
import string
import copy
import logging
import nltk
import re
import os
from chunking import get_chunks
from miscc.utils import get_sentences

class parser2:
    def __init__(self):
        print('nlp init')
        self.nlp = StanfordCoreNLP('stanford-corenlp-4.3.1')
        print('nlp finish init')
        self.arr={}
        self.num=0
        self.special_char=['HYPH','\'\'','``','-RRB-','-LRB-','SYM','/']

    def dfs(self,node):
        cnt=0
        cur_num=self.num
        for child in node:
            if type(child)==nltk.tree.Tree:
                if child.label()=='NP':
                    cnt+=1
                cnt+=self.dfs(child)
            elif node.label() not in string.punctuation and node.label() not in self.special_char:
                self.num+=1

        if cnt==0 and node.label()=='NP':
            lst=node.leaves()
            for i in self.special_char:
                if i in lst:
                    lst.remove(i)
            for i in string.punctuation:
                if i in lst:
                    lst.remove(i)
            self.arr[cur_num]=(self.num-cur_num,' '.join(lst))
        return cnt


    def parse(self,caption):
        self.arr.clear()
        self.num=0
        for i in string.punctuation:
            caption=caption.replace(i,', ')
        caption=caption.replace('-',' ')
        caption=caption.replace('&',' ')
        caption=caption.replace('\'',' ')
        caption=caption.replace('cannot','can')
        t=self.nlp.parse(caption)
        tree=nltk.tree.Tree.fromstring(t)

        self.dfs(tree)
        self.tree=tree
        return copy.deepcopy(self.arr)


    def __del__(self):
        try:
            self.nlp.close()
        except KeyError:
            return


class parser:
    def __init__(self):
        print('nlp init')
        self.nlp = StanfordCoreNLP('code/stanford-corenlp-4.3.1')
        print('nlp finish init')
        self.arr={}
        self.num=0
        self.num_p=0
        self.last_phrase_is_np=False
        self.np_pos=-1
        self.noun_list=['NN','NNP','NNS','NNPS']
        self.merge_labels=['VP','PP','ADJP','SBAR','JJ']
        # self.back_labels=['HYPH']
        self.back=False
        self.chunks=[]
        self.cur_chunk_pos=0

        self.result=[]
        self.pattern=re.compile(r'\W')

    def dfs(self,node):
        cnt=0
        cur_num=self.num
        has_independent_np=False
        nums=[]
        child_cnts=[]
        child_has_independent_np=[]
        for child in node:
            cur_child_num=self.num
            nums.append(cur_child_num)

            if type(child)==nltk.tree.Tree:
                if child.label() == 'NP':
                    cnt+=1
                cnt_child,ok=self.dfs(child)
                cnt+=cnt_child
                child_cnts.append(cnt_child)
                child_has_independent_np.append(ok)
                if ok:
                    has_independent_np=True
            else:
                child_cnts.append(0)
                child_has_independent_np.append(False)
                self.num+=1

        put=False
        if has_independent_np:
            for i,child in enumerate(node):
                if child_has_independent_np[i]:
                    continue
                if type(child)==nltk.tree.Tree and child.label() in self.merge_labels:
                    if nums[i] in self.arr:
                        del self.arr[nums[i]]
                    cur_p=-1
                    for p in self.arr:
                        if p<nums[i]:
                            cur_p=p
                    lst=self.words[cur_p:nums[i]]
                    new_str=' '.join(lst)+' '+' '.join(child.leaves())
                    adj_pos,noun_words=self.get_pos(new_str)
                    self.arr[cur_p]=(adj_pos,noun_words,new_str)
                    # print('p merge',child.leaves())
        else:        
            for i,child in enumerate(node):
                if nums[i] in self.arr and self.arr[nums[i]][0]==-1:
                    # print('dependent np up',child.leaves())
                    del self.arr[nums[i]]
                    put=True

        if put or cnt==0 and node.label() == 'NP':
            lst=node.leaves()
            attr=' '.join(lst)
            adj_pos,noun_words=self.get_pos(attr)
            self.arr[cur_num]=(adj_pos, noun_words, attr)
            if adj_pos!=-1:
                has_independent_np=True

        return cnt, has_independent_np

    def update_chunk(self):
        if self.cur_chunk_pos>=len(self.chunks):
            return
        if self.num>self.chunks[self.cur_chunk_pos]['end_pos']:
            self.cur_chunk_pos+=1

    def adjust_tree(self,node):
        if self.cur_chunk_pos>=len(self.chunks):
            return

        self.update_chunk()
        if self.cur_chunk_pos>=len(self.chunks):
            return
        cur_chunk=self.chunks[self.cur_chunk_pos]
        
        b=0
        for i in range(len(node)):
            self.update_chunk()
            if self.cur_chunk_pos>=len(self.chunks):
                return
            cur_chunk=self.chunks[self.cur_chunk_pos]

            i1=i+b
            if type(node[i1])==str:
                self.num+=1
                continue

            child_str=node[i1].leaves()
            if cur_chunk['start_pos']==self.num and cur_chunk['end_pos']==self.num+len(child_str)-1:
                # print('same',child_str)
                node[i1]=nltk.tree.Tree(node[i1].label(), [cur_chunk['text']])
                self.num+=len(child_str)
                continue
            elif self.num==cur_chunk['start_pos'] and cur_chunk['end_pos']>self.num+len(child_str)-1:
                # print('222',child_str)
                node[i1]=nltk.tree.Tree(cur_chunk['label'], [cur_chunk['text']])
                self.num+=len(child_str)
                continue
            elif self.num>cur_chunk['start_pos'] and cur_chunk['end_pos']>=self.num+len(child_str)-1:
                # print('del',child_str)
                self.num+=len(child_str)
                del node[i1]
                b-=1
                continue

            self.adjust_tree(node[i1])

    def parse(self,captions):
        self.result.clear()
        for caption in captions:
            self.arr.clear()
            self.num=0
            self.cur_chunk_pos=0
            caption=caption.replace('.',', ')
            caption=caption.replace('!',', ')
            t=self.nlp.parse(caption)
            self.chunks=get_chunks(caption)
            # print(self.chunks)

            tree=nltk.tree.Tree.fromstring(t)
            self.adjust_tree(tree)
            self.words=tree.leaves()

            # tree.draw()
            self.num=0
            self.dfs(tree)
            self.tree=tree
            for item in self.arr:
                adj_pos,noun_words,cap=self.arr[item]
                if adj_pos!=-1 and len(noun_words)!=0 and self.arr[item] not in self.result:
                    self.result.append(self.arr[item])

        self.result.sort(key=lambda x:(len(x[1]),len(x[2].split(' '))))
        return copy.deepcopy(self.result)


    def get_pos(self,cap):
        # tokens = nltk.word_tokenize(cap)
        # lst = nltk.pos_tag(tokens)

        t=self.nlp.parse(cap)
        node=nltk.tree.Tree.fromstring(t)
        # node.pretty_print()
        lst=node.pos()
        # print(lst)
        p=-1
        b=0
        noun_words=[]
        for i,pos in enumerate(node.treepositions('leaves')):
            label=node[pos[:-1]].label()
            label2=node[pos[:-2]].label()
            word=node[pos]
            word2=word.replace("\ufffd\ufffd", " ")
            word2=self.pattern.sub(' ',word2)
            if word2 != word:
                b-=1
                continue
            if label in ['JJ'] and p==-1:
                p=i+b

            if label in self.noun_list and label2=='NP':
                noun_words.append(i+b)
        
        # for i,(word, label) in enumerate(lst):
        #     word2=word.replace("\ufffd\ufffd", " ")
        #     word2=self.pattern.sub(' ',word2)
        #     if word2 != word:
        #         b-=1
        #         continue
        #     if label in ['JJ'] and p==-1:
        #         p=i+b

        #     if label in self.noun_list:
        #         noun_words.append(i+b)
        return p,noun_words

    # def __del__(self):
    #     try:
    #         self.nlp.close()
    #     except KeyError:
    #         return


if __name__ == '__main__':
    sentences = get_sentences('datasets/birds/text/095.Baltimore_Oriole/Baltimore_Oriole_0127_87560.txt')
    p=parser()
    print(p.parse(sentences))
    # p.tree.draw()
