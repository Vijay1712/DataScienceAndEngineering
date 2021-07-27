import numpy as np
from collections import OrderedDict

class Lab1(object):
    def parse_reads_illumina(self,reads) :
        '''
        Input - Illumina reads file as a string
        Output - list of DNA reads
        '''
        x = reads.split('\n')
        print(x)
        return x[1::4]
        #end code here

    def unique_lengths(self,dna_reads) :
        '''
        Input - list of dna reads
        Output - set of counts of reads
        '''
        #start code here
        count = set()
        for i in dna_reads:
            length = len(i)
            count.add(length)
        return count
        #end code here

    def check_impurity(self,dna_reads) :
        '''
        Input - list of dna reads
        Output - list of reads which have impurities, a set of impure chars 
        '''
        #start code here
        impurities = []
        impureChar = set()
        for dna in dna_reads:
            impure = False
            for char in dna:
                if char not in 'ACGTacgt':
                    impureChar.add(char)
                    impure = True
            if impure:
                impurities.append(dna)
                # if char not in 'ACGT':
        return impurities, impureChar

        #end code here

    def get_read_counts(self,dna_reads) :
        '''
        Input - list of dna reads
        Output - dictionary with key as read and value as the no. of times it occurs
        '''
        #start code here
        x = {}
        for i in dna_reads:
            if i in x:
                x[i] += 1
            else:
                x[i] = 1
        return x
        #end code here

    def parse_reads_pac(self,reads_pac) :
        '''
        Input - pac bio reads file as a string
        Output - list of dna reads
        '''
        #start code here
        output = []
        x = reads_pac.split(">")
        for i in range(1, len(x)):
            line = x[i].split('\n')
            dna = ''
            for j in range(1, len(line)):
                dna+=line[j]
            # print(dna)
            output.append(dna)
        return output
        
        #end code here