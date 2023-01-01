import pandas as pd
import scipy.misc
import numpy as np
import matplotlib
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from random import random


class Proteins():
   def __init__(self):
       self.sequence =  ""
       self.position = list()
       self.phostype =  list()
       self.species =  ""
       self.kinas = ""
class DataPrepration():
    def __init__(self):
        self.proteinlist = dict()
        self.aminoacid = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']
        self.aminoacidName = ['alanine','arginine','asparagine ','aspartic acid','cysteine','glutamine ','glutamic acid','glycine','histidine','isoleucine ','leucine ','lysine ','methionine','phenylalanine','proline','serine','threonine','tryptophan','tyrosine','valine']
    def PrintList(self):
        for item in self.proteinlist.keys():
            ff = elm[item]
            print(ff.acc)
            print(ff.sequence)
            print(ff.position)
            print(ff.phostype)
    def FrequencyOfPhospho(self):
        sumt =0
        sumy =0
        sums =0
        for item in self.proteinlist.keys():
            ff = elm[item]
            for ii in ff.phostype:
                if ii == 'T' or ii== 't':
                    sumt+=1;
                if ii == 'Y' or ii== 'y':
                    sumy+=1;
                if ii == 'S' or ii== 's':
                    sums+=1;
        print("T {} , Y {} , S {}".format(sumt,sumy,sums))
    def ReadProteinsSequence_ELM(self,filename):
        df = pd.read_excel(filename, sheet_name='Sheet1')
        data = df.to_numpy()
        for i,item in enumerate(data):

            if (not(item[0] in self.proteinlist.keys())):
                protein = Proteins()
                protein.acc = item[0]
                protein.sequence =item[1]
                protein.position.append(item[2])
                protein.phostype.append(item[3])
                protein.species =item[7]
                self.proteinlist[item[0]] =protein
            else:
                self.proteinlist[item[0]].position.append(item[2])
                self.proteinlist[item[0]].phostype.append(item[3])
        return self.proteinlist
    def SlideStringToSequence(self,windowsize):

        filePho = open("PreparedData/ELM/ELM_Phos_windows_{}.fasta".format(windowsize), "w")
        fileNonPho = open("PreparedData/ELM/ELM_NonPhos_windows_{}.fasta".format(windowsize), "w")

        max_count = 1000

        noph_county=0
        noph_counts=0
        noph_countt=0

        ph_county=0
        ph_counts=0
        ph_countt=0


        for item in self.proteinlist.keys():
            protein = self.proteinlist[item]
            for i,aminoacid in enumerate(protein.sequence):
                if aminoacid=='S' and (i+1) not in protein.position :
                    noph_counts+=1
                if aminoacid=='Y'  and (i+1) not in protein.position :
                    noph_county+=1
                if aminoacid=='T' and (i+1) not in protein.position :
                    noph_countt+=1

                if aminoacid=='S' and (i+1) in protein.position :
                    ph_counts+=1
                if aminoacid=='Y'  and (i+1) in protein.position :
                    ph_county+=1
                if aminoacid=='T' and (i+1)  in protein.position :
                    ph_countt+=1

                if (aminoacid=='S' or aminoacid == 'Y' or aminoacid=='T'):
                    middle = windowsize // 2
                    sequence = ">{}_{}\n".format(protein.acc,aminoacid)
                    for index in range(i-middle,i+middle+1,1):
                        if (index>=0 and index<len(protein.sequence)):
                            sequence += (protein.sequence[index])
                        else:
                            sequence += "N"
                    sequence += ("\n")

                    if (i+1 in protein.position):
                        if (random()>0.5):
                            if (aminoacid == 'S'  and ph_counts < max_count):
                                filePho.write(sequence)
                            if (aminoacid == 'Y' and ph_county < max_count):
                                filePho.write(sequence)
                            if (aminoacid == 'T' and ph_countt < max_count):
                                filePho.write(sequence)
                    else:
                        if (random()>0.5):
                            if (aminoacid == 'S'  and noph_counts<max_count):
                                fileNonPho.write(sequence)
                            if (aminoacid == 'Y' and noph_county<max_count):
                                fileNonPho.write(sequence)
                            if (aminoacid == 'T' and noph_countt<max_count):
                                fileNonPho.write(sequence)
        filePho.close()
        fileNonPho.close()

prepare = DataPrepration()
elm = prepare.ReadProteinsSequence_ELM("DataSet/ELm/Dataset.xlsx")
prepare.FrequencyOfPhospho()
prepare.SlideStringToSequence(47)
