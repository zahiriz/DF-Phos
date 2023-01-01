import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
file = open('pvalues.txt',"r")
lines = file.readlines()
file.close()
pvalues = np.zeros((24,24))
for item in lines:
    data = item.split(",")
    i = int(data[0])
    j = int(data[1])
    v = float (data[2])
    pvalues[i,j]= v

labels = ['N_1','N_2','N_3','N_4','N_5','N_6','N_7','N_8','M_1','M_2','M_3','M_4','M_5','M_6','M_7','M_8','C_1','C_2','C_3','C_4','C_5','C_6','C_7','C_8']

import numpy as np; np.random.seed(0)
import seaborn as sns
mask = np.zeros_like(pvalues)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    f, ax = plt.subplots(figsize=(7, 5))
    ax = sns.heatmap(pvalues, mask=mask, vmax=.3, square=True,xticklabels=labels,yticklabels=labels)

plt.show()
df = pd.DataFrame(pvalues,index=labels,columns=labels)
df.to_excel('R/pvalues.xlsx')

