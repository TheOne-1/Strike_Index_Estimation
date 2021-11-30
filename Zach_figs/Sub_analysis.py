from numpy.core.defchararray import index
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from num2words import num2words
from scipy.optimize import curve_fit


mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["savefig.dpi"] = 200
column_names = [num2words(x) for x in range(1,10)]
data_path = "C:\\Users\\Zachs\\Documents\\GitHub\\ThesisCode\\result_conclusion"
i = 1
j = 1
base_name = f"\\predict_result_conclusion_RealziesSigmoid_{i}_training_with_{j}.csv"
sub_names =pd.read_csv(data_path+base_name, usecols=["Subject Name"]).values.ravel()
master_df_ave = pd.DataFrame(columns=column_names, index=sub_names)
master_df_max = pd.DataFrame(columns=column_names, index=sub_names)
master_df_min = pd.DataFrame(columns=column_names, index=sub_names)
for j in range(1,10):
    cor_max_collector = np.zeros([10,1])
    cor_ave_collector = np.zeros([10,1])
    cor_min_collector = np.ones([10,1])
    for num, i in enumerate(range(1,10)):
        base_name = f"\\predict_result_conclusion_RealziesSigmoid_{i}_training_with_{j}.csv"
        cor_values = pd.read_csv(data_path + base_name, usecols=["correlation"]).values
        print(cor_values.mean())
        cor_ave_collector += (cor_values - cor_ave_collector)/(num+1)
        cor_max_collector = np.max((cor_max_collector, cor_values), axis=0)
        cor_min_collector = np.min((cor_min_collector, cor_values), axis=0)

    
    master_df_ave[num2words(j)] = pd.Series(cor_ave_collector.ravel(), index=sub_names)
    master_df_max[num2words(j)] = pd.Series(cor_max_collector.ravel(), index=sub_names)
    master_df_min[num2words(j)] = pd.Series(cor_min_collector.ravel(), index=sub_names)
print(master_df_ave)
masterT = master_df_ave.T
masterT.plot()
plt.show()

basename = "out4"
master_df_ave.to_csv(f"Zach_figs\\data\\{basename}_ave.csv")
master_df_max.to_csv(f"Zach_figs\\data\\{basename}_max.csv")
master_df_min.to_csv(f"Zach_figs\\data\\{basename}_min.csv")

master_df = master_df_min

means = master_df.mean()
stds = master_df.std()


upper_band = means + stds
lower_band = means - stds
master_df.mean().plot()

plt.fill_between(range(0,9), upper_band.values, lower_band.values, alpha=0.3)
plt.xticks(np.arange(9),np.arange(1,10))
plt.xlabel("Number of subjects in train set")
plt.ylabel("Average correlation for 9 iterations \n of random selection of training subjects")
plt.title("Effect of training the model with different numbers of subjects")
plt.savefig("Zach_figs\\figs\\Done\\SubSufficiently.png", bbox_inches='tight')
plt.show()






