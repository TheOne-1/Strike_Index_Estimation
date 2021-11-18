import pandas as pd
# from const import SUB_NAMES
import matplotlib.pyplot as plt
import string
from math import sqrt
import numpy as np

import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report, accuracy_score

mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["savefig.dpi"] = 200



SUB_NAMES = ('190521GongChangyang', '190523ZengJia', '190522QinZhun', '190522YangCan', '190521LiangJie','190510HeMing', '190517ZhangYaqian', '190518MouRongzi', '190518FuZhinan', '190522SunDongxiao')


path_to_sub_stats = "D:\\PhaseIData\\SubjectInfo.csv"
path_to_correlations_folder = "C:\\Users\\Zachs\\Documents\\GitHub\\ThesisCode\\result_conclusion"
correlations_file = "predict_result_conclusion_RealziesSigmoid.csv"
path_to_discrete_results = "C:\\Users\\Zachs\\Documents\\GitHub\\ThesisCode\\results"
discrete_results_prefix = "RealziesSigmoid"
discrete_results_types = ["_24","_28", "_Minim", "_Trad"]

def barplot_annotate_brackets(num1, num2, data, center, height, yerr=None, dh=.05, barh=.05, fs=None, maxasterix=None, prev_y = 0):
    """ 
    Annotate barplot with p-values.

    :param num1: number of left bar to put bracket over
    :param num2: number of right bar to put bracket over
    :param data: string to write or number for generating asterixes
    :param center: centers of all bars (like plt.bar() input)
    :param height: heights of all bars (like plt.bar() input)
    :param yerr: yerrs of all bars (like plt.bar() input)
    :param dh: height offset over bar / bar + yerr in axes coordinates (0 to 1)
    :param barh: bar height in axes coordinates (0 to 1)
    :param fs: font size
    :param maxasterix: maximum number of asterixes to write (for very small p-values)
    """

    if type(data) is str:
        text = data
    else:
        # * is p < 0.05
        # ** is p < 0.005
        # *** is p < 0.0005
        # etc.
        text = ''
        p = .05

        while data < p:
            text += '*'
            p /= 10.

            if maxasterix and len(text) == maxasterix:
                break

        if len(text) == 0:
            text = 'n. s.'

    lx, ly = center[num1], height[num1]
    rx, ry = center[num2], height[num2]

    if yerr is not None:
        ly += yerr[num1]
        ry += yerr[num2]

    ax_y0, ax_y1 = plt.gca().get_ylim()
    dh *= (ax_y1 - ax_y0)
    barh *= (ax_y1 - ax_y0)

    y = max(ly, ry, prev_y) + dh


    barx = [lx, lx, rx, rx]
    bary = [y, y+barh, y+barh, y]
    mid = ((lx+rx)/2, y+barh)

    plt.plot(barx, bary, c='black')

    kwargs = dict(ha='center', va='bottom')
    if fs is not None:
        kwargs['fontsize'] = fs

    plt.text(*mid, text, **kwargs)
    return y


def table1(): # This is just a table of all the subject data and correlations
    sub_data = pd.read_csv(path_to_sub_stats)
    names_dict = dict(enumerate(SUB_NAMES,1))
    sub_data["Subject Name"] = sub_data["Subject ID"].map(names_dict)
    correlation_data = pd.read_csv(path_to_correlations_folder + "\\" + correlations_file)
    all_data = sub_data.merge(correlation_data, on="Subject Name")
    all_data.drop(["Subject Name", "RMSE", "Mean Error"], "columns", inplace=True)
    summary_row = {"Subject ID": "Mean" ,
                    "Sex" : "" , 
                    "Age": f"{all_data.Age.mean():.1f}",
                    "Weight": f"{all_data.Weight.mean():.1f}",
                    "Height": f"{all_data.Height.mean():.1f}",
                    "correlation":f"{all_data.correlation.mean():.2f}",
                    "Train_Size":f"{all_data.Train_Size.mean():.1f}",
                    "Test_Size":f"{all_data.Test_Size.mean():.1f}"

                    }
    all_data = all_data.append(summary_row, ignore_index=True)
    return all_data

def fig1(zipped_discrete_kinds, legend_entries, save_name=None): # This is a fig that plots scatter for different conditions
    for discrete_results_type, marker_type in zipped_discrete_kinds :
        master_df = pd.DataFrame(columns=["true", "pred"])
        for sub in SUB_NAMES:
            path = path_to_discrete_results + "\\" + discrete_results_prefix + discrete_results_type + "\\" + sub + ".csv"
            sub_df = pd.read_csv(path)
            master_df = master_df.append(sub_df, ignore_index=True)
            # master_df = master_df[master_df.true != 1]
        plt.plot(master_df.true.values, master_df.pred.values, marker=marker_type[0], color=marker_type[1], linestyle="None", alpha=0.5, markeredgewidth=0.0)[0]
    if legend_entries is not None:
        plt.legend(legend_entries)
    plt.plot([0,1],[0,1],"r")
    plt.xlabel("True Strike Index")
    plt.ylabel("Predicted Strike Index")
    ax = plt.gca()
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    vals = ax.get_xticks()
    ax.set_xticklabels(['{:,.0%}'.format(x) for x in vals])
    if save_name is None:
        plt.show()
    else:
        plt.savefig(save_name)
        plt.close("all")

def fig2(save_name=None): # This is a bar chart with whiskers for strike patterns
    # discrete_results_types = ["_24","_28", "_Minim", "_Trad"]
    master_df_rf = pd.DataFrame(columns=["true", "pred"])
    master_df_mf = pd.DataFrame(columns=["true", "pred"])
    master_df_ff = pd.DataFrame(columns=["true", "pred"])
    master_df = pd.DataFrame(columns=["rf_error", "mf_error", "ff_error"]) 
    # for discrete_results_type in discrete_results_types:
    discrete_results_prefix = "4_Delay_Project"
    for sub in SUB_NAMES:
        path = path_to_discrete_results + "\\" + discrete_results_prefix + "\\" + sub + ".csv"
        sub_df = pd.read_csv(path)
        master_df_rf = master_df_rf.append(sub_df[sub_df.true < .33], ignore_index=True)
        master_df_mf = master_df_mf.append(sub_df[sub_df.true.between(.33,.66)], ignore_index=True)
        master_df_ff = master_df_ff.append(sub_df[sub_df.true > .66], ignore_index=True)
        
    master_df.rf_error = (((master_df_rf.pred - master_df_rf.true)**2).mean())**.5
    master_df.mf_error = (((master_df_mf.pred - master_df_mf.true)**2).mean())**.5
    master_df.ff_error = (((master_df_ff.pred - master_df_ff.true)**2).mean())**.5

    standard_error =  master_df.std()/sqrt(len(master_df.index))
    pos = range(1,4)
    means = master_df.mean().values
    plt.bar(pos, means, width=.6 ,  tick_label= ["Rearfoot","Midfoot","Forefoot"], color="grey",  yerr=standard_error, capsize=10)
    plt.axhline(y = 0 , color = 'k', linestyle = '-')
    plt.xlabel("Strike Pattern")
    plt.ylabel("Average Strike Index Error")
    if save_name is None:
        plt.show()
    else:
        plt.savefig(save_name)
        plt.close("all")

def fig3(discrete_results_types, legend_entries, save_name=None): # This gives a bar plot with error bars for different conditions
    columns = list(string.ascii_lowercase[0:len(legend_entries[0])])
    master_df = pd.DataFrame(columns=columns) 
    for i, discrete_results_type in enumerate(discrete_results_types):
        minor_df = pd.DataFrame(columns=["true", "pred"])
        for sub in SUB_NAMES:
            path = path_to_discrete_results + "\\" + discrete_results_prefix + discrete_results_type + "\\" + sub + ".csv"
            sub_df = pd.read_csv(path)
            minor_df = minor_df.append(sub_df, ignore_index=True)
        master_df[columns[i]] = minor_df.pred - minor_df.true

    standard_dev =  master_df.std()
    pos = range(1,len(legend_entries[0])+1)
    means = ((master_df**2).mean())**.5
    plt.bar(pos, means, width=.6, tick_label= legend_entries[0], color="0.6",  yerr=standard_dev, capsize=10)
    plt.axhline(y = 0 , color = 'k', linestyle = '-')
    plt.xlabel(legend_entries[1])
    plt.ylabel("Average Strike Index Error")
    if save_name is None:
        plt.show()
    else:
        plt.savefig(save_name)
        plt.close("all")

def fig4(discrete_results_types, save_name=None): # This is a histogram for the true value for all the data
    for discrete_results_type in discrete_results_types :
        master_df = pd.DataFrame(columns=["true", "pred"])
        for sub in SUB_NAMES:
            path = path_to_discrete_results + "\\" + discrete_results_prefix + discrete_results_type + "\\" + sub + ".csv"
            sub_df = pd.read_csv(path)
            master_df = master_df.append(sub_df, ignore_index=True)
    plt.hist(master_df.true.values, bins=100) 
    # plt.xlabel("True Strike Index")
    # plt.ylabel("Predicted Strike Index")
    if save_name is None:
        plt.show()
    else:
        plt.savefig(save_name)
        plt.close("all")

def fig5(save_name=None): # This is a bar chart with whiskers for strike patterns
    master_df = pd.DataFrame(columns=["true", "pred"])
    discrete_results_prefix = "4_Delay_Project"
    cm_collector = np.zeros((3,3))
    for num, sub in enumerate(SUB_NAMES):
        path = path_to_discrete_results + "\\" + discrete_results_prefix + "\\" + sub + ".csv"
        sub_df = pd.read_csv(path)
        sub_df["pred_footstrike_type"] = pd.Series(np.round(sub_df["pred"].values/.33-.5), index=None)
        sub_df["true_footstrike_type"] = pd.Series(np.round(sub_df["true"].values/.33-.5), index=None)
        sub_df.pred_footstrike_type.mask(sub_df.pred_footstrike_type > 2, 2, inplace=True)
        sub_df.pred_footstrike_type.mask(sub_df.pred_footstrike_type < 0, 0, inplace=True)
        sub_df.true_footstrike_type.mask(sub_df.true_footstrike_type > 2, 2, inplace=True)
        sub_df.true_footstrike_type.mask(sub_df.true_footstrike_type < 0, 0, inplace=True)
        cm = confusion_matrix(sub_df.true_footstrike_type.values, sub_df.pred_footstrike_type.values, normalize="true")
        cm_collector += (cm - cm_collector)/(num+1)

    # print(classification_report(master_df.true_footstrike_type.values, master_df.pred_footstrike_type.values))
    disp = ConfusionMatrixDisplay(cm_collector)
    disp.plot()
    plt.show()

def fig6(discrete_results_types, legend_entries, result_type , save_name=None, axis_limits = None, sig_list=None): # This gives a bar plot with error bars for different conditions
    columns = list(string.ascii_lowercase[0:len(legend_entries[0])])
    master_df = pd.DataFrame(columns=columns, index= range(0,len(SUB_NAMES))) 
    for i, discrete_results_type in enumerate(discrete_results_types):
        path = path_to_correlations_folder + "\\predict_result_conclusion_" + discrete_results_prefix + discrete_results_type +".csv"
        con_df = pd.read_csv(path, usecols=[result_type[0]])
        master_df[columns[i]] = con_df

    standard_dev =  master_df.std()
    pos = range(1,len(legend_entries[0])+1)
    means = master_df.mean()
    fig, ax = plt.subplots()
    plt.bar(pos, means, width=.6, tick_label= legend_entries[0], color="0.6",  yerr=standard_dev, capsize=10)
    # plt.axhline(y = 0 , color = 'k', linestyle = '-')
    
    plt.xlabel(legend_entries[1])
    plt.ylabel(result_type[1])
    if result_type[0] == "RMSE":
        vals = ax.get_yticks()
        ax.set_yticklabels(['{:,.1%}'.format(x) for x in vals])
    else:
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    if axis_limits is not None:
        plt.ylim(bottom = axis_limits[0])
    if sig_list is not None:
        prev_y = 0
        for pairs, sig in sig_list:
            prev_y = barplot_annotate_brackets(pairs[0],pairs[1],sig, pos, means, standard_dev, prev_y=prev_y)
    
    if save_name is None:
        plt.show()
    else:
        plt.savefig(save_name, bbox_inches='tight')
        plt.close("all")

def fig7(discrete_results_types, entries, result_type , save_name=None, axis_limits = None): # This gives a bar plot with error bars for different conditions
    columns = list(string.ascii_lowercase[0:len(discrete_results_types)])
    master_df = pd.DataFrame(columns=columns, index= range(0,len(SUB_NAMES))) 
    for i, discrete_results_type in enumerate(discrete_results_types):
        path = path_to_correlations_folder + "\\predict_result_conclusion_" + discrete_results_prefix + discrete_results_type +".csv"
        con_df = pd.read_csv(path, usecols=[result_type[0]])
        master_df[columns[i]] = con_df

    standard_dev =  master_df.std()
    means = master_df.mean()
    fig = plt.figure()
    axs = fig.add_subplot(221), fig.add_subplot(222), fig.add_subplot(223), fig.add_subplot(224)
    idxs = [[0,3], [2,1], [0,3], [3,1]]
    label_idxs = [[0,1],[0,1],[2,3],[2,3]] 
    ttl_idxs = [2,3,0,1]
    bunched_idxs = zip(idxs, ttl_idxs, axs,label_idxs)
    for idx, ttl_idx, ax, label_idx in bunched_idxs:
        ax.bar([1,2], means[idx], width=.6, tick_label= [entries[i] for i in label_idx], color="0.6", yerr=standard_dev[idx], capsize=5)
        ax.title.set_text(entries[ttl_idx])   
        ax.set_ylim((.9,1))
    
    plt.show()


    
    
    # plt.xlabel(legend_entries[1])
    # plt.ylabel(result_type[1])
    # if axis_limits is not None:
    #     plt.ylim((axis_limits[0],axis_limits[1]))
    # if save_name is None:
    #     plt.show()
    # else:
    #     plt.savefig(save_name, bbox_inches='tight')
    #     plt.close("all")



if __name__ == "__main__":
    # # Table Data
    # # all_data = table1()
    # # print(all_data)

    # # Fig that shows scatter of all data
    # discrete_results_types = ["_24","_28","_Minim", "_Trad"]
    # legend_entries = [r"2.4 m/s", r"2.8 m/s","Minimalist Running Shoes","Traditional Running Shoes"]
    # marker_types = [("s","0.7" ), ("s","0.3" ), ("o","0.7" ), ("o","0.3" )]
    # # discrete_results_types = ["_SI_Only"]
    # # legend_entries = [r""]
    # # marker_types = ["xb"]
    # zipped_discrete_results = zip(discrete_results_types,marker_types)
    # save_name = "Misc_tools\\figs\\Done\\all_scatter.png"
    # # save_name = None
    # fig1(zipped_discrete_results, legend_entries, save_name)

    # Fig that shows scatter of all data
    discrete_results_types = ["_24","_28","_Minim", "_Trad"]
    legend_entries = [r"2.4 m/s", r"2.8 m/s","Minimalist Running Shoes","Traditional Running Shoes"]
    marker_types = [("s","0.7" ), ("s","0.3" ), ("o","0.7" ), ("o","0.3" )]
    discrete_results_types = [""]
    legend_entries = None
    marker_types = ["ok"]
    zipped_discrete_results = zip(discrete_results_types,marker_types)
    save_name = "Misc_tools\\figs\\Done\\all_scatter_percent.png"
    # save_name = None
    fig1(zipped_discrete_results, legend_entries, save_name)

    # # Fig that shows scatter of all data for two speeds
    # discrete_results_types = ["_24","_28"]
    # legend_entries = [r"2.4 m/s", r"2.8 m/s"]
    # marker_types = [("s","0.7" ), ("o","0.3" )]
    # zipped_discrete_results = zip(discrete_results_types,marker_types)
    # save_name = "Misc_tools\\figs\\speeds_scatter.png"
    # # save_name = None
    # fig1(zipped_discrete_results, legend_entries, save_name)

    # # Fig that shows scatter of all data for two shoe types
    # discrete_results_types = ["_Minim", "_Trad"]
    # legend_entries = ["Minimalist Running Shoes","Traditional Running Shoes"]
    # marker_types = [("s","0.7" ), ("o","0.3" )]
    # zipped_discrete_results = zip(discrete_results_types,marker_types)
    # save_name = "Misc_tools\\figs\\shoe_type_scatter.png"
    # # save_name = None
    # fig1(zipped_discrete_results, legend_entries, save_name)

    # # # Fig that shows box plot for strike types
    # save_name = "Misc_tools\\figs\\Strike_Type_Bar.png"
    # fig2(save_name)
    # save_name = None
    # Fig that shows box plot for shoe types
    # discrete_results_types = ["_Minim", "_Trad"]
    # legend_entries = [["Minimalist Running Shoes","Traditional Running Shoes"], "Shoe Type"]
    # save_name = "Misc_tools\\figs\\Shoe_Type_Bar.png"
    # fig3(discrete_results_types, legend_entries, save_name)

    # Fig that shows box plot for speeds
    # discrete_results_types = ["_24","_28"]
    # legend_entries = [["2.4 m/s","2.8 m/s"], "Running Speed"]
    # save_name = "Misc_tools\\figs\\Speed_Bar.png"
    # fig3(discrete_results_types, legend_entries, save_name)

    # save_name = None
    discrete_results_types = ["_24","_28","_Minim", "_Trad"]
    legend_entries = [["2.4 m/s","2.8 m/s","Minimalist \n Running Shoes","Traditional \n Running Shoes"], "Running Condition"]
    result_type = ["correlation","Average correlation coefficient"] 
    axis_limits = [.8,1]
    save_name = "Misc_tools\\figs\\Done\\correlation.png"
    fig6(discrete_results_types, legend_entries, result_type , save_name=save_name, axis_limits = [.9,1.05])
    save_name = "Misc_tools\\figs\\Done\\RMSE_percent.png"
    result_type = ["RMSE"," Average Root Mean Square error"]
    fig6(discrete_results_types, legend_entries, result_type , save_name=save_name)

    legend_entries = [["Train with 2.4 m/s \n Test with 2.4 m/s","Train with 2.8 m/s \n Test with 2.8 m/s","Train with 2.4 m/s \n Test with 2.8 m/s ","Train with 2.8 m/s \n Test with 2.4 m/s "], "Condition Combination"]
    discrete_results_types = ["_24", "_28", "_tr24te28", "_tr28te24"]
    result_type = ["correlation","Average correlation coefficient"] 


    # fig6(discrete_results_types, legend_entries, result_type , save_name=save_name,axis_limits=[.9, None], sig_list=list(zip([[0,1],[1,2],[0,3]],["*","*","*"])))

    # legend_entries = [["Train with Minimalist \n Test with Minimalist","Train with Traditional \n Test with Traditional","Train with Minimalist \n Test with Traditional ","Train with Traditional \n Test with Minimalist"], "Condition Combinations"]
    # discrete_results_types = [ "_Minim", "_Trad", "_trMinimteTrad", "_trTradteMinim"]
    # result_type = ["correlation","Average correlation coefficient"] 
    # fig6(discrete_results_types, legend_entries, result_type , save_name=save_name,axis_limits=[.85, None], sig_list=list(zip([[2,3],[1,3],[0,3]],["*","*","*"])))


    # discrete_results_types = ["_24", "_28", "_tr24te28", "_tr28te24"]
    # result_type = ["correlation","Average correlation coefficient"] 
    # entries = ["Train with 2.4 m/s","Train with 2.8 m/s", "Test with 2.4 m/s","Test with 2.8 m/s"]
    # fig7(discrete_results_types, entries, result_type , save_name=save_name)
    # discrete_results_types = [ "_Minim", "_Trad", "_trMinimteTrad", "_trTradteMinim"]
    # entries = ["Train with Minimalist","Train with Traditional", "Test with Minimalist","Test with Traditional"]
    # fig7(discrete_results_types, entries, result_type , save_name=save_name)


    # # Fig that histogram of data
    # discrete_results_types = ["_24","_28","_Minim", "_Trad"]
    # save_name = "Misc_tools\\figs\\Strike_Hist.png"
    # save_name = None
    # fig4(discrete_results_types, save_name)

    # fig5(save_name=None)
    # def rmse(predictions, targets):
    # return np.sqrt(((predictions - targets) ** 2).mean())
