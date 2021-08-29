#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import math
import numpy as np
from sklearn.cluster import KMeans
from sklearn import cluster, metrics
import matplotlib.pyplot as plt
import itertools as it
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
import xgboost
import random
from sklearn.metrics import explained_variance_score
from xgboost import plot_importance
from matplotlib import pyplot
from sklearn.metrics import r2_score
import tkinter as tk
import warnings
warnings.filterwarnings("ignore")

def datapreprocess(df):
    # data preprocess
    twenty_features = ['Input_A6_024','Input_A3_016','Input_C_013','Input_A2_016','Input_A3_017',
                       'Input_C_050','Input_A6_001','Input_C_096','Input_A3_018','Input_A6_019',
                       'Input_A1_020','Input_A6_011','Input_A3_015','Input_C_046','Input_C_049',
                       'Input_A2_024','Input_C_058','Input_C_057','Input_A3_013','Input_A2_017']
    output = ['Output_A1','Output_A2','Output_A3','Output_A4','Output_A5','Output_A6']
    # fetch 20 features and the add in the end
    X_feature = pd.concat([df.drop(twenty_features,axis = 1),df.loc[:,twenty_features]],axis = 1).drop(output,axis = 1)
    # select direction features
    direction_features = ['Input_C_0' + str(i) for i in it.chain(range(15,39),range(63,83))]
    # transform above direction features
    concat_data = [X_feature]
    num,dire = dict(),dict()
    num['L'] = num['D'] = -1; num['R'] = num['U'] = 1; num['N'] = dire['N'] = 0
    dire['L'] = dire['R'] = 0; dire['U'] = dire['D'] = 1
    for j in direction_features:
        x_axis = []; y_axis = []; ed = []
        for i in range(X_feature.loc[:,j].shape[0]):
            if X_feature.loc[:,j].isnull()[i] == True:
                x_axis.append(np.nan); y_axis.append(np.nan); ed.append(np.nan)
            else:
                a = X_feature.loc[:,j][i].split(';')
                b = [a[0],a[2]]
                c = [float(a[1]),float(a[3])]
                d = dict(zip(b, c))
                cord = [0,0]
                for e in d:
                    f = [e,d[e]]
                    cord[dire[f[0]]] += num[f[0]]*f[1]
                edu = math.sqrt(sum(map(lambda x : x * x, cord)))
                x_axis.append(cord[0]) ; y_axis.append(cord[1]) ; ed.append(edu)
        new = pd.DataFrame({ j + str('_x_axis'): x_axis, j + str('_y_axis'): y_axis, j  + str('_Euclidean'): ed})
        concat_data.append(new)
    df_all = pd.concat(concat_data,axis = 1)
    direction_features.extend(['Number']) 
    delete_fea = direction_features
    #delete Number and direction feature(string)
    complete_data_X = df_all.drop(delete_fea,axis = 1)
    complete_data_Y = df.loc[:,output]
    return complete_data_X, complete_data_Y

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def train_model():
    df = pd.read_csv(train_file.get())
    complete_data_X, complete_data_Y = datapreprocess(df)
    fixed_X = [i for i in complete_data_X.columns if i[6] == 'C']
    Input_A_tail = [df.columns[1:25].str.split('_')[i][2] for i in range(24)]
    output = ['Output_A1','Output_A2','Output_A3','Output_A4','Output_A5','Output_A6']
    model = []
    performance_r2 = []; performance_rmse = []
    for i,j in enumerate(output):
        xgb = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,colsample_bytree=1, max_depth=7)
        X_train = ['Input_A' + str(i+1) + '_' + k for k in Input_A_tail] ; X_train.extend(fixed_X)
        X_train = complete_data_X[X_train]
        xgb.fit(X_train, complete_data_Y[j])
        performance_r2.append(r2_score(complete_data_Y[j],xgb.predict(X_train)))
        performance_rmse.append(rmse(xgb.predict(X_train), complete_data_Y[j]))
        model.append(xgb)
    metrics = pd.DataFrame({"R_suqared":performance_r2,"RMSE":performance_rmse}).T
    metrics.columns = output
    print(metrics)
    return model

def totalError(model, complete_data_X, twenty_fea,test):
    df_test = pd.concat([pd.DataFrame(test).T.reset_index(),twenty_fea],axis = 1)
    test_x, test_y = datapreprocess(df_test)
    error_set = []
    predict_output = []
    output = ['Output_A1','Output_A2','Output_A3','Output_A4','Output_A5','Output_A6']
    fixed_X = [i for i in complete_data_X.columns if i[6] == 'C']
    df = pd.read_csv(train_file.get())
    Input_A_tail = [df.columns[1:25].str.split('_')[i][2] for i in range(24)]
    for i,j in enumerate(output):
        X_test = ['Input_A' + str(i+1) + '_' + k for k in Input_A_tail] ; X_test.extend(fixed_X)
        X_test = test_x[X_test]
        X_test = X_test.astype(float)
        predict_output.append(list(model[i].predict(X_test))[0])
    real_pred = pd.DataFrame(({"real":test_y.iloc[0,:],"pred":predict_output}))
    error = math.sqrt(((real_pred['real']-real_pred['pred'])**2).mean())
    return error, predict_output

def create_parent_generation(n_parents,twenty_variables):
    
    parents = []
    for i in range(n_parents):
        chromosome = [np.random.normal(twenty_variables.iloc[0,x],twenty_variables.iloc[1,x], size=1) for x in range(20)]
        parents.append(chromosome)
    return np.asarray(parents)

def random_combine(twenty_variables,parents,n_offspring):
    n_parents = len(parents)
    n_periods = len(parents[0])
    offspring = []
    for i in range(n_offspring):
        rdn = random.sample(range(n_parents),2)
        random_dad = parents[rdn[0]]
        random_mom = parents[rdn[1]]
                            
        dad_mask = np.random.randint(0,2, size = np.array(random_dad).shape)
        mom_mask = np.logical_not(dad_mask)
        
        child = np.add(np.multiply(random_dad, dad_mask), np.multiply(random_mom,mom_mask))
        offspring.append(child)
    return np.r_[parents,np.asarray(offspring)]

def mutate_parent(twenty_variables, parent, n_mutations):
    size = parent.shape[0]
    for i in range(n_mutations):
        rand1 = np.random.randint(0,size)
        parent[rand1,0] = np.random.normal(twenty_variables.iloc[0,rand1],twenty_variables.iloc[1,rand1], size=1)
    return parent

def mutate_gen(twenty_variables, parent_gen,n_mutations):
    mutated_parent_gen = []
    for parent in parent_gen:
        mutated_parent_gen.append(mutate_parent(twenty_variables, parent, n_mutations))
    return np.r_[parent_gen,np.asarray(mutated_parent_gen)]

def select_best(model, complete_data_X, parent_gen,one_test,n_best):
    twenty_features = ['Input_A6_024','Input_A3_016','Input_C_013','Input_A2_016','Input_A3_017',
                       'Input_C_050','Input_A6_001','Input_C_096','Input_A3_018','Input_A6_019',
                       'Input_A1_020','Input_A6_011','Input_A3_015','Input_C_046','Input_C_049',
                       'Input_A2_024','Input_C_058','Input_C_057','Input_A3_013','Input_A2_017']
    twenty_fea = [pd.DataFrame(parent_gen[i].T,columns = twenty_features)for i in range(len(parent_gen))]
    error = [totalError(model, complete_data_X, twenty_fea[i],one_test) for i in range(parent_gen.shape[0])]
    error_tmp = pd.DataFrame(error).sort_values(by = 0, ascending = True)
    selected_parents_idx = sorted(list(error_tmp.index.values[range(n_best)]))
    selected_parents = [parent for idx, parent in enumerate(parent_gen) if idx in selected_parents_idx]
    error_tmp.reset_index(drop=True)
    return np.array(selected_parents), np.array(error_tmp.iloc[range(n_best),:])

def gen_algo(model, complete_data_X, generation_size, n_iterations,test,twenty_variables):
    parent_gen = create_parent_generation(generation_size,twenty_variables)
    i = 0
    for i in range(n_iterations):
        parent_gen = random_combine(twenty_variables, parent_gen,n_offspring = generation_size)
        parent_gen = mutate_gen(twenty_variables, parent_gen, n_mutations = 1)
        parent_gen = select_best(model, complete_data_X, parent_gen, test, n_best = generation_size)[0]
        i=i+1
        
    best_child = select_best(model, complete_data_X, parent_gen, test, n_best = 1)
    return best_child

def Run():
    generation_size = int(g_num.get())
    n_iteration = int(ite_num.get())
    df = pd.read_csv(train_file.get())
    complete_data_X, complete_data_Y = datapreprocess(df)
    all_data = complete_data_X.agg(["mean", "std"], axis= 0)
    twenty_features = ['Input_A6_024','Input_A3_016','Input_C_013','Input_A2_016','Input_A3_017',
                       'Input_C_050','Input_A6_001','Input_C_096','Input_A3_018','Input_A6_019',
                       'Input_A1_020','Input_A6_011','Input_A3_015','Input_C_046','Input_C_049',
                       'Input_A2_024','Input_C_058','Input_C_057','Input_A3_013','Input_A2_017']
    output = ['Output_A1','Output_A2','Output_A3','Output_A4','Output_A5','Output_A6']
    twenty_variables = all_data.loc[:,twenty_features]
    test = pd.read_csv(test_file.get())
    test_output = test.iloc[range(2),test.shape[1]-6:test.shape[1]]
    model = train_model()
    answer = [(gen_algo(model, complete_data_X,generation_size,n_iteration,test.iloc[i,],twenty_variables)) for i in range(95)]
    best_answer = pd.DataFrame([answer[i][0][0].T[0] for i in range(len(answer))],columns = twenty_features)
    out = pd.DataFrame([answer[i][1][0][1] for i in range(len(answer))],columns=output)
    loss = pd.DataFrame([answer[i][1][0][0] for i in range(len(answer))])
    test_rmse = []
    test_rmse.append(rmse(out.iloc[:,i],test_output.iloc[:,i]) for i in range(6))
    test_rmse = pd.DataFrame(test_rmse)
    test_rmse.columns = output
    print("best answer:")
    print(best_answer)
    print("predicted output:")
    print(out)
    print("Loss:")
    print(loss)
    print("RMSE:")
    print(test_rmse)
    return best_answer, out, loss, test_rmse


window = tk.Tk()
window.title('GA')
window.geometry('300x200')
tk.Label(window, text="Training data file:").grid(row=0)
tk.Label(window, text="Testing data file:").grid(row=1)
train_path = tk.StringVar()
train_file = tk.Entry(window, show='', font=('Arial', 12))
train_file.grid(row=0, column=1)
test_path = tk.StringVar()
test_file = tk.Entry(window, show='', font=('Arial', 12))
test_file.grid(row=1, column=1)
tk.Label(window, text="Generation size:").grid(row=2)
tk.Label(window, text="Iteration:").grid(row=3)
g_num = tk.Entry(window, show='', font=('Arial', 12))
g_num.grid(row=2, column=1)
ite_num = tk.Entry(window, show='', font=('Arial', 12))
ite_num.grid(row=3, column=1)
t = tk.Button(window, text='Train Model', font=('Arial', 12), width=10, height=2, command=train_model)
t.grid(row=4, column=0)
r = tk.Button(window, text='Run', font=('Arial', 12), width=10, height=2, command=Run)
r.grid(row=4, column=1)
window.mainloop()


# In[ ]:




