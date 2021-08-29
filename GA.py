{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 啟發式演算法前的資料前處理跟XGBOOST"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 160,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import math\n",
    "import numpy as np\n",
    "from sklearn.cluster import KMeans\n",
    "from sklearn import cluster, metrics\n",
    "import matplotlib.pyplot as plt\n",
    "import itertools as it\n",
    "from sklearn.linear_model import LassoCV\n",
    "from sklearn.linear_model import Lasso\n",
    "import xgboost\n",
    "import random\n",
    "from sklearn.metrics import explained_variance_score\n",
    "from xgboost import plot_importance\n",
    "from matplotlib import pyplot\n",
    "from toggle_cell import toggle_code as hide_sloution"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 143,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv(\"0714train.csv\")\n",
    "def datapreprocess(df):\n",
    "    # data preprocess\n",
    "    twenty_features = ['Input_A6_024','Input_A3_016','Input_C_013','Input_A2_016','Input_A3_017',\n",
    "                       'Input_C_050','Input_A6_001','Input_C_096','Input_A3_018','Input_A6_019',\n",
    "                       'Input_A1_020','Input_A6_011','Input_A3_015','Input_C_046','Input_C_049',\n",
    "                       'Input_A2_024','Input_C_058','Input_C_057','Input_A3_013','Input_A2_017']\n",
    "    output = ['Output_A1','Output_A2','Output_A3','Output_A4','Output_A5','Output_A6']\n",
    "    # fetch 20 features and the add in the end\n",
    "    X_feature = pd.concat([df.drop(twenty_features,axis = 1),df.loc[:,twenty_features]],axis = 1).drop(output,axis = 1)\n",
    "    # select direction features\n",
    "    direction_features = ['Input_C_0' + str(i) for i in it.chain(range(15,39),range(63,83))]\n",
    "    # transform above direction features\n",
    "    concat_data = [X_feature]\n",
    "    num,dire = dict(),dict()\n",
    "    num['L'] = num['D'] = -1; num['R'] = num['U'] = 1; num['N'] = dire['N'] = 0\n",
    "    dire['L'] = dire['R'] = 0; dire['U'] = dire['D'] = 1\n",
    "    for j in direction_features:\n",
    "        x_axis = []; y_axis = []; ed = []\n",
    "        for i in range(X_feature.loc[:,j].shape[0]):\n",
    "            if X_feature.loc[:,j].isnull()[i] == True:\n",
    "                x_axis.append(np.nan); y_axis.append(np.nan); ed.append(np.nan)\n",
    "            else:\n",
    "                a = X_feature.loc[:,j][i].split(';')\n",
    "                b = [a[0],a[2]]\n",
    "                c = [float(a[1]),float(a[3])]\n",
    "                d = dict(zip(b, c))\n",
    "                cord = [0,0]\n",
    "                for e in d:\n",
    "                    f = [e,d[e]]\n",
    "                    cord[dire[f[0]]] += num[f[0]]*f[1]\n",
    "                edu = math.sqrt(sum(map(lambda x : x * x, cord)))\n",
    "                x_axis.append(cord[0]) ; y_axis.append(cord[1]) ; ed.append(edu)\n",
    "        new = pd.DataFrame({ j + str('_x_axis'): x_axis, j + str('_y_axis'): y_axis, j  + str('_Euclidean'): ed})\n",
    "        concat_data.append(new)\n",
    "    df_all = pd.concat(concat_data,axis = 1)\n",
    "    direction_features.extend(['Number']) \n",
    "    delete_fea = direction_features\n",
    "    #delete Number and direction feature(string)\n",
    "    complete_data_X = df_all.drop(delete_fea,axis = 1)\n",
    "    complete_data_Y = df.loc[:,output]\n",
    "    return complete_data_X, complete_data_Y\n",
    "complete_data_X, complete_data_Y = datapreprocess(df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 144,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "# model\n",
    "fixed_X = [i for i in complete_data_X.columns if i[6] == 'C']\n",
    "Input_A_tail = [df.columns[1:25].str.split('_')[i][2] for i in range(24)]\n",
    "output = ['Output_A1','Output_A2','Output_A3','Output_A4','Output_A5','Output_A6']\n",
    "model = []\n",
    "for i,j in enumerate(output):\n",
    "    xgb = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,colsample_bytree=1, max_depth=7)\n",
    "    X_train = ['Input_A' + str(i+1) + '_' + k for k in Input_A_tail] ; X_train.extend(fixed_X)\n",
    "    X_train = complete_data_X[X_train]\n",
    "    xgb.fit(X_train, complete_data_Y[j])\n",
    "    model.append(xgb)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 啟發式演算法範例 還沒做"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 145,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Input_A6_024</th>\n",
       "      <th>Input_A3_016</th>\n",
       "      <th>Input_C_013</th>\n",
       "      <th>Input_A2_016</th>\n",
       "      <th>Input_A3_017</th>\n",
       "      <th>Input_C_050</th>\n",
       "      <th>Input_A6_001</th>\n",
       "      <th>Input_C_096</th>\n",
       "      <th>Input_A3_018</th>\n",
       "      <th>Input_A6_019</th>\n",
       "      <th>Input_A1_020</th>\n",
       "      <th>Input_A6_011</th>\n",
       "      <th>Input_A3_015</th>\n",
       "      <th>Input_C_046</th>\n",
       "      <th>Input_C_049</th>\n",
       "      <th>Input_A2_024</th>\n",
       "      <th>Input_C_058</th>\n",
       "      <th>Input_C_057</th>\n",
       "      <th>Input_A3_013</th>\n",
       "      <th>Input_A2_017</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>0.005862</td>\n",
       "      <td>-0.030000</td>\n",
       "      <td>0.004687</td>\n",
       "      <td>-0.029566</td>\n",
       "      <td>-0.028121</td>\n",
       "      <td>0.005289</td>\n",
       "      <td>0.012716</td>\n",
       "      <td>0.011420</td>\n",
       "      <td>-0.027572</td>\n",
       "      <td>-0.029480</td>\n",
       "      <td>0.521387</td>\n",
       "      <td>0.004517</td>\n",
       "      <td>0.005315</td>\n",
       "      <td>0.000896</td>\n",
       "      <td>0.000634</td>\n",
       "      <td>0.006060</td>\n",
       "      <td>0.008688</td>\n",
       "      <td>0.011549</td>\n",
       "      <td>0.005037</td>\n",
       "      <td>-0.027919</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>0.013393</td>\n",
       "      <td>0.013641</td>\n",
       "      <td>0.000647</td>\n",
       "      <td>0.013257</td>\n",
       "      <td>0.013284</td>\n",
       "      <td>0.002891</td>\n",
       "      <td>0.035926</td>\n",
       "      <td>0.007194</td>\n",
       "      <td>0.012962</td>\n",
       "      <td>0.013199</td>\n",
       "      <td>0.796117</td>\n",
       "      <td>0.002353</td>\n",
       "      <td>0.030754</td>\n",
       "      <td>0.000333</td>\n",
       "      <td>0.000284</td>\n",
       "      <td>0.013866</td>\n",
       "      <td>0.005639</td>\n",
       "      <td>0.007660</td>\n",
       "      <td>0.001645</td>\n",
       "      <td>0.013821</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      Input_A6_024  Input_A3_016  Input_C_013  Input_A2_016  Input_A3_017  \\\n",
       "mean      0.005862     -0.030000     0.004687     -0.029566     -0.028121   \n",
       "std       0.013393      0.013641     0.000647      0.013257      0.013284   \n",
       "\n",
       "      Input_C_050  Input_A6_001  Input_C_096  Input_A3_018  Input_A6_019  \\\n",
       "mean     0.005289      0.012716     0.011420     -0.027572     -0.029480   \n",
       "std      0.002891      0.035926     0.007194      0.012962      0.013199   \n",
       "\n",
       "      Input_A1_020  Input_A6_011  Input_A3_015  Input_C_046  Input_C_049  \\\n",
       "mean      0.521387      0.004517      0.005315     0.000896     0.000634   \n",
       "std       0.796117      0.002353      0.030754     0.000333     0.000284   \n",
       "\n",
       "      Input_A2_024  Input_C_058  Input_C_057  Input_A3_013  Input_A2_017  \n",
       "mean      0.006060     0.008688     0.011549      0.005037     -0.027919  \n",
       "std       0.013866     0.005639     0.007660      0.001645      0.013821  "
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "twenty_features = ['Input_A6_024','Input_A3_016','Input_C_013','Input_A2_016','Input_A3_017',\n",
    "                   'Input_C_050','Input_A6_001','Input_C_096','Input_A3_018','Input_A6_019',\n",
    "                   'Input_A1_020','Input_A6_011','Input_A3_015','Input_C_046','Input_C_049',\n",
    "                   'Input_A2_024','Input_C_058','Input_C_057','Input_A3_013','Input_A2_017']\n",
    "all_data = complete_data_X.agg([\"mean\", \"std\"], axis= 0)\n",
    "twenty_variables = all_data.loc[:,twenty_features]\n",
    "twenty_variables"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 146,
   "metadata": {},
   "outputs": [],
   "source": [
    "def create_parent_generation(n_parents):\n",
    "    parents = []\n",
    "    for i in range(n_parents):\n",
    "        chromosome = [np.random.normal(twenty_variables.iloc[0,x],twenty_variables.iloc[1,x], size=1) for x in range(20)]\n",
    "        parents.append(chromosome)\n",
    "    return np.asarray(parents)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 147,
   "metadata": {},
   "outputs": [],
   "source": [
    "parents = create_parent_generation(10)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 2 — How to evaluate the Genetic Algorithm’s solution?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 148,
   "metadata": {},
   "outputs": [],
   "source": [
    "test = pd.read_csv(\"0728test.csv\")\n",
    "twenty_fea = [pd.DataFrame(parents[i].T,columns = twenty_features)for i in range(len(parents))]\n",
    "def totalError(twenty_fea,test):\n",
    "    df = pd.concat([pd.DataFrame(test).T,twenty_fea],axis = 1)\n",
    "    test_x, test_y = datapreprocess(df)\n",
    "    error_set = []\n",
    "    for i,j in enumerate(output):\n",
    "        X_test = ['Input_A' + str(i+1) + '_' + k for k in Input_A_tail] ; X_test.extend(fixed_X)\n",
    "        X_test = test_x[X_test]\n",
    "        X_test = X_test.astype(float)\n",
    "        predict_output = model[i].predict(X_test)\n",
    "        error = (test_y[j] - predict_output)[0]**2\n",
    "        error_set.append(error)\n",
    "    return sum(error_set)/6"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 149,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.concat([pd.DataFrame(one_test).T,twenty_fea[0]],axis = 1)\n",
    "test_x, test_y = datapreprocess(df)\n",
    "X_test = ['Input_A' + str(i+1) + '_' + k for k in Input_A_tail] ; X_test.extend(fixed_X)\n",
    "X_test = test_x[X_test]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 3 — How to code Mating (Cross-Over) for the Genetic Algorithm?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 112,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "\n",
       "<form action=\"javascript:code_toggle()\"><input type=\"submit\" id=\"toggleButton\" value=\"Show Sloution\"></form>\n"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "\"\"\"\n",
    "for each iteration, select randomly two parents and make a random combination of those two parents\n",
    "by applying a randomly generated yes/no mask to the two selected parents\n",
    "\"\"\"\n",
    "def random_combine(parents,n_offspring):\n",
    "    n_parents = len(parents)\n",
    "    n_periods = len(parents[0])\n",
    "    \n",
    "    offspring = []\n",
    "    for i in range(n_offspring):\n",
    "        rdn = random.sample(range(n_parents),2)\n",
    "        random_dad = parents[rdn[0]]\n",
    "        random_mom = parents[rdn[1]]\n",
    "                            \n",
    "        dad_mask = np.random.randint(0,2, size = np.array(random_dad).shape)\n",
    "        mom_mask = np.logical_not(dad_mask)\n",
    "        \n",
    "        child = np.add(np.multiply(random_dad, dad_mask), np.multiply(random_mom,mom_mask))\n",
    "        offspring.append(child)\n",
    "    return np.r_[parents,np.asarray(offspring)]\n",
    "hide_sloution()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 4 — How to code Mutations for the genetic algorithm?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 113,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "\n",
       "<form action=\"javascript:code_toggle()\"><input type=\"submit\" id=\"toggleButton\" value=\"Show Sloution\"></form>\n"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "def mutate_parent(parent, n_mutations):\n",
    "    size = parent.shape[0]\n",
    "    \n",
    "    for i in range(n_mutations):\n",
    "        \n",
    "        rand1 = np.random.randint(0,size)\n",
    "        \n",
    "        parent[rand1,0] = np.random.normal(twenty_variables.iloc[0,rand1],twenty_variables.iloc[1,rand1], size=1)\n",
    "        \n",
    "    return parent\n",
    "\n",
    "def mutate_gen(parent_gen,n_mutations):\n",
    "    mutated_parent_gen = []\n",
    "    for parent in parent_gen:\n",
    "        mutated_parent_gen.append(mutate_parent(parent, n_mutations))\n",
    "    return np.r_[parent_gen,np.asarray(mutated_parent_gen)]\n",
    "hide_sloution()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 5 — How to define Selection for the Genetic Algorithm?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 163,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "\n",
       "<form action=\"javascript:code_toggle()\"><input type=\"submit\" id=\"toggleButton\" value=\"Show Sloution\"></form>\n"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "def select_best(parent_gen,one_test,n_best):\n",
    "    twenty_fea = [pd.DataFrame(parent_gen[i].T,columns = twenty_features)for i in range(len(parent_gen))]\n",
    "    error = [totalError(twenty_fea[i],one_test) for i in range(parent_gen.shape[0])]\n",
    "    error_tmp = pd.DataFrame(error).sort_values(by = 0, ascending = True).reset_index(drop=True)\n",
    "    selected_parents_idx = range(n_best)\n",
    "    selected_parents = [parent for idx, parent in enumerate(parent_gen) if idx in selected_parents_idx]\n",
    "    return np.array(selected_parents)\n",
    "hide_sloution()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 6 — How to define iterations and stopping for the Genetic Algorithm?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 164,
   "metadata": {},
   "outputs": [],
   "source": [
    "\"\"\"\n",
    "the overall function\n",
    "\"\"\"\n",
    "def gen_algo(generation_size, n_iterations, test):\n",
    "    parent_gen = create_parent_generation(generation_size)\n",
    "    i = 0\n",
    "    for i in range(n_iterations):\n",
    "        parent_gen = random_combine(parent_gen, n_offspring = generation_size)\n",
    "        parent_gen = mutate_gen(parent_gen, n_mutations = 1)\n",
    "        parent_gen = select_best(parent_gen, test, n_best = generation_size)\n",
    "        i=i+1\n",
    "        \n",
    "    best_child = select_best(parent_gen, test, n_best = 1)\n",
    "    return best_child"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "answer = [(gen_algo(50,100,test.iloc[i,])) for i in range(95)]"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
