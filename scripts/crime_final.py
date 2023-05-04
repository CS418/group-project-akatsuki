# %%
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import RandomizedSearchCV

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

import catboost
from catboost import *
from catboost.utils import eval_metric

import textwrap

import shap
shap.initjs()

import jenkspy

from BorutaShap import BorutaShap
mpl.rcParams.update({'font.size':13})

# %%
df = pd.read_csv("./Data/All_Parameter_CSV.csv", index_col=0).sort_values(by='NAME10').drop_duplicates('NAME10').reset_index(drop=True)
svi_2020 = pd.read_csv('Data/svi_2020.csv', index_col=0).sort_values(by='NAME10')

# %%
df.drop(['Socioeconomic\xa0Status', 'Household Composition & Disability', 'Minority Status & Language',
       'Housing Type & Transportation', 'Diabetes'], axis=1, inplace=True)

# %%
df = df.merge(right=svi_2020, left_on='NAME10', right_on='NAME10', how='inner')

# %%
df

# %%
df_minorities = df[['NAME10', 'EP_AFAM', 'EP_HISP', 'EP_ASIAN', 'EP_AIAN']]
df_svi = df[['NAME10', 'SVI']]

# %%
df=df.rename(columns={"Solar_DNI":"Direct Normal Irradiation"})

# %%
rearranged_columns = ['TRACTCE10',
                      'GEOID10',
                      'NAME10',
                      'NAMELSAD10',
                      'Long',
                      'Lat',
                      'State',
                      'County',
                      'Smoking',
                      'Binge Drinking',
                      'Physical Inactivity',
                      'Sleep<7hours',
                      'Opioid Overdose',
                      'High Cholesterol',
                      'Obesity',
                      'High Blood Pressure',
                      'Depression',
                      'Kidney Diseases',
                      'Land Surface Temperature',
                      'Air Temperature',
                      'Direct Normal Irradiation',
                      'Wind Speed',
                      'Precipitation',
                      'Socioeconomic Status',
                      'Household Characteristics',
                      'Racial & Ethnic Minority Status',
                      'Housing Type & Transportation',
                      'Aerosol ',
                      'Nitrogen Dioxide ',
                      'Carbon Monoxide ',
                      'Sulfur Dioxide ',
                      'Formaldehyde ',
                      'Ozone ',
                      'Crime']

# %%
df_all_param = df[rearranged_columns]

# %%
df_parameters = df_all_param.iloc[:,8:]

# %%
sns.pairplot(df_parameters[['Smoking', 'Binge Drinking', 'Physical Inactivity', 
                            'Sleep<7hours', 'Opioid Overdose', 'Crime']])
plt.savefig('EDA Behavior and Crime.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
sns.pairplot(df_parameters[['High Cholesterol', 'Obesity', 'High Blood Pressure',
                            'Depression', 'Kidney Diseases', 'Crime']])
plt.savefig('EDA Health and Crime.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
sns.pairplot(df_parameters[['Land Surface Temperature',
       'Air Temperature', 'Direct Normal Irradiation', 'Wind Speed',
       'Precipitation', 'Crime']])
plt.savefig('EDA Weather and Crime.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
df_parameters.columns

# %%
import textwrap

g = sns.pairplot(df_parameters[['Socioeconomic Status', 'Household Characteristics',
       'Racial & Ethnic Minority Status', 'Housing Type & Transportation', 'Crime']])

for ax in g.axes.flat:
    ax.set_xlabel("\n".join(textwrap.wrap(ax.get_xlabel(), width=20)))
    ax.set_ylabel("\n".join(textwrap.wrap(ax.get_ylabel(), width=20)))

plt.savefig('EDA SVI and Crime.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
sns.pairplot(df_parameters[['Aerosol ', 'Nitrogen Dioxide ',
       'Carbon Monoxide ', 'Sulfur Dioxide ', 'Formaldehyde ', 'Ozone ', 'Crime']])
plt.savefig('EDA Air Pollution and Crime.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
pearson_corr = df_parameters.corr(method='pearson')
kendall_corr = df_parameters.corr(method='kendall')

# corr = df_parameters
corr = pd.DataFrame()
corr['Pearson'] = pearson_corr['Crime'].round(2)
corr['Kendall'] = kendall_corr['Crime'].round(2)


# %%
corr = corr.iloc[:25,:]
corr.sort_values('Pearson', inplace=True, ascending=False)

# %%
cmap = sns.color_palette('coolwarm', as_cmap=True)

# %%
%matplotlib inline
plt.figure(figsize=(7, 8))
p = sns.heatmap(corr, 
                annot=True, 
                cmap=cmap,
                vmax=1,
                vmin=-1,
                fmt='.2f',
                annot_kws={'color':'black'},
                cbar_kws={'shrink': 0.6,
                          'label': 'Correlation Coefficient',
                          'ticks': [-1.0, -0.5, 0.0, 0.5, 1.0]
                         }
               )
# plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig("1. Correlation.jpg", dpi=300)
plt.show()

# %%
# Normalizing the Diabetes with max value
# df_parameters['Crime'] = MinMaxScaler().fit_transform(df[['Crime']]) 

df_parameters['Crime'] = np.log(df['Crime'])
#Sorting the dataframe with the normalized diabetes values and sampling 350 datapoints
df_parameters.sort_values(by='Crime', inplace=True)
df_train = df_parameters.sample(400, random_state=1234)

# %%
#Check the histogram/kde of normalized values
df_train['Crime'].plot(kind='hist')

# %%
#Splitting the data into independent and dependent variables
X_, y_ = df_train.iloc[:,:-1], df_train.iloc[:,-1]

#Training and validation set split
x_train, x_test, y_train, y_test = train_test_split(X_, y_, test_size=0.4, shuffle=True, random_state=14)

# %%
lr = LinearRegression()
lr.fit(x_train, y_train)
y_test_pred = lr.predict(x_test)
print("R2: ", np.round(r2_score(y_test, y_test_pred), 2))
print("MSE: ", np.round(mean_squared_error(y_test, y_test_pred), 2))
print("MAE:", np.round(mean_absolute_error(y_test, y_test_pred), 2))
print("MAPE:", np.round(mean_absolute_percentage_error(y_test, y_test_pred), 2))

# %%
rf = RandomForestRegressor()
rf.fit(x_train, y_train)
y_test_pred = rf.predict(x_test)
print("R2: ", np.round(r2_score(y_test, y_test_pred), 2))
print("MSE: ", np.round(mean_squared_error(y_test, y_test_pred), 2))
print("MAE:", np.round(mean_absolute_error(y_test, y_test_pred), 2))
print("MAPE:", np.round(mean_absolute_percentage_error(y_test, y_test_pred), 2))

# %%
catboost = CatBoostRegressor(verbose=False)
catboost.fit(x_train, y_train)
y_test_pred = catboost.predict(x_test)
print("R2: ", np.round(r2_score(y_test, y_test_pred), 2))
print("MSE: ", np.round(mean_squared_error(y_test, y_test_pred), 2))
print("MAE:", np.round(mean_absolute_error(y_test, y_test_pred), 2))
print("MAPE:", np.round(mean_absolute_percentage_error(y_test, y_test_pred), 2))

# %%
param_distributions = {
#     'iterations': [1200],
# #     'depth': [3, 4, 5],
# #     'min_data_in_leaf':[9, 11, 13],
#     'learning_rate': [0.005],
#     'l2_leaf_reg': [5, 7],
# #     'colsample_bylevel': [0.3, 0.4],
#     'bagging_temperature': [1.5],
# #     'random_strength': [1],
#     'loss_function': ['MAE'],
#     'od_type': ["IncToDec", "Iter"],
#     'od_wait':[30, 40 ,50]
}

catboost = CatBoostRegressor(verbose=False)
# Randomized grid search
random_search = RandomizedSearchCV(catboost, param_distributions, n_iter=10, cv=5, n_jobs=-1, random_state=42)
random_search.fit(x_train, y_train)

# %%
model = random_search.best_estimator_
model.fit(x_train, y_train, eval_set=(x_test, y_test), )

# %%
print(random_search.best_params_)
# print(model.get_all_params())

# %%
r2_score(y_train, model.predict(x_train))

# %%
y_test_pred = model.predict(x_test)
r2_score(y_test, y_test_pred)

# %%
y_train.describe()

# %%
y_test.describe()

# %%
d = df.copy(deep=True)

# %%
d = d.drop(columns=['TRACTCE10', 'GEOID10', 'NAMELSAD10', 'Long', 'Lat', 'State', 'County','EP_AFAM', 'EP_HISP',
       'EP_ASIAN', 'EP_AIAN', 'SVI'], axis=1)

# %%
d['Crime'] = np.log(d['Crime'])

# %%
d['Crime_pred'] = model.predict(d.iloc[:,1:-1])

# %%
regression_roc_auc_score(approxes=d['Crime_pred'], targets=d['Crime'])

# %%
r2_score(y_pred=d['Crime_pred'], y_true=d['Crime'])

# %%
d['Crime_pred'].describe()

# %%
d['Crime'].describe()

# %%
d[['Crime','Crime_pred']].plot(kind='kde')

# %%
X_, y, y_hat = d.iloc[:,:-2], d.iloc[:,-2], d.iloc[:,-1]
X = X_.iloc[:, 1:]
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

shap.force_plot(explainer.expected_value, shap_values, X)

# %%
exclude = df_train.index.to_list()

mask = ~y.index.isin(exclude)
Y_test = y[mask]
Y_hat = y_hat[mask]


# %%
#MSE
mean_squared_error(y_true=Y_test,
                   y_pred=Y_hat
                  )

# %%
#MAE
mean_absolute_error(y_true=Y_test,
                    y_pred=Y_hat
                   )

# %%
#MAPE
mean_absolute_percentage_error(y_true=Y_test,
                               y_pred=Y_hat
                              )

# %%
#R2 Score
r2_score(y_true=Y_test, y_pred=Y_hat)

# %%
d.reset_index(drop=True, inplace=True)

# %%
shap_df = pd.DataFrame(shap_values, columns=X.columns)

# %%
shap.summary_plot(shap_values, 
                  X, 
                  plot_type='bar',
                  color_bar=True,
                  color="#3182bd",
                  show=False,
                  max_display=25
                 )
# plt.title("Parameter contribution for High Diabetes")
plt.tight_layout()
plt.xticks(color='black')
plt.yticks(color='black')
plt.savefig(f'2. Mean_shap_values.jpg', dpi=300, bbox_inches='tight')
plt.show()

# %%
shap.summary_plot(shap_values, X, cmap=cmap, show=False, max_display=25, color_bar_label='Parameter Value')
plt.tight_layout()
plt.xticks(fontsize=13, color='black')
plt.yticks(fontsize=13, color='black')
plt.savefig(f'3. Mean_shap_values_beeplot.jpg', dpi=300, bbox_inches='tight')
plt.show()

# %%
df_plot = X_.join(shap_df, rsuffix='_shap').set_index('NAME10')
df_plot

# %%
df_plot['proba'] = np.round(model.predict(X), decimals=2)

# %%
df_plot.head()

# %%
df_plot['proba'].describe()

# %%
df_plot['proba'] = MinMaxScaler().fit_transform(df_plot[['proba']])

# %%
from matplotlib.ticker import FormatStrFormatter

fig, axs = plt.subplots(nrows=5, ncols=5, figsize=(21, 21))
ax = axs.ravel()
columns = x_train.columns

sci_notation_variables = ['Nitrogen Dioxide ', 'Carbon Monoxide ', 'Sulfur Dioxide ',
                          'Formaldehyde ', 'Ozone ']

for i, col in enumerate(columns):
    
    wrapped_title = "\n".join(textwrap.wrap(f"{col}", width=25))
    ax[i].set_title(wrapped_title, fontdict={'fontsize':20}, y=1.12)
#     ax2.title.set_position([.5, 1.05])

    p = ax[i].scatter(data=df_plot, x=col, y=f"{col}_shap", c="proba", cmap=cmap)
    
    ax[i].set(xlabel=None, ylabel=None, label=None)
    ax[i].set_xticks(np.linspace(np.min(df_plot[col]), np.max(df_plot[col]), 5))
    plt.setp(ax[i].get_xticklabels(), rotation=45, horizontalalignment='right')
    ax[i].tick_params(axis='x', labelsize=19)
    
    if col in sci_notation_variables:
        ax[i].ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
        
    else:
        ax[i].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    
    ax[i].tick_params(axis='y', labelsize=19)
    ax[i].set_yticks(np.linspace(np.min(df_plot[f"{col}_shap"]), np.max(df_plot[f"{col}_shap"]), 5))
    ax[i].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    ax[i].set_box_aspect(1)
    
    
    
    
# fig.suptitle("SHAP Values for Parameters", fontsize=30)
fig.text(0.5, 0.0, 'Parameter Value', ha='center', va='center', fontsize=25)
fig.text(0.0, 0.5, 'SHAP Value', ha='center', va='center', rotation='vertical', fontsize=25)


# cax = fig.add_axes([1, 0.1, 0.02, 0.8])
# cb = ColorbarBase(cax, cmap=cmap, orientation='vertical', ticks=np.linspace(0, 1, 5))
# cb.ax.tick_params(labelsize=18)

# cbar = fig.colorbar(p, ax=axs, cmap=cmap, shrink=0.6, anchor=(2.4,0.5))
# cbar = fig.colorbar(p, ax=axs, cmap=cmap, shrink=0.4, anchor=(2.4,0.5))
# cbar.set_label('Crime Vulnerability', fontsize=25, loc='center', labelpad=15)
# cbar.ticks = ticks=np.linspace(0, 1, 5) 
# cbar.ax.tick_params(labelsize=18)
fig.tight_layout(pad=0.06)
# plt.axis('square')
plt.xticks(color='black')
plt.yticks(color='black')
plt.savefig(f'4. SHAP values for individual Parameters.jpg', dpi=300, bbox_inches='tight')
plt.show()

# %%
fig, axs = plt.subplots(nrows=5, ncols=5, figsize=(20, 20))
ax = axs.ravel()
columns = x_train.columns

for i, col in enumerate(columns):
    
    wrapped_title = "\n".join(textwrap.wrap(f"{col}", width=25))
    ax[i].set_title(wrapped_title, fontdict={'fontsize':20}, y=1.05)
    
    ax[i].set(xlabel=None, ylabel=None, label=None)
    ax[i].set_xticks(np.linspace(np.min(df_plot[col]), np.max(df_plot[col]), 5))
    ax[i].tick_params(axis='x', labelsize=19)
    plt.setp(ax[i].get_xticklabels(), rotation=45, horizontalalignment='right')
    
    if col in sci_notation_variables:
        ax[i].ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
        
    else:
        ax[i].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    ax[i].set_yticks(np.linspace(0, 1, 5))
    ax[i].tick_params(axis='y', labelsize=19)
    ax[i].yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))
    
    ax[i].set_box_aspect(1)
    
    x, y = df_plot[col], df_plot['proba']
    m, b = np.polyfit(x, y, 1)
    
#     p = ax[i].scatter(data=df_plot, x=col, y="proba", c="proba", cmap=cmap)
    ax[i].scatter(data=df_plot, x=col, y="proba", alpha=0.5, edgecolors='white')
    ax[i].plot(x, m*x + b, color='red', )
    
    
# fig.suptitle("Diabetes Vulnerability v/s Parameters", fontsize=30, y=1.01)
fig.text(0.5, 0.0, 'Parameter Value', ha='center', va='center', fontsize=25)
fig.text(0.0, 0.5, 'Crime Vulnerability', ha='center', va='center', rotation='vertical', fontsize=25)

# cbar = fig.colorbar(p, ax=axs, cmap=cmap, shrink=0.6, anchor=(2.4,0.5) )
# cbar.set_label('Predicted Proba', fontsize=25)
# cbar.ax.tick_params(labelsize=18)
# cbar.ticks = ticks=np.linspace(0,1, 5) 
fig.tight_layout(pad=0.06)
plt.savefig(f'5. Vulnerability values vs Parameters.jpg', dpi=300, bbox_inches='tight')
plt.show()

# %%
vul_breaks = jenkspy.jenks_breaks(d['Crime_pred'], 5)

d['Vulnerability_Class'] = pd.cut(d['Crime_pred'], 
                                  bins=vul_breaks,
                                  labels=['Very Low', 'Low', 'Moderate', 'High', 'Very High'],
                                  include_lowest=True)

d['Vulnerability_ClassNum'] = pd.cut(d['Crime_pred'], 
                                     bins=vul_breaks,
                                     labels=range(1, len(vul_breaks)),
                                     include_lowest=True)

prev_breaks = jenkspy.jenks_breaks(d['Crime'], 5)

d['Crime_Class'] = pd.cut(d['Crime'],
                             bins=prev_breaks,
                             labels=['Very Low', 'Low', 'Moderate', 'High', 'Very High'],
                             include_lowest=True)

d['Crime_ClassNum'] = pd.cut(d['Crime'],
                                bins=prev_breaks,
                                labels=range(1, len(prev_breaks)),
                                include_lowest=True)

# %%
d[d['Vulnerability_Class'].isin(['High', 'Very High', 'Moderate'])]['Crime_Class'].value_counts()

# %%
def annotate_fig (fig):
  for p in fig.patches:
      fig.annotate('{:.0f}'.format(p.get_height()), 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', 
                    xytext = (0, 9), 
                    textcoords = 'offset points')

# %%
plt.figure(figsize=(10,8))
vulnerability_val_counts = d['Vulnerability_Class'].value_counts(sort=False)

colors = sns.color_palette(['#1b62a5', '#5592cc', '#f9ad10', '#cd5258', '#e00021'])

plot = sns.barplot(x=vulnerability_val_counts.index, y=vulnerability_val_counts.values, palette=colors)
annotate_fig(plot)
plt.xticks(rotation=0)
plt.xlabel('Crime Vulnerability Class', labelpad=5)
plt.ylabel('No. of Census Tracts')
plt.tight_layout()
plt.savefig(f'Diabetes Vulnerability Classes barplot.jpg', dpi=300)
plt.show()

# %%
plt.figure(figsize=(10,8))

prevelence_val_counts= d['Crime_Class'].value_counts(sort=False)

colors = sns.color_palette(['#1b62a5', '#5592cc', '#f9ad10', '#cd5258', '#e00021'])

plot = sns.barplot(x=prevelence_val_counts.index, y=prevelence_val_counts.values, palette=colors)
annotate_fig(plot)
plt.xticks(rotation=0)
plt.xlabel('Crime Prevalence Class', labelpad=5)
plt.ylabel('No. of Census Tracts')
plt.tight_layout()
plt.savefig('Crime Prevalence Classes barplot.jpg', dpi=300)
plt.show()

# %%
import pickle
model.save_model(f'Regressor_model')

with open(f'SHAP_explainer.pkl', 'wb') as f:
    pickle.dump(explainer, f)

# %%
d

# %%
d.to_csv('AllParameters and Vulnerability Classes.csv')
pd.read_csv('AllParameters and Vulnerability Classes.csv', index_col=0)

# %%
df_cvi = d[['NAME10', 'Crime_pred', 'Vulnerability_Class', 'Vulnerability_ClassNum']]
df_svi = df_svi.merge(df_cvi, left_on='NAME10', right_on='NAME10')

# %%
df_minorities = df_minorities.merge(df_cvi, left_on='NAME10', right_on='NAME10')

# %%
def getClassforCols(df_, cols):
    df = df_.copy(deep=True)
    n_breaks=5
    breaks = np.linspace(0, 100, n_breaks+1)
    for col in cols:
#         df[col] = MinMaxScaler().fit_transform(df[[col]])
        if col != 'EP_AIAN1':
            df[f'{col}_Class'] = pd.cut(df[col],
                                        bins=breaks,
                                        labels=['Very Low', 'Low', 'Moderate', 'High', 'Very High'],
                                        include_lowest=True,
                                        duplicates='drop'
                                       )

            df[f'{col}_ClassNum'] = pd.cut(df[col],
                                           bins=breaks,
                                           labels=range(1, n_breaks+1),
                                           include_lowest=True,
                                           duplicates='drop'
                                          )

        else:
            breaks = (-1, 0, 100)
            df[f'{col}_Class'] = pd.cut(df[col],
                                        bins=breaks,
                                        labels=['Not Present', 'Present'],
                                        include_lowest=True,
                                        duplicates='drop'
                                       )

            df[f'{col}_ClassNum'] = pd.cut(df[col],
                                           bins=breaks,
                                           labels=[0, 1],
                                           include_lowest=True,
                                           duplicates='drop'
                                          )
        
    return df

# %%
cols_to_break = ['EP_AFAM', 'EP_HISP', 'EP_ASIAN', 'EP_AIAN']
# cols_to_break = ['E_AFAM', 'E_HISP', 'E_ASIAN', 'E_AIAN']

df_minorities_Class = getClassforCols(df_minorities, cols_to_break)

# %%
col_broken = [f"{col}_Class" for col in cols_to_break]

dfs = []
for i in range(len(col_broken)):
    a = df_minorities_Class.groupby(by=['Vulnerability_Class', col_broken[i]])[cols_to_break[i]].count()
    dfs.append(a)

# %%
colors = ['#1b62a5', '#5592cc', '#f9ad10', '#cd5258', '#e00021']
titles = ['Black/African American', 'Hispanic or Latino', 'Asian American', 'Native American or Alska Native']
legend_labels = ["Very Low (0-20)", "Low (20-40)", "Moderate (40-60)", "High (60-80)", "Very High (80-100)"]
for i, col in enumerate(col_broken):
    fig, ax = plt.subplots(figsize=(11, 9))
    if col == 'EP_AIAN_Class':
        unstacked = dfs[i].unstack(level=0)
        stacked_percent = (unstacked / unstacked.sum()) * 100
        
    else:
        unstacked = dfs[i].unstack()
        unstacked.index = pd.Categorical(unstacked.index, ordered=True, categories=['Very Low', 'Low', 'Moderate', 'High', 'Very High'])
        unstacked = unstacked.sort_index()
        stacked_percent = unstacked.div(unstacked.sum(axis=1), axis=0) * 100
    
    stacked_percent.plot(kind='bar', stacked=True, fontsize=13, rot=0, color=colors, ax=ax, edgecolor='white')
    for j, (name, values) in enumerate(stacked_percent.iterrows()):
        total = values.sum()
        for k, value in enumerate(values):
            if value > 1.5:
                percent = value / total * 100
                ax.text(j, sum(values[:k+1]) - value/2, f'{percent:.1f}%', ha='center', va='center', fontdict={'fontsize':9}, color='black')
    ax.legend(title=f"Minority population Class", labels=legend_labels, fontsize=8, title_fontsize=9, bbox_to_anchor=(1., 0.6))
    fig.set_size_inches(10,7, forward=False)
    plt.title(titles[i])
    plt.xlabel('Crime Vulnerability Class')
    plt.ylabel('% of Census Tract')
    plt.tight_layout()
    fig.savefig(f'{col}.jpg', bbox_inches='tight',dpi=300)
    plt.show()

# %%


# %%


# %%


# %% [markdown]
# ## Exporting the classes of vulnerability

# %%
d = d.sort_values("NAME10").reset_index(drop=True)

# %%
d.to_csv('AllParameters and Vulnerability Classes.csv')
pd.read_csv('AllParameters and Vulnerability Classes.csv', index_col=0)

# %%
df_meta = df.iloc[:,:8].sort_values('NAME10').reset_index(drop=True)

# %%
df_VulClass = d[['NAME10', 'Crime_pred', 'Vulnerability_Class', 'Vulnerability_ClassNum']]

# %%
df_cencsusVulnerability = df_meta.join(other=df_VulClass, rsuffix='r')
df_cencsusVulnerability

# %%
df_cencsusVulnerability.drop(labels='NAME10r', inplace=True, axis=1)
df_cencsusVulnerability.to_csv('Vulnerability Classes and CensusTractInfo.csv')
pd.read_csv('Vulnerability Classes and CensusTractInfo.csv', index_col=0)


# %%
model = CatBoostRegressor()

# %%
model.load_model('Final/Regressor_model')

# %% [markdown]
# ## Implementing SHAP on individual Census Tracts

# %%
df_all_params = pd.read_csv('Final/Data/AllParameters and Vulnerability Classes.csv', index_col=0)
df_all_params.head()

# %%
X, y = d.iloc[:, :26], df_all_params['Diabetes_pred']
df_renamed=X.rename(columns={"Household Composition & Disability":"Household Comp. & Disability"})

# %%
X[X['NAME10'].isin([611, 318, 8362, 3817, 5001, 104, 2831, 8233.04, 6810, 3406, 2604])]

# %%
import pickle
with open(f'Final/SHAP_explainer.pkl', 'rb') as f:
    explainer = pickle.load(f)

# %%
model = CatBoostRegressor()
model.load_model('Final/Regressor_model')

# %%
X.columns

# %%
feature_d = {'Household Composition & Disability': 'Household Comp. & Disability'}


# %%
X['Socioeconomic Status']

# %%
from shapash import SmartExplainer
import plotly.io as pio

# %%
xpl = SmartExplainer(model=model, features_dict=feature_d)
xpl.compile(X.iloc[:,1:])

# %%
xpl.run_app()

# %%
xpl.plot.features_importance()

# %%
low_DVI_low_SVI = [611, 318, 8362]
high_DVI_low_SVI = [3817, 5001]
low_DVI_high_SVI = [104, 2831, 8233.04]
high_DVI_high_SVI = [6810, 3406, 2604]

# %%
X_low_DVI_low_SVI = X[X['NAME10'].isin(low_DVI_low_SVI)]
X_high_DVI_low_SVI = X[X['NAME10'].isin(high_DVI_low_SVI)]
X_low_DVI_high_SVI = X[X['NAME10'].isin(low_DVI_high_SVI)]
X_high_DVI_high_SVI = X[X['NAME10'].isin(high_DVI_high_SVI)]

# X_v_low = X[X['NAME10'].isin(v_low)]

# %%
X_low_DVI_high_SVI.iloc[0, 16:20].plot.barh()

# %%
X_low_DVI_high_SVI

# %%
xpl.filter(max_contrib=25)

# %% [markdown]
# ## Low DVI Low SVI

# %%
from matplotlib import rcParams

rcParams['font.family'] = 'sans-serif'
rcParams['font.weight'] = 'normal'

# %%
X_low_DVI_low_SVI

# %%
shap_explainer = explainer(X.iloc[:, 1:])
shap_values = shap_explainer.values
shap_explainer.feature_names

# %%
df_svi[df_svi['NAME10']==611]['SVI'].values

# %%
y.iloc[84]

# %%
shap_df = pd.DataFrame(shap_values, columns=X.columns[1:])

def plot_parameter_importance(df, i, dvi, svi, align='top left'):

        
    index = df.index[i]
    name = df.iloc[i,0]
    
    dvi = np.round(y.iloc[index], 2)
    svi = np.round(df_svi[df_svi['NAME10']==name]['SVI'].values, 2)
    
    fig, ax = plt.subplots(figsize=(15, 10))
    if align == 'top left':
        ax2 = fig.add_axes([0.47, 0.72, 0.19, 0.16])
    else:
        ax2 = fig.add_axes([0.77, 0.15, 0.19, 0.16])
    color = list(map(lambda x: '#d95f02' if x >= 0 else '#1f78b4', shap_df.iloc[index].sort_values()))
    shap_df.iloc[index].sort_values().plot.barh(width=0.8, color=color, edgecolor='black', ax=ax)
    df.iloc[i, 16:20].plot.barh(width=0.8, align='center', ax=ax2)
    ax2.set_title(f"SVI: {svi}")
    ax2.set_xlabel('Contribution')
    ax.set_xlabel('Contribution')
    ax.set_title(f'Census Tract: {name} \n DVI: {dvi}')
    fig.tight_layout()
    plt.savefig(f'Final/Individual Census Tracts/{name}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return None

# %%
X_low_DVI_low_SVI

# %%
X[X['NAME10'] == 3817]

# %%
X_low_DVI_low_SVI

# %%
# top_left = [0.47, 0.72, 0.19, 0.16]
# bottom_right = [0.77, 0.15, 0.19, 0.16]

plot_parameter_importance(X_low_DVI_low_SVI, 0, 0.00, 0.05)
plot_parameter_importance(X_low_DVI_low_SVI, 1, 0.06, 0.31)
plot_parameter_importance(X_low_DVI_low_SVI, 2, 0.09, 0.37)

plot_parameter_importance(X_low_DVI_high_SVI, 0, 0.16, 0.71)
plot_parameter_importance(X_low_DVI_high_SVI, 1, 0.27, 0.88)
plot_parameter_importance(X_low_DVI_high_SVI, 2, 0.38, 0.85, align='bottom right')

plot_parameter_importance(X_high_DVI_low_SVI, 0, 0.60, 0.00, align='bottom right')
plot_parameter_importance(X_high_DVI_low_SVI, 1, 0.61, 0.25, align='bottom right')

plot_parameter_importance(X_high_DVI_high_SVI, 0, 0.93, 0.90, align='bottom right')
plot_parameter_importance(X_high_DVI_high_SVI, 1, 0.99, 0.95, align='bottom right')
plot_parameter_importance(X_high_DVI_high_SVI, 2, 1.00, 0.82, align='bottom right')

# %%
plot_parameter_importance(X_low_DVI_low_SVI, 1, 0.06, 0.31)

# %%
fig = xpl.plot.local_plot(index=X_low_DVI_low_SVI.index[1], show_predict=False, width=650, height=700)
fig.update_layout(font=dict(family="sans-serif",
                            color="Black",
                            size=14
                           ),
                  margin=dict(t=110),
                  title=dict(text=f"Census Tract: {X_low_DVI_low_SVI.iloc[1,0]} <br><sup>DVI: 0.06</sup>" )
                 )
fig.update_xaxes(title_font_family="sans-serif")
pio.write_image(fig, file=f'Final/Individual Census Tracts/Low DVI Low SVI {X_low_DVI_low_SVI.iloc[1,0]}.png', scale=3)

# %%
fig, ax = plt.subplots(figsize=(6, 4))
X_low_DVI_low_SVI.iloc[1, 16:20].plot.barh(width=0.8, align='center')
plt.xlabel("Contribution", fontdict={'fontsize':24, 'weight':'bold'})
plt.xticks(fontsize=24)
plt.yticks(fontsize=24, weight='bold')
plt.title(f"SVI: 0.31", fontdict={'fontsize':24, 'weight':'bold'})
# plt.savefig(f'Final/Individual Census Tracts/SVI Themes {X_low_DVI_low_SVI.iloc[1,0]}.png', dpi=300, bbox_inches='tight')
plt.tight_layout()
plt.show()

# %%
plot_parameter_importance(X_low_DVI_low_SVI, 2, 0.09, 0.37)

# %%
fig = xpl.plot.local_plot(index=X_low_DVI_low_SVI.index[2], show_predict=False, width=650, height=700)
fig.update_layout(font=dict(family="sans-serif",
                            color="Black",
                            size=14,
                           ),
                  margin=dict(t=110),
                  title=dict(text=f"Census Tract: {X_low_DVI_low_SVI.iloc[2,0]} <br><sup>DVI: 0.09</sup>" )
                 )
fig.update_xaxes(title_font_family="sans-serif")
pio.write_image(fig, file=f'Final/Individual Census Tracts/Low DVI Low SVI {X_low_DVI_low_SVI.iloc[2,0]}.png', scale=3)

# %%
fig, ax = plt.subplots(figsize=(6, 4))
X_low_DVI_low_SVI.iloc[2, 16:20].plot.barh(width=0.8, align='center')
plt.xlabel("Contribution", fontdict={'fontsize':24, 'weight':'bold'})
plt.xticks(fontsize=24)
plt.yticks(fontsize=24, weight='bold')
plt.title(f"SVI: 0.37", fontdict={'fontsize':24, 'weight':'bold'})
plt.savefig(f'Final/Individual Census Tracts/SVI Themes {X_low_DVI_low_SVI.iloc[2,0]}.png', dpi=300, bbox_inches='tight')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Low DVI High SVI

# %%
plot_parameter_importance(X_low_DVI_high_SVI, 0, 0.16, 0.71)
plot_parameter_importance(X_low_DVI_high_SVI, 1, 0.27, 0.88)
plot_parameter_importance(X_low_DVI_high_SVI, 2, 0.38, 0.85)

# %%
plot_parameter_importance(X_low_DVI_high_SVI, 1, 0.27, 0.88)

# %%
plot_parameter_importance(X_low_DVI_high_SVI, 2, 0.38, 0.85)

# %%
fig = xpl.plot.local_plot(index=X_low_DVI_high_SVI.index[0], show_predict=False, width=650, height=700)
fig.update_layout(font=dict(family="sans-serif",
                            color="Black",
                            size=14,
                           ),
                  margin=dict(t=110),
                  title=dict(text=f"Census Tract: {X_low_DVI_high_SVI.iloc[0,0]} <br><sup>DVI: 0.16</sup>" )
                 )
fig.update_xaxes(title_font_family="sans-serif")
pio.write_image(fig, file=f'Final/Individual Census Tracts/Low DVI High SVI {X_low_DVI_high_SVI.iloc[0,0]}.png', scale=3)

# %%
fig, ax = plt.subplots(figsize=(6, 4))
X_low_DVI_high_SVI.iloc[0, 16:20].plot.barh(width=0.8, align='center')
plt.xlabel("Contribution", fontdict={'fontsize':24, 'weight':'bold'})
plt.xticks(fontsize=24)
plt.yticks(fontsize=24, weight='bold')
plt.title(f"SVI: 0.71", fontdict={'fontsize':24, 'weight':'bold'})
plt.savefig(f'Final/Individual Census Tracts/SVI Themes {X_low_DVI_high_SVI.iloc[0,0]}.png', dpi=300, bbox_inches='tight')
plt.tight_layout()
plt.show()

# %%
fig = xpl.plot.local_plot(index=X_low_DVI_high_SVI.index[1], show_predict=False, width=650, height=700)
fig.update_layout(font=dict(family="sans-serif",
                            color="Black",
                            size=14,
                           ),
                  margin=dict(t=110),
                  title=dict(text=f"Census Tract: {X_low_DVI_high_SVI.iloc[1,0]} <br><sup>DVI: 0.27</sup>" )
                 )
fig.update_xaxes(title_font_family="sans-serif")
pio.write_image(fig, file=f'Final/Individual Census Tracts/Low DVI High SVI {X_low_DVI_high_SVI.iloc[1,0]}.png', scale=3)

# %%
fig, ax = plt.subplots(figsize=(6, 4))
X_low_DVI_high_SVI.iloc[1, 16:20].plot.barh(width=0.8, align='center')
plt.xlabel("Contribution", fontdict={'fontsize':24, 'weight':'bold'})
plt.xticks(fontsize=24)
plt.yticks(fontsize=24, weight='bold')
plt.title(f"SVI: 0.88", fontdict={'fontsize':24, 'weight':'bold'})
plt.savefig(f'Final/Individual Census Tracts/SVI Themes {X_low_DVI_high_SVI.iloc[1,0]}.png', dpi=300, bbox_inches='tight')
plt.tight_layout()
plt.show()

# %%
fig = xpl.plot.local_plot(index=X_low_DVI_high_SVI.index[2], show_predict=False, width=650, height=700)
fig.update_layout(font=dict(family="sans-serif",
                            color="Black",
                            size=14,
                           ),
                  margin=dict(t=110),
                  title=dict(text=f"Census Tract: {X_low_DVI_high_SVI.iloc[2,0]} <br><sup>DVI: 0.38</sup>" )
                 )
fig.update_xaxes(title_font_family="sans-serif")
pio.write_image(fig, file=f'Final/Individual Census Tracts/Low DVI High SVI {X_low_DVI_high_SVI.iloc[2,0]}.png', scale=3)

# %%
fig, ax = plt.subplots(figsize=(6, 4))
X_low_DVI_high_SVI.iloc[2, 16:20].plot.barh(width=0.8, align='center')
plt.xlabel("Contribution", fontdict={'fontsize':24, 'weight':'bold'})
plt.xticks(fontsize=24)
plt.yticks(fontsize=24, weight='bold')
plt.title(f"SVI: 0.85", fontdict={'fontsize':24, 'weight':'bold'})
plt.savefig(f'Final/Individual Census Tracts/SVI Themes {X_low_DVI_high_SVI.iloc[2,0]}.png', dpi=300, bbox_inches='tight')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## High DVI Low SVI

# %%
plot_parameter_importance(X_high_DVI_low_SVI, 0, 0.60, 0.00, align='bottom right')
plot_parameter_importance(X_high_DVI_low_SVI, 1, 0.61, 0.25, align='bottom right')

# %%
fig = xpl.plot.local_plot(index=X_high_DVI_low_SVI.index[0], show_predict=False, width=650, height=700)
fig.update_layout(font=dict(family="sans-serif",
                            color="Black",
                            size=14,
                           ),
                  margin=dict(t=110),
                  title=dict(text=f"Census Tract: {X_high_DVI_low_SVI.iloc[0,0]} <br><sup>DVI: 0.60</sup>" )
                 )
fig.update_xaxes(title_font_family="sans-serif")
pio.write_image(fig, file=f'Final/Individual Census Tracts/High DVI Low SVI {X_high_DVI_low_SVI.iloc[0,0]}.png', scale=3)

# %%
fig, ax = plt.subplots(figsize=(6, 4))
X_high_DVI_low_SVI.iloc[0, 16:20].plot.barh(width=0.8, align='center')
plt.xlabel("Contribution", fontdict={'fontsize':24, 'weight':'bold'})
plt.xticks(fontsize=24)
plt.yticks(fontsize=24, weight='bold')
plt.title(f"SVI: 0.00", fontdict={'fontsize':24, 'weight':'bold'})
plt.savefig(f'Final/Individual Census Tracts/SVI Themes {X_high_DVI_low_SVI.iloc[0,0]}.png', dpi=300, bbox_inches='tight')
plt.tight_layout()
plt.show()

# %%
fig = xpl.plot.local_plot(index=X_high_DVI_low_SVI.index[1], show_predict=False, width=650, height=700)
fig.update_layout(font=dict(family="sans-serif",
                            color="Black",
                            size=14,
                           ),
                  margin=dict(t=110),
                  title=dict(text=f"Census Tract: {X_high_DVI_low_SVI.iloc[1,0]} <br><sup>DVI: 0.61</sup>" )
                 )
fig.update_xaxes(title_font_family="sans-serif")
pio.write_image(fig, file=f'Final/Individual Census Tracts/High DVI Low SVI {X_high_DVI_low_SVI.iloc[1,0]}.png', scale=3)

# %%
fig, ax = plt.subplots(figsize=(6, 4))
X_high_DVI_low_SVI.iloc[1, 16:20].plot.barh(width=0.8, align='center')
plt.xlabel("Contribution", fontdict={'fontsize':24, 'weight':'bold'})
plt.xticks(fontsize=24)
plt.yticks(fontsize=24, weight='bold')
plt.title(f"SVI: 0.25", fontdict={'fontsize':24, 'weight':'bold'})
plt.savefig(f'Final/Individual Census Tracts/SVI Themes {X_high_DVI_low_SVI.iloc[1,0]}.png', dpi=300, bbox_inches='tight')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## High DVI High SVI

# %%
plot_parameter_importance(X_high_DVI_high_SVI, 0, 0.93, 0.90, align='bottom right')
plot_parameter_importance(X_high_DVI_high_SVI, 1, 0.99, 0.95, align='bottom right')
plot_parameter_importance(X_high_DVI_high_SVI, 2, 1.00, 0.82, align='bottom right')

# %%
fig = xpl.plot.local_plot(index=X_high_DVI_high_SVI.index[0], show_predict=False, width=650, height=700)
fig.update_layout(font=dict(family="sans-serif",
                            color="Black",
                            size=14,
                           ),
                  margin=dict(t=110),
                  title=dict(text=f"Census Tract: {X_high_DVI_high_SVI.iloc[0,0]} <br><sup>DVI: 0.93</sup>" )
                 )
fig.update_xaxes(title_font_family="sans-serif")
pio.write_image(fig, file=f'Final/Individual Census Tracts/High DVI High SVI {X_high_DVI_high_SVI.iloc[0,0]}.png', scale=3)

# %%
fig, ax = plt.subplots(figsize=(6, 4))
X_high_DVI_high_SVI.iloc[0, 16:20].plot.barh(width=0.8, align='center')
plt.xlabel("Contribution", fontdict={'fontsize':24, 'weight':'bold'})
plt.xticks(fontsize=24)
plt.yticks(fontsize=24, weight='bold')
plt.title(f"SVI: 0.90", fontdict={'fontsize':24, 'weight':'bold'})
plt.savefig(f'Final/Individual Census Tracts/SVI Themes {X_high_DVI_high_SVI.iloc[0,0]}.png', dpi=300, bbox_inches='tight')
plt.tight_layout()
plt.show()

# %%
fig = xpl.plot.local_plot(index=X_high_DVI_high_SVI.index[1], show_predict=False, width=650, height=700)
fig.update_layout(font=dict(family="sans-serif",
                            color="Black",
                            size=14,
                           ),
                  margin=dict(t=110),
                  title=dict(text=f"Census Tract: {X_high_DVI_high_SVI.iloc[1,0]} <br><sup>DVI: 0.99</sup>" )
                 )
fig.update_xaxes(title_font_family="sans-serif")
pio.write_image(fig, file=f'Final/Individual Census Tracts/High DVI High SVI {X_high_DVI_high_SVI.iloc[1,0]}.png', scale=3)

# %%
fig, ax = plt.subplots(figsize=(6, 4))
X_high_DVI_high_SVI.iloc[1, 16:20].plot.barh(width=0.8, align='center')
plt.xlabel("Contribution", fontdict={'fontsize':24, 'weight':'bold'})
plt.xticks(fontsize=24)
plt.yticks(fontsize=24, weight='bold')
plt.title(f"SVI: 0.95", fontdict={'fontsize':24, 'weight':'bold'})
plt.savefig(f'Final/Individual Census Tracts/SVI Themes {X_high_DVI_high_SVI.iloc[1,0]}.png', dpi=300, bbox_inches='tight')
plt.tight_layout()
plt.show()

# %%
fig = xpl.plot.local_plot(index=X_high_DVI_high_SVI.index[2], show_predict=False, width=650, height=700)
fig.update_layout(font=dict(family="sans-serif",
                            color="Black",
                            size=14,
                           ),
                  margin=dict(t=110),
                  title=dict(text=f"Census Tract: {X_high_DVI_high_SVI.iloc[2,0]} <br><sup>DVI: 1.00</sup>" )
                 )
fig.update_xaxes(title_font_family="sans-serif")
pio.write_image(fig, file=f'Final/Individual Census Tracts/High DVI High SVI {X_high_DVI_high_SVI.iloc[2,0]}.png', scale=3)

# %%
fig, ax = plt.subplots(figsize=(6, 4))
X_high_DVI_high_SVI.iloc[2, 16:20].plot.barh(width=0.8, align='center')
plt.xlabel("Contribution", fontdict={'fontsize':24, 'weight':'bold'})
plt.xticks(fontsize=24)
plt.yticks(fontsize=24, weight='bold')
plt.title(f"SVI: 0.82", fontdict={'fontsize':24, 'weight':'bold'})
plt.savefig(f'Final/Individual Census Tracts/SVI Themes {X_high_DVI_high_SVI.iloc[2,0]}.png', dpi=300, bbox_inches='tight')
plt.tight_layout()
plt.show()

# %%
df_test = pd.read_csv('Final/Data/AllParameters and Vulnerability Classes.csv', index_col=0)
df_test

# %%


# %%
sel = df_all_params[df_all_params['Vulnerability_ClassNum'] == 1].index
fig = xpl.plot.features_importance(selection=sel, max_features=25, width=600, height=750)
fig.update_traces(name='Very Low', selector=dict(name='Subset'))
fig.update_layout(
    font=dict(
        family="sans-serif",
        size=18,  # Set the font size here
        color="Black"
    ),
    margin=dict(t=110)
)
fig.update_xaxes(title_font_family="sans-serif")
pio.write_image(fig, file='Very Low Vulnerability.png', scale=3)

# %%
sel = df_all_params[df_all_params['Vulnerability_ClassNum'] == 2].index
fig = xpl.plot.features_importance(selection=sel, max_features=25, width=600, height=750)
fig.update_traces(name='Low', selector=dict(name='Subset'))
fig.update_layout(
    font=dict(
        family="sans-serif",
        size=18,  # Set the font size here
        color="Black"
    ),
    margin=dict(t=110)
)
fig.update_xaxes(title_font_family="sans-serif")
pio.write_image(fig, file='Low Vulnerability.png', scale=3)

# %%
sel = df_all_params[df_all_params['Vulnerability_ClassNum'] == 3].index
fig = xpl.plot.features_importance(selection=sel, max_features=25, width=600, height=750)
fig.update_traces(name='Moderate', selector=dict(name='Subset'))
fig.update_layout(
    font=dict(
        family="sans-serif",
        size=18,  # Set the font size here
        color="Black"
    ),
    margin=dict(t=110)
)
fig.update_xaxes(title_font_family="sans-serif")
pio.write_image(fig, file='Moderate Vulnerability.png', scale=3)

# %%
sel = df_all_params[df_all_params['Vulnerability_ClassNum'] == 4].index
fig = xpl.plot.features_importance(selection=sel, max_features=25, width=600, height=750)
fig.update_traces(name='High', selector=dict(name='Subset'))
fig.update_layout(
    font=dict(
        family="sans-serif",
        size=18,  # Set the font size here
        color="Black"
    ),
    margin=dict(t=110)
)
fig.update_xaxes(title_font_family="sans-serif")
pio.write_image(fig, file='High Vulnerability.png', scale=3)

# %%
sel = df_all_params[df_all_params['Vulnerability_ClassNum'] == 5].index
fig = xpl.plot.features_importance(selection=sel, max_features=25, width=600, height=750)
fig.update_traces(name='Very High', selector=dict(name='Subset'))
fig.update_layout(
    font=dict(
        family="sans-serif",
        size=18,  # Set the font size here
        color="Black"
    ),
    margin=dict(t=110)
)
fig.update_xaxes(title_font_family="sans-serif")
pio.write_image(fig, file='Very High Vulnerability.png', scale=3)

# %%


# %%


# %%


# %%
import shap

# %%
shap_values = explainer.shap_values(X)

shap.force_plot(explainer.expected_value, shap_values, X)

# %%
# V. High: 6810, 3406, 2604
# Moderate: 8305, 5202, 2306
# V. Low: 611, 318, 8362
v_high = [6810, 3406, 2604]
moderate = [8305, 5202, 2306]
v_low = [611, 318, 8362]

# %%
X_v_high = X[X['NAME10'].isin(v_high)]
X_moderate = X[X['NAME10'].isin(moderate)]
X_v_low = X[X['NAME10'].isin(v_low)]

# %%
shap_explainer = explainer(X.iloc[:, 1:])
shap_values = shap_explainer.values
shap_explainer.feature_names
shap_values.shape

# %%


# %%
shap_explainer.feature_names

# %%
# ax = axs.ravel()
for i in range(len(X_v_high)):
    fig, axs = plt.subplots()
    shap.plots.waterfall(shap_explainer[X_v_high.index[0]], max_display=10, show=False)
#     shap.plots.waterfall(shap_explainer[X_v_high.index[1]], max_display=25, show=True, ax=ax[1])
    plt.title(f"Census Tract: {X_v_high.iloc[i]['NAME10']}")
fig.savefig(f"../../Data/saved_data/Census Tract/High_{X_v_high.iloc[i]['NAME10']}.jpg", dpi=300, bbox_inches='tight')
plt.show()

# %%
for i in range(len(X_moderate)):
    fig, ax = plt.subplots(figsize=(15, 16))
    shap.plots.waterfall(shap_explainer[X_moderate.index[i]], max_display=10, show=False)
    plt.title(f"Census Tract: {X_moderate.iloc[i]['NAME10']}")
    fig.savefig(f"../../Data/saved_data/Census Tract/Moderate_{X_moderate.iloc[i]['NAME10']}.jpg", dpi=300, bbox_inches='tight')
    plt.show()

# %%
for i in range(len(X_v_low)):
    fig, ax = plt.subplots(figsize=(15, 16))
    shap.plots.waterfall(shap_explainer[X_v_low.index[i]], max_display=10, show=False)
    plt.title(f"Census Tract: {X_v_low.iloc[i]['NAME10']}")
    fig.savefig(f"../../Data/saved_data/Census Tract/Low_{X_v_low.iloc[i]['NAME10']}.jpg", dpi=300, bbox_inches='tight')
    plt.show()

# %%
shap_explainer[X_moderate.index[0]]

# %%


shap_values = shap_explainer[X_moderate.index[0]].values

# Define the feature names
feature_names = X.columns[1:]

# Calculate the base value and the final prediction value
base_value = shap_explainer[X_moderate.index[0]].base_values
final_value = base_value + np.sum(shap_values)


# Sort the SHAP values in descending order of absolute value
sort_inds = np.argsort(np.abs(shap_values))#[::-1]
sorted_shap_values = shap_values[sort_inds]
sorted_feature_names = np.array(feature_names)[sort_inds]

# Define the colors for positive and negative contributions
colors = ['#1f77b4' if val >= 0 else '#d62728' for val in sorted_shap_values]

# Create the waterfall plot
fig, ax = plt.subplots(figsize=(10, 12))

# Plot the base value
ax.axhline(y=base_value, linestyle='--', color='#AAAAAA', linewidth=1)
# ax.text(-0.1, base_value, 'Base Value', ha='right', va='center', fontsize=12)

# Plot the feature contributions
for i, (shap_val, name, color) in enumerate(zip(sorted_shap_values, sorted_feature_names, colors)):
    y_start = base_value + np.sum(sorted_shap_values[:i])
    y_end = y_start + shap_val
    ax.errorbar(x=(y_start + y_end) / 2, y=name, xerr=shap_val / 2, fmt='o', color=color, ecolor='black')
    ax.text(y_start + 0.5 * (y_end - y_start), i, f'{shap_val:.2f}', ha='center', va='center', color='white', fontsize=12)

# Plot the final prediction value
ax.axhline(y=final_value, linestyle='--', color='#AAAAAA', linewidth=1)
# ax.text(-0.1, final_value, 'Final Prediction', ha='right', va='center', fontsize=12)

# Set the axis labels and title
ax.set_xlabel('Contribution to Prediction', fontsize=12)
ax.set_ylabel('Feature', fontsize=12)
ax.set_title('SHAP Values Waterfall Plot', fontsize=14)

# Set the y-axis limits
ax.set_ylim([sorted_feature_names[0], sorted_feature_names[-1]])
plt.tight_layout()
# Show the plot
plt.show()

# %%
y = pd.DataFrame(model.predict(X.iloc[:, 1:]), index=X['NAME10'])
y.reset_index(inplace=True)

# %%
y.columns = ['NAME10', 'vul']

# %%
y['vul'] = MinMaxScaler().fit_transform(y[['vul']])

# %%
y[y["NAME10"] == 318.00]

# %%
y[y['vul'] < 0.05].sort_values(by='vul')

# %%
y['vul'].plot(kind='kde')
plt.show()

# %%
import shap

# %%
help(shap)

# %%
help(shap.plots)

# %%
for i in range(len(X_v_low)):
    fig, ax = plt.subplots(figsize=(15, 16))
    shap_values = shap_explainer[X_v_low.index[i]].values
    base_value = shap_explainer[X_v_low.index[i]].base_values
    fx = np.round(base_value + np.sum(shap_values), 2)
#     print(fx)
    vul = y[y['NAME10'] == X_v_low.iloc[i]['NAME10']]['vul'].values
    scaling_factor = np.round(vul/fx, 6)
#     print(scaling_factor)
    print(vul)
    shap.plots.waterfall(shap_explainer[X_v_low.index[i]], max_display=10, show=False)
    plt.title(f"Census Tract: {X_v_low.iloc[i]['NAME10']}")
#     print(ax.get_shared_x_axes)
#     xtick_labels = [2*float(label.get_text()) for label in ax.get_xticklabels()]
#     ax.set_xticklabels(xtick_labels)
    print(np.round(np.linspace(vul, base_value, len(ax.get_xticklabels())), 2))
#     ax.set_xticklabels(np.round(np.linspace(vul, base_value, 6), 2))
    ax.set_xticklabels(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
#     fig.savefig(f"../../Data/saved_data/Census Tract/Low_{X_v_low.iloc[i]['NAME10']}.jpg", dpi=300, bbox_inches='tight')
    plt.show()
    break

# %%


# %%
y

# %%
clustering = shap.utils.hclust(X.iloc[:,1:], y.iloc[:,1])
shap.plots.bar(shap_explainer, max_display=25, clustering=clustering, clustering_cutoff=0.7)

# %%
expected_value = explainer.expected_value
shap.decision_plot(expected_value, shap_values, X.columns[1:], feature_order='hclust')

# %%
for i in range(len(X_v_high)):
    shap.plots.waterfall(shap_explainer[X_v_high.index[i]], max_display=25)

# %%
for i in range(len(X_v_high)):
    shap.plots.waterfall(shap_explainer[X_v_high.index[i]], max_display=25)

# %%
for i in range(len(X_v_high)):
    shap.plots.waterfall(shap_explainer[X_v_high.index[i]], max_display=25)

# %%
X_v_high.index[0]

# %%
X_v_high.iloc[0,:].index.values

# %%



