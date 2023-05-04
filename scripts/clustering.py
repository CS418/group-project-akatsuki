# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import ParameterGrid
from sklearn.cluster import KMeans

# %%
Crime_df = pd.read_csv("./Data/All_Parameter_CSV.csv", index_col=0).sort_values(by='NAME10').drop_duplicates('NAME10').reset_index(drop=True)
svi_2020 = pd.read_csv('Data/svi_2020.csv', index_col=0).sort_values(by='NAME10')

# %%
Crime_df.drop(['Socioeconomic\xa0Status', 'Household Composition & Disability', 'Minority Status & Language',
       'Housing Type & Transportation', 'Diabetes'], axis=1, inplace=True)

# %%
Crime_df = Crime_df.merge(right=svi_2020, left_on='NAME10', right_on='NAME10', how='inner')

# %%
Crime_df.head(5)

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
Crime_df = Crime_df.rename(columns={"Solar_DNI":"Direct Normal Irradiation"})

# %%
Crime_df_all_params = Crime_df[rearranged_columns]

# %%
Crime_df_all_params.head(5)

# %%
Crime_df_all_params_scaled = Crime_df_all_params.copy()

# %%
Crime_df_all_params_scaled = Crime_df_all_params_scaled.iloc[:,8:]

# %%
Crime_df_all_params_scaled[Crime_df_all_params_scaled.columns] = StandardScaler().fit_transform(Crime_df_all_params_scaled)
Crime_df_all_params_scaled

# %%
len(Crime_df_all_params_scaled.columns)

# %%
# Importing Modules
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

clustering = AgglomerativeClustering(n_clusters=5)
clustering.fit(Crime_df_all_params_scaled)

# %%
plt.scatter(Crime_df_all_params_scaled['Smoking'], Crime_df_all_params_scaled['Binge Drinking'], c=clustering.labels_)
plt.xlabel('Smoking')
plt.ylabel('Binge Drinking')
plt.show()

# %%
from sklearn.decomposition import PCA

# Perform PCA with all components
pca = PCA(n_components=26, random_state=42)
pca.fit(Crime_df_all_params_scaled)
X_pca = pca.transform(Crime_df_all_params_scaled)

print("Explained variance ratio: ", sum(pca.explained_variance_ratio_ * 100))



# %%
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Number of Components")
plt.ylabel("Explained variance")

#from the below plot we can see that the first 15 components explain about 98% of variance in data

# %%
#perform PCA with just 2 components
pca_2 = PCA(n_components=10, random_state=42)
pca_2.fit(Crime_df_all_params_scaled)
X_pca = pca_2.transform(Crime_df_all_params_scaled)

# %%
import seaborn as sns

# Get the explained variance ratio for each component
variance_ratio = pca_2.explained_variance_ratio_

# Plot the first 10 components and color by cluster labels
plt.figure(figsize=(10,7))
sns.scatterplot(x= X_pca[:,0],y= X_pca[:,1], s=70, hue='Crime', data=Crime_df_all_params_scaled, c=clustering.labels_)
plt.xlabel(f'PCA Component 1 (Explains {variance_ratio[0]*100:.2f}% Variance)')
plt.ylabel(f'PCA Component 2 (Explains {variance_ratio[1]*100:.2f}% Variance)')
plt.show()

# %%
#listing component names 

pc_names = ["PC"+str(i+1) for i in range(len(pca_2.components_))]
column_names = Crime_df_all_params_scaled.columns.tolist()

pc_col_names = dict(zip(pc_names, pca_2.components_))
for name, col_names in pc_col_names.items():
    sorted_col_names = sorted(zip(col_names, column_names), reverse=True)
    print(f"{name}: {[col[1] for col in sorted_col_names]}")

# %%
plt.scatter(Crime_df_all_params_scaled['Obesity'], Crime_df_all_params_scaled['Physical Inactivity'], c=clustering.labels_)
plt.xlabel('Obesity')
plt.ylabel('Physical Inactivity')
plt.show()

# %%
plt.scatter(Crime_df_all_params_scaled['Carbon Monoxide '], Crime_df_all_params_scaled['Sulfur Dioxide '], c=clustering.labels_)
plt.xlabel('Carbon Monoxide')
plt.ylabel('Sulfur Dioxide ')
plt.show()


