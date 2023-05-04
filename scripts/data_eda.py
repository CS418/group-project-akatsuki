# %% [markdown]
# ### Team : Akatsuki
# ### Project Title : Chicago Crime Index
# ### Members : Karan Jogi, Nikita Thakur
# 
# 

# %% [markdown]
# ## Project/Problem Introduction
# 
# 
# The big idea of this project is to study the crime rates in Chicago city based on various variables such as weather, Demographics, Behavior and socioeconomic factors. The aim is to develop a tool that can assist policymakers, law enforcement agencies, and community members in identifying areas with high crime rates and prioritizing resources to reduce crime.
# 
# The problem we want to solve is  Identification of high crime rate areas in Chicago, which can negatively impact the safety and well-being of residents, economic development, and community cohesion. By providing a tool that can analyze and visualize the relationships between crime rates and various factors, we hope to support decision-making processes that can lead to the reduction of crime rates and the improvement of the quality of life for residents.
# 
# This project is essential because high crime rates are a significant concern for many urban areas worldwide. In Chicago, for instance, the city has struggled with high crime rates for many years, leading to social and economic challenges. Therefore, addressing this issue can have a positive impact on many aspects of society, including safety, security, economic development, and community well-being.
# 
# We chose this problem because it is a complex and challenging issue that requires innovative solutions. By utilizing data-driven approaches and advanced analytical tools, we believe that we can contribute to the existing efforts of reducing crime rates in Chicago.

# %% [markdown]
# ## Data
# 
# We have extracted columns that were relevant from larger datasets that are listed below. Each such feature is being grouped by census tracts in the city of Chicago. The original and final data sizes are present in the Data Cleaning part of the notebook.
# 
# | Themes          | Features                             | Source                                             |
# | ---------------- | ---------------------------------------- | -------------------------------------------------- |
# | Behavior         | Smoking                                  | [CDC places](http://www.cdc.gov/places)            |
# |                  | Binge Drinking                           |                |
# |                  | Opioid Overdose                          |               |
# |                  | Sleep<7hours                             |                |
# |                  | Physical Inactivity                      | [CDPH](http://www.cdc.gov/physicalactivity)        |
# | Climate          | Surface Temperature                      | [MODIS](http://www.modis.gsfc.nasa.gov)             |
# |                  | Wind Speed                               | [WorldClim](http://www.worldclim.org)              |
# |                  | Precipitation                            | [SOLARGIS](http://www.solargis.com)                |
# |                  | Snow                                     | [Global Wind Atlas](http://www.globalwindatlas.info)|
# | Socio-economic   | Socioeconomic Status                     | [CDC SVI](http://www.atsdr.cdc.gov/placeandhealth/svi)|
# |                  | Household composition and Disability     |              |
# |                  | Minority Status & Language               |                |
# |                  | Housing Type & Transportation            |               |
# | Surveillance      | Surveillance Cameras                      | [pending]                 |
# |                  | Lighting/Street Lights                   | [pending]                  |
# |                  | Police stations/ Patrolling routes       | [City of Chicago](http://www.chicago.gov/police)   |
# | Health           | Depression                               | [CDC places](http://www.cdc.gov/places)            |
# 
# 
# 

# %% [markdown]
# ## Research Questions
# 
# 1. What is the Crime Vulnerability for the Census Tracts in Chicago based on Social Vulnerability Index, Climate, Behavior (and Surveillance)?
# 2. What are the most influential features that contribute to the Crime Vulnerability Index in Chicago?
# 3. What are the racially marginalized communities suffering from the disproportionate burden of the Crime in Chicago?

# %% [markdown]
# ## Data cleaning

# %% [markdown]
# Data cleaning and processing of multiple datasets related to various health and social indicators was performed for the study. Here's a brief overview of the steps taken:
# 1. Data was filtered based on specific criteria, such as StateAbbr and CountyName, to only include relevant rows using boolean indexing.
# 2. Data types of certain columns were converted to the appropriate data types using the "astype()" method.
# 3. Crime data was geographically joined with the census tract boundaries using the "sjoin()" function from geopandas.
# 4. Data from different datasets was merged based on a common column, such as FIPS or TractFIPS, using the "merge()" function from pandas.
# 5. Column names were renamed to more descriptive names.
# 6. Unwanted columns are dropped using indexing and the resulting Data Frame was saved to a CSV File.
# 

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd

# %%
# Data for Binge Drinking, Smoking, Physical inactivity, Sleep and Mental health
cols_to_read = ['StateAbbr', 'StateDesc', 'CountyName', 'CountyFIPS', 'TractFIPS', 'BINGE_CrudePrev', 'CSMOKING_CrudePrev', 'LPA_CrudePrev', 'SLEEP_CrudePrev', 'DEPRESSION_CrudePrev' ,'Geolocation']
df_behavior = pd.read_csv('Data/PLACES__Census_Tract_Data__GIS_Friendly_Format___2022_release.csv', usecols=cols_to_read)

# Filter the DataFrame to only include rows where the StateAbbr column is 'IL'
df_behavior = df_behavior[df_behavior['StateAbbr'] == 'IL']

# Filter the DataFrame again to only include rows where the CountyName column is 'Cook'
df_behavior = df_behavior[df_behavior['CountyName'] == 'Cook'].reset_index(drop=True)
df_behavior.head()

# %%
#Data for Socioeconomic status, Household Composition & Disability, Minority Status & Language and Housing & Transportation 
cols_to_read = ['NAME10', 'GEOID10','EP_AFAM', 'EP_AIAN', 'EP_ASIAN', 'EP_HISP','RPL_THEMES', 'RPL_THEME1', 'RPL_THEME2', 'RPL_THEME3', 'RPL_THEME4', 'geometry']

df_svi = gpd.read_file('./Data/Chicago/Chicago_SVI_2020.shp', include_fields = cols_to_read).drop_duplicates('NAME10')


# %%
#Crime Data by Chicago Census Tracts
cols_to_read = ['ID', 'Primary Type', 'Latitude', 'Longitude']
df_crime = pd.read_csv('/Users/karanjogi/Downloads/Crimes_-_2001_to_Present (1).csv', usecols=cols_to_read)
df_crime.dropna(inplace=True)
df_crime.head()

# %%
crime_types = ['BATTERY', 'THEFT', 'ASSAULT', 'BURGLARY', 'ROBBERY',
               'OTHER OFFENSE', 'CRIMINAL DAMAGE', 'WEAPONS VIOLATION',
               'CRIMINAL TRESPASS', 'MOTOR VEHICLE THEFT',
               'SEX OFFENSE', 'OFFENSE INVOLVING CHILDREN', 'PUBLIC PEACE VIOLATION',
               'GAMBLING', 'CRIM SEXUAL ASSAULT', 'LIQUOR LAW VIOLATION', 'ARSON',
               'STALKING', 'KIDNAPPING', 'INTIMIDATION', 'CONCEALED CARRY LICENSE VIOLATION',
               'HUMAN TRAFFICKING', 'OBSCENITY', 'CRIMINAL SEXUAL ASSAULT', 'PUBLIC INDECENCY',
               'HOMICIDE', 'RITUALISM', 'DOMESTIC VIOLENCE']

df_crime = df_crime[df_crime['Primary Type'].isin(crime_types)]

# %%
#Grouping crime data by Census tracts

from shapely.geometry import Point
geometry = [Point(xy) for xy in zip(df_crime['Longitude'], df_crime['Latitude'])]
gdf = gpd.GeoDataFrame(df_crime, geometry=geometry)

gdf.crs = df_svi.crs

joined = gpd.sjoin(gdf, df_svi, how="inner", op="within")

grouped = joined.groupby('NAME10').agg({'ID': 'count'}).reset_index()

# %%
grouped.columns = ['NAME10', 'Crime']
df_svi = df_svi.merge(grouped, left_on='NAME10', right_on='NAME10')

# %%
df_svi.columns

# %%
index = df_svi['Crime'].idxmax()
df_svi.loc[index, 'Crime'] = 50000

# %%
svi_2020 = pd.DataFrame(data={'NAME10': df_svi['NAME10'],
                              'Socioeconomic Status': df_svi['RPL_THEME1'],
                              'Household Characteristics': df_svi['RPL_THEME2'],
                              'Racial & Ethnic Minority Status': df_svi['RPL_THEME3'],
                              'Housing Type & Transportation': df_svi['RPL_THEME4'],
                              'Crime': df_svi['Crime'],
                              'EP_AFAM': df_svi['EP_AFAM'],
                              'EP_HISP': df_svi['EP_HISP'],
                              'EP_ASIAN': df_svi['EP_ASIAN'],
                              'EP_AIAN': df_svi['EP_AIAN'],
                              'SVI': df_svi['RPL_THEMES']
                             }
                       )

svi_2020.replace(-999, 0, inplace=True)
svi_2020.to_csv('Data/svi_2020.csv')

# %%
#Joining Tables based on FIPS number
df_all_parameters = pd.merge(left=df_behavior, right=df_svi, left_on='TractFIPS', right_on='GEOID10', how='right')

# %%
df_all_parameters

# %%
# Rename columns 
# Socioeconomic Status – RPL_THEME1
# Household Characteristics – RPL_THEME2
# Racial & Ethnic Minority Status – RPL_THEME3
# Housing Type & Transportation – RPL_THEME4

rename_dict = {
    'RPL_THEME1': 'Socioeconomic Status',
    'RPL_THEME2': 'Household Characteristics',
    'RPL_THEME3': 'Racial & Ethnic Minority Status',
    'RPL_THEME4': 'Housing Type & Transportation',
    'geoid10': 'FIPS',
    'ID': 'Crime prevelance',
    'BINGE_CrudePrev': 'Binge Drinking',
    'CSMOKING_CrudePrev': 'Smoking',
    'DEPRESSION_CrudePrev': 'Depression',
    'LPA_CrudePrev': 'Physical Inactivity',
    'SLEEP_CrudePrev': 'Sleep<7hours',
    
}

# %%
df_all_parameters.drop_duplicates('GEOID10')

# %%
df_all_parameters.to_csv('Data/All_parameters.csv')

# %%
df_all_parameters.shape

# %%
df_all_parameters.dtypes

# %%
df_crime

# %% [markdown]
# ## Exploratory Data Analysis
# 
# Following is a summary of the EDA tasks performed:
# 1. Data visualization: Bar chart to show the frequency of different crime types, pairplot to visualize the relationships between different behavioral health factors 
# 2. Pairplot to visualize the relationships between different social vulnerability index (SVI) themes, and boxplot to visualize the distribution of crime prevalence.
# 3. Data correlation analysis: Calculating Pearson and Kendall correlation coefficients between crime prevalence and other parameters.
# 4. Heatmap visualization: Creating a heatmap to visualize the correlation coefficients between crime prevalence and other parameters.
# 5. Data visualization on map: Plotting crime prevalence on the map of Chicago using shapefile data.
# 

# %% [markdown]
# #### Table: Crime Prevelance 
# #### Author: Nikita Thakur

# %%
crime_counts = df_crime['Primary Type'].value_counts()

fig, ax = plt.subplots(figsize=(14, 8))
ax.bar(crime_counts.index, crime_counts.values, log=True)

plt.xlabel('Crime Type')
plt.ylabel('Frequency')
plt.xticks(rotation=90)
plt.title('Crimes')
plt.savefig('Crimes.png', dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
# #### Table: Behavior Parameters
# #### Author: Nikita Thakur

# %%
sns.pairplot(df_behavior[['BINGE_CrudePrev', 'CSMOKING_CrudePrev', 'DEPRESSION_CrudePrev',
       'LPA_CrudePrev', 'SLEEP_CrudePrev']])
plt.show()

# %% [markdown]
# #### Table: SVI Variables
# #### Author: Karan Jogi

# %%
df_svi.columns = ['ST_ABBR', 'COUNTY', 'FIPS', 'Socioeconomic Status', 'Household Characteristics', 
                  'Racial & Ethnic Minority Status', 'Housing Type & Transportation', 
                  'EP_AFAM', 'EP_HISP', 'EP_ASIAN', 'EP_AIAN', 'geometry']

# %%
df_svi.replace(to_replace=-999, value=np.NaN, inplace=True)
sns.pairplot(df_svi[['Socioeconomic Status', 'Household Characteristics', 
                  'Racial & Ethnic Minority Status', 'Housing Type & Transportation']])
plt.show()

# %% [markdown]
# #### Table: All parameters with Crime 
# #### Author: Karan Jogi

# %%
df = pd.read_csv("./Data/All_Parameter_CSV.csv", index_col=0).sort_values(by='NAME10').drop_duplicates('GEOID10').reset_index(drop=True)
grouped['geoid10'] = grouped['geoid10'].astype('int64')

df = df.merge(right=grouped, left_on='GEOID10', right_on='geoid10', how='inner').drop(['geoid10', 'Diabetes'], axis=1)
df=df.rename(columns={"Solar_DNI":"Direct Normal Irradiation",
                      "ID": 'Crime'})

df_params = df.drop(columns=['TRACTCE10', 'GEOID10', 'NAME10', 'NAMELSAD10', 'Long', 'Lat', 'State',
       'County'], axis=1)

# %%
df_params.info()

# %%
df_params.describe()

# %%
sns.pairplot(df_params, )
plt.show()

# %%
sns.boxplot(data=df_params, y='Crime')
plt.title('Crime Prevelance Boxplot')
plt.show()

# %%
index = df_params['Crime'].idxmax()
df_params.loc[index, 'Crime'] = 50000

# %%
sns.boxplot(data=df_params, y='Crime')
plt.title('Crime Prevelance Boxplot')
plt.show()

# %%
index = df['Crime'].idxmax()
df.loc[index, 'Crime'] = 50000

# %%
census_tracts_shape['geoid10'] = census_tracts_shape['geoid10'].astype('int64')
merged_all_parameters = census_tracts_shape.join(df.set_index('GEOID10'), on='geoid10', how='inner')

# %%
fig, ax = plt.subplots()
df_svi.plot('Crime', 
            edgecolor='white',
            linewidth=0.2,
            legend=True,
            ax=ax,
            cmap='Reds',
            legend_kwds={"shrink":0.5}
           )
plt.title('Crime Prevelance in Chicago')

# get the current size of the figure
current_size = fig.get_size_inches()

# set the scaling factor for the size
scaling_factor = 2

# calculate the new size of the figure
new_size = (current_size[0] * scaling_factor, current_size[1] * scaling_factor)

# set the new size of the figure
fig.set_size_inches(new_size)
plt.savefig('Chicago Crime prevelance')
plt.show()

# %% [markdown]
# ## Export SVI Classes

# %%
df_censusVul = pd.read_csv('Vulnerability Classes and CensusTractInfo.csv', index_col=0)
df_param_Vul = pd.read_csv('AllParameters and Vulnerability Classes.csv', index_col=0).sort_values('NAME10').reset_index(drop=True)

# %%
df_censusVul

# %%
df_svi.merge(df_censusVul, left_on='FIPS', right_on='GEOID10', how='inner')

# %% [markdown]
# ## ML/Stats
# 
# ### Code & Visualizations:
# 
# 1. Regression, Catboost, SHAP - https://github.com/CS418/group-project-akatsuki/blob/main/Crime_Final.ipynb
# 2. PCA, Clustering - https://github.com/CS418/group-project-akatsuki/blob/main/clustering.ipynb
# 
# ### Results
# 
# | Data Science Question                                                                       | ML Task              | ML Techniques                      | Outcome                                                                  |
# | -------------------------------------------------------------------------------------------| ------------------- | ---------------------------------- | ------------------------------------------------------------------------ |
# | What is the Crime Vulnerability for the Census Tracts in Chicago based on Social Vulnerability Index, Climate, Behavior, Health and Air Pollution? | Regression           | Linear Regression, Random Forest Regression, Catboost Regression | Catboost regression performed the best as compared to Random Forest and Linear Regression |
# | What are the most influential features that contribute to the Crime Vulnerability Index in Chicago? | Explainable AI (XAI) | SHapley Additive exPlanations (SHAP) | The top 5 features affecting the vulnerability of the crime in census tracts are: Minority Status and Language, Opioid Overdose, Socioeconomic Status, Sleep<7hours, Carbon Monoxide |
# | What are the most influential features that contribute to the Crime Vulnerability Index in Chicago?| Dimensionality reduction| PCA & Agglomerative clustering         | About 98% of the data variance is explained by 15 most influential features. The top features reported by this method were Obesity, Physical Inactivity, Sleep<7hours, Carbon Monoxide, Sulpher Dioxide and High Blood Pressure|
# | What are the racially marginalized communities suffering from the disproportionate burden of the Crime in Chicago? | Regression and XAI           | Catboost Regression + SHAP | 72.2% of Census Tracts with more than 80% of African American population are under Very High Class. 18.8% of Census Tracts with more than 80% of African American population are under Very High Class. |
# 

# %% [markdown]
# ## Literary Survey
# 
# The literary survey includes 20 research papers that focus on using machine learning methods to predict and analyze crime in urban areas. The variables considered in these studies include demographic and socioeconomic factors, crime types and locations, weather conditions, social media activity, and environmental data such as the proximity of public transit and recreational areas. The most commonly used machine learning algorithms are decision trees, random forests, logistic regression, K-nearest neighbors, support vector machines, and neural networks. The importance of temporal features such as day of the week and time of day in predicting crime occurrences is also highlighted in some studies. Overall, these studies provide valuable insights into the use of machine learning for crime analysis and prediction in urban areas.
# 
# - Ahamed, J., & Roy, D. (2019). Crime analysis using machine learning techniques. In 2019 IEEE International Conference on Electrical, Computer and Communication Technologies (ICECCT) (pp. 1-5). IEEE. [Link]( https://ieeexplore.ieee.org/abstract/document/8769541)
# - Al-Zeibak, R., & Haque, M. A. (2019). Prediction of crime occurrence using machine learning algorithms. Journal of King Saud University-Computer and Information Sciences, 31(3), 325-334. [Link]( https://www.sciencedirect.com/science/article/pii/S1319157817304765)
# - Bajaj, A., & Saini, J. S. (2018). Crime prediction using machine learning. In 2018 9th International Conference on Computing, Communication and Networking Technologies (ICCCNT) (pp. 1-5). IEEE. [Link]( https://ieeexplore.ieee.org/abstract/document/8494084)
# - Bhatia, P., & Kaur, M. (2019). Crime prediction using machine learning. In 2019 IEEE 7th International Conference on Advanced Computing (IACC) (pp. 76-81). IEEE. [Link]( https://ieeexplore.ieee.org/abstract/document/8692196)
# - Burian, J., & Wozniak, M. (2019). Predicting crime using machine learning methods. In 2019 IEEE 12th International Conference on Humanoid, Nanotechnology, Information Technology, Communication and Control, Environment and Management (HNICEM) (pp. 1-4). IEEE. [Link]( https://ieeexplore.ieee.org/abstract/document/9032632)
# - Cardoso, M. J., Ferreira, H. R., & Ferreira, A. J. (2019). Crime prediction in smart cities using machine learning algorithms. In Proceedings of the 2019 International Conference on Cyber-Enabled Distributed Computing and Knowledge Discovery (CyberC) (pp. 14-21). IEEE. [Link]( https://ieeexplore.ieee.org/abstract/document/8919271)
# - Cheng, L., Liu, C., & Lu, J. (2018). Predicting crime occurrences using temporal features and machine learning techniques. IEEE Transactions on Intelligent Transportation Systems, 19(12), 3971-3980. [Link]( https://ieeexplore.ieee.org/abstract/document/8342953)
# - Demir, I., & Kose, U. (2019). Crime prediction using machine learning: A review. In 2019 3rd International Symposium on Multidisciplinary Studies and Innovative Technologies (ISMSIT) (pp. 82-86). IEEE. [Link]( https://ieeexplore.ieee.org/abstract/document/8945002)
# - Dilawar, N., & Hussain, S. (2019). Crime prediction using machine learning techniques. In 2019 4th International Conference on Computer and Communication Systems (ICCCS) (pp. 194-198). IEEE. [Link]( https://ieeexplore.ieee.org/abstract/document/8759598)
# - Dong, W., Tang, L., Huang, H., & Cheng, J. (2019). Crime prediction using machine learning algorithms with decision rules. Journal of Ambient Intelligence and Humanized Computing, 
# - Abadi, M., & Singh, M. P. (2019). Crime prediction using machine learning algorithms. International Journal of Computer Science and Information Security, 17(9), 42-48. [Link](https://ijcsis.org/papers/vol17no9/ijcsis-vol17no9-p03.pdf)
# - Adderley, R. J., Morris, A., & Schneider, M. (2017). Crime prediction in London using machine learning techniques. Journal of Maps, 13(2), 246-252. [Link](https://www.tandfonline.com/doi/full/10.1080/17445647.2017.1370453)
# - Alemi, F., & Itoga, C. (2018). Predicting crime using machine learning and city data. International Journal of Big Data Intelligence, 5(1), 22-31. [Link](https://www.inderscienceonline.com/doi/abs/10.1504/IJBDI.2018.090516)
# - Borrion, H., & Andresen, M. A. (2017). Applying machine learning techniques to crime data in the city of Vancouver. Security Informatics, 6(1), 5. [Link](https://link.springer.com/article/10.1186/s13388-017-0058-2)
# - Cao, X., Wang, J., & Li, X. (2019). Crime prediction using spatiotemporal data: A deep learning approach. IEEE Transactions on Intelligent Transportation Systems, 20(6), 2199-2208. [Link](https://ieeexplore.ieee.org/abstract/document/8708807)
# - Chang, C. C., Chen, T. Y., & Huang, Y. W. (2019). Crime prediction in a smart city using machine learning techniques. In 2019 IEEE International Conference on Industrial Engineering and Engineering Management (IEEM) (pp. 1179-1183). IEEE. [Link](https://ieeexplore.ieee.org/abstract/document/8978738)
# - Chaturvedi, A., Roy, P. K., & Khan, A. (2019). Machine learning based crime prediction using spatiotemporal data. In 2019 3rd International Conference on Trends in Electronics and Informatics (ICOEI) (pp. 628-632). IEEE. [Link](https://ieeexplore.ieee.org/abstract/document/8862768)
# - Chikaraishi, M., Shimizu, H., & Shibasaki, R. (2018). Crime prediction with deep learning and feature engineering using 911 calls for service. In 2018 21st International Conference on Information Fusion (Fusion) (pp. 1146-1153). IEEE. [Link](https://ieeexplore.ieee.org/document/8455716)
# - Dayal, R., & Cuddihy, E. (2018). Crime prediction using machine learning on social media data. In 2018 9th IEEE Annual Information Technology, Electronics and Mobile Communication Conference (IEMCON) (pp. 449-453). IEEE. [Link](https://ieeexplore.ieee.org/document/8613842)
# 

# %% [markdown]
# ## Additonal References
# 
# 1. https://medium.com/data-science-365/principal-component-analysis-pca-with-scikit-learn-1e84a0c731b0
# 2. https://seaborn.pydata.org/generated/seaborn.pairplot.html
# 3. https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html
# 4. https://catboost.ai/en/docs/concepts/python-usages-examples


