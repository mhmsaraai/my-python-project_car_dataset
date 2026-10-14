import pandas as pd
df=pd.read_csv('new_claen_car.csv')
print(df.shape[0])

numeric_cols=df.select_dtypes(include=['int64','float64'])
print(numeric_cols)

categorical_cols=df.select_dtypes(include=['object'])
#print(categorical_cols.columns)

target=df['price']
#print(target)
#print(df['engine'].haed(10))

df['horsepower']=df['engine'].str.extract(r'(/d+/.?/d*)HP')
df['horsepower']=pd.to_numeric(df['horsepower'])

df['engine_size']=df['engine'].str.extract(r'(/d+/.?/d*)L')
df['engine_size']=pd.to_numeric(df['engine_size'])

df['engine_type']=df['engine'].str.extract(r'(v/d+I/d+)')
print(df[['engine','horsepower','engine_size','engine_type']])

print(df['transmission'].value_counts)

def claen_transmission(value):

    if pd.isna(value):
        return 'others'
    
    value=str(value).lower()

    if 'dual shift' in value:
        return 'dual shift'

    elif 'cvt' in value:
        return 'cvt'
    
    elif 'm/t' in value or 'manual' in value:
       return 'Manual'
    
    elif 'a/t' in value or 'automatic' in value:
        return 'Automatic'
    
    else:
        return 'other'

df['transmission_clean']=df['transmission'].apply(claen_transmission)
print(df[['transmission','transmission_clean']])    

cols=['brand','fuel_type','accident','clean_title','engine_type','transmission_clean']

for col in cols:
    print(f'{col}: {df[col].nunique()}')

#label encoder 

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

df['accident']=le.fit_transform(df['accident'])
df['clean_title']=le.fit_transform(df['clean_title']) 

#one_hot

df=pd.get_dummies(
    data=df,
    columns=['brand','fuel_type','engine_type','transmission_clean']
)
print(df.head(10))

df.drop(columns=['transmission','engine'],inplace=True)
print(df.columns)

print(df.shape)
print(df.info)
print(df.isnull().sum())

import matplotlib.pyplot as plt 
import seaborn as sns 


correlation = df.select_dtypes(include='number').corr()
plt.figure(figsize=(8,4))
sns.heatmap(correlation)
plt.title('haetmap of carr')
plt.show

print(correlation['price'].sort_values(ascending=False))
df.to_csv('cars_features.csv',index=False)
