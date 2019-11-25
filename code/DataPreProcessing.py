import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import category_encoders as ce 

from collections import Counter 

from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split 
from sklearn import neural_network as nn 

# reading data from csv 
ori_data = pd.read_csv('../data/tcd-ml-1920-group-income-train.csv', low_memory=False) 
test_data = pd.read_csv('../data/tcd-ml-1920-group-income-test.csv', low_memory=False) 

# renaming the columns for easier use 
rename_dic = { 
    "Year of Record":"YEAR", 
    "Housing Situation":"HOUSE", 
    "Crime Level in the City of Employement":"CRIME", 
    "Work Experience in Current Job [years]":"WORK", 
    "Satisfation with employer":"SATIS", 
    "Gender":"SEX", 
    "Age":"AGE", 
    "Country":"COUNT", 
    "Size of City":"CITY", 
    "Profession":"PROF", 
    "University Degree":"UNI", 
    "Wears Glasses":"GLASS", 
    "Hair Color":"HAIR", 
    "Body Height [cm]":"BODY", 
    "Yearly Income in addition to Salary (e.g. Rental Income)":'ADD', 
    "Total Yearly Income [EUR]":'TOTAL'} 

ori_data = ori_data.rename(columns=rename_dic) 
test_data = test_data.rename(columns=rename_dic) 
ori_data=ori_data[ori_data['WORK']!= '#NUM!'] 
test_data['WORK'].replace(['#NUM!'], [0], inplace=True) 
ori_data['ADD'] = ori_data['ADD'].map(lambda x: float(x.rstrip(' EUR'))) 
test_data['ADD'] = test_data['ADD'].map(lambda x: float(x.rstrip(' EUR'))) 
ori_data['IsTrain']=1 
test_data['IsTrain']=0 

# Kill outliers 
condition1 = (ori_data['TOTAL']>400000) & (ori_data['YEAR']<1990) | (ori_data['TOTAL']>250000) & (ori_data['YEAR']<1980) 
ori_data_1 = ori_data.drop(ori_data[condition1].index) 


def subset_by_iqr(df, column, whisker_width=1.1): 
    """Remove outliers from a dataframe by column, including optional 
       whiskers, removing rows for which the column value are 
       less than Q1-1.5IQR or greater than Q3+1.5IQR. 
    Args: 
        df (`:obj:pd.DataFrame`): A pandas dataframe to subset 
        column (str): Name of the column to calculate the subset from. 
        whisker_width (float): Optional, loosen the IQR filter by a 
                               factor of `whisker_width` * IQR. 
    Returns: 
        (`:obj:pd.DataFrame`): Filtered dataframe 
    """ 
    # Calculate Q1, Q2 and IQR 
    q1 = df[column].quantile(0.25)                  
    q3 = df[column].quantile(0.75) 
    iqr = q3 - q1 
    # Apply filter with respect to IQR, including optional whiskers 
    filter = (df[column] >= q1 - whisker_width*iqr) & (df[column] <= q3 + whisker_width*iqr) 
    return df.loc[filter] 

# if you want to remove more outliers then uncomment the below code 
#ori_data_1 = subset_by_iqr(ori_data, 'Total Yearly Income [EUR]', 4.0) 

        
#concat test and train dataset 
data = pd.concat([ori_data_1,test_data],ignore_index=True) 
data.isnull().sum() 

# handling NAs, unknowns, 0's 
data['HOUSE'].unique() 
data['HOUSE'].replace(['0',0,'nA'], ['unknown','unknown','unknown'], inplace=True) 
data['HOUSE'].replace(['unknown'], ['unknownHousingSituation'], inplace=True) 

data['SEX'].unique() 
data['SEX'].replace(['other','0','unknown','f'], ['unknownGender','unknownGender','unknownGender','female'], inplace=True) 

data['UNI'].unique() 
data['UNI'].replace(['0'], ['unknownDegree'], inplace=True) 

data['HAIR'].unique() 
data['HAIR'].replace(['0','Unknown'], ['unknownHairColor','unknownHairColor'], inplace=True) 

#Handling NAN's (i.e actual null values) 
fill_col_dict = { 
    'YEAR': 1979, 
    'SATIS': 'Average', 
    'SEX': 'male', 
    'COUNT': 'Honduras', 
    'PROF': 'payment analyst', 
    'UNI': 'Bachelor', 
    'HAIR': 'Black', 
    } 
for col in fill_col_dict.keys(): 
    data[col] = data[col].fillna(fill_col_dict[col]) 
     
    
# Apply bins to continuous data 
data['BODY'] = pd.qcut(data['BODY'], 3, labels=['Short','Normal','Tall']) 
data['CRIME'] = pd.qcut(data['CRIME'],3, labels=['Low', 'Medium', 'High']) 
data['AGE'] = pd.qcut(data['AGE'], 13,labels=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm']) 

# concat features function     
def create_cat_con(df,cats,cons,normalize=True):   
     for i,cat in enumerate(cats): 
        vc = df[cat].value_counts(dropna=False, normalize=normalize).to_dict() 
        nm = cat + '_FE_FULL' 
        df[nm] = df[cat].map(vc) 
        df[nm] = df[nm].astype('float32') 
        for j,con in enumerate(cons): 
            new_col = cat +'_'+ con 
            print('timeblock frequency encoding:', new_col) 
            df[new_col] = df[cat].astype(str)+'_'+df[con].astype(str)   
            temp_df = df[new_col] 
            fq_encode = temp_df.value_counts(normalize=True).to_dict() 
            df[new_col] = df[new_col].map(fq_encode) 
            df[new_col] = df[new_col]/df[cat+'_FE_FULL'] 
     return df 

cats = ['YEAR', 'HOUSE', 'WORK', 
        'SATIS', 'SEX', 'AGE', 
        'COUNT', 'PROF', 'UNI', 'GLASS', 'HAIR'] 
cons = ['CRIME', 'CITY', 'BODY', 'ADD'] 

#concating features 
data = create_cat_con(data,cats,cons) 

#deleting irrelevant features 

del_col = set(['GLASS', 'GLASS_BODY', 'WORK_CITY', 'GLASS_FE_FULL', 'HAIR_BODY', 
               'HAIR_FE_FULL', 'UNI_BODY', 'SEX_BODY', 'SATIS_BODY', 'HAIR']) 
delete = set(del_col) 
feature =  list(set(data) - delete) 
data = data[feature] 

# uncomment this code if you want to do label encoding 
#for col in data.dtypes[data.dtypes == 'object'].index.tolist(): 
    #feat_le = LabelEncoder() 
    #feat_le.fit(data[col].unique().astype(str)) 
    #data[col] = feat_le.transform(data[col].astype(str)) 
     
#one hot encoding for variables with less unique categories 
one_hot = pd.get_dummies(data['SATIS']) 
data = data.drop('SATIS',axis = 1) 
data = data.join(one_hot) 

one_hot = pd.get_dummies(data['SEX']) 
data = data.drop('SEX',axis = 1) 
data = data.join(one_hot) 

one_hot = pd.get_dummies(data['BODY']) 
data = data.drop('BODY',axis = 1) 
data = data.join(one_hot) 

one_hot = pd.get_dummies(data['CRIME']) 
data = data.drop('CRIME',axis = 1) 
data = data.join(one_hot) 
     
#Splitting back into train and test (needs to be done earlier to 
#Target encoding but before label encoding(in case you use label encoding 
#thats why between both these codes)) 
X_train=data[data['IsTrain']==1] 
Y_train= X_train['TOTAL'] 

# target encoding for variables having too many unique values 
te = ce.TargetEncoder(cols=['HOUSE','COUNT','PROF','UNI','AGE']).fit(X_train, Y_train) 
data = te.transform(data) 

X_train=data[data['IsTrain']==1] 
X_test=data[data['IsTrain']==0] 
X_test_id = X_test['Instance'] 

del X_train['TOTAL'] 
del X_train['Instance'] 
del X_train['IsTrain'] 
del X_test['TOTAL'] 
del X_test['Instance'] 
del X_test['IsTrain'] 

#splitting train set into training and validating set 
x_train, x_val, y_train, y_val = train_test_split(X_train,Y_train,test_size=0.2,random_state=1234) 

# Create Neural Network, use function as may use multiple instances
def setupNN(): 
    return nn.MLPRegressor( 
            hidden_layer_sizes = (100, 100, 100), 
            max_iter = 100, 
            tol = 0.00001, 
            n_iter_no_change = 10, 
            early_stopping = True, 
            learning_rate = 'adaptive', 
            learning_rate_init = 0.0005, 
            )

neuralNet = setupNN() 

print("Starting Training") 
neuralNet.fit(x_train, y_train) 
print("Iterations: ", neuralNet.n_iter_) 

#Calculating both RMSE and MAE 
from sklearn.metrics import mean_absolute_error 

y_val_test = neuralNet.predict(x_val) 
val_mae = mean_absolute_error(y_val, y_val_test) 

print("Validation MAE :", val_mae) 

y_test = neuralNet.predict(X_test) 

sub_df = pd.DataFrame({'Instance':X_test_id, 
                       'Total Yearly Income [EUR]':y_test}) 
sub_df.head() 
sub_df.to_csv("../output.csv",index=False) 
