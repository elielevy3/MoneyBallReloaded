# pour multiplier/div un df par une ligne/colonne

final_advanced = agg_ad.div((games) , axis=0)
df = df.mul(games, axis=0)


# pour multiplier toute une col/ligne par le meme scalaire

 df['quantity'] = df['quantity'].apply(lambda x: x*-1)  


#access the nth element of df
df.iloc[n]

#access df like a 2D array
df.iloc[n][m]

# pour drop une colonne

final_advanced = final_advanced.drop(columns=["G"])


#Pour concat des df en ligne ou en colonne
final_advanced = pd.concat([final_advanced, games], axis=1)

#from dict of dict to list of list
[list(z.values()) for y,z in x.items()]


#Pour aggreger des données avec des fonctions différentes selon la colonne

agr = {'MP':['sum'],'G': ['sum'], 'PER':['sum'],'TS%':['sum'],'3PAr':['sum'],'TRB%':['sum'],'USG%':['sum'], 'OWS':['sum'], 'DWS':['sum']}

agg_ad =summed_ad.groupby(['Player']).agg(agr)


#Pour appliquer une fonction a tous les éléments d'un df

final_advanced = final_advanced.apply(round, args=[2])

# pour renommer les colonnes d'un df

res.columns = ["Player", "G", "MP", "PER", "TS%", "3PAr", "TRB%", "USG%", "OWS", "DWS"]


# pour ne garder que certaines col d'un df => reste des df et non des séries
df = df.loc[:,(df.columns != "Player") & (df.columns != "G")]

df = df[["NomCol1", "NomCol2"]]

df = df.NomCol1

# filtrer un df par rapport aux valeurs des colonnes

df = df[df["G"] > 100 ]

# pour concat en ligne des df ayant les memes colonnes

df = df.append(df_2).append(df_3).append(df_4).append(df_5)

# Pour scaler les donnes 

final_advanced = final_advanced - final_advanced.min()
final_advanced = final_advanced / ( final_advanced.max() - final_advanced.min() )
final_advanced


# Pour transformer un np array en df

df = pd.DataFrame(np_arr)

# pour obtenir des stats de bases sur les colonnes

df.describre()

# pour avoir les premieres lignes

df.head()

# To print lines which fulfil a column condition

df_2020[df_2020["Player"] == "Kyle Alexander"]

# keep index while for-loop iterating

for index, item in enumerate(array):
	print(index)
	print(item)


# get the number of rows from df

len(df.index)

# change the type of a whole column
df = df.astype({"Column_name": str}, errors='raise')


# merge 2 df by column values
# to only keep row in the original df which players value are in the second
df_fcm = pd.merge(df_fcm, unclustered_players, on="Player")


#sort dataframe

df = df.sort_values(by=["col_name"])

# to keep rows from df which column value are not in a list
			
df = df[~df['col'].isin(list_of_values)]


#tranform series to list
df = df["Nomcol"].tolist()



#to set the df index
df.set_index("name_col")

#rename one single col
df1 = df.rename(columns={'Name': 'EmpName'})    


# faire une selection sur les rows avec plusieurs conditions => ATTENTION AUX PARENTHESES ET AUX OP BINAIRES

agg_advanced = agg_advanced.loc[(agg_advanced["MP"]["sum"] > 2500) & (agg_advanced["G"]["sum"] > 100)]


# to convert a dict of dict into a list of list of number => works under the assomption that keys are in the same order in both dimension
only_number_matrix = [list(value.values()) for key, value in dict_of_dict.items()]


# extract value from df
df.iloc[0]['col']


 #########################################################
 ############# MACHINE LEARNING WITH SKLEARN #############
 #########################################################


# to build a decision tree regressor with sklearn
# without max_leaves parameter it stops until theres is only one x(i) per leaves

from sklearn.tree import DecisionTreeRegressor  
#specify the model. 
model = DecisionTreeRegressor(random_state=0)

# Fit the model
model.fit(X, y)

# Predict => return array-like of predicted values

predictions = model.predict(X)


# split the data for validation
from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

# calculate the Mean Square Error

from sklearn.metrics import mean_absolute_error
mean_absolute_error(y, predicted)

# calculate the Mean Absolute Error

from sklearn.metrics import mean_absolute_error
val_mae = mean_absolute_error(val_y, val_predictions)

# dict omprehension

candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
scores = {leaf_size: get_mae(leaf_size, train_X, val_X, train_y, val_y) for leaf_size in candidate_max_leaf_nodes}
best_model = min(scores, key=scores.get)

# use random forest

from sklearn.ensemble import RandomForestRegressor

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)



# Get names of columns with missing values
cols_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()]


# get the number of missing value per columns

X.isnull().sum()

# get the total number of missing value in the dataset

X.isnull().sum().sum()

# nb of rows with at least a na value

sum([True for idx,row in X_full.iterrows() if any(row.isnull())])


# impute missing values

from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()
# first we fit
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
# then we apply we the fittest value
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))



# iterate over series

for index, elem in s.items()


# to replace categorical data with binary valie (one extra columns for each possible value)

X = pd.get_dummies(X)

# get the value of the nth column (except the index) in a pd serie

result.get(n) 

# to only keep numerical value

X_test = X_test.select_dtypes(exclude=['object'])


# to remove columns that are not in name list

df = df[df.columns.intersection(final_table_columns)]


# Get list of categorical variables
s = (X_train.dtypes == 'object')
object_cols = list(s[s].index)


# label encoding => from "low", "medium", "high" to 1, 2, 3
label_X_train[col] = label_encoder.fit_transform(X_train[col])
label_X_valid[col] = label_encoder.transform(X_valid[col])


#checking if catagorical column has the same range from 2 df 
# with set, we remove duplicate to only get the values taken in the column
good_label_cols = [col for col in object_cols if set(X_train[col]) == set(X_valid[col])]
        

#get the cardinality of every categorical columns
s = (X_train.dtypes == 'object')
object_cols = list(s[s].index)
object_nunique = [X_train[col].nunique() for col in object_cols]


# get the low and high cardinality columns

card_limit = 10
# Columns that will be one-hot encoded
low_cardinality_cols = [col for col in object_cols if X_train[col].nunique() < card_limit]

# Columns that will be dropped from the dataset
high_cardinality_cols = list(set(object_cols)-set(low_cardinality_cols))


# to One-hot encode categorical data

# Remove the high cardinality columns
X_train = X_train.drop(high_cardinality_cols, axis=1)
# OH encode low cardinality columns
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[low_cardinality_cols]))
# retrieve the index because fit_transform lose it 
OH_cols_train.index = X_train.index
# remove columns that have been OH encoded
num_cols_X_train = X_train.drop(low_cardinality_cols, axis=1)
# concat OH columns with numerical columns
OH_X_train = pd.concat([num_cols_X_train, OH_cols_train] , axis=1)


# pipeline

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])


# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])


# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

# Preprocessing of training data, fit model 
my_pipeline.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds = my_pipeline.predict(X_valid)


# fold the dataset for cross validation => array of metrics values
scores = -1 * cross_val_score(my_pipeline, X, y, cv=3, scoring='neg_mean_absolute_error')



# XGBoosting
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
my_model.fit(X_train, y_train, 
             early_stopping_rounds=5, 
             eval_set=[(X_valid, y_valid)], 
             verbose=False)

