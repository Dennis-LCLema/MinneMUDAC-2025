import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression


BBBS_train_df = pd.read_excel("Training-Restated.xlsx")
BBBS_test_df = pd.read_excel("Test-Truncated-Restated.xlsx")
predictions = "Testset_Predictions_Submit.csv"



#converts county name for Big to a factor
BBBS_train_df["Big County"] = pd.factorize(BBBS_train_df["Big County"])[0]

#Printing to confirm change(s)
print(BBBS_train_df.dtypes.to_string())



X = BBBS_train_df[["Big County", "Big Age"]]
y = BBBS_train_df["Match Length"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

#Ridge Model
ridge = Ridge(alpha=1)
ridge.fit(X_train, y_train)

y_pred_ridge = ridge.predict(X_val)
rmse_ridge = np.sqrt(mean_squared_error(y_val, y_pred_ridge))
print(f"Ridge RMSE: {rmse_ridge:.2f}")


#Lasso Model
lasso = Lasso(alpha=1)
lasso.fit(X_train, y_train)

y_pred_lasso = lasso.predict(X_val)
rmse_lasso = np.sqrt(mean_squared_error(y_val, y_pred_lasso))
print(f"Lasso RMSE: {rmse_lasso:.2f}")


#ElasticNet Model
elastic = ElasticNet(alpha=1, l1_ratio=0.5)
elastic.fit(X_train, y_train)

y_pred_elastic = elastic.predict(X_val)
rmse_elastic = np.sqrt(mean_squared_error(y_val, y_pred_elastic))
print(f"ElasticNet RMSE: {rmse_elastic:.2f}")