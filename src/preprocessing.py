from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def get_preprocessor(X):
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    cat_cols = X.columns.drop(num_cols).tolist()
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])
    return preprocessor
