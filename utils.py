import pandas as pd
import category_encoders as ce
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

def split_data_num_cat(data: pd.DataFrame):
    data_numerical, data_categorical = None, None

    data_numerical = data.select_dtypes(include=['int32', 'int64', 'float64'])
    data_categorical = data.select_dtypes(include=['object'])

    return data_numerical, data_categorical


def split_data_train_test(X, y, ratio=0.8):
    X_train, y_train, X_test, y_test = None, None, None, None

    X_train = X[:int(ratio * X.shape[0])]
    y_train = y[:int(ratio * y.shape[0])]
    X_test = X[int(ratio * X.shape[0]):]
    y_test = y[int(ratio * y.shape[0]):]

    return X_train, y_train, X_test, y_test

def replace_missing_values(data: pd.DataFrame):
    data_numerical, data_categorical = split_data_num_cat(data)

    col_median_numerical = data_numerical.mean().round(2)
    col_median_categorical = data_categorical.mode().iloc[0]

    data_numerical_filled = data_numerical.fillna(col_median_numerical)
    data_categorical_filled = data_categorical.fillna(col_median_categorical)

    return pd.concat([data_numerical_filled, data_categorical_filled], axis=1)

def encode_categorical(data_train: pd.DataFrame, data_test: pd.DataFrame, cols: list) -> tuple[
    pd.DataFrame, pd.DataFrame]:
    encoder = ce.OneHotEncoder(cols=cols, use_cat_names=True)
    return_train_data = encoder.fit_transform(data_train)
    return_test_data = encoder.transform(data_test)

    return return_train_data, return_test_data

def scale_features(data_train: pd.DataFrame, data_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    scaler = StandardScaler()
    return_train_data_scaled = scaler.fit_transform(data_train)
    return_test_data_scaled = scaler.transform(data_test)

    # reconstruct DataFrame with original column names and index
    return_train_data_scaled_df = pd.DataFrame(return_train_data_scaled, columns=data_train.columns, index=data_train.index)
    return_test_data_scaled_df = pd.DataFrame(return_test_data_scaled, columns=data_test.columns, index=data_test.index)
    return return_train_data_scaled_df, return_test_data_scaled_df


def plot_outlier_boxplots(data, n_cols=3):
    numerical, categorical = split_data_num_cat(data)

    nr_cols = len(numerical.columns)
    n_rows = (nr_cols // n_cols) + (nr_cols % n_cols > 0)

    plt.figure(figsize=(n_cols * 5, n_rows * 5))

    for i, col in enumerate(numerical.columns):
        plt.subplot(n_rows, n_cols, i + 1)
        sns.boxplot(y=numerical[col])
        plt.title(col)

    plt.tight_layout()
    plt.show()