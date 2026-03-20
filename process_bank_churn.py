# Imports
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns that are not needed for modeling.

    Args:
        df: Input dataframe.

    Returns:
        DataFrame without unnecessary columns.
    """
    return df.drop(columns=['Surname'], errors='ignore')


def get_feature_and_target_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Define input and target columns.

    Args:
        df: Input dataframe.

    Returns:
        Tuple of (input columns, target column).
    """
    input_cols = list(df.columns)[2:-1]
    target_col = list(df.columns)[-1:]
    return input_cols, target_col


def split_numeric_categorical(
    df: pd.DataFrame, input_cols: List[str]
) -> Tuple[List[str], List[str]]:
    """
    Split columns into numeric and categorical.

    Args:
        df: Input dataframe.
        input_cols: Feature column names.

    Returns:
        Tuple of (numeric columns, categorical columns).
    """
    numeric_cols = df[input_cols].select_dtypes('number').columns.tolist()
    categorical_cols = df[input_cols].select_dtypes('object').columns.tolist()
    return numeric_cols, categorical_cols


def split_data(
    df: pd.DataFrame, target_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataframe into train and validation sets.

    Args:
        df: Input dataframe.
        target_col: Target column name.

    Returns:
        Train and validation dataframes.
    """
    return train_test_split(
        df,
        test_size=0.25,
        random_state=42,
        stratify=df[target_col]
    )


def create_inputs_targets(
    df: pd.DataFrame, input_cols: List[str], target_col: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separate inputs and targets.

    Args:
        df: Input dataframe.
        input_cols: Feature columns.
        target_col: Target column.

    Returns:
        Tuple of (inputs, targets).
    """
    return df[input_cols].copy(), df[target_col].copy()


def fit_onehot_encoder(
    df: pd.DataFrame, categorical_cols: List[str]
) -> OneHotEncoder:
    """
    Fit OneHotEncoder on training data.

    Args:
        train_inputs: Training features.
        categorical_cols: Categorical column names.

    Returns:
        Fitted OneHotEncoder.
    """
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    if not categorical_cols:
        return None
    
    if categorical_cols:
       encoder.fit(df[categorical_cols])
    
    return encoder


def apply_onehot_encoding(
    df: pd.DataFrame,
    encoder: OneHotEncoder,
    categorical_cols: List[str]
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Apply one-hot encoding to dataframe.

    Args:
        df: Input dataframe.
        encoder: Fitted encoder.
        categorical_cols: Categorical columns.

    Returns:
        Updated dataframe and list of encoded column names.
    """
    df = df.copy()
    
    # Ensure all categorical columns exist
    for col in categorical_cols:
        if col not in df.columns:
            df[col] = None

    encoded_array = encoder.transform(df[categorical_cols])
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))

    encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols, index=df.index)

    df = df.drop(columns=categorical_cols)
    df = pd.concat([df, encoded_df], axis=1)

    return df, encoded_cols


def fit_scaler(
    df: pd.DataFrame,
    columns: List[str]
) -> MinMaxScaler:
    """
    Fit and apply MinMaxScaler.

    Args:
        train_df: Training features.
        val_df: Validation features.
        columns: Columns to scale.

    Returns:
        Scaled train, scaled validation, and fitted scaler.
    """
    scaler = MinMaxScaler()

    if columns:
        scaler.fit(df[columns])

    return scaler


def apply_scaler(
    df: pd.DataFrame,
    scaler: MinMaxScaler,
    columns: List[str]
) -> pd.DataFrame:
    """
    Apply pre-trained MinMaxScaler.

    Args:
        df: Input dataframe.
        scaler: Pre-trained scaler.
        columns: Columns to scale.

    Returns:
        Scaled dataframe.
    """
    data = df.copy()
    data.loc[:, columns] = scaler.transform(data[columns])
    return data


def preprocess_data(
    raw_df: pd.DataFrame,
    scale_numeric: bool = False
) -> Dict[str, Any]:
    """
    Training preprocessing pipeline.

    Returns:
        Processed train/validation sets and fitted transformers.
    """
    df = drop_unnecessary_columns(raw_df)

    input_cols, target_col = get_feature_and_target_columns(df)
    numeric_cols, categorical_cols = split_numeric_categorical(df, input_cols)

    train_df, val_df = split_data(df, target_col[0])

    X_train, y_train = create_inputs_targets(train_df, input_cols, target_col)
    X_val, y_val = create_inputs_targets(val_df, input_cols, target_col)

    encoder = fit_onehot_encoder(X_train, categorical_cols)

    X_train, encoded_cols = apply_onehot_encoding(X_train, encoder, categorical_cols)
    X_val, _ = apply_onehot_encoding(X_val, encoder, categorical_cols)

    feature_cols = numeric_cols + encoded_cols
    scaler: Optional[MinMaxScaler] = None

    if scale_numeric:
        scaler = fit_scaler(X_train, feature_cols)
        X_train= apply_scaler(X_train, scaler, feature_cols)
        X_val = apply_scaler(X_val, scaler, feature_cols)

    return {
        'X_train': X_train,
        'train_targets': y_train,
        'X_val': X_val,
        'val_targets': y_val,
        'input_cols': feature_cols,
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols,
        'scaler': scaler,
        'encoder': encoder
    }


def preprocess_new_data(
    raw_df: pd.DataFrame,
    input_cols: List[str],
    numeric_cols: List[str],
    categorical_cols: List[str],
    encoder: OneHotEncoder,
    scaler: Optional[MinMaxScaler] = None,
    scale_numeric: bool = False,
) -> Dict[str, Any]:
    """
    Inference preprocessing pipeline using pre-trained transformers.

    Args:
        raw_df: New data.
        numeric_cols: Numeric columns from training.
        categorical_cols: Categorical columns from training.
        encoder: Pre-trained encoder.
        scaler: Pre-trained scaler.
        scale_numeric: Whether to apply scaling.

    Returns:
        Processed feature matrix.
    """
    df = drop_unnecessary_columns(raw_df)

    X = df.copy()

    X, encoded_cols = apply_onehot_encoding(X, encoder, categorical_cols)

    feature_cols = numeric_cols + encoded_cols

    # Ensure feature consistency
    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0

    X = X[feature_cols]

    if scale_numeric and scaler is not None:
        X = apply_scaler(X, scaler, feature_cols)

    return {
        'X': X
    }