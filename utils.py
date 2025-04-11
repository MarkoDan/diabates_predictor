import pandas as pd
from sklearn.model_selection import train_test_split
import torch

def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    # Load the data
    df = pd.read_csv(csv_path)

    #Create the target label based on medical thresholds
    df["Diabetes"] = ((df["HbA1c"] >= 6.5) | (df["Fasting_Blood_Glucose"] >= 126)).astype(int)

    return df

def select_and_clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Selects only user-friendly columns and drops clinical data
    that a normal user wouldn't know.
    """
    selected_columns = [
        "Age",
        "Sex",
        "Ethnicity",
        "BMI",
        "Physical_Activity_Level",
        "Dietary_Intake_Calories",
        "Alcohol_Consumption",
        "Smoking_Status",
        "Family_History_of_Diabetes",
        "Previous_Gestational_Diabetes",
        "Diabetes"  # keep the target column too
    ]
    
    # Filter the dataframe
    df = df[selected_columns].copy()

    # Drop rows with any missing values (simpler for now)
    df = df.dropna()

    return df


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:

    """
    Encodes all categories (text) columns into numeric values.
    Uses simple label encoding (e.g. Male -> 0, Female -> 1)
    """
    categorical_columns = [
        "Sex", 
        "Ethnicity",
        "Physical_Activity_Level",
        "Alcohol_Consumption",
        "Smoking_Status"
    ]

    for col in categorical_columns:
        df[col] = df[col].astype("category").cat.codes
    
    return df

def normalize_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize numerical features to the range [0, 1].
    This improves training stability
    """

    numeric_columns = [
        "Age",
        "BMI",
        "Dietary_Intake_Calories"
    ]

    for col in numeric_columns:
        min_val = df[col].min()
        max_val = df[col].max()
        df[col] = (df[col] - min_val) / (max_val - min_val)
    
    return df

def split_and_tensorize(df, test_size=0.2, random_state=42):
    """
    Splits the data into training and testing sets,
    separates features from labels, and converts everything to PyTorch tensors
    """

    # Separates features and target
    X = df.drop("Diabetes", axis=1).values
    y = df["Diabetes"].values

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    #Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1) # shape (N, 1)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor 