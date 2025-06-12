import pytest
import pandas as pd
from engines.preprocessing_engine import PreprocessingEngine


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "age": ["25", "30", "   ", "40", "35"],
        "income": ["50K", "60K", "$55,000", None, "70K"],
        "gender": ["Male", "Female", "male", None, "FEMALE"],
        "zip": ["12345", "12345", "54321", "00000", "99999"],
        "target": ["yes", "no", "yes", "no", None]
    })


def test_initialization(sample_df):
    """TC-09 Verify engine initializes with correct task type and column setup."""
    engine = PreprocessingEngine(
        sample_df, target_column="target", categorical_columns=["gender", "target"])
    assert engine.task_type == "classification"
    assert "gender" in engine.categorical_columns
    assert engine.df.shape == sample_df.shape


def test_remove_unwanted_columns(sample_df):
    """TC-10 Verify specified columns are removed from the DataFrame."""
    engine = PreprocessingEngine(
        sample_df, "target", columns_to_remove=["zip"])
    df = engine.remove_unwanted_columns(sample_df)
    assert "zip" not in df.columns


def test_clean_categorical_columns(sample_df):
    """TC-11 Verify categorical columns columns are lowercased and stripped of whitespace."""
    engine = PreprocessingEngine(
        sample_df, "target", categorical_columns=["gender"])
    df = engine.clean_categorical_columns(sample_df)
    assert df["gender"].iloc[0] == "male"


def test_clean_continuous_columns(sample_df):
    """TC-12 Verify numeric-looking strings are converted to numeric dtype."""
    engine = PreprocessingEngine(
        sample_df, "target", categorical_columns=["gender", "target"])
    df = engine.clean_continuous_columns(sample_df)
    assert pd.api.types.is_numeric_dtype(df["income"])

def test_drop_missing_columns(sample_df):
    """TC-14 Verify columns with excessive missing data are dropped."""
    df = sample_df.copy()
    df["mostly_missing"] = [None, None, None, 1, None]
    engine = PreprocessingEngine(
        df, "target", categorical_columns=["gender", "target"])
    df_clean, columns_to_drop = engine.drop_missing_columns(df)
    assert "mostly_missing" not in df_clean.columns

def test_drop_missing_rows(sample_df):
    """TC-15 Verify rows with missing target values are removed."""
    engine = PreprocessingEngine(
        sample_df, "target", categorical_columns=["gender", "target"])
    original_len = len(engine.df)
    df = engine.drop_missing_rows(sample_df)
    assert len(df) < original_len

def test_handle_missing_data(sample_df):
    """TC-16 Verify missing values are imputed for both numeric and categorical columns."""
    engine = PreprocessingEngine(
        sample_df, "target", categorical_columns=["gender", "target"])
    df = engine.drop_missing_rows(sample_df)
    df = engine.handle_missing_data(df)
    assert df.isnull().sum().sum() == 0

def test_encode_target_column(sample_df):
    """TC-17 Verify categorical target column is encoded and renamed correctly."""
    engine = PreprocessingEngine(
        sample_df, "target", categorical_columns=["target"])
    engine = PreprocessingEngine(sample_df, "target", categorical_columns=["target"])
    engine.encode_target_column()
    assert engine.target_column == "target_encoded"
    assert "target_encoded" in engine.df.columns
    assert "target" not in engine.df.columns

def test_split_features_and_target(sample_df):
    """TC-18 Verify features and target are split correctly after encoding."""
    engine = PreprocessingEngine(sample_df, "target", categorical_columns=["gender", "target"])
    X, y = engine.split_features_and_target()
    assert list(y) == ['yes', 'no', 'yes', 'no', None]
    assert y.name == "target"
    assert "target" not in X.columns
    assert list(X.columns) == ["age", "income", "gender", "zip"]

def test_scale_continuous_columns(sample_df):
    """TC-19 Verify numeric columns are standardized (mean ≈ 0)."""
    df = pd.DataFrame({
        "age": [20, 30, 40, 50, 60],
        "income": [1000, 2000, 3000, 4000, 5000],
        "target": [1, 0, 1, 0, 1]
    })

    engine = PreprocessingEngine(df, "target", categorical_columns=["target"])
    engine.scale_continuous_features()
    assert abs(engine.df["age"].mean()) < 1e-6
    assert abs(engine.df["income"].mean()) < 1e-6
    assert abs(engine.df["age"].std(ddof=0) - 1) < 1e-6
    assert abs(engine.df["income"].std(ddof=0) - 1) < 1e-6

def test_fit_one_hot_encode_categoricals(sample_df):
    """TC-20 Verify categorical columns are one-hot encoded correctly."""
    df = pd.DataFrame({
        "age": [25, 30, 35],
        "gender": ["Male", "Female", "Other"],
        "target": [1, 0, 1]
    })
    engine = PreprocessingEngine(df, "target", categorical_columns=["gender"])
    engine.fit_categorical_encoder()
    assert "gender" not in engine.df.columns
    expected_columns = {"gender_Male", "gender_Other"} # dropped first category
    assert expected_columns.issubset(engine.df.columns)
    assert "gender_Female" not in engine.df.columns

def test_train_test_split_data(sample_df):
    """TC-21 Verify train/test split returns correct row counts and structure."""
    df = pd.DataFrame({
        "age": [25, 30, 35, 40],
        "income": [50000, 60000, 70000, 80000],
        "gender": ["Male", "Female", "Other", "Male"],
        "target": [1, 0, 1, 0]
    })
    engine = PreprocessingEngine(df, "target", categorical_columns=["gender"])
    X, y = engine.split_features_and_target()
    X_train, X_test, y_train, y_test = engine.train_test_split_data(X, y)
    assert len(X_train) + len(X_test) == len(X)
    assert len(y_train) + len(y_test) == len(y)
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]

def test_run_preprocessing_engine(sample_df):
    """TC-22 Verify full preprocessing pipeline runs and returns expected outputs."""
    engine = PreprocessingEngine(
        sample_df, "target", categorical_columns=["gender", "target"])
    X_train, X_test, y_train, y_test, task_type = engine.run_preprocessing_engine()
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(task_type, str)

def test_summary_outputs_correct_structure_and_values(sample_df):
    """TC-23Verify summary output is a dictionary with expected keys and values."""
    engine = PreprocessingEngine(
        df=sample_df,
        target_column="target",
        categorical_columns=["target"]
    )
    engine.run_preprocessing_engine()
    summary = engine.summary()
    assert isinstance(summary, dict)
    assert summary["task_type"] == "classification"
    assert "features" in summary
    assert "missing_values" in summary
    assert "label_mapping" in summary

def test_decode_target_returns_original_labels(sample_df):
    """TC-24 Verify encoded target values can be decoded back to original labels."""
    df = pd.DataFrame({
        "age": [25, 30, 35, 40],
        "gender": ["Male", "Female", "Female", "Male"],
        "target": ["yes", "no", "yes", "no"]
    })
    engine = PreprocessingEngine(df, "target", categorical_columns=["target"])
    engine.encode_target_column()
    encoded_values = engine.df[engine.target_column].tolist()
    decoded = engine.decode_target(encoded_values)
    assert decoded.tolist() == ["yes", "no", "yes", "no"]
