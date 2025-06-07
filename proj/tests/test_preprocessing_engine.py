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
    engine.remove_unwanted_columns()
    assert "zip" not in engine.df.columns


def test_clean_text_columns_lowercases_and_strips(sample_df):
    """TC-11 Verify text columns are lowercased and stripped of whitespace."""
    engine = PreprocessingEngine(
        sample_df, "target", categorical_columns=["gender"])
    engine.clean_text_columns()
    assert engine.df["gender"].iloc[0] == "male"


def test_clean_numeric_columns_converts_strings_to_numbers(sample_df):
    """TC-12 Verify numeric-looking strings are converted to numeric dtype."""
    engine = PreprocessingEngine(
        sample_df, "target", categorical_columns=["gender", "target"])
    engine.clean_numeric_columns()
    assert pd.api.types.is_numeric_dtype(engine.df["income"])


def test_detect_and_convert_categoricals(sample_df):
    """TC-13 Verify specified categorical columns are converted to category dtype."""
    engine = PreprocessingEngine(
        sample_df, "target", categorical_columns=["gender", "target"])
    engine.clean_numeric_columns()
    engine.clean_text_columns()
    engine.detect_and_convert_categoricals()
    assert all(engine.df[col].dtype.name == "category"
               for col in engine.categorical_columns if col in engine.df)


def test_drop_missing_columns(sample_df):
    """TC-14 Verify columns with excessive missing data are dropped."""
    df = sample_df.copy()
    df["mostly_missing"] = [None, None, None, 1, None]
    engine = PreprocessingEngine(
        df, "target", categorical_columns=["gender", "target"])
    engine.drop_missing_columns()
    assert "mostly_missing" not in engine.df.columns


def test_drop_missing_rows_removes_rows_missing_target(sample_df):
    """TC-15 Verify rows with missing target values are removed."""
    engine = PreprocessingEngine(
        sample_df, "target", categorical_columns=["gender", "target"])
    original_len = len(engine.df)
    engine.drop_missing_rows()
    assert len(engine.df) < original_len


def test_handle_missing_data_fills_values(sample_df):
    """TC-16 Verify missing values are imputed for both numeric and categorical columns."""
    engine = PreprocessingEngine(
        sample_df, "target", categorical_columns=["gender", "target"])
    engine.drop_missing_rows()
    engine.handle_missing_data()
    assert engine.df.isnull().sum().sum() == 0


def test_encode_target_column(sample_df):
    """TC-17 Verify categorical target column is encoded and renamed correctly."""
    engine = PreprocessingEngine(
        sample_df, "target", categorical_columns=["target"])
    engine.drop_missing_rows()
    engine.handle_missing_data()
    engine.encode_target_column()
    assert engine.target_column == "target_encoded"
    assert "target_encoded" in engine.df.columns


def test_split_features_and_target(sample_df):
    """TC-18 Verify features and target are split correctly after encoding."""
    engine = PreprocessingEngine(
        sample_df, "target", categorical_columns=["target"])
    engine.drop_missing_rows()
    engine.handle_missing_data()
    engine.encode_target_column()
    X, y = engine.split_features_and_target()
    assert engine.target_column in y.name
    assert engine.target_column not in X.columns


def test_scale_numerical_columns_standardizes_values(sample_df):
    """TC-19 Verify numeric columns are standardized (mean ≈ 0)."""
    engine = PreprocessingEngine(
        sample_df, "target", categorical_columns=["gender", "target"])
    engine.clean_numeric_columns()
    engine.handle_missing_data()
    engine.encode_target_column()
    engine.scale_numerical_columns()
    assert abs(engine.df["age"].mean()) < 1e-6  # Mean ~ 0


def test_one_hot_encode_categorical_features(sample_df):
    """TC-20 Verify categorical columns are one-hot encoded correctly."""
    engine = PreprocessingEngine(
        sample_df, "target", categorical_columns=["gender", "target"])
    engine.drop_missing_rows()
    engine.handle_missing_data()
    engine.one_hot_encode_categorical_features()
    assert any("gender_" in col for col in engine.df.columns)


def test_train_test_split_data_returns_correct_shapes(sample_df):
    """TC-21 Verify train/test split returns correct row counts and structure."""
    engine = PreprocessingEngine(
        sample_df, "target", categorical_columns=["target"])
    engine.drop_missing_rows()
    engine.handle_missing_data()
    engine.encode_target_column()
    X, y = engine.split_features_and_target()
    X_train, X_test, y_train, y_test = engine.train_test_split_data(X, y)
    assert len(X_train) + len(X_test) == len(X)


def test_run_preprocessing_engine_completes(sample_df):
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
    engine = PreprocessingEngine(
        df=sample_df,
        target_column="target",
        categorical_columns=["target"]
    )
    engine.run_preprocessing_engine()

    encoded = [0, 1]
    decoded = engine.decode_target(encoded)
    assert set(decoded).issubset({"yes", "no"})
