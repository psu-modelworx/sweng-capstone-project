import pytest
import pandas as pd
import json
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from joblib import load
from engines.preprocessing_engine import PreprocessingEngine
from unittest.mock import mock_open, patch, MagicMock


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "age": ["25", "30", "   ", "40", "35"],
        "income": ["50K", "60K", "$55,000", None, "70K"],
        "gender": ["Male", "Female", "male", None, "FEMALE"],
        "zip": ["12345", "12345", "54321", "00000", "99999"],
        "target": ["yes", "no", "yes", "no", None]
    })

@pytest.fixture
def engine():
    df = pd.DataFrame({
        'age': [25, 30],
        'income': [50000, 60000],
        'gender': ['M', 'F'],
        'label': [1, 0]
    })
    target_column = 'label'

    engine = PreprocessingEngine(df=df, target_column=target_column)
    engine.categorical_columns = ['gender']
    engine.target_column = target_column
    engine.original_target_column = target_column
    engine.target_is_categorical = True
    engine.dropped_columns = ['unwanted']
    engine.original_columns = ['age', 'income', 'gender', 'label', 'unwanted']
    engine.final_columns = ['age', 'income', 'gender_F', 'gender_M', 'label_encoded']

    return engine


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
    """TC-11 Verify categorical columns are lowercased and stripped of whitespace."""
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


def test_load_from_files_method():
    """TC-31 Test load_from_files initializes PreprocessingEngine correctly with provided meta and encoders."""
    
    # Prepare dummy metadata dictionary (like loaded from a JSON)
    meta_data = {
        "target_column": "label",
        "categorical_columns": ["gender"],
        "columns_to_remove": ["unwanted"],
        "original_target_column": "label",
        "target_is_categorical": True,
        "original_columns": ["age", "income", "gender", "label", "unwanted"],
        "final_columns": ["age", "income", "gender_F", "gender_M", "label_encoded"],
        "task_type": "classification",
        "dropped_columns": ["unwanted"]
    }
    dummy_feature_encoder = MagicMock(name="feature_encoder")
    dummy_scaler = MagicMock(name="scaler")
    dummy_label_encoder = MagicMock(name="label_encoder")
    engine = PreprocessingEngine.load_from_files(
        clean_df=sample_df(),
        meta=meta_data,
        feature_encoder=dummy_feature_encoder,
        scaler=dummy_scaler,
        label_encoder=dummy_label_encoder
    )

    # make sure all of the props are set correctly
    assert isinstance(engine, PreprocessingEngine)
    assert engine.target_column == "label"
    assert engine.categorical_columns == ["gender"]
    assert engine.columns_to_remove == ["unwanted"]
    assert engine.original_target_column == "label"
    assert engine.target_is_categorical is True
    assert engine.original_columns == ["age", "income", "gender", "label", "unwanted"]
    assert engine.final_columns == ["age", "income", "gender_F", "gender_M", "label_encoded"]
    assert engine.task_type == "classification"
    assert engine.dropped_columns == ["unwanted"]
    assert engine.feature_encoder is dummy_feature_encoder
    assert engine.scaler is dummy_scaler
    assert engine.label_encoder is dummy_label_encoder

def test_scale_continuous_features_in_new_df(engine):
    """TC-32 Verify continuous features in new DataFrame are scaled correctly."""
    engine.scaler = StandardScaler()
    df_train = pd.DataFrame({'age': [20, 30, 40], 'income': [100, 200, 300], 'gender': ['M', 'F', 'F'], 'label': [1, 0, 1]})
    engine.scaler.fit(df_train[['age', 'income']])
    new_df = pd.DataFrame({'age': [25], 'income': [150], 'gender': ['M'], 'label': [1]})
    scaled_df = engine.scale_continuous_features_in_new_df(new_df)

    assert 'age' in scaled_df.columns
    assert abs(scaled_df['age'].values[0]) < 1 

def test_transform_categoricals_in_new_df(engine):
    """TC-33 Verify categorical features in new DataFrame are transformed correctly."""
    engine.feature_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    engine.feature_encoder.fit(pd.DataFrame({'gender': ['M', 'F']}))
    new_df = pd.DataFrame({'age': [25], 'income': [150], 'gender': ['F'], 'label': [1]})
    transformed = engine.transform_categoricals_in_new_df(new_df)

    assert 'gender_F' in transformed.columns
    assert 'gender_M' in transformed.columns
    assert 'gender' not in transformed.columns

def test_encode_target_in_new_df(engine):
    """TC-34 Verify target column in new DataFrame is encoded correctly."""
    engine.label_encoder = LabelEncoder()
    engine.label_encoder.fit(['yes', 'no'])
    df = pd.DataFrame({'label': ['yes', 'no']})
    encoded = engine.encode_target_in_new_df(df)
    
    assert 'label_encoded' in encoded.columns
    assert 'label' not in encoded.columns
    assert list(encoded['label_encoded']) == [1, 0]

def test_remove_dropped_columns(engine):
    """TC-35 Verify unwanted columns are removed from new DataFrame."""
    df = pd.DataFrame({'age': [25], 'unwanted': ['dropped data']})
    cleaned = engine.remove_dropped_columns(df)

    assert 'unwanted' not in cleaned.columns
    assert 'age' in cleaned.columns

def test_clean_new_dataset_column_check(engine, monkeypatch):
    """TC-36 Verify new dataset cleaning results in expected final columns."""
    df = pd.DataFrame({
        'age': [25], 'income': [150], 'gender': ['M'],
        'label': ['yes'], 'unwanted': ['drop']
    })

    # avoids having to fit the encoders and scalers with real data
    monkeypatch.setattr(engine, 'remove_unwanted_columns', lambda d: d)
    monkeypatch.setattr(engine, 'remove_dropped_columns', lambda d: d.drop(columns=['unwanted']))
    monkeypatch.setattr(engine, 'clean_categorical_columns', lambda d: d)
    monkeypatch.setattr(engine, 'clean_continuous_columns', lambda d: d)
    monkeypatch.setattr(engine, 'encode_target_in_new_df', lambda d: d.assign(label_encoded=[1]).drop(columns=['label']))
    monkeypatch.setattr(engine, 'scale_continuous_features_in_new_df', lambda d: d)
    monkeypatch.setattr(engine, 'transform_categoricals_in_new_df', lambda d: d.assign(gender_F=[0], gender_M=[1]).drop(columns=['gender']))

    cleaned = engine.clean_new_dataset(df)
    assert set(cleaned.columns) == set(engine.final_columns)