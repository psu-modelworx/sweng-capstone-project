import pytest
import numpy as np
from sklearn.datasets import make_classification, make_regression
from engines.modeling_engine import ModelingEngine


@pytest.fixture
def sample_data_classification():
    """ Generate synthetic classification dataset """
    X, y = make_classification(n_samples=100, n_features=10)
    return X[:80], y[:80], X[80:], y[80:]


@pytest.fixture
def sample_data_regression():
    """ Generate synthetic regression dataset """
    X, y = make_regression(n_samples=100, n_features=10)
    return X[:80], y[:80], X[80:], y[80:]


def test_modeling_engine_initialization():
    """ TC-01 Ensure ModelingEngine initializes correctly """
    engine = ModelingEngine(np.array([[1, 2]]), np.array([1]), np.array([[3, 4]]), np.array([0]), task_type="classification")
    assert engine.task_type == 'classification'
    assert isinstance(engine.models, dict)


def test_model_selection():
    """ TC-02 Validate correct model selection based on task type """
    engine_class = ModelingEngine(np.array([[1, 2]]), np.array([1]), np.array([[3, 4]]), np.array([0]), task_type="classification")
    engine_reg = ModelingEngine(np.array([[1, 2]]), np.array([1]), np.array([[3, 4]]), np.array([0]), task_type="regression")

    assert "LogisticRegression" in engine_class.models
    assert "RandomForestRegressor" in engine_reg.models


def test_evaluate_models(sample_data_classification):
    """ TC-03 Verify cross-validation scores are computed correctly """
    X_train, y_train, X_test, y_test = sample_data_classification
    engine = ModelingEngine(X_train, y_train, X_test,
                            y_test, task_type="classification")

    engine.evaluate_models()
    assert isinstance(engine.results["model_scores"], dict)
    assert engine.results["best_model_name"] in engine.results["model_scores"]


def test_tune_best_model(sample_data_classification):
    """ TC-04 Ensure hyperparameter tuning selects optimal parameters """
    X_train, y_train, X_test, y_test = sample_data_classification
    engine = ModelingEngine(X_train, y_train, X_test,
                            y_test, task_type="classification")

    engine.evaluate_models()
    engine.tune_best_model()

    assert isinstance(engine.results["best_params"], dict)


def test_evaluate_final_model(sample_data_classification):
    """ TC-05 Validate final model evaluation computes scores """
    X_train, y_train, X_test, y_test = sample_data_classification
    engine = ModelingEngine(X_train, y_train, X_test,
                            y_test, task_type="classification")

    engine.evaluate_models()
    engine.tune_best_model()
    engine.evaluate_final_model()

    assert "train" in engine.results["final_scores"]
    assert "test" in engine.results["final_scores"]
