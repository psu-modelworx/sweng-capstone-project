import pytest
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
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
    engine_reg = ModelingEngine(np.array([[1, 2]]), np.array([1]), np.array([[3, 4]]), np.array([0]), task_type="regression")
    assert "RandomForestRegressor" in engine_reg.models
    assert isinstance(engine_reg.models, dict)


def test_tune_model(sample_data_classification):
    """ TC-02 Validate tune model functionality for classification """

    X_train, y_train, X_test, y_test = sample_data_classification

    engine = ModelingEngine(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        task_type='classification'
    )
    model = LogisticRegression(max_iter=1000)
    tuned_model = engine.tune_model(model)

    assert tuned_model is not None
    tuned_results = engine.results.get('tuned', {})
    assert 'LogisticRegression' in tuned_results
    log_reg_info = tuned_results['LogisticRegression']
    assert 'best_params' in log_reg_info
    assert 'best_score' in log_reg_info
    assert 'optimized_model' in log_reg_info
    assert isinstance(log_reg_info['optimized_model'], LogisticRegression)
    assert log_reg_info['best_score'] >= 0
    

def test_fit_model(sample_data_classification):
    """ TC-03 Verify that the model has been fit and populated with coefficients """
    X_train, y_train, X_test, y_test = sample_data_classification

    engine = ModelingEngine(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        task_type='classification'
    )
    model = LogisticRegression(max_iter=1000)
    engine.fit_model(model)

    assert hasattr(model, "coef_"), "Model should have been fitted and have a coef_ attribute"


def test_evaluate_model(sample_data_classification):
    """ TC-04 Validate model evaluation computes train and test scores """
    X_train, y_train, X_test, y_test = sample_data_classification

    engine = ModelingEngine(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        task_type='classification'
    )
    model = LogisticRegression(max_iter=1000)
    engine.fit_model(model) # need a fitted model to evaluate
    model_name = model.__class__.__name__
    engine.results['tuned'] = {
        model_name: {"optimized_model": model}
    }
    engine.evaluate_model(model)

    tuned_model_info = engine.results['tuned'][model_name]

    assert "train_accuracy" in tuned_model_info["final_scores"]
    assert "test_accuracy" in tuned_model_info["final_scores"]
    assert 0.0 <= tuned_model_info["final_scores"]["train_accuracy"] <= 1.0
    assert 0.0 <= tuned_model_info["final_scores"]["test_accuracy"] <= 1.0


def test_get_best_untuned_model():
    """ TC-05 Ensure the best untuned model is returned correctly """
    
    lr_model = LogisticRegression(max_iter=1000)
    rf_model = RandomForestClassifier()
    engine = ModelingEngine(
        X_train=[[0, 1], [1, 0]], y_train=[0, 1], X_test=[[0, 1], [1, 0]], y_test=[0, 1]
    )

    # input to simulate results
    engine.results['untuned'] = {
        "LogisticRegression": {
            "mean_score": 0.85,
            "model": lr_model,
            "cv_scores": [0.8, 0.9, 0.85]
        },
        "RandomForestClassifier": {
            "mean_score": 0.90,
            "model": rf_model,
            "cv_scores": [0.89, 0.91, 0.90]
        }
    }

    best_untuned = engine.get_best_untuned_model()

    assert best_untuned is not None
    assert best_untuned['model_name'] == "RandomForestClassifier"
    assert best_untuned['mean_score'] == 0.90
    assert isinstance(best_untuned['model'], RandomForestClassifier)

def test_get_best_tuned_model():
    """ TC-47 Ensure the best tuned model is returned correctly """
    engine = ModelingEngine(
        X_train=[[0, 0], [1, 1]], y_train=[0, 1],
        X_test=[[0, 0], [1, 1]], y_test=[0, 1],
        task_type='classification'
    )
    engine.results['tuned'] = {
        "LogisticRegression": {
            "best_params": {"C": 1},
            "best_score": 0.90,
            "optimized_model": LogisticRegression()
        },
        "RandomForestClassifier": {
            "best_params": {"n_estimators": 100},
            "best_score": 0.95,
            "optimized_model": RandomForestClassifier()
        }
    }
    best_model_info = engine.get_best_tuned_model()

    assert best_model_info is not None
    assert best_model_info['model_name'] == "RandomForestClassifier"
    assert best_model_info['best_score'] == 0.95
    assert best_model_info['best_params'] == {"n_estimators": 100}
    assert isinstance(best_model_info['model'], RandomForestClassifier)

def test_compare_untuned_models(sample_data_classification):
    """ TC-48 Ensure compare_untuned_models runs and populates results """
    X_train, y_train, X_test, y_test = sample_data_classification
    engine = ModelingEngine(X_train, y_train, X_test, y_test, task_type='classification')

    engine.compare_untuned_models(cv_folds=3)

    assert 'untuned' in engine.results
    assert isinstance(engine.results['untuned'], dict)
    assert "LogisticRegression" in engine.results['untuned']
    assert "RandomForestClassifier" in engine.results['untuned']
    for model_name, info in engine.results['untuned'].items():
        assert "mean_score" in info
        assert "cv_scores" in info
        assert "model" in info
        assert isinstance(info['cv_scores'], list)
        assert isinstance(info['mean_score'], float)

def test_tune_all_models(sample_data_classification):
    """ TC-49 Validate tuning all models works as expected """
    X_train, y_train, X_test, y_test = sample_data_classification
    engine = ModelingEngine(
        X_train, y_train, X_test, y_test, task_type='classification'
    )
    engine.tune_all_models()
    tuned_results = engine.results.get('tuned', {})
    assert isinstance(tuned_results, dict)
    logreg_info = tuned_results['LogisticRegression']
    assert isinstance(logreg_info['best_params'], dict)
    assert isinstance(logreg_info['best_score'], float)
    assert logreg_info['optimized_model'] is not None
    assert 'tuned' in engine.results

def test_fit_tuned_models(sample_data_classification):
    """ TC-50 Validate fit_tuned_models works on all models in tuned dict """
    X_train, y_train, X_test, y_test = sample_data_classification
    engine = ModelingEngine(
        X_train, y_train, X_test, y_test, task_type='classification'
    )
    engine.results['tuned'] = {
        "LogisticRegression": {
            "optimized_model": LogisticRegression(max_iter=1000),
            "best_params": {"C": 1.0, "solver": "lbfgs"},
            "best_score": 0.9
        }
    }
    engine.fit_tuned_models()

    model = engine.results['tuned']['LogisticRegression']['optimized_model']
    preds = model.predict(X_test)
    assert len(preds) == len(y_test), "Predictions must match number of test samples"

def test_evaluate_tuned_models(sample_data_classification):
    """ TC-51 Validate evaluate_tuned_models computes train and test scores """
    X_train, y_train, X_test, y_test = sample_data_classification

    engine = ModelingEngine(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        task_type='classification'
    )
    model = LogisticRegression(max_iter=1000)
    engine.results['tuned'] = {
        "LogisticRegression": {"optimized_model": model}
    }
    
    engine.fit_tuned_models()
    engine.evaluate_tuned_models()

    tuned_model_info = engine.results['tuned']['LogisticRegression']

    assert "train_accuracy" in tuned_model_info["final_scores"]
    assert "test_accuracy" in tuned_model_info["final_scores"]
    assert 0.0 <= tuned_model_info["final_scores"]["train_accuracy"] <= 1.0
    assert 0.0 <= tuned_model_info["final_scores"]["test_accuracy"] <= 1.0