import pytest
from engines.modeling_engine import ModelingEngine

def test_pytest_setup():
    """ Basic test to confirm pytest is running. """
    assert 1 + 1 == 2
    
def test_modeling_engine_initialization():
    """ Test the initialization of the ModelingEngine class. """
    engine = ModelingEngine(X_train=[[1, 2]], y_train=[1], X_test=[[3, 4]], y_test=[0], task_type='classification')
    assert engine.task_type == 'classification'
    assert isinstance(engine.models, dict)

