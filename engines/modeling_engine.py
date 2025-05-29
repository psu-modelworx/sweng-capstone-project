from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, r2_score
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class ModelingEngine:
    # Class constant for default parameter grids
    PARAM_GRIDS = {
        'LogisticRegression': {
            'C': [0.01, 0.1, 1, 10],
            'solver': ['liblinear', 'lbfgs']
        },
        'RandomForestClassifier': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20]
        },
        'GradientBoostingClassifier': {
            'learning_rate': [0.01, 0.1],
            'n_estimators': [50, 100]
        },
        'SVC': {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf']
        },
        'LinearRegression': {
        },
        'RandomForestRegressor': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'GradientBoostingRegressor': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0]
        },
        'SVR': {
            'kernel': ['linear', 'rbf'],
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto']
        }
    }

    def __init__(self, X_train, y_train, X_test, y_test, task_type='classification'):
        """
        Initializes the ModelingEngine with task type and datasets.
        """
        self.X_train, self.y_train, self.X_test, self.y_test = X_train, y_train, X_test, y_test
        self.task_type = task_type
        self.models = self.get_models(task_type)
        self.results = {'model_scores': {}, 'best_model_name': None, 'best_params': None,
                        'optimized_model': None, 'final_scores': {}}

    def get_models(self, task_type):
        """ Returns a dictionary of models based on task type. """
        classifiers = {
            'LogisticRegression': LogisticRegression(max_iter=1000),
            'RandomForestClassifier': RandomForestClassifier(),
            'GradientBoostingClassifier': GradientBoostingClassifier(),
            'SVC': SVC()
        }
        regressors = {
            'LinearRegression': LinearRegression(),
            'RandomForestRegressor': RandomForestRegressor(),
            'GradientBoostingRegressor': GradientBoostingRegressor(),
            'SVR': SVR()
        }
        return classifiers if task_type == 'classification' else regressors

    def evaluate_models(self, cv_folds=5):
        """ Evaluates models and selects the best one using cross-validation. """
        scoring_metric = 'accuracy' if self.task_type == 'classification' else 'r2'

        for name, model in self.models.items():
            try:
                scores = cross_val_score(model, self.X_train, self.y_train, cv=cv_folds, scoring=scoring_metric)
                self.results['model_scores'][name] = scores.mean()
                logging.info(f"{name}: Mean {scoring_metric} = {scores.mean():.3f}")
            except Exception as e:
                logging.warning(f"Error evaluating {name}: {e}")

        self.results['best_model_name'] = max(self.results['model_scores'], key=self.results['model_scores'].get, default=None)

        if self.results['best_model_name']:
            logging.info(f"Best model selected: {self.results['best_model_name']} with {scoring_metric} = {self.results['model_scores'][self.results['best_model_name']]:.3f}")
        else:
            logging.error("No suitable model found.")

    def tune_best_model(self):
        """ Tunes hyperparameters for the best model using GridSearchCV. """
        if not self.results['best_model_name']:
            logging.error("Must run evaluate_models before tuning.")
            return

        param_grid = self.PARAM_GRIDS.get(self.results['best_model_name'], {})
        if not param_grid:
            logging.warning(f"No parameter grid available for {self.results['best_model_name']}. Skipping tuning.")
            return

        grid_search = GridSearchCV(self.models[self.results['best_model_name']], param_grid, cv=5,
                                   scoring='accuracy' if self.task_type == 'classification' else 'r2', n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)

        self.results['best_params'] = grid_search.best_params_
        self.results['optimized_model'] = grid_search.best_estimator_

        logging.info(f"Best params for {self.results['best_model_name']}: {self.results['best_params']}")
        logging.info(f"Best CV score: {grid_search.best_score_:.3f}")

    def evaluate_final_model(self):
        if not self.results['optimized_model']:
            logging.error("Run tune_best_model before final evaluation.")
            return

        model = self.results['optimized_model']
        model.fit(self.X_train, self.y_train)
        y_train_pred = model.predict(self.X_train)
        y_test_pred = model.predict(self.X_test)

        metric_fn = accuracy_score if self.task_type == 'classification' else r2_score
        self.results['final_scores']['train'] = metric_fn(self.y_train, y_train_pred)
        self.results['final_scores']['test'] = metric_fn(self.y_test, y_test_pred)

        logging.info(f"Final Model Train Score: {self.results['final_scores']['train']:.4f}")
        logging.info(f"Final Model Test Score: {self.results['final_scores']['test']:.4f}")

    def run_modeling_engine(self):
        self.evaluate_models()
        self.tune_best_model()
        self.evaluate_final_model()
        return self.results