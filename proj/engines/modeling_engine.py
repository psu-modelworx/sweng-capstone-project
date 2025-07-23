import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, precision_score, recall_score, accuracy_score, r2_score
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
            'fit_intercept': [True, False],
            'positive': [True, False]
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
        self.results = {}

    def run_modeling_engine(self):
        self.compare_untuned_models(cv_folds=5)
        self.tune_all_models()
        self.best_tuned = self.get_best_tuned_model()
        self.fit_tuned_models()
        self.evaluate_tuned_models()

    def get_models(self, task_type):
        """ Returns a dictionary of models based on task type. """
        classifiers = {
            'LogisticRegression': LogisticRegression(max_iter=1000),
            'RandomForestClassifier': RandomForestClassifier(),
            'GradientBoostingClassifier': GradientBoostingClassifier(),
            'SVC': SVC(probability=True)
        }
        regressors = {
            'LinearRegression': LinearRegression(),
            'RandomForestRegressor': RandomForestRegressor(),
            'GradientBoostingRegressor': GradientBoostingRegressor(),
            'SVR': SVR()
        }
        return classifiers if task_type == 'classification' else regressors

    def tune_model(self, model=None):
        """ Tunes hyperparameters for the given model using GridSearchCV. """
        if model is None:
            logging.error("No model available to fit.")
            return

        model_name = model.__class__.__name__
        param_grid = self.PARAM_GRIDS.get(model_name, {})
        if not param_grid:
            logging.warning(f"No parameter grid available for {model_name}. Skipping tuning.")
            return

        grid_search = GridSearchCV(model, param_grid, cv=5,
                                   scoring='accuracy' if self.task_type == 'classification' else 'r2', n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)

        self.results['tuned'] = self.results.get('tuned', {})
        self.results['tuned'][model_name] = {
            "best_params": grid_search.best_params_,
            "best_score": grid_search.best_score_,
            "optimized_model": grid_search.best_estimator_,
        }

        logging.info(f"Best params for {model_name}: {grid_search.best_params_}")
        logging.info(f"Best CV score for {model_name}: {grid_search.best_score_:.3f}")

        return self.results['tuned'][model_name]['optimized_model']

    def fit_model(self, model=None):
        """ Fits the model to the training data. """

        if not model:
            logging.error("No model available to fit.")
            return

        model.fit(self.X_train, self.y_train)
        logging.info(f"Model {model.__class__.__name__} fitted to training data.")

    def evaluate_model(self, model=None):
        """Evaluates the model on training and test sets, computes and stores predictions and metrics."""
        if not model:
            logging.error("No model available for evaluation.")
            return

        model_name = model.__class__.__name__

        # Predictions
        y_train_pred = model.predict(self.X_train)
        y_test_pred = model.predict(self.X_test)

        self.results.setdefault('tuned', {}).setdefault(model_name, {})

        # Store predictions
        self.results['tuned'][model_name].update({
            "y_train_pred": y_train_pred,
            "y_test_pred": y_test_pred,
        })

        # Compute and store metrics
        if self.task_type == 'classification':
            if hasattr(model, "predict_proba"):
                self.results['tuned'][model_name].update({
                    "y_train_proba": model.predict_proba(self.X_train),
                    "y_test_proba": model.predict_proba(self.X_test),
                })
            elif hasattr(model, "decision_function"):
                y_train_scores = model.decision_function(self.X_train)
                y_test_scores = model.decision_function(self.X_test)

                n_classes = len(np.unique(self.y_train))
                if n_classes == 2 and y_train_scores.ndim == 1:
                    y_train_scores = np.vstack([1 - y_train_scores, y_train_scores]).T
                    y_test_scores = np.vstack([1 - y_test_scores, y_test_scores]).T

                self.results['tuned'][model_name].update({
                    "y_train_decision": y_train_scores,
                    "y_test_decision": y_test_scores,
                })
            acc = accuracy_score(self.y_test, y_test_pred)
            prec = precision_score(self.y_test, y_test_pred, average='weighted', zero_division=0)
            rec = recall_score(self.y_test, y_test_pred, average='weighted', zero_division=0)
            f1 = f1_score(self.y_test, y_test_pred, average='weighted', zero_division=0)

            self.results['tuned'][model_name].update({
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "train_accuracy": accuracy_score(self.y_train, y_train_pred),
                "test_accuracy": acc,
            })

            logging.info(f"{model_name} - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

        elif self.task_type == 'regression':
            mse = mean_squared_error(self.y_test, y_test_pred)
            mae = mean_absolute_error(self.y_test, y_test_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(self.y_test, y_test_pred)
            n, p = self.X_test.shape
            adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else r2  # fallback if n too small

            self.results['tuned'][model_name].update({
                "rmse": rmse,
                "r2": r2,
                "mse": mse,
                "mae": mae,
                "adjusted_r2": adj_r2,
                "train_r2": r2_score(self.y_train, y_train_pred),
                "test_r2": r2,
            })

            logging.info(f"{model_name} - RMSE: {rmse:.4f}, R²: {r2:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}, Adjusted R²: {adj_r2:.4f}")



    def get_results(self):
        """ Returns the results of the modeling process. """
        return self.results
    
    def get_best_untuned_model(self):
        """ Returns the best untuned model. """
        if 'untuned' not in self.results or not self.results['untuned']:
            logging.error("No untuned models found. Run compare_untuned_models() first.")
            return None

        best_model_name = None
        best_score = float('-inf')
        best_model = None

        for name, info in self.results['untuned'].items():
            mean_score = info.get('mean_score', float('-inf'))
            if mean_score > best_score:
                best_score = mean_score
                best_model_name = name
                best_model = info.get('model')

        if best_model_name:
            logging.info(f"Best untuned model is {best_model_name} with mean_score = {best_score:.3f}")
            return {
                "model_name": best_model_name,
                "mean_score": best_score,
                "model": best_model
            }
        else:
            logging.warning("No suitable untuned model was found.")
            return None
        
    def get_best_tuned_model(self):
        """Returns the best tuned model info based on best_score."""
        tuned_results = self.results.get('tuned', {})
        if not tuned_results:
            logging.error("No tuned models found in results.")
            return None

        best_model_name = None
        best_score = float('-inf')
        best_model = None
        best_params = None

        for model_name, info in tuned_results.items():
            score = info.get('best_score', float('-inf'))
            if score > best_score:
                best_score = score
                best_model_name = model_name
                best_model = info.get('optimized_model')
                best_params = info.get('best_params')

        logging.info(f"Best tuned model: {best_model_name} with score {best_score:.4f}")
        return {
            "model_name": best_model_name,
            "best_score": best_score,
            "best_params": best_params,
            "model": best_model
        }
    
    def get_tuned_models(self):
        """ Returns the tuned models with their parameters. """
        return self.results.get('tuned', {})
    
    def compare_untuned_models(self, cv_folds=5):
        """ Compares all untuned models using cross-validation and returns the best one. """
        if not hasattr(self, 'models') or not self.models:
            logging.error("No untuned models found in self.models. Make sure to set self.models first.")
            return None

        if not hasattr(self, 'models') or not self.models:
            logging.error("No untuned models found in self.models. Make sure to set self.models first.")
            return

        model_list = list(self.models.items())
        logging.info(f"Found {len(model_list)} untuned models to evaluate: {', '.join([m for m, _ in model_list])}")
        logging.info(f"Evaluating {len(model_list)} models using {cv_folds}-fold CV...")

        scoring_metric = 'accuracy' if self.task_type == 'classification' else 'r2'
        self.results['untuned'] = {}

        for name, model in model_list:
            try:
                scores = cross_val_score(model, self.X_train, self.y_train, cv=cv_folds, scoring=scoring_metric)
                mean_score = scores.mean()

                self.results['untuned'][name] = {
                    "mean_score": mean_score,
                    "cv_scores": scores.tolist(),
                    "model": model
                }
                logging.info(f"{name}: Mean {scoring_metric} = {mean_score:.3f}")

            except Exception as e:
                logging.warning(f"Error evaluating {name}: {e}")
    
    def tune_all_models(self):
        """ Tunes all models and returns the results. """
        tuned_results = {}

        if not self.models:
            logging.warning("No models found to tune. Make sure self.models is populated.")
            return tuned_results

        logging.info(f"Starting hyperparameter tuning for {len(self.models)} models...")

        for model_name, model in self.models.items():
            logging.info(f"Tuning model: {model_name}")

            tuned_model = self.tune_model(model)
            if tuned_model:
                tuned_results[model_name] = tuned_model
                logging.info(f"Finished tuning {model_name}. Added to tuned results.")
            else:
                logging.warning(f"Could not tune {model_name}. Check parameter grid or model type.")

        logging.info(f"Completed tuning all models. Successfully tuned {len(tuned_results)} models.")
        return tuned_results

    def fit_tuned_models(self):
        """ Fits all tuned models to the training data. """
        if 'tuned' not in self.results or not self.results['tuned']:
            logging.error("No tuned models found. Run tune_all_models() first.")
            return

        for model_name, model_info in self.results['tuned'].items():
            model = model_info.get('optimized_model')
            if model:
                self.fit_model(model)
            else:
                logging.warning(f"No optimized model found for {model_name}. Skipping fitting and evaluation.")

    def evaluate_tuned_models(self):
        """ Evaluates all tuned models on the training and test datasets. """
        if 'tuned' not in self.results or not self.results['tuned']:
            logging.error("No tuned models found. Run tune_all_models() first.")
            return

        self.results['final_scores'] = {}

        for model_name, model_info in self.results['tuned'].items():
            model = model_info.get('optimized_model')
            if model:
                self.evaluate_model(model)
            else:
                logging.warning(f"No optimized model found for {model_name}. Skipping evaluation.")
    
        