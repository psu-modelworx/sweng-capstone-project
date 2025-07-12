import logging
import os
import tempfile
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import label_binarize
from sklearn.metrics import auc, average_precision_score, confusion_matrix, f1_score, mean_squared_error, precision_recall_curve, precision_score, recall_score, roc_curve, accuracy_score, r2_score
import seaborn as sns

class ReportingEngine:
    """Generates a PDF report for preprocessing info and model generation info."""
    def __init__(self, preprocessor, modeler):
        """ initializes the reporting engine with preprocessor and modeler objects."""
        self.preprocessor = preprocessor
        self.modeler = modeler
        self.pdf = FPDF()
        self.pdf.add_page()
        self.pdf.set_auto_page_break(auto=True, margin=15)
        self.pdf.set_font("Helvetica", size=12)

    def generate_full_report(self):
        self.write_title("modelworx Job Report")
        self.write_introduction()
        self.write_preprocessing_summary()
        self.write_modeling_summary()
        self.write_visuals_section()

        # self.pdf.output("model_report.pdf")

    def write_title(self, title):
        self.pdf.set_font("Helvetica", 'B', 16)
        self.pdf.multi_cell(0, 10, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        self.pdf.ln(10)

    def write_introduction(self):
        self.section_header("Introduction")
        self.pdf.set_font("Helvetica", size=12)
        task_type = self.preprocessor.task_type.capitalize()
        self.pdf.multi_cell(0, 10, 
            f"This report summarizes the preprocessing and modeling steps performed during the modelworx job execution. "
            f"It includes details about the data preprocessing engine, modeling generation, and model recommendation. "
            f"The models generated are based on the provided dataset and the task type is set to: {task_type}."
        )
        self.pdf.ln(5)

    def write_preprocessing_summary(self):
        self.section_header("Preprocessing Summary")

        self.subsection("Initial Data Overview")
        num_rows = self.preprocessor.df.shape[0]
        num_columns = self.preprocessor.df.shape[1]
        self.add_bullet(f"Number of rows in the dataset: {num_rows}")
        self.add_bullet(f"Number of columns in the dataset: {num_columns}")
        # Insert figures for data types and distribution
        if self.preprocessor.task_type == 'classification':
            self.plot_class_distribution()

        self.subsection("Data Cleaning")
        self.add_bullet("Handled missing numerical values by replacing with mean.")
        self.add_bullet("Handled missing categorical values by replacing with mode.")
        removed = []
        if hasattr(self.preprocessor, 'dropped_columns') and self.preprocessor.dropped_columns:
            removed.extend(self.preprocessor.dropped_columns)
        if hasattr(self.preprocessor, 'cols_to_remove') and self.preprocessor.cols_to_remove:
            removed.extend(self.preprocessor.cols_to_remove)
        removed = list(set(removed))
        self.add_bullet(f"Removed columns from the dataset: {removed if removed else 'None'}")
        # consider handling outliers
        # consider checking for duplicates

        self.subsection("Feature Engineering")
        if hasattr(self.preprocessor.feature_encoder, "categories_"):
            self.add_buller(f"Feature encoding method: {self.preprocessor.feature_encoder.__class__.__name__}")
            for i, cats in enumerate(self.preprocessor.feature_encoder.categories_):
                self.add_bullet(f"Categorical feature {i + 1} categories: {', '.join(map(str, cats))}")
            self.add_bullet(f"Encoded column names: {', '.join(self.preprocessor.get_feature_names_out)}")
        else:
            self.add_bullet("No feature encoding applied.")
        if hasattr(self.preprocessor.scaler, 'scale_'):
            self.add_bullet(f"Feature scaling applied using {self.preprocessor.scaler.__class__.__name__}.")
            self.add_bullet(f"Feature scales: {', '.join([f'{s:.4f}' for s in self.preprocessor.scaler.scale_])}")
        if hasattr(self.preprocessor.label_encoder, 'classes_'):
            self.add_bullet(f"Label encoding applied for target variable: {self.preprocessor.label_encoder.__class__.__name__}")
            self.add_bullet(f"Encoded target classes: {', '.join(map(str, self.preprocessor.label_encoder.classes_))}")
        else:
            self.add_bullet("Target variable is not categorical. No label encoding applied for target variable.")

        self.subsection("Data Splitting")
        self.add_bullet(f"Train set size: {self.preprocessor.X_train.shape[0]} rows")
        self.add_bullet(f"Test set size: {self.preprocessor.X_test.shape[0]} rows") 
        

        

    def write_modeling_summary(self):
        self.section_header("3. Modeling Engine Summary")
        me = self.modeler

        self.subsection("Task Info")
        self.add_bullet(f"Task type: {me.task_type.capitalize()}")

        # Untuned models summary
        untuned = me.results.get('untuned', {})
        if not untuned:
            self.add_bullet("No untuned model results available.")
        else:
            self.subsection("Untuned Models Performance")
            self.add_bullet(f"Number of untuned models evaluated: {len(untuned)}")
            best_untuned = me.get_best_untuned_model()
            if best_untuned:
                self.add_bullet(f"Best untuned model: {best_untuned['model_name']} with mean CV score {best_untuned['mean_score']:.4f}")
            for model_name, info in untuned.items():
                mean_score = info.get('mean_score', None)
                cv_scores = info.get('cv_scores', [])
                if mean_score is not None:
                    self.subsection(f"Model: {model_name}")
                    self.add_bullet(f"Mean CV score: {mean_score:.4f}")
                    self.add_bullet(f"CV scores: {', '.join(f'{s:.4f}' for s in cv_scores)}")

            

        # Tuned models summary
        tuned = me.results.get('tuned', {})
        if not tuned:
            self.add_bullet("No tuned model results available.")
        else:
            self.subsection("Tuned Models Performance")
            self.add_bullet(f"Number of tuned models evaluated: {len(tuned)}")
            for model_name, info in tuned.items():
                best_params = info.get('best_params', {})
                best_score = info.get('best_score', None)
                train_score = info.get('train', None)
                test_score = info.get('test', None)

                self.subsection(f"Model: {model_name}")
                self.add_bullet(f"Best hyperparameters: {best_params if best_params else 'N/A'}")
                if best_score is not None:
                    self.add_bullet(f"Best CV score: {best_score:.4f}")
                if train_score is not None:
                    self.add_bullet(f"Train set performance: {train_score:.4f}")
                if test_score is not None:
                    self.add_bullet(f"Test set performance: {test_score:.4f}")

            best_tuned = me.get_best_tuned_model()
            if best_tuned:
                self.section_header("Best Tuned Model")
                self.add_bullet(f"Model: {best_tuned['model_name']}")
                self.add_bullet(f"Best score: {best_tuned['best_score']:.4f}")
                self.add_bullet(f"Hyperparameters: {best_tuned['best_params']}")
            else:
                self.add_bullet("No best tuned model found.")

    def write_visuals_section(self):
        self.section_header("3. Figures and Visualizations")

        if self.preprocessor.task_type == 'classification':
            logging.info("Generating classification visualizations...")

            self.subsection("Confusion Matrix Heatmap")
            self.generate_conf_matrix()
            self.subsection("Class Distribution Plot")
            self.plot_class_distribution()
            self.subsection("Feature Importance Plot")
            self.plot_feature_importance()
            self.subsection("ROC Curve")
            self.plot_roc_curve()
            self.subsection("Precision-Recall Curve")
            self.plot_precision_recall_curve()
            self.subsection("Classification Report Heatmap")
            self.plot_classification_report()
            self.subsection("Model Performance Bar Chart")
            self.plot_model_performance_bar_chart()
            self.subsection("Cross-Validation Score Boxplots")
            self.plot_cv_score_boxplots()

        elif self.preprocessor.task_type == 'regression':
            logging.info("Generating regression visualizations...")

            self.subsection("Residuals Plot")
            self.plot_residuals()
            self.subsection("Actual vs Predicted Plot")
            self.plot_actual_vs_predicted()
            self.subsection("Error Distribution Plot")
            self.plot_error_distribution()
            self.subsection("Feature Importance Plot")
            self.plot_feature_importance()
            self.subsection("Metrics Bar Chart")
            self.plot_metrics_bar_chart()
            self.subsection("Model Performance Bar Chart")
            self.plot_cv_score_boxplots()
            self.subsection("Model Performance Bar Chart")
            self.plot_model_performance_bar_chart()

        else:
            logging.warning("Unsupported task type for visualizations: %s", ppe.task_type)


    def generate_conf_matrix(self):
        """Generates confusion matrix heatmaps for tuned models."""
        if self.preprocessor.task_type != 'classification':
            self.add_bullet("Confusion matrix heatmap: Not applicable for regression task.")
            return

        tuned_models = self.modeler.results.get('tuned', {})
        if not tuned_models:
            self.add_bullet("Confusion matrix heatmap: No tuned models available to generate matrices.")
            return

        X_test = self.preprocessor.X_test
        y_test = self.preprocessor.y_test

        if X_test is None or y_test is None:
            self.add_bullet("Confusion matrix heatmap: Test data not available.")
            return

        for model_name, info in tuned_models.items():
            model = info.get('optimized_model')
            if model is None:
                self.add_bullet(f"Confusion matrix heatmap: No model available for {model_name}. Skipping.")
                continue

            y_pred_encoded = model.predict(X_test)
            try:
                y_pred = self.preprocessor.decode_target(y_pred_encoded)
                y_true = self.preprocessor.decode_target(y_test)
            except Exception:
                y_pred = y_pred_encoded
                y_true = y_test

            cm = confusion_matrix(y_true, y_pred)
            labels = sorted(list(set(y_true)))

            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=labels, yticklabels=labels)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title(f'Confusion Matrix Heatmap: {model_name}')

            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmpfile:
                plt.savefig(tmpfile.name, bbox_inches='tight')
                plt.close()
                self.pdf.image(tmpfile.name, x=15, w=180)
            os.remove(tmpfile.name)

    def plot_class_distribution(self):
        """Plots the distribution of classes in the target variable."""
        if self.preprocessor.task_type != 'classification':
            self.add_bullet("Class distribution plot: Not applicable for regression task.")
            return

        y_encoded = self.preprocessor.y
        if y_encoded is None:
            self.add_bullet("Class distribution plot: Target variable not available.")
            return

        # Decode target classes
        y = self.preprocessor.decode_target(y_encoded)

        plt.figure(figsize=(8, 6))
        sns.countplot(x=y, hue=y, palette='viridis', legend=False)
        plt.title('Class Distribution')
        plt.xlabel('Classes')
        plt.ylabel('Count')

        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmpfile:
            plt.savefig(tmpfile.name, bbox_inches='tight')
            plt.close()
            self.pdf.image(tmpfile.name, x=15, w=180)
        os.remove(tmpfile.name)

    def plot_feature_importance(self):
        """Plots feature importance for tuned models."""
        if self.preprocessor.task_type not in ['classification', 'regression']:
            self.add_bullet("Feature importance plot: Not applicable for this task type.")
            return

        tuned_models = self.modeler.results.get('tuned', {})
        if not tuned_models:
            self.add_bullet("Feature importance plot: No tuned models available.")
            return

        X_train = self.preprocessor.X_train
        if X_train is None:
            self.add_bullet("Feature importance plot: Training data not available.")
            return

        for model_name, info in tuned_models.items():
            model = info.get('optimized_model')
            if model is None:
                self.add_bullet(f"Feature importance plot: No model available for {model_name}. Skipping.")
                continue

            try:
                importances = model.feature_importances_
                feature_names = self.preprocessor.final_columns
                indices = importances.argsort()[::-1]

                plt.figure(figsize=(10, 6))
                plt.title(f'Feature Importance: {model_name}')
                plt.bar(range(len(importances)), importances[indices], align='center')
                plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
                plt.xlabel('Features')
                plt.ylabel('Importance')

                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmpfile:
                    plt.savefig(tmpfile.name, bbox_inches='tight')
                    plt.close()
                    self.pdf.image(tmpfile.name, x=15, w=180)
                os.remove(tmpfile.name)

            except AttributeError:
                self.add_bullet(f"Feature importance plot: Model {model_name} does not support feature importance.")

    def plot_roc_curve(self):
        """Plots ROC curve for classification models."""
        if self.preprocessor.task_type != 'classification':
            self.add_bullet("ROC curve plot: Not applicable for regression task.")
            return

        tuned_models = self.modeler.results.get('tuned', {})
        if not tuned_models:
            self.add_bullet("ROC curve plot: No tuned models available.")
            return

        X_test = self.preprocessor.X_test
        y_test = self.preprocessor.y_test

        if X_test is None or y_test is None:
            self.add_bullet("ROC curve plot: Test data not available.")
            return

        for model_name, info in tuned_models.items():
            model = info.get('optimized_model')
            if model is None:
                self.add_bullet(f"ROC curve plot: No model available for {model_name}. Skipping.")
                continue

            try:
                from sklearn.metrics import roc_curve, auc
                y_score = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_score)
                roc_auc = auc(fpr, tpr)

                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], color='red', linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'ROC Curve: {model_name}')
                plt.legend(loc='lower right')

                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmpfile:
                    plt.savefig(tmpfile.name, bbox_inches='tight')
                    plt.close()
                    self.pdf.image(tmpfile.name, x=15, w=180)
                os.remove(tmpfile.name)

            except Exception as e:
                self.add_bullet(f"ROC curve plot: Error generating ROC curve for {model_name}: {str(e)}")

    def plot_precision_recall_curve(self):
        """Plots Precision-Recall curve for classification models."""
        if self.preprocessor.task_type != 'classification':
            self.add_bullet("Precision-Recall curve plot: Not applicable for regression task.")
            return

        tuned_models = self.modeler.results.get('tuned', {})
        if not tuned_models:
            self.add_bullet("Precision-Recall curve plot: No tuned models available.")
            return

        X_test = self.preprocessor.X_test
        y_test = self.preprocessor.y_test

        if X_test is None or y_test is None:
            self.add_bullet("Precision-Recall curve plot: Test data not available.")
            return

        for model_name, info in tuned_models.items():
            model = info.get('optimized_model')
            if model is None:
                self.add_bullet(f"Precision-Recall curve plot: No model available for {model_name}. Skipping.")
                continue

            try:
                from sklearn.metrics import precision_recall_curve
                y_score = model.predict_proba(X_test)[:, 1]
                precision, recall, _ = precision_recall_curve(y_test, y_score)

                plt.figure(figsize=(8, 6))
                plt.plot(recall, precision, color='blue')
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title(f'Precision-Recall Curve: {model_name}')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])

                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmpfile:
                    plt.savefig(tmpfile.name, bbox_inches='tight')
                    plt.close()
                    self.pdf.image(tmpfile.name, x=15, w=180)
                os.remove(tmpfile.name)

            except Exception as e:
                self.add_bullet(f"Precision-Recall curve plot: Error generating Precision-Recall curve for {model_name}: {str(e)}")

    def plot_classification_report(self):
        """Plots classification report for tuned models."""
        if self.preprocessor.task_type != 'classification':
            self.add_bullet("Classification report plot: Not applicable for regression task.")
            return

        tuned_models = self.modeler.results.get('tuned', {})
        if not tuned_models:
            self.add_bullet("Classification report plot: No tuned models available.")
            return

        X_test = self.preprocessor.X_test
        y_test = self.preprocessor.y_test

        if X_test is None or y_test is None:
            self.add_bullet("Classification report plot: Test data not available.")
            return

        for model_name, info in tuned_models.items():
            model = info.get('optimized_model')
            if model is None:
                self.add_bullet(f"Classification report plot: No model available for {model_name}. Skipping.")
                continue

            try:
                from sklearn.metrics import classification_report
                y_pred_encoded = model.predict(X_test)
                y_pred = self.preprocessor.decode_target(y_pred_encoded)
                y_true = self.preprocessor.decode_target(y_test)

                report = classification_report(y_true, y_pred, output_dict=True)
                sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True, cmap='Blues')

                plt.title(f'Classification Report: {model_name}')
                plt.xlabel('Metrics')
                plt.ylabel('Classes')

                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmpfile:
                    plt.savefig(tmpfile.name, bbox_inches='tight')
                    plt.close()
                    self.pdf.image(tmpfile.name, x=15, w=180)
                os.remove(tmpfile.name)

            except Exception as e:
                self.add_bullet(f"Classification report plot: Error generating classification report for {model_name}: {str(e)}")
    
    def plot_model_performance_bar_chart(self):
        """Plots a bar chart of model performance metrics."""
        if self.preprocessor.task_type not in ['classification', 'regression']:
            self.add_bullet("Model performance bar chart: Not applicable for this task type.")
            return

        tuned_models = self.modeler.results.get('tuned', {})
        if not tuned_models:
            self.add_bullet("Model performance bar chart: No tuned models available.")
            return

        model_names = []
        scores = []

        for model_name, info in tuned_models.items():
            best_score = info.get('best_score', None)
            if best_score is not None:
                model_names.append(model_name)
                scores.append(best_score)

        if not model_names:
            self.add_bullet("Model performance bar chart: No scores available for tuned models.")
            return

        plt.figure(figsize=(10, 6))
        sns.barplot(x=model_names, y=scores, hue=model_names, palette='viridis', legend=False)
        plt.title('Model Performance Comparison')
        plt.xlabel('Models')
        plt.ylabel('Best CV Score')
        plt.xticks(rotation=45)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmpfile:
            plt.savefig(tmpfile.name, bbox_inches='tight')
            plt.close()
            self.pdf.image(tmpfile.name, x=15, w=180)
        os.remove(tmpfile.name)

    def plot_cv_score_boxplots(self):
        """Plots boxplots of cross-validation scores for tuned models."""
        if self.preprocessor.task_type not in ['classification', 'regression']:
            self.add_bullet("CV score boxplots: Not applicable for this task type.")
            return

        tuned_models = self.modeler.results.get('tuned', {})
        if not tuned_models:
            self.add_bullet("CV score boxplots: No tuned models available.")
            return

        model_names = []
        cv_scores = []

        for model_name, info in tuned_models.items():
            scores = info.get('cv_scores', [])
            if scores:
                model_names.append(model_name)
                cv_scores.append(scores)

        if not model_names:
            self.add_bullet("CV score boxplots: No CV scores available for tuned models.")
            return

        plt.figure(figsize=(10, 6))
        sns.boxplot(data=cv_scores)
        plt.title('Cross-Validation Score Distribution')
        plt.xlabel('Models')
        plt.ylabel('CV Scores')
        plt.xticks(range(len(model_names)), model_names, rotation=45)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmpfile:
            plt.savefig(tmpfile.name, bbox_inches='tight')
            plt.close()
            self.pdf.image(tmpfile.name, x=15, w=180)
        os.remove(tmpfile.name)

    def plot_residuals(self):
        """Plots residuals for regression models."""
        if self.preprocessor.task_type != 'regression':
            self.add_bullet("Residuals plot: Not applicable for classification task.")
            return

        X_test = self.preprocessor.X_test
        y_test = self.preprocessor.y_test

        if X_test is None or y_test is None:
            self.add_bullet("Residuals plot: Test data not available.")
            return

        tuned_models = self.modeler.results.get('tuned', {})
        if not tuned_models:
            self.add_bullet("Residuals plot: No tuned models available.")
            return

        for model_name, info in tuned_models.items():
            model = info.get('optimized_model')
            if model is None:
                self.add_bullet(f"Residuals plot: No model available for {model_name}. Skipping.")
                continue

            try:
                y_pred = model.predict(X_test)
                residuals = y_test - y_pred

                plt.figure(figsize=(8, 6))
                sns.scatterplot(x=y_pred, y=residuals)
                plt.axhline(0, color='red', linestyle='--')
                plt.title(f'Residuals Plot: {model_name}')
                plt.xlabel('Predicted Values')
                plt.ylabel('Residuals')

                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmpfile:
                    plt.savefig(tmpfile.name, bbox_inches='tight')
                    plt.close()
                    self.pdf.image(tmpfile.name, x=15, w=180)
                os.remove(tmpfile.name)

            except Exception as e:
                self.add_bullet(f"Residuals plot: Error generating residuals plot for {model_name}: {str(e)}")

    def plot_actual_vs_predicted(self):
        """Plots actual vs predicted values for regression models."""
        if self.preprocessor.task_type != 'regression':
            self.add_bullet("Actual vs Predicted plot: Not applicable for classification task.")
            return

        X_test = self.preprocessor.X_test
        y_test = self.preprocessor.y_test

        if X_test is None or y_test is None:
            self.add_bullet("Actual vs Predicted plot: Test data not available.")
            return

        tuned_models = self.modeler.results.get('tuned', {})
        if not tuned_models:
            self.add_bullet("Actual vs Predicted plot: No tuned models available.")
            return

        for model_name, info in tuned_models.items():
            model = info.get('optimized_model')
            if model is None:
                self.add_bullet(f"Actual vs Predicted plot: No model available for {model_name}. Skipping.")
                continue

            try:
                y_pred = model.predict(X_test)

                plt.figure(figsize=(8, 6))
                sns.scatterplot(x=y_test, y=y_pred)
                plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
                plt.title(f'Actual vs Predicted: {model_name}')
                plt.xlabel('Actual Values')
                plt.ylabel('Predicted Values')

                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmpfile:
                    plt.savefig(tmpfile.name, bbox_inches='tight')
                    plt.close()
                    self.pdf.image(tmpfile.name, x=15, w=180)
                os.remove(tmpfile.name)

            except Exception as e:
                self.add_bullet(f"Actual vs Predicted plot: Error generating plot for {model_name}: {str(e)}")
    
    def plot_error_distribution(self):
        """Plots the distribution of errors for regression models."""
        if self.preprocessor.task_type != 'regression':
            self.add_bullet("Error distribution plot: Not applicable for classification task.")
            return

        X_test = self.preprocessor.X_test
        y_test = self.preprocessor.y_test

        if X_test is None or y_test is None:
            self.add_bullet("Error distribution plot: Test data not available.")
            return

        tuned_models = self.modeler.results.get('tuned', {})
        if not tuned_models:
            self.add_bullet("Error distribution plot: No tuned models available.")
            return

        for model_name, info in tuned_models.items():
            model = info.get('optimized_model')
            if model is None:
                self.add_bullet(f"Error distribution plot: No model available for {model_name}. Skipping.")
                continue

            try:
                y_pred = model.predict(X_test)
                errors = y_test - y_pred

                plt.figure(figsize=(8, 6))
                sns.histplot(errors, kde=True, color='blue')
                plt.title(f'Error Distribution: {model_name}')
                plt.xlabel('Error')
                plt.ylabel('Frequency')

                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmpfile:
                    plt.savefig(tmpfile.name, bbox_inches='tight')
                    plt.close()
                    self.pdf.image(tmpfile.name, x=15, w=180)
                os.remove(tmpfile.name)

            except Exception as e:
                self.add_bullet(f"Error distribution plot: Error generating plot for {model_name}: {str(e)}")


    def plot_metrics_bar_chart(self):
        """Plots a bar chart of regression metrics."""
        if self.preprocessor.task_type != 'regression':
            self.add_bullet("Metrics bar chart: Not applicable for classification task.")
            return

        tuned_models = self.modeler.results.get('tuned', {})
        if not tuned_models:
            self.add_bullet("Metrics bar chart: No tuned models available.")
            return

        model_names = []
        metrics = []

        for model_name, info in tuned_models.items():
            train_score = info.get('train', None)
            test_score = info.get('test', None)
            if train_score is not None and test_score is not None:
                model_names.append(model_name)
                metrics.append((train_score, test_score))

        if not model_names:
            self.add_bullet("Metrics bar chart: No scores available for tuned models.")
            return

        train_scores, test_scores = zip(*metrics)

        plt.figure(figsize=(10, 6))
        x = range(len(model_names))
        plt.bar(x, train_scores, width=0.4, label='Train Score', align='center')
        plt.bar([i + 0.4 for i in x], test_scores, width=0.4, label='Test Score', align='center')
        plt.title('Model Performance Comparison')
        plt.xlabel('Models')
        plt.ylabel('Scores')
        plt.xticks([i + 0.2 for i in x], model_names, rotation=45)
        plt.legend()

        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmpfile:
            plt.savefig(tmpfile.name, bbox_inches='tight')
            plt.close()
            self.pdf.image(tmpfile.name, x=15, w=180)
        os.remove(tmpfile.name)

    def save_report(self, filename="model_report.pdf"):
        """Saves the generated PDF report to a file."""
        if not filename.endswith('.pdf'):
            filename += '.pdf'
        self.pdf.output(filename)
        logging.info(f"Report saved to {filename}")

    # Utility Methods
    def section_header(self, text):
        self.pdf.set_font("Helvetica", 'B', 14)
        self.pdf.multi_cell(0, 10, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.pdf.set_font("Helvetica", size=12)
        self.pdf.ln(2)

    def subsection(self, text):
        self.pdf.set_font("Helvetica", 'B', 12)
        self.pdf.multi_cell(0, 8, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.pdf.set_font("Helvetica", size=12)

    def add_bullet(self, text):
        self.pdf.set_x(self.pdf.get_x() + 5)
        self.pdf.multi_cell(0, 8, f"- {text}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
