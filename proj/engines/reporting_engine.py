import os
import tempfile
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from datetime import datetime

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
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
        self.write_preprocessing_summary()
        self.write_modeling_summary()
        self.write_visuals_section()

        # self.pdf.output("model_report.pdf")

    def write_title(self, title):
        self.pdf.set_font("Helvetica", 'B', 16)
        self.pdf.multi_cell(0, 10, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        self.pdf.ln(10)

    def write_preprocessing_summary(self):
        self.section_header("1. Preprocessing Engine Summary")

        ppe = self.preprocessor

        self.subsection("General Information")
        self.add_bullet(f"Target column and its type: {ppe.target_column} ({'categorical' if ppe.target_is_categorical else 'continuous'})")
        self.add_bullet(f"Task type: {ppe.task_type}")
        original_shape = ppe.original_df.shape
        final_shape = ppe.final_df.shape if ppe.final_df is not None else ("Not available",)
        self.add_bullet(f"Original dataset shape: {original_shape}")
        self.add_bullet(f"Final dataset shape after preprocessing: {final_shape}")
        self.add_bullet(f"Columns removed or dropped: {ppe.dropped_columns if ppe.dropped_columns else 'None'}")
        final_feature_count = len(ppe.final_columns) if ppe.final_columns else "Not computed yet"
        self.add_bullet(f"Final feature count after encoding: {final_feature_count}")
        self.add_bullet("Missing value handling: Imputed numeric columns with mean and categorical columns with mode if missing values were present.")

        self.subsection("Encoding Details")
        self.add_bullet(f"Categorical columns encoded: {ppe.categorical_columns if ppe.categorical_columns else 'None'}")
        self.add_bullet(f"Encoding method: {str(ppe.feature_encoder)}")

        self.subsection("Encoding Details")

        if not ppe.categorical_columns or not hasattr(ppe, 'encoded_categorical_columns') or not ppe.encoded_categorical_columns:
            self.add_bullet("Categorical columns encoded: None")
            self.add_bullet("Encoding method: None (no categorical columns to encode)")
            self.add_bullet("New column names created: None")
        else:
            self.add_bullet(f"Categorical columns encoded: {ppe.encoded_categorical_columns}")
            try:
                encoded_column_names = ppe.feature_encoder.get_feature_names_out(
                    ppe.encoded_categorical_columns).tolist()
                self.add_bullet(f"New column names created: {encoded_column_names}")
            except Exception as e:
                self.add_bullet(f"Failed to get encoded feature names: {str(e)}")

            self.add_bullet(f"Encoding method: {str(ppe.feature_encoder)}")

        self.add_bullet("Scaler used: StandardScaler")
        self.add_bullet(f"Final feature names: {ppe.final_columns if ppe.final_columns else 'Not yet generated'}")

        self.subsection("Column Information")
        self.add_bullet(f"original_columns: {ppe.original_columns}")
        self.add_bullet(f"final_columns: {ppe.final_columns if ppe.final_columns else 'Not yet generated'}")
        self.add_bullet(f"dropped_columns: {ppe.dropped_columns if ppe.dropped_columns else 'None'}")
        self.add_bullet(f"columns_to_remove: {ppe.columns_to_remove if ppe.columns_to_remove else 'None'}")

    def write_train_test_split_info(self):
        self.section_header("2. Train/Test Split Info")
        self.add_bullet("Train/test set sizes: [placeholder]")
        self.add_bullet("Random seed used: [placeholder]")

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
            for model_name, info in untuned.items():
                mean_score = info.get('mean_score', None)
                cv_scores = info.get('cv_scores', [])
                if mean_score is not None:
                    self.subsubsection(f"Model: {model_name}")
                    self.add_bullet(f"Mean CV score: {mean_score:.4f}")
                    self.add_bullet(f"CV scores: {', '.join(f'{s:.4f}' for s in cv_scores)}")

            best_untuned = me.get_best_untuned_model()
            if best_untuned:
                self.add_bullet(f"Best untuned model: {best_untuned['model_name']} with mean CV score {best_untuned['mean_score']:.4f}")

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

                self.subsubsection(f"Model: {model_name}")
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
        self.section_header("7. Figures and Visualizations")
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
        self.add_bullet("ROC curve: [placeholder]")
        self.add_bullet("Feature importance bar plot: [placeholder]")
        self.add_bullet("Distributions before/after scaling or encoding: [placeholder]")

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

    def subsubsection(self, title):
        self.pdf.set_font("Helvetica", 'B', 12)
        self.pdf.multi_cell(0, 8, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.pdf.ln(2)
        self.pdf.set_font("Helvetica", size=12)

    def add_bullet(self, text):
        self.pdf.multi_cell(10)
        self.pdf.multi_cell(0, 8, f"- {text}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
