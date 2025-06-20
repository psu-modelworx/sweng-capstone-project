from fpdf import FPDF
from fpdf.enums import XPos, YPos
from datetime import datetime

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
        self.write_train_test_split_info()
        self.write_modeling_summary()
        self.write_versioning_info()
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

        encoded_column_names = ppe.feature_encoder.get_feature_names_out(ppe.categorical_columns).tolist()
        self.add_bullet(f"New column names created: {encoded_column_names}")
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

        self.subsection("Model Info")
        self.add_bullet("Model type: [placeholder]")
        self.add_bullet("Hyperparameters used: [placeholder]")
        self.add_bullet("Training duration: [placeholder]")

        self.subsection("Performance Metrics")
        self.add_bullet("Accuracy / R²: [placeholder]")
        self.add_bullet("Precision, Recall, F1-score / MAE, MSE, RMSE: [placeholder]")
        self.add_bullet("Confusion Matrix / Residual plot: [placeholder]")
        self.add_bullet("Class distribution (if classification): [placeholder]")

    def write_versioning_info(self):
        self.section_header("6. Versioning & Reproducibility")
        self.add_bullet(f"Timestamp of report: {datetime.now().isoformat()}")
        self.add_bullet("Python version: [placeholder]")
        self.add_bullet("scikit-learn version: [placeholder]")
        self.add_bullet("pandas version: [placeholder]")

    def write_visuals_section(self):
        self.section_header("7. Figures and Visualizations")
        self.add_bullet("Confusion matrix heatmap: [placeholder]")
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

    def add_bullet(self, text):
        self.pdf.multi_cell(10)
        self.pdf.multi_cell(0, 8, f"- {text}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
