import re
from unittest.mock import MagicMock, patch
import numpy as np
from pypdf import PdfReader
from engines.reporting_engine import ReportingEngine


def test_initialization():
    """TC-75: Ensure ReportingEngine initializes with preprocessor and modeler."""
    preprocessor_mock = MagicMock()
    modeler_mock = MagicMock()
    re = ReportingEngine(preprocessor=preprocessor_mock, modeler=modeler_mock)
    assert re.preprocessor is preprocessor_mock
    assert re.modeler is modeler_mock
    assert re.pdf is not None
    assert re.pdf.page_no() == 1


def test_write_title(tmp_path):
    """TC-76: Ensure title is written to PDF."""
    dummy_preprocessor = MagicMock()
    dummy_modeler = MagicMock()

    report = ReportingEngine(dummy_preprocessor, dummy_modeler)
    report.write_title("My Report Title")
    output_path = tmp_path / "title_test.pdf"
    report.pdf.output(str(output_path))

    assert output_path.exists()
    assert output_path.stat().st_size > 0

    reader = PdfReader(str(output_path))
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    assert "My Report Title" in text

def normalize_text(text):
    return re.sub(r'\s+', ' ', text).strip()

def test_write_introduction_section(tmp_path):
    """TC-77: Ensure introduction section is written to PDF."""
    dummy_preprocessor = MagicMock()
    dummy_preprocessor.task_type = "classification"
    dummy_modeler = MagicMock()

    report = ReportingEngine(dummy_preprocessor, dummy_modeler)
    report.write_introduction()
    output_path = tmp_path / "intro_test.pdf"
    report.pdf.output(str(output_path))

    assert output_path.exists()
    assert output_path.stat().st_size > 0

    reader = PdfReader(str(output_path))
    pdf_text = "".join(page.extract_text()
                       for page in reader.pages if page.extract_text())
    normalized_text = re.sub(r'\s+', ' ', pdf_text).strip()
    expected_phrase = "This report summarizes the preprocessing and modeling steps performed during the modelworx job execution"

    assert expected_phrase in normalized_text
    assert dummy_preprocessor.task_type.capitalize() in normalized_text


def test_section_header(tmp_path):
    """TC-78: Ensure section header is written to PDF."""
    dummy_preprocessor = MagicMock()
    dummy_modeler = MagicMock()
    report = ReportingEngine(dummy_preprocessor, dummy_modeler)

    report.section_header("Section Header Test")
    output_path = tmp_path / "section_header_test.pdf"
    report.pdf.output(str(output_path))

    reader = PdfReader(str(output_path))
    text = "".join(p.extract_text() for p in reader.pages if p.extract_text())
    assert "Section Header Test" in text


def test_subsection(tmp_path):
    """TC-79: Ensure subsection is written to PDF."""
    dummy_preprocessor = MagicMock()
    dummy_modeler = MagicMock()
    report = ReportingEngine(dummy_preprocessor, dummy_modeler)

    report.subsection("Subsection Test")
    output_path = tmp_path / "subsection_test.pdf"
    report.pdf.output(str(output_path))

    reader = PdfReader(str(output_path))
    text = "".join(p.extract_text() for p in reader.pages if p.extract_text())
    assert "Subsection Test" in text


def test_add_bullet(tmp_path):
    """TC-80: Ensure bullet point is added to PDF."""
    dummy_preprocessor = MagicMock()
    dummy_modeler = MagicMock()
    report = ReportingEngine(dummy_preprocessor, dummy_modeler)

    report.add_bullet("Test bullet point")
    output_path = tmp_path / "bullet_test.pdf"
    report.pdf.output(str(output_path))

    reader = PdfReader(str(output_path))
    text = "".join(p.extract_text() for p in reader.pages if p.extract_text())
    assert "- Test bullet point" in text


def test_add_table(tmp_path):
    """TC-81: Ensure table is added to PDF."""
    dummy_preprocessor = MagicMock()
    dummy_modeler = MagicMock()
    report = ReportingEngine(dummy_preprocessor, dummy_modeler)

    header = ["Name", "Score"]
    data = [["Dan", 90], ["Jean", 85]]
    col_widths = [40, 30]

    report.add_table(header, data, col_widths)
    output_path = tmp_path / "table_test.pdf"
    report.pdf.output(str(output_path))

    reader = PdfReader(str(output_path))
    text = "".join(p.extract_text() for p in reader.pages if p.extract_text())
    assert "Name" in text and "Score" in text
    assert "Dan" in text and "Jean" in text
    assert "90" in text and "85" in text


def test_write_preprocessing_summary():
    """TC-82: Ensure preprocessing summary is written to PDF."""
    # set up
    mock_preprocessor = MagicMock()
    mock_preprocessor.df.shape = (100, 10)
    mock_preprocessor.task_type = 'classification'
    mock_preprocessor.dropped_columns = ['col']
    mock_preprocessor.cols_to_remove = []
    mock_preprocessor.feature_encoder.categories_ = [
        ['A', 'B'], ['X', 'Y']]
    mock_preprocessor.feature_encoder.__class__.__name__ = 'OneHotEncoder'
    mock_preprocessor.feature_encoder.feature_names_in_ = [
        'feature1', 'feature2']
    mock_preprocessor.scaler.scale_ = [1.0, 2.0]
    mock_preprocessor.scaler.__class__.__name__ = 'StandardScaler'
    mock_preprocessor.label_encoder.classes_ = ['yes', 'no']
    mock_preprocessor.label_encoder.__class__.__name__ = 'LabelEncoder'
    mock_preprocessor.X_train.shape = (80, 10)
    mock_preprocessor.X_test.shape = (20, 10)
    mock_modeler = MagicMock()
    report = ReportingEngine(mock_preprocessor, mock_modeler)
    report.preprocessor = mock_preprocessor
    report.section_header = MagicMock()
    report.subsection = MagicMock()
    report.add_bullet = MagicMock()
    report.plot_class_distribution = MagicMock()
    report.plot_reg_distribution = MagicMock()

    # run
    report.write_preprocessing_summary()

    # assertions
    report.section_header.assert_called_with("Preprocessing Summary")
    report.subsection.assert_any_call("Initial Data Overview")
    report.add_bullet.assert_any_call("Number of rows in the dataset: 100")
    report.add_bullet.assert_any_call(
        "Number of columns in the dataset: 10")
    report.plot_class_distribution.assert_called_once()
    report.plot_reg_distribution.assert_not_called()
    report.subsection.assert_any_call("Data Cleaning")
    report.add_bullet.assert_any_call(
        "Handled missing numerical values by replacing with mean.")
    report.add_bullet.assert_any_call(
        "Removed columns from the dataset: ['col']")
    report.subsection.assert_any_call("Feature Engineering")
    report.add_bullet.assert_any_call(
        "Feature encoding method: OneHotEncoder")
    report.add_bullet.assert_any_call("feature1: A, B")
    report.add_bullet.assert_any_call("feature2: X, Y")
    report.add_bullet.assert_any_call(
        "Feature scaling applied using StandardScaler.")
    report.add_bullet.assert_any_call(
        "Label encoding applied for target variable: LabelEncoder")
    report.add_bullet.assert_any_call("Encoded target classes: yes, no")
    report.subsection.assert_any_call("Data Splitting")
    report.add_bullet.assert_any_call("Train set size: 80 rows")
    report.add_bullet.assert_any_call("Test set size: 20 rows")

def test_classification_modeling():
    """TC-83: Ensure classification modeling summary is written to PDF."""
    report = ReportingEngine(preprocessor=MagicMock(), modeler=MagicMock())
    report.preprocessor.task_type = 'classification'
    report.modeler.task_type = 'classification'
    report.add_bullet = MagicMock()

    dummy_model = MagicMock()
    dummy_model.predict.return_value = np.array([1, 0, 1, 1])

    report.modeler.X_test = np.array([[1], [2], [3], [4]])
    report.modeler.y_test = np.array([1, 0, 0, 1])

    report.modeler.results = {
        'tuned': {
            'LogisticRegression': {
                'optimized_model': dummy_model
            }
        }
    }

    report.modeler.get_best_tuned_model.return_value = {
        'model_name': 'LogisticRegression',
        'best_score': 0.85,
        'best_params': {'C': 1.0, 'solver': 'lbfgs'}
    }

    report.write_modeling_summary()

    print(report.add_bullet.call_args_list)

    report.add_bullet.assert_any_call("Classification task detected. The following models were considered:")
    report.add_bullet.assert_any_call("Accuracy measures the proportion of correct predictions out of all predictions made by the model.")
    report.add_bullet.assert_any_call("The best tuned model was LogisticRegression, which achieved the highest cross-validated accuracy of 85.00%.")

def test_regression_modeling():
    """TC-84: Ensure regression modeling summary is written to PDF."""
    report = ReportingEngine(preprocessor=MagicMock(), modeler=MagicMock())
    report.preprocessor.task_type = 'regression'
    report.modeler.task_type = 'regression'
    report.add_bullet = MagicMock()

    # Dummy regression model
    dummy_model = MagicMock()
    dummy_model.predict.return_value = np.array([3.2, 4.1, 5.0, 6.2])

    report.modeler.X_test = np.array([[1], [2], [3], [4]])
    report.modeler.y_test = np.array([3.0, 4.0, 5.0, 6.0])

    report.modeler.results = {
        'tuned': {
            'LinearRegression': {
                'optimized_model': dummy_model
            }
        }
    }

    report.modeler.get_best_tuned_model.return_value = {
        'model_name': 'LinearRegression',
        'best_score': 0.91,
        'best_params': {}
    }

    report.write_modeling_summary()

    print(report.add_bullet.call_args_list)

    report.add_bullet.assert_any_call("Regression task detected. The following models were considered:")
    report.add_bullet.assert_any_call("R-squared (R²) measures the proportion of variance in the target variable explained by the regression model, indicating how well the model fits the data.")
    report.add_bullet.assert_any_call("The best tuned model was LinearRegression, which achieved the highest cross-validated R² score of 0.9100.")

def test_write_visuals_section_classification():
    """TC-85: Ensure visuals section is written for classification tasks."""
    report = ReportingEngine(preprocessor=MagicMock(), modeler=MagicMock())
    report.preprocessor.task_type = 'classification'

    report.section_header = MagicMock()
    report.generate_conf_matrix = MagicMock()
    report.plot_feature_importance = MagicMock()
    report.plot_roc_curve = MagicMock()
    report.plot_precision_recall_curve = MagicMock()
    report.plot_classification_report = MagicMock()
    report.plot_classification_model_performance_bar_chart = MagicMock()
    report.plot_cv_score_boxplots = MagicMock()

    report.write_visuals_section()

    report.section_header.assert_called_once_with("3. Additional Visuals")
    report.generate_conf_matrix.assert_called_once()
    report.plot_feature_importance.assert_called_once()
    report.plot_roc_curve.assert_called_once()
    report.plot_precision_recall_curve.assert_called_once()
    report.plot_classification_report.assert_called_once()
    report.plot_classification_model_performance_bar_chart.assert_called_once()
    report.plot_cv_score_boxplots.assert_called_once()

def test_write_visuals_section_regression():
    """TC-86: Ensure visuals section is written for regression tasks."""
    report = ReportingEngine(preprocessor=MagicMock(), modeler=MagicMock())
    report.preprocessor.task_type = 'regression'

    report.section_header = MagicMock()
    report.plot_residuals = MagicMock()
    report.plot_actual_vs_predicted = MagicMock()
    report.plot_error_distribution = MagicMock()
    report.plot_feature_importance = MagicMock()
    report.plot_cv_score_boxplots = MagicMock()

    report.write_visuals_section()

    report.section_header.assert_called_once_with("3. Additional Visuals")
    report.plot_residuals.assert_called_once()
    report.plot_actual_vs_predicted.assert_called_once()
    report.plot_error_distribution.assert_called_once()
    report.plot_feature_importance.assert_called_once()
    report.plot_cv_score_boxplots.assert_called_once()

def test_plot_reg_distribution():
    """TC-87: Ensure regression distribution plot is generated."""
    report = ReportingEngine(preprocessor=MagicMock(), modeler=MagicMock())
    report.preprocessor.y = [1, 2, 3, 4, 5]
    report.pdf = MagicMock()

    with patch('matplotlib.pyplot.savefig'), patch('matplotlib.pyplot.close'), patch('os.remove'):
        report.plot_reg_distribution()

    report.pdf.image.assert_called_once()

def test_plot_residuals():
    """TC-88: Ensure residuals plot is generated for regression tasks."""
    report = ReportingEngine(preprocessor=MagicMock(), modeler=MagicMock())
    report.preprocessor.task_type = 'regression'
    report.preprocessor.X_test = np.array([[1], [2], [3]])
    report.preprocessor.y_test = np.array([2, 4, 6])
    report.modeler.results = {
        'tuned': {
            'LinearRegression': {
                'optimized_model': MagicMock(predict=MagicMock(return_value=np.array([1.8, 4.2, 5.9])))
            }
        }
    }
    report.pdf = MagicMock()
    report.add_bullet = MagicMock()
    report.subsection = MagicMock()

    with patch('matplotlib.pyplot.savefig'), patch('matplotlib.pyplot.close'), patch('os.remove'):
        report.plot_residuals()

    report.subsection.assert_called_once_with("Residuals Plot")
    report.add_bullet.assert_any_call("The residuals plot displays the difference between predicted and actual values, helping to diagnose non-linearity, unequal error variance, and outliers in regression models.")
    report.pdf.image.assert_called_once()

def test_plot_actual_vs_predicted():
    """TC-89: Ensure actual vs predicted plot is generated for regression tasks."""
    report = ReportingEngine(preprocessor=MagicMock(), modeler=MagicMock())
    report.preprocessor.task_type = 'regression'
    report.preprocessor.X_test = np.array([[1], [2], [3]])
    report.preprocessor.y_test = np.array([1.5, 2.5, 3.5])
    report.modeler.results = {'tuned': {'LinearRegression': {'optimized_model': MagicMock()}}}
    report.modeler.results['tuned']['LinearRegression']['optimized_model'].predict.return_value = np.array([1.4, 2.6, 3.6])
    report.pdf = MagicMock()
    report.add_bullet = MagicMock()
    report.subsection = MagicMock()

    with patch('matplotlib.pyplot.savefig'), patch('matplotlib.pyplot.close'), patch('os.remove'):
        report.plot_actual_vs_predicted()

    report.subsection.assert_called_once_with("Actual vs. Predicted Plot")
    report.add_bullet.assert_any_call("The actual vs. predicted plot compares the model's predictions to the true values, providing a visual assessment of prediction accuracy and potential bias in regression models.")
    report.pdf.image.assert_called_once()

def test_plot_error_distribution():
    """TC-90: Ensure error distribution plot is generated for regression tasks."""
    report = ReportingEngine(preprocessor=MagicMock(), modeler=MagicMock())
    report.preprocessor.task_type = 'regression'
    report.preprocessor.X_test = np.array([[1], [2], [3]])
    report.preprocessor.y_test = np.array([1.5, 2.5, 3.5])
    report.modeler.results = {'tuned': {'LinearRegression': {'optimized_model': MagicMock()}}}
    report.modeler.results['tuned']['LinearRegression']['optimized_model'].predict.return_value = np.array([1.4, 2.6, 3.6])
    report.pdf = MagicMock()
    report.add_bullet = MagicMock()
    report.subsection = MagicMock()

    with patch('matplotlib.pyplot.savefig'), patch('matplotlib.pyplot.close'), patch('os.remove'):
        report.plot_error_distribution()

    report.subsection.assert_called_once_with("Error Distribution Plot")
    report.add_bullet.assert_any_call("The error distribution plot shows the frequency distribution of prediction errors, helping to evaluate model bias and detect if errors are normally distributed.")
    report.pdf.image.assert_called_once()

def test_plot_class_distribution():
    """TC-91: Ensure class distribution plot is generated for classification tasks."""
    report = ReportingEngine(preprocessor=MagicMock(), modeler=MagicMock())
    report.preprocessor.task_type = 'classification'
    report.preprocessor.y = np.array([0, 1, 1, 0, 2])
    report.preprocessor.decode_target = MagicMock(return_value=report.preprocessor.y)
    report.pdf = MagicMock()
    report.add_bullet = MagicMock()

    with patch('matplotlib.pyplot.savefig'), patch('matplotlib.pyplot.close'), patch('os.remove'):
        report.plot_class_distribution()

    report.add_bullet.assert_not_called()  # No bullets for normal path
    report.pdf.image.assert_called_once()

def test_generate_conf_matrix():
    """TC-92: Ensure confusion matrix heatmap is generated for classification tasks."""
    report = ReportingEngine(preprocessor=MagicMock(), modeler=MagicMock())
    report.preprocessor.task_type = 'classification'
    report.preprocessor.X_test = np.array([[1], [2]])
    report.preprocessor.y_test = np.array([0, 1])
    report.preprocessor.decode_target = MagicMock(side_effect=lambda x: x)
    report.modeler.results = {'tuned': {'ModelA': {'optimized_model': MagicMock()}}}
    report.modeler.results['tuned']['ModelA']['optimized_model'].predict.return_value = np.array([0, 1])
    report.pdf = MagicMock()
    report.add_bullet = MagicMock()
    report.subsection = MagicMock()

    with patch('matplotlib.pyplot.savefig'), patch('matplotlib.pyplot.close'), patch('os.remove'):
        report.generate_conf_matrix()

    report.subsection.assert_called_once_with("Confusion Matrix Heatmap")
    report.add_bullet.assert_any_call("A confusion matrix heatmap visualizes the number of correct and incorrect predictions for each class, helping to identify patterns of misclassification.")
    report.pdf.image.assert_called_once()

def test_plot_roc_curve():
    """TC-93: Ensure ROC curve is generated for classification tasks."""
    report = ReportingEngine(preprocessor=MagicMock(), modeler=MagicMock())
    report.preprocessor.task_type = 'classification'
    report.preprocessor.X_test = np.array([[1], [2], [3], [4]])
    report.preprocessor.y_test = np.array([0, 1, 0, 1])
    report.modeler.results = {'tuned': {'ModelA': {'optimized_model': MagicMock()}}}
    
    report.modeler.results['tuned']['ModelA']['optimized_model'].predict_proba.return_value = np.array([
        [0.6, 0.4],
        [0.3, 0.7],
        [0.8, 0.2],
        [0.1, 0.9]
    ])
    
    report.pdf = MagicMock()
    report.add_bullet = MagicMock()
    report.subsection = MagicMock()

    with patch('matplotlib.pyplot.savefig'), patch('matplotlib.pyplot.close'), patch('os.remove'):
        report.plot_roc_curve()

    report.subsection.assert_called_once_with("ROC Curve")
    report.add_bullet.assert_any_call("The ROC (Receiver Operating Characteristic) curve illustrates the trade-off between true positive rate and false positive rate across different thresholds, helping assess the model's ability to distinguish between classes.")
    report.pdf.image.assert_called_once()

def test_plot_precision_recall_curve():
    """TC-94: Ensure precision-recall curve is generated for classification tasks."""
    report = ReportingEngine(preprocessor=MagicMock(), modeler=MagicMock())
    report.preprocessor.task_type = 'classification'
    report.preprocessor.X_test = np.array([[1], [2], [3], [4]])
    report.preprocessor.y_test = np.array([0, 1, 0, 1])
    report.modeler.results = {'tuned': {'ModelA': {'optimized_model': MagicMock()}}}

    model = report.modeler.results['tuned']['ModelA']['optimized_model']
    model.predict_proba.return_value = np.array([
        [0.6, 0.4],
        [0.3, 0.7],
        [0.8, 0.2],
        [0.1, 0.9]
    ])
    report.pdf = MagicMock()
    report.add_bullet = MagicMock()
    report.subsection = MagicMock()

    with patch('matplotlib.pyplot.savefig'), patch('matplotlib.pyplot.close'), patch('os.remove'):
        report.plot_precision_recall_curve()

    report.subsection.assert_called_once_with("Precision-Recall Curve")
    report.add_bullet.assert_any_call("The Precision-Recall curve shows the tradeoff between precision and recall for different decision thresholds, helping assess model performance on imbalanced classification tasks.")
    report.pdf.image.assert_called_once()


def test_plot_classification_report():
    """TC-95: Ensure classification report heatmap is generated for classification tasks."""
    report = ReportingEngine(preprocessor=MagicMock(), modeler=MagicMock())
    report.preprocessor.task_type = 'classification'
    report.preprocessor.X_test = np.array([[1], [2]])
    report.preprocessor.y_test = np.array([0, 1])
    report.preprocessor.decode_target = MagicMock(side_effect=lambda x: x)
    report.modeler.results = {'tuned': {'ModelA': {'optimized_model': MagicMock()}}}
    model = report.modeler.results['tuned']['ModelA']['optimized_model']
    model.predict.return_value = np.array([0, 1])
    report.pdf = MagicMock()
    report.add_bullet = MagicMock()
    report.subsection = MagicMock()

    with patch('matplotlib.pyplot.savefig'), patch('matplotlib.pyplot.close'), patch('seaborn.heatmap'), patch('os.remove'), patch('sklearn.metrics.classification_report') as mock_cr:
        mock_cr.return_value = {
            '0': {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 1},
            '1': {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 1},
            'accuracy': 1.0,
            'macro avg': {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 2},
            'weighted avg': {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 2}
        }
        report.plot_classification_report()

    report.subsection.assert_called_once_with("Classification Report Heatmap")
    report.add_bullet.assert_any_call("The classification report heatmap visualizes key metrics like precision, recall, and F1-score for each class, making it easier to compare performance across categories.")
    report.pdf.image.assert_called_once()


def test_plot_classification_model_performance_bar_chart():
    """TC-96: Ensure classification model performance bar chart is generated."""
    report = ReportingEngine(preprocessor=MagicMock(), modeler=MagicMock())
    report.preprocessor.task_type = 'classification'
    report.modeler.results = {
        'tuned': {
            'ModelA': {'train': 0.9, 'test': 0.85},
            'ModelB': {'train': 0.92, 'test': 0.88},
        }
    }
    report.pdf = MagicMock()
    report.add_bullet = MagicMock()
    report.subsection = MagicMock()

    with patch('matplotlib.pyplot.savefig'), patch('matplotlib.pyplot.close'), patch('os.remove'):
        report.plot_classification_model_performance_bar_chart()

    report.subsection.assert_called_once_with("Metrics Bar Chart")
    report.add_bullet.assert_any_call("The metrics bar chart compares train and test scores for each classification model, helping assess potential overfitting and generalization performance.")
    report.pdf.image.assert_called_once()

def test_plot_feature_importance():
    """TC-97: Ensure feature importance plot is generated for classification tasks."""
    report = ReportingEngine(preprocessor=MagicMock(), modeler=MagicMock())
    report.preprocessor.task_type = 'classification'
    report.preprocessor.X_train = np.array([[1, 2], [3, 4]])
    report.preprocessor.final_columns = ['feat1', 'feat2']
    report.modeler.results = {
        'tuned': {
            'ModelA': {'optimized_model': MagicMock(feature_importances_=np.array([0.7, 0.3]))}
        }
    }
    report.pdf = MagicMock()
    report.add_bullet = MagicMock()
    report.subsection = MagicMock()

    with patch('matplotlib.pyplot.savefig'), patch('matplotlib.pyplot.close'), patch('os.remove'):
        report.plot_feature_importance()

    report.subsection.assert_called_once_with("Feature Importance Plot")
    report.add_bullet.assert_any_call("A feature importance plot highlights which input features had the most influence on the model's predictions, offering insights into key drivers of the outcome.")
    report.pdf.image.assert_called_once()

def test_plot_cv_score_boxplots():
    """TC-98: Ensure cross-validation score boxplots are generated for regression tasks."""
    report = ReportingEngine(preprocessor=MagicMock(), modeler=MagicMock())
    report.preprocessor.task_type = 'regression'
    report.modeler.results = {
        'untuned': {
            'ModelA': {'cv_scores': [0.8, 0.82, 0.78]},
            'ModelB': {'cv_scores': [0.75, 0.77, 0.74]}
        }
    }
    report.pdf = MagicMock()
    report.add_bullet = MagicMock()
    report.subsection = MagicMock()

    with patch('matplotlib.pyplot.savefig'), patch('matplotlib.pyplot.close'), patch('seaborn.boxplot'), patch('os.remove'):
        report.plot_cv_score_boxplots()

    report.subsection.assert_called_once_with("Cross-Validation Score Boxplots")
    report.add_bullet.assert_any_call("The cross-validation box plot visualizes the distribution of scores across CV folds for each model, highlighting variability, consistency, and potential outliers in performance.")
    report.pdf.image.assert_called_once()
