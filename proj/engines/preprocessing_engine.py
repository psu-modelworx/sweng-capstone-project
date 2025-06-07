import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


class PreprocessingEngine:
    """ Handles data preprocessing, including missing data handling, feature scaling, encoding, and train-test splitting. """

    DEFAULT_TEST_SIZE = 0.2
    DEFAULT_RANDOM_STATE = 42
    MAX_UNIQUE_CATEGORICAL_VALUES = 10
    MISSING_INDICATORS = {"none", "n/a", "na", "null", "missing", "unknown", "", "<na>", "nan"}

    def __init__(self, df, target_column, categorical_columns=None, columns_to_remove=None, test_size=DEFAULT_TEST_SIZE, random_state=DEFAULT_RANDOM_STATE):
        """ Initializes the preprocessing engine with required properties. """
        self.original_df = df.copy()  # Stores the original dataset before preprocessing
        self.df = df.copy()  # Working copy for transformations
        self.final_df = None  # Placeholder for final processed dataset
        if categorical_columns is None:
            self.categorical_columns = []
        elif isinstance(categorical_columns, str):
            self.categorical_columns = [categorical_columns]
        else:
            self.categorical_columns = list(categorical_columns)
        self.target_column = target_column
        self.original_target_column = target_column  # Store original target column name for later reference
        self.target_is_categorical = target_column in (categorical_columns if isinstance(
            categorical_columns, list) else [categorical_columns])
        self.task_type = 'classification' if self.target_is_categorical else 'regression'
        self.label_mapping_df = None
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        # Default to empty list
        self.columns_to_remove = columns_to_remove if columns_to_remove else []
        self.X, self.y = None, None  # Placeholders for features and target

        logging.info(
            f"Initialized PreprocessingEngine with task type: {self.task_type}")

    def remove_unwanted_columns(self):
        """ Removes specified columns from the dataset before further processing. """
        if self.columns_to_remove:
            self.df.drop(columns=self.columns_to_remove,
                         errors="ignore", inplace=True)
            logging.info(f"Removed unwanted columns: {self.columns_to_remove}")
        else:
            logging.info("No columns specified for removal.")

    def detect_and_convert_categoricals(self):
        """Detects categorical columns based on unique value counts and converts them."""

        # convert explicitly labeled categorical columns
        self.df = self.df.copy()
        for col in self.categorical_columns:
            if col in self.df:
                self.df[col] = self.df[col].astype(str).str.strip().str.lower().astype("category")

        # Detect additional categorical columns
        unique_counts = self.df.nunique()
        detected_categorical_columns = [
            col for col in unique_counts.index
            if col not in self.categorical_columns
            and unique_counts[col] < self.MAX_UNIQUE_CATEGORICAL_VALUES
            and self.df[col].dtype != "object"
        ]

        if detected_categorical_columns:
            self.categorical_columns = list(set(self.categorical_columns + detected_categorical_columns))
            for col in detected_categorical_columns:
                self.df[col] = self.df[col].astype("category")
            logging.info(f"Detected and converted categorical columns: {', '.join(detected_categorical_columns)}")
        else:
            logging.info("No categorical columns detected.")

    def clean_text_columns(self):
        text_columns = [col for col in self.df.columns if self.df[col].dtype in ['object', 'category']]
        if text_columns:
            for col in text_columns:
                # Convert all to string, lowercase, strip
                self.df[col] = self.df[col].astype(str).str.strip().str.lower()
                self.df.loc[self.df[col].isin(self.MISSING_INDICATORS), col] = pd.NA
            logging.info(f"Cleaned all text columns: {', '.join(text_columns)}")
        else:
            logging.info("No text columns detected.")

    def clean_numeric_columns(self):
        """Cleans and converts numeric-like columns stored as text."""
        self.df = self.df.copy()
        fixed_columns = []

        for col in self.df.columns:
            if col not in self.categorical_columns:
                # Preserve original values
                original_values = self.df[col].copy()

                self.df[col] = (
                    self.df[col]
                    .astype(str)
                    .str.strip()
                    .str.replace(r"[^\d\.\-]", "", regex=True)
                    .apply(pd.to_numeric, errors="coerce")
                )

                # Check if values changed after processing
                if not original_values.equals(self.df[col]):
                    fixed_columns.append(col)

        if fixed_columns:
            logging.info(
                f"Fixed non-numeric values in columns: {', '.join(fixed_columns)}")
        else:
            logging.info("No non-numeric values found in numeric columns.")

    def drop_missing_columns(self, threshold=0.6):
        """Drops columns that have more than the given threshold of missing values."""

        missing_percentage = self.df.isnull().mean()
        columns_to_drop = missing_percentage[missing_percentage > threshold].index.tolist(
        )

        if columns_to_drop:
            self.df.drop(columns=columns_to_drop, inplace=True)
            logging.info(
                f"Dropped columns with excessive missing values: {columns_to_drop}")
        else:
            logging.info("No columns dropped due to missing values.")

    def drop_missing_rows(self, threshold=0.4):
        """Drops rows missing the target or with excessive missing values."""
        import numpy as np

        initial_row_count = len(self.df)

        # Handle target
        self.df[self.target_column] = self.df[self.target_column].astype(str).str.strip().str.lower()
        self.df[self.target_column] = self.df[self.target_column].replace(list(self.MISSING_INDICATORS), np.nan)
        self.df.dropna(subset=[self.target_column], inplace=True)
        target_dropped = initial_row_count - len(self.df)

        # Handle rest
        row_missing_percentage = self.df.isnull().mean(axis=1)
        self.df = self.df.loc[row_missing_percentage < threshold]
        excessive_missing_dropped = initial_row_count - target_dropped - len(self.df)

        if target_dropped > 0:
            logging.info(f"Dropped {target_dropped} rows missing the target variable ({self.target_column}).")
        if excessive_missing_dropped > 0:
            logging.info(f"Dropped {excessive_missing_dropped} rows with excessive missing values.")
        if target_dropped == 0 and excessive_missing_dropped == 0:
            logging.info("No rows dropped due to missing values.")

    def handle_missing_data(self):
        """Handles missing values by imputing numerical features with mean and categorical features with mode."""

        numeric_missing_handled = []
        categorical_missing_handled = []

        for col in self.df.columns:
            if col in self.categorical_columns:
                if self.df[col].isnull().any():
                    self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
                    categorical_missing_handled.append(col)
            else:
                if self.df[col].isnull().any():
                    self.df[col] = self.df[col].fillna(self.df[col].mean())
                    numeric_missing_handled.append(col)

        if numeric_missing_handled or categorical_missing_handled:
            if numeric_missing_handled:
                logging.info(
                    f"Imputed missing values in numeric columns using mean: {numeric_missing_handled}")
            if categorical_missing_handled:
                logging.info(
                    f"Imputed missing values in categorical columns using mode: {categorical_missing_handled}")
        else:
            logging.info("No missing data to handle.")

    def verify_final_dataset(self):
        """Logs remaining missing data counts after cleaning."""

        remaining_missing = self.df.isnull().sum()
        logging.info(f"Final dataset missing values:\n{remaining_missing}")

    def encode_target_column(self):
        """ Encodes a categorical target column if applicable. """
        if not self.target_is_categorical:
            logging.info("Skipping target encoding (not categorical).")
            return

        # Save original target column name for decoding
        self.original_target_column = self.target_column

        encoded_col = f"{self.target_column}_encoded"
        self.df[encoded_col] = self.df[self.target_column].astype('category').cat.codes

        self.label_mapping_df = (
            self.df[[self.target_column, encoded_col]]
            .drop_duplicates()
            .sort_values(encoded_col)
            .reset_index(drop=True)
        )

        if encoded_col not in self.categorical_columns:
            self.categorical_columns.append(encoded_col)

        self.df.drop(columns=[self.target_column], inplace=True)
        self.target_column = encoded_col

        logging.info(f"Encoded target column '{self.target_column}'.")

    def split_features_and_target(self):
        """ Splits data into features (X) and target (y)"""
        X = self.df.drop(columns=self.target_column, errors="ignore")
        y = self.df[self.target_column]

        self.X, self.y = X, y

        logging.info(f"Split dataset. Using '{self.target_column}' as target.")
        return X, y

    def scale_numerical_columns(self):
        """ Scales all numerical features using StandardScaler, excluding categorical and target columns. """
        self.df = self.df.copy()
        original_numerical_columns = [
            col for col in self.df.columns
            if col not in self.categorical_columns and col != self.target_column
        ]

        if original_numerical_columns:
            self.df[original_numerical_columns] = self.scaler.fit_transform(
                self.df[original_numerical_columns])
            logging.info(
                f"Scaled numerical columns: {original_numerical_columns}")
        else:
            logging.warning("No numerical columns found to scale.")

    def one_hot_encode_categorical_features(self):
        """ Applies one-hot encoding to categorical features in self.df, excluding the target column. """

        self.df = self.df.copy()

        categorical_cols_to_encode = [
            col for col in self.categorical_columns
            if col in self.df.columns and col != self.target_column and self.df[col].nunique() > 2
        ]

        if categorical_cols_to_encode:
            self.df = pd.get_dummies(
                self.df, columns=categorical_cols_to_encode, drop_first=True)
            logging.info(
                f"One-hot encoded categorical features: {categorical_cols_to_encode}")
        else:
            logging.warning(
                "No categorical columns found in features to encode. Skipping one-hot encoding.")

    def train_test_split_data(self, X, y):
        """ Splits dataset into training and testing sets. """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state)
        logging.info(
            f"Split data into train ({1 - self.test_size:.0%}) and test ({self.test_size:.0%}) sets.")
        return X_train, X_test, y_train, y_test

    def run_preprocessing_engine(self):
        """ Runs the full preprocessing pipeline and saves final processed data. """
        logging.info("Running PreprocessingEngine...")

        self.remove_unwanted_columns()
        self.clean_text_columns()
        self.detect_and_convert_categoricals()
        self.clean_numeric_columns()
        self.detect_and_convert_categoricals() # second pass after cleaning numeric columns
        self.drop_missing_columns()
        self.drop_missing_rows()
        self.handle_missing_data()
        self.verify_final_dataset()
        self.encode_target_column()
        self.scale_numerical_columns()
        self.one_hot_encode_categorical_features()
        X, y = self.split_features_and_target()
        X_train, X_test, y_train, y_test = self.train_test_split_data(X, y)

        # Save final dataset
        self.final_df = pd.concat([X, y], axis=1)
        logging.info(
            "Preprocessing completed successfully. Final dataset stored.")

        return X_train, X_test, y_train, y_test, self.task_type

    def summary(self):
        return {
            "task_type": self.task_type,
            "features": list(self.final_df.columns.difference([self.target_column])),
            "missing_values": self.df.isnull().sum().sum(),
            "label_mapping": self.label_mapping_df if self.label_mapping_df is not None else "N/A"
        }

    def decode_target(self, encoded_values):
        """ Maps encoded target values back to original labels. """
        if not self.target_is_categorical:
            logging.warning("Target is not categorical. skipping decoding.")
            return encoded_values
    
        if self.label_mapping_df is not None:
            original_col = self.original_target_column
            encoded_col = self.target_column

            inverse_mapping = dict(zip(
                self.label_mapping_df[encoded_col],
                self.label_mapping_df[original_col]
            ))

            return [inverse_mapping.get(val, val) for val in encoded_values]
        else:
            return encoded_values
        