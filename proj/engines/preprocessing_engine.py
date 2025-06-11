import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
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
        self.feature_encoder = OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore')
        self.target_encoder = LabelEncoder()
        # Default to empty list
        self.columns_to_remove = columns_to_remove if columns_to_remove else []
        self.X, self.y = None, None  # Placeholders for features and target

        logging.info(
            f"Initialized PreprocessingEngine with task type: {self.task_type}")

    def remove_unwanted_columns(self, df):
        """Removes specified columns from the input DataFrame and returns the cleaned DataFrame."""
        if self.columns_to_remove:
            df_cleaned = df.drop(columns=self.columns_to_remove, errors="ignore")
            logging.info(f"Removed unwanted columns: {self.columns_to_remove}")
        else:
            df_cleaned = df.copy()
            logging.info("No columns specified for removal.")

    return df_cleaned

    def clean_categorical_columns(self, df):
        """Cleans categorical columns in the input DataFrame and returns the cleaned DataFrame."""
        if not self.categorical_columns:
            logging.info("No categorical columns specified.")
            return df.copy()

        valid_cols = [col for col in self.categorical_columns if col in df.columns]
        if not valid_cols:
            logging.info("No valid categorical columns found in the input DataFrame.")
            return df.copy()

        df_cleaned = df.copy()
        for col in valid_cols:
            df_cleaned[col] = df_cleaned[col].astype(str).str.strip().str.lower()
            df_cleaned[col] = df_cleaned[col].replace(self.MISSING_INDICATORS, pd.NA)

        logging.info(f"Cleaned categorical columns: {', '.join(valid_cols)}")
        return df_cleaned

        
    def clean_continuous_columns(self, df):
        """Cleans and converts continuous columns stored as text in the input DataFrame."""
        df_cleaned = df.copy()
        fixed_columns = []

        continuous_columns = [
            col for col in df_cleaned.columns if col not in self.categorical_columns
        ]

        for col in continuous_columns:
            original_values = df_cleaned[col].copy()

            df_cleaned[col] = (
                df_cleaned[col]
                .astype(str)
                .str.strip()
                .str.replace(r"[^\d\.\-]", "", regex=True)
                .apply(pd.to_numeric, errors="coerce")
            )

            if not original_values.equals(df_cleaned[col]):
                fixed_columns.append(col)

        if fixed_columns:
            logging.info(f"Cleaned continuous columns: {', '.join(fixed_columns)}")
        else:
            logging.info("No continuous columns required cleaning.")

        return df_cleaned

    def drop_missing_columns(self, df, threshold=0.6):
        """ Drops columns in the input DataFrame that have more than the given threshold of missing values. """
        df_cleaned = df.copy()
        missing_percentage = df_cleaned.isnull().mean()
        columns_to_drop = missing_percentage[missing_percentage > threshold].index.tolist()

        if columns_to_drop:
            df_cleaned.drop(columns=columns_to_drop, inplace=True)
            logging.info(f"Dropped columns with excessive missing values: {columns_to_drop}")
        else:
            logging.info("No columns dropped due to missing values.")

        return df_cleaned, columns_to_drop

    def drop_missing_rows(self, df, threshold=0.4):
        """ Drops rows from the input DataFrame that are missing the target or have excessive missing values. """
        df_cleaned = df.copy()
        initial_row_count = len(df_cleaned)

        # Drop rows missing the target
        df_cleaned.dropna(subset=[self.target_column], inplace=True)
        target_dropped = initial_row_count - len(df_cleaned)

        # Drop rows with excessive missing data (excluding target)
        row_missing_percentage = df_cleaned.drop(columns=[self.target_column]).isnull().mean(axis=1)
        df_cleaned = df_cleaned.loc[row_missing_percentage < threshold]
        excessive_missing_dropped = initial_row_count - target_dropped - len(df_cleaned)

        if target_dropped > 0:
            logging.info(f"Dropped {target_dropped} rows missing the target variable ({self.target_column}).")
        if excessive_missing_dropped > 0:
            logging.info(f"Dropped {excessive_missing_dropped} rows with excessive missing values.")
        if target_dropped == 0 and excessive_missing_dropped == 0:
            logging.info("No rows dropped due to missing values.")

        return df_cleaned

    def handle_missing_data(self, df):
        """
        Handles missing values in the input DataFrame by:
        - Imputing numeric columns with the mean
        - Imputing categorical columns with the mode

        Returns the cleaned DataFrame and a dictionary with lists of columns that were imputed.
        """
        df_cleaned = df.copy()
        numeric_missing_handled = []
        categorical_missing_handled = []

        for col in df_cleaned.columns:
            if col in self.categorical_columns:
                if df_cleaned[col].isnull().any():
                    mode_val = df_cleaned[col].mode(dropna=True)
                    if not mode_val.empty:
                        df_cleaned[col] = df_cleaned[col].fillna(mode_val[0])
                        categorical_missing_handled.append(col)
            else:
                if df_cleaned[col].isnull().any():
                    mean_val = df_cleaned[col].mean()
                    if pd.notnull(mean_val):
                        df_cleaned[col] = df_cleaned[col].fillna(mean_val)
                        numeric_missing_handled.append(col)

        if numeric_missing_handled or categorical_missing_handled:
            if numeric_missing_handled:
                logging.info(f"Imputed missing values in numeric columns using mean: {numeric_missing_handled}")
            if categorical_missing_handled:
                logging.info(f"Imputed missing values in categorical columns using mode: {categorical_missing_handled}")
        else:
            logging.info("No missing data to handle.")

        return df_cleaned
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

    def scale_continuous_features(self):
        """ Scales all continuous features using StandardScaler, excluding possible continuous target columns. """
        self.df = self.df.copy()
        cont_features = [
            col for col in self.df.columns
            if col not in self.categorical_columns and col != self.target_column
        ]

        if not cont_features:
            logging.warning("No continuous features found to scale.")
            return

        self.df[cont_features] = self.scaler.fit_transform(self.df[cont_features])
        logging.info(f"Scaled continuous features: {cont_features}")

    def one_hot_encode_categorical_features(self):
        """ Applies one-hot encoding to categorical features in self.df, excluding the target column and binaries. """

        self.df = self.df.copy()

        binary_feat = [
            col for col in self.categorical_columns
            if col in self.df.columns and col != self.target_column and self.df[col].nunique() == 2
        ]
        multi_cat_feat = [
            col for col in self.categorical_columns
            if col in self.df.columns and col != self.target_column and self.df[col].nunique() > 2
        ]
        for col in binary_feat:
            self.df[col] = self.df[col].astype('category').cat.codes

        if multi_cat_feat:
            self.df = pd.get_dummies(self.df, columns=multi_cat_feat, drop_first=True)
            logging.info(f"One-hot encoded features: {multi_cat_feat}")

        if not binary_feat and not multi_cat_feat:
            logging.warning("No categorical features found to encode.")
        else:
            logging.info(f"Label encoded binary columns: {binary_feat}")

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

        self.remove_unwanted_columns() # drops columns specified for removal
        self.clean_categorical_columns()
        self.clean_continuous_columns()
        self.drop_missing_columns() # drop columns with excessive missing values
        self.drop_missing_rows() # drop rows with excessive missing values or missing target
        self.handle_missing_data() # fill in missing values
        self.verify_final_dataset() # print out of remaining missing values
        self.encode_target_column()
        self.scale_continuous_features()
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
        