import pandas as pd
import joblib
import json
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
        self.dropped_columns = None
        self.original_df = df.copy()  # Stores the original dataset before preprocessing
        self.original_columns = df.columns.tolist()  # Store original columns for reference
        self.df = df.copy()  # Working copy for transformations
        self.final_df = None  # Placeholder for final processed dataset
        self.final_columns = []  # Placeholder for final columns after preprocessing
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
        self.feature_encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        self.label_encoder = LabelEncoder()
        # Default to empty list
        self.columns_to_remove = columns_to_remove if columns_to_remove else []
        self.X, self.y = None, None  # Placeholders for features and target
        self.X_train, self.X_test = None, None  # Placeholders for train-test split features
        self.y_train, self.y_test = None, None  # Placeholders for train-test split target

        logging.info(
            f"Initialized PreprocessingEngine with task type: {self.task_type}")

    @classmethod
    def load_from_files(cls, clean_df,  meta: dict, feature_encoder, scaler, label_encoder=None):
        """ Initializes the PreprocessingEngine from saved metadata and encoders/scalers. """
        engine = cls(df=clean_df, 
                 target_column=meta['target_column'],
                 categorical_columns=meta.get('categorical_columns', []),
                 columns_to_remove=meta.get('columns_to_remove', []))

        engine.original_target_column = meta.get('original_target_column', meta['target_column'])
        engine.target_is_categorical = meta.get('target_is_categorical', False)
        engine.original_columns = meta.get('original_columns', [])
        engine.final_columns = meta.get('final_columns', [])
        engine.task_type = meta.get('task_type', 'regression')
        engine.dropped_columns = meta.get('dropped_columns', [])
        
        engine.feature_encoder = feature_encoder
        engine.scaler = scaler
        engine.label_encoder = label_encoder if engine.target_is_categorical else None

        return engine
    

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

    def encode_target_column(self):
        """Encodes a categorical target column and saves the mapping for reuse."""
        if not self.target_is_categorical:
            logging.info("Skipping target encoding (not categorical).")
            return

        self.original_target_column = self.target_column
        self.df[f"{self.target_column}_encoded"] = self.label_encoder.fit_transform(self.df[self.target_column])

        if f"{self.target_column}_encoded" not in self.categorical_columns:
            self.categorical_columns.append(f"{self.target_column}_encoded")

        self.df.drop(columns=[self.target_column], inplace=True)
        self.target_column = f"{self.target_column}_encoded"

        logging.info(f"Encoded target column '{self.target_column}' with classes: {list(self.label_encoder.classes_)}")

    def decode_target(self, values):
        return self.label_encoder.inverse_transform(values)

    

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

    
    
    def fit_categorical_encoder(self):
        """Fit OneHotEncoder on categorical columns excluding target, save the encoder, and transform self.df."""
        
        cols_to_encode = [
            col for col in self.categorical_columns
            if col in self.df.columns and col != self.target_column
        ]

        if not cols_to_encode:
            logging.info("No categorical features found for one-hot encoding.")
            return

        self.feature_encoder.fit(self.df[cols_to_encode])

        encoded_array = self.feature_encoder.transform(self.df[cols_to_encode])
        encoded_df = pd.DataFrame(
            encoded_array,
            columns=self.feature_encoder.get_feature_names_out(cols_to_encode),
            index=self.df.index
        )

        # Drop original categorical columns and add encoded
        self.df.drop(columns=cols_to_encode, inplace=True)
        self.df = pd.concat([self.df, encoded_df], axis=1)

        logging.info(f"One-hot encoded columns: {cols_to_encode}")

    
    def split_features_and_target(self):
        """ Splits data into features (X) and target (y)"""
        X = self.df.drop(columns=self.target_column, errors="ignore")
        y = self.df[self.target_column]

        self.X, self.y = X, y

        logging.info(f"Split dataset. Using '{self.target_column}' as target.")
        return X, y

    def train_test_split_data(self, X, y):
        """ Splits dataset into training and testing sets. """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        logging.info(
            f"Split data into train ({1 - self.test_size:.0%}) and test ({self.test_size:.0%}) sets.")
        return X_train, X_test, y_train, y_test
    
    def split_data(self):
        """ Splits the dataset into training and testing sets, returning the split data. """
        self.split_features_and_target()
        self.train_test_split_data(self.X, self.y)
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def to_meta_dict(self):
        """ Returns a dictionary with metadata about the preprocessing steps. """
        return {
            "target_column": self.target_column,
            "categorical_columns": self.categorical_columns,
            "columns_to_remove": self.columns_to_remove,
            "original_target_column": getattr(self, "original_target_column", self.target_column),
            "target_is_categorical": getattr(self, "target_is_categorical", False),
            "original_columns": getattr(self, "original_columns", []),
            "final_columns": getattr(self, "final_columns", []),
            "task_type": getattr(self, "task_type", "regression"),
            "dropped_columns": getattr(self, "dropped_columns", []),
        }

    def save_preprocessing_artifacts(self, path_prefix=""):
        """Saves encoders, scaler, and metadata for preprocessing. This is a testing artifact only."""

        joblib.dump(self.feature_encoder, f'{path_prefix}feature_encoder.pkl')
        joblib.dump(self.scaler, f'{path_prefix}scaler.pkl')
        joblib.dump(self.label_encoder, f'{path_prefix}label_encoder.pkl')

        meta = self.to_meta_dict()

        with open(f'{path_prefix}preprocessing_meta.json', 'w') as f:
            json.dump(meta, f, indent=4)

        logging.info(f"Metadata saved to {path_prefix}preprocessing_meta.json")

    def run_preprocessing_engine(self):
        """ Runs the full preprocessing pipeline and saves final processed data. """
        logging.info("Running PreprocessingEngine...")

        self.df = self.remove_unwanted_columns(self.df) # remove unwanted columns
        self.df = self.clean_categorical_columns(self.df)
        self.df = self.clean_continuous_columns(self.df)
        self.df, self.dropped_columns = self.drop_missing_columns(self.df) # drop columns with excessive missing values
        self.df = self.drop_missing_rows(self.df)
        self.df = self.handle_missing_data(self.df)
        self.encode_target_column()
        self.scale_continuous_features()
        self.fit_categorical_encoder()

        self.split_features_and_target() # split into features and target
        self.train_test_split_data(self.X, self.y) # split into train and test sets

        # Save final dataset
        self.final_df = self.df.copy()
        self.final_columns = self.final_df.columns.tolist()

        logging.info("Preprocessing completed successfully. Final dataset stored.")

        return self.X_train, self.X_test, self.y_train, self.y_test, self.task_type # pass right into modeling engine


    # Methods for accessing and summarizing the final dataset

    def summary(self):
        return {
            "task_type": self.task_type,
            "features": list(self.final_df.columns.difference([self.target_column])),
            "missing_values": self.df.isnull().sum().sum(),
            "label_mapping": self.label_mapping_df if self.label_mapping_df is not None else "N/A"
        }

    def verify_final_dataset(self, df):
        """Logs remaining missing data counts after cleaning."""

        remaining_missing = df.isnull().sum()
        logging.info(f"Final dataset missing values:\n{remaining_missing}")


    # Methods for transforming new data using the fitted preprocessing steps

    def scale_continuous_features_in_new_df(self, new_df):
        """Applies the previously fitted scaler to continuous features of a new dataframe."""
        
        cont_features = [
            col for col in new_df.columns
            if col not in self.categorical_columns and col != self.target_column
        ]
        
        if not cont_features:
            logging.warning("No continuous features found to scale in new DataFrame.")
            return new_df
        
        if not hasattr(self, 'scaler'):
            raise ValueError("Scaler not fitted. Call scale_continuous_features() on training data first.")
        
        new_df = new_df.copy()
        new_df[cont_features] = self.scaler.transform(new_df[cont_features])
        
        logging.info(f"Scaled continuous features in new DataFrame: {cont_features}")
        return new_df

    def transform_categoricals_in_new_df(self, new_df):
        """Apply the previously fitted OneHotEncoder to a new DataFrame."""
        
        if not hasattr(self, 'feature_encoder'):
            raise ValueError("OneHotEncoder not fitted yet. Call fit_one_hot_encoder first.")

        cols_to_encode = [
            col for col in self.categorical_columns
            if col in new_df.columns and col != self.target_column
        ]

        if not cols_to_encode:
            logging.info("No categorical features found in new data for one-hot encoding.")
            return new_df

        encoded_array = self.feature_encoder.transform(new_df[cols_to_encode])
        encoded_df = pd.DataFrame(
            encoded_array,
            columns=self.feature_encoder.get_feature_names_out(cols_to_encode),
            index=new_df.index
        )

        new_df = new_df.drop(columns=cols_to_encode)
        new_df = pd.concat([new_df, encoded_df], axis=1)

        logging.info(f"One-hot encoded categorical columns in new DataFrame: {cols_to_encode}")

        return new_df

    def encode_target_in_new_df(self, new_df):
        if not self.target_is_categorical:
            logging.info("Skipping target encoding for new DataFrame, target not categorical.")
            return new_df

        if not hasattr(self, "label_encoder"):
            raise ValueError("No saved label encoder found. Call encode_target_column first.")

        target_col = self.original_target_column
        if target_col not in new_df.columns:
            raise ValueError(f"Target column '{target_col}' not found in new DataFrame.")

        try:
            new_df[f"{target_col}_encoded"] = self.label_encoder.transform(new_df[target_col])
        except ValueError:
            new_df[f"{target_col}_encoded"] = new_df[target_col].map(
                {label: i for i, label in enumerate(self.label_encoder.classes_)}
            ).fillna(-1).astype(int)

        new_df.drop(columns=[target_col], inplace=True)

        logging.info(f"Encoded target column in new DataFrame using saved label encoder.")
        return new_df

    def remove_dropped_columns(self, new_df):
        """Removes columns that were dropped during preprocessing from a new DataFrame."""
        
        if not self.dropped_columns:
            logging.info("No columns to remove from new DataFrame.")
            return new_df
        
        new_df = new_df.drop(columns=self.dropped_columns, errors="ignore")
        logging.info(f"Removed dropped columns from new DataFrame: {self.dropped_columns}")
        
        return new_df

    def clean_new_dataset(self, new_data):
        """Cleans a new dataset using the same preprocessing steps as the original."""

        if set(new_data.columns) != set(self.original_columns):
            raise ValueError(
                f"Input columns do not match expected columns.\n"
                f"Expected: {sorted(self.original_columns)}\n"
                f"Received: {sorted(new_data.columns)}"
            )

        new_data = new_data.copy()
        new_data = self.remove_unwanted_columns(new_data)
        new_data = self.remove_dropped_columns(new_data)
        new_data = self.clean_categorical_columns(new_data)
        new_data = self.clean_continuous_columns(new_data)
        new_data = self.encode_target_in_new_df(new_data)
        new_data = self.scale_continuous_features_in_new_df(new_data)
        new_data = self.transform_categoricals_in_new_df(new_data)

        if set(new_data.columns) != set(self.final_columns):
            raise ValueError(
                f"Output columns do not match expected columns.\n"
                f"Expected: {sorted(self.final_columns)}\n"
                f"Calculated: {sorted(new_data.columns)}"
            )

        return new_data

    def transform_single_row(self, new_df):
        # Clean new dataframe
        new_df = self.clean_new_dataset(new_df)
        
        # Drop output column if it exists
        new_df = new_df.drop(self.target_column, axis=1)
        
        return new_df