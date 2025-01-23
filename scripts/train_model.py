import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import xgboost as xgb
import joblib
from datetime import datetime
import logging
import argparse
import yaml
from typing import Dict, Any, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, config_path: str = 'config/training_config.yaml'):
        self.config = self._load_config(config_path)
        self.models = {
            'random_forest': RandomForestClassifier,
            'gradient_boosting': GradientBoostingClassifier,
            'xgboost': xgb.XGBClassifier
        }
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Could not load config file: {e}. Using default configuration.")
            return {
                'model_type': 'random_forest',
                'test_size': 0.2,
                'random_state': 42,
                'hyperparameters': {
                    'random_forest': {
                        'n_estimators': [100, 200],
                        'max_depth': [10, 20, None],
                        'min_samples_split': [2, 5]
                    }
                }
            }

    def prepare_data(self, data_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        features = self._engineer_features(df)
        if 'label' not in df.columns:
            raise ValueError("Training data must contain 'label' column")
        y = df['label'].values
        return train_test_split(
            features, 
            y,
            test_size=self.config['test_size'],
            random_state=self.config['random_state']
        )

    def _engineer_features(self, df: pd.DataFrame) -> np.ndarray:
        df['amount_log'] = np.log1p(df['amount'])
        df['is_round_number'] = df['amount'].apply(
            lambda x: abs(x - round(x, 0)) < 1e-8
        ).astype(int)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['time_diff'] = df.groupby('ownerAddress')['timestamp'].diff().dt.total_seconds()
        df['transaction_velocity'] = 1 / df['time_diff'].fillna(1e6)
        address_counts = df.groupby('ownerAddress').size()
        df['address_frequency'] = df['ownerAddress'].map(address_counts)
        feature_columns = [
            'amount_log', 'is_round_number', 'hour', 'is_weekend',
            'transaction_velocity', 'address_frequency'
        ]
        return df[feature_columns].fillna(0).values

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
        logger.info(f"Training {self.config['model_type']} model")
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', self.models[self.config['model_type']]())
        ])
        param_grid = {
            f'classifier__{k}': v 
            for k, v in self.config['hyperparameters'][self.config['model_type']].items()
        }
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=5,
            n_jobs=-1,
            verbose=2
        )
        grid_search.fit(X_train, y_train)
        logger.info(f"Best parameters: {grid_search.best_params_}")
        return grid_search.best_estimator_

    def evaluate_model(self, model: Pipeline, X_test: np.ndarray, y_test: np.ndarray) -> None:
        logger.info("Evaluating model performance")
        y_pred = model.predict(X_test)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

    def save_model(self, model: Pipeline, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(output_dir, f'model_{timestamp}.joblib')
        joblib.dump(model, model_path)
        metadata = {
            'timestamp': timestamp,
            'model_type': self.config['model_type'],
            'config': self.config
        }
        metadata_path = os.path.join(output_dir, f'metadata_{timestamp}.yaml')
        with open(metadata_path, 'w') as f:
            yaml.dump(metadata, f)
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Metadata saved to {metadata_path}")

def main():
    parser = argparse.ArgumentParser(description='Train transaction analysis model')
    parser.add_argument('--data', required=True, help='Path to training data CSV')
    parser.add_argument('--config', default='config/training_config.yaml', help='Path to training configuration')
    parser.add_argument('--output', default='models', help='Output directory for model files')
    args = parser.parse_args()

    try:
        trainer = ModelTrainer(args.config)
        X_train, X_test, y_train, y_test = trainer.prepare_data(args.data)
        model = trainer.train_model(X_train, y_train)
        trainer.evaluate_model(model, X_test, y_test)
        trainer.save_model(model, args.output)
        logger.info("Training completed successfully")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
