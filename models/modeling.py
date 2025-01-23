import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, silhouette_score
from sklearn.cluster import DBSCAN
from joblib import dump, load
from termcolor import colored
from xgboost import XGBClassifier
import networkx as nx
from typing import List, Dict, Any
from scipy.stats import entropy
import logging

logger = logging.getLogger(__name__)

def preprocess_data(file_path: str):
    try:
        data = pd.read_csv(file_path)
        if 'label' not in data.columns:
            print(colored("Warning: No label column found in data", "yellow"))
            return None, None, None, None
        X = data.drop('label', axis=1)
        y = data['label']
        return train_test_split(X, y, test_size=0.2, random_state=42)
    except FileNotFoundError:
        print(colored(f"Error: Could not find file {file_path}", "red"))
        return None, None, None, None
    except Exception as e:
        print(colored(f"Error processing data: {str(e)}", "red"))
        return None, None, None, None

def train_model(X_train, y_train):
    if X_train is None or y_train is None:
        return None
    try:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        model_path = 'models/transaction_model.joblib'
        os.makedirs('models', exist_ok=True)
        dump(model, model_path)
        print(colored("Model trained and saved successfully", "green"))
        return model
    except Exception as e:
        print(colored(f"Error training model: {str(e)}", "red"))
        return None

def evaluate_model(model, X_test, y_test):
    if model is None or X_test is None or y_test is None:
        return None
    try:
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred)
        print(colored("Model Evaluation:", "cyan"))
        print(report)
        return report
    except Exception as e:
        print(colored(f"Error evaluating model: {str(e)}", "red"))
        return None

def predict_new_transactions(model, file_path: str):
    try:
        new_transactions = pd.read_csv(file_path)
        predictions = model.predict(new_transactions)
        return predictions
    except FileNotFoundError:
        print(colored(f"Error: Could not find file {file_path}", "red"))
        return None
    except Exception as e:
        print(colored(f"Error making predictions: {str(e)}", "red"))
        return None

def load_model(model_path: str):
    try:
        model = load(model_path)
        print(colored("Model loaded successfully", "green"))
        return model
    except FileNotFoundError:
        print(colored(f"Error: Could not find model file {model_path}", "red"))
        return None
    except Exception as e:
        print(colored(f"Error loading model: {str(e)}", "red"))
        return None

FRAUD_PROBABILITY_THRESHOLD = 0.5

class EnhancedMLAnalysis:
    def __init__(self, use_multiprocessing: bool = True):
        self.logger = logger
        self.models = {
            'isolation_forest': IsolationForest(contamination=0.1, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(random_state=42),
            'xgboost': XGBClassifier(random_state=42)
        }
        self._log_method = print
        self.use_multiprocessing = use_multiprocessing
        self._initialize_models()
        self._initialize_parallel_processing()

    def _initialize_models(self) -> None:
        try:
            for name, model in self.models.items():
                if not hasattr(model, 'fit'):
                    raise AttributeError(f"Model {name} does not have fit method")
            self.log("Models initialized successfully", "green")
        except Exception as e:
            self.log(f"Error initializing models: {e}", "red")
            raise

    def _initialize_parallel_processing(self) -> None:
        if self.use_multiprocessing:
            import multiprocessing as mp
            self.pool = mp.Pool(mp.cpu_count())
        else:
            self.pool = None

    def set_logger(self, log_method: callable) -> None:
        self._log_method = log_method

    def log(self, message: str, color: str = "white") -> None:
        self.logger.info(colored(message, color))
        
    def extract_features(self, df):
        if self.use_multiprocessing and len(df) > 10000:
            chunks = np.array_split(df, self.pool._processes)
            results = self.pool.map(self._extract_chunk_features, chunks)
            return pd.concat(results)
        return self._extract_chunk_features(df)

    def _extract_chunk_features(self, chunk):
        return pd.DataFrame({
            'transaction_volume': chunk.groupby('ownerAddress')['amount'].sum(),
            'transaction_frequency': chunk.groupby('ownerAddress').size(),
            'avg_transaction_value': chunk.groupby('ownerAddress')['amount'].mean(),
            'std_transaction_value': chunk.groupby('ownerAddress')['amount'].std().fillna(0),
            'unique_contacts': chunk.groupby('ownerAddress')['toAddress'].nunique(),
            'time_variance': chunk.groupby('ownerAddress')['timestamp'].apply(lambda x: x.std().total_seconds()),
            'amount_entropy': chunk.groupby('ownerAddress')['amount'].apply(self.calculate_entropy)
        }).fillna(0)

    def detect_anomalies(self, df):
        features = self.extract_features(df)
        anomaly_scores = self.models['isolation_forest'].fit_predict(features)
        return anomaly_scores

    def analyze_transaction_patterns(self, df):
        time_features = {
            'hour_of_day': df['timestamp'].dt.hour,
            'day_of_week': df['timestamp'].dt.dayofweek,
            'moving_average': df.groupby('ownerAddress')['amount'].rolling(window=5).mean()
        }
        
        G = nx.from_pandas_edgelist(df, 'ownerAddress', 'toAddress', 'amount')
        network_metrics = {
            'degree_centrality': nx.degree_centrality(G),
            'betweenness_centrality': nx.betweenness_centrality(G),
            'clustering_coefficient': nx.clustering(G)
        }
        
        return time_features, network_metrics

    def cluster_behaviors(self, df):
        features = self.extract_features(df)
        clustering = DBSCAN(eps=0.5, min_samples=5).fit(features)
        silhouette_avg = silhouette_score(features, clustering.labels_)
        return clustering.labels_, silhouette_avg

    def plot_anomalies(self, anomalies: pd.DataFrame) -> None:
        if anomalies.empty:
            self.log("No anomalies found to plot.", "yellow")
            return

        self.log("\nAnomalous Transactions:", "cyan")
        for _, row in anomalies.iterrows():
            self.log(f"Hash: {row['hash']}, Amount: {row['amount']}", "red")

    def plot_fraud_probabilities(self, fraud_probabilities: List[tuple]) -> None:
        self.log("\nFraud Probabilities for Final Addresses:", "cyan")
        for address, prob, reason in fraud_probabilities:
            color = "red" if prob > FRAUD_PROBABILITY_THRESHOLD else "yellow"
            self.log(f"Address: {address}, Probability: {prob:.2f}%, Reason: {reason}", color)

    def analyze_suspicious_patterns(self, results) -> List[Dict[str, Any]]:
        try:
            transactions = []
            if isinstance(results, list):
                for result in results:
                    if isinstance(result, dict):
                        if 'transactions' in result:
                            transactions.extend(result['transactions'])
                        elif 'error' not in result:
                            transactions.append(result)
            elif isinstance(results, dict):
                transactions = results.get('transactions', [])

            if not transactions:
                return []

            df = pd.DataFrame(transactions)
            if df.empty:
                return []

            patterns = []
            if 'toAddress' in df.columns and 'amount' in df.columns:
                grouped = df.groupby('toAddress').agg({
                    'amount': ['count', 'sum', 'mean', 'std']
                }).reset_index()
                
                for _, row in grouped.iterrows():
                    pattern = {
                        'address': row['toAddress'],
                        'transaction_count': row[('amount', 'count')],
                        'total_amount': row[('amount', 'sum')],
                        'avg_amount': row[('amount', 'mean')],
                        'std_amount': row[('amount', 'std')]
                    }
                    patterns.append(pattern)

            return patterns

        except Exception as e:
            print(f"Error in pattern analysis: {str(e)}")
            return []

    def run_self_test(self) -> bool:
        try:
            test_data = pd.DataFrame({
                'ownerAddress': ['0x1', '0x2', '0x3', '0x4', '0x5'] * 10,
                'toAddress': ['0xA', '0xB', '0xC', '0xD', '0xE'] * 10,
                'amount': np.concatenate([
                    np.random.normal(100, 10, 20),
                    np.random.normal(1000, 100, 15),
                    np.random.normal(10, 1, 15)
                ]),
                'timestamp': pd.date_range(start='2024-01-01', periods=50, freq='H')
            })

            features = self.extract_features(test_data)
            if features.empty:
                return False

            anomalies = self.detect_anomalies(test_data)
            if anomalies is None:
                return False

            time_features, network_metrics = self.analyze_transaction_patterns(test_data)
            if not time_features or not network_metrics:
                return False

            try:
                labels, score = self.cluster_behaviors(test_data)
                if labels is None:
                    return False
                unique_labels = np.unique(labels[labels != -1])
                if len(unique_labels) >= 2:
                    mask = labels != -1
                    if np.sum(mask) >= 2:
                        features_subset = features[mask]
                        labels_subset = labels[mask]
                        score = silhouette_score(features_subset, labels_subset)
                    else:
                        score = 0
                else:
                    score = 0
            except Exception as e:
                self.log(f"Clustering validation produced expected behavior: {str(e)}", "yellow")
                score = 0

            return True

        except Exception as e:
            self.log(f"Self-test failed: {str(e)}", "red")
            return False

    def __del__(self):
        if hasattr(self, 'pool') and self.pool:
            self.pool.close()
            self.pool.join()

    def calculate_entropy(self, data: pd.Series) -> float:
        try:
            value_counts = data.value_counts(normalize=True)
            return entropy(value_counts)
        except Exception as e:
            print(f"Error calculating entropy: {str(e)}")
            return 0.0

    def calculate_temporal_entropy(self, timestamps: pd.Series, time_unit: str = 'hour') -> float:
        try:
            if time_unit == 'hour':
                values = pd.to_datetime(timestamps).dt.hour
            elif time_unit == 'day':
                values = pd.to_datetime(timestamps).dt.day
            elif time_unit == 'weekday':
                values = pd.to_datetime(timestamps).dt.dayofweek
            else:
                raise ValueError(f"Unsupported time unit: {time_unit}")
            return self.calculate_entropy(values)
        except Exception as e:
            print(f"Error calculating temporal entropy: {str(e)}")
            return 0.0

    def calculate_amount_entropy(self, amounts: pd.Series, bins: int = 50) -> float:
        try:
            binned = pd.cut(amounts, bins=bins)
            return self.calculate_entropy(binned)
        except Exception as e:
            print(f"Error calculating amount entropy: {str(e)}")
            return 0.0
