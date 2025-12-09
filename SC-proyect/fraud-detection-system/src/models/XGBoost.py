import sys
import os
import json
import pickle
import warnings
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Optional
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix, fbeta_score
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore', category=UserWarning)

class XGBoostFraudModel:
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.pipeline = None
        self.metrics = {}
        self.optimal_threshold = 0.5
        self.feature_columns = None
        self._initialize_pipeline()
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        default_config = {
            "n_estimators": 850,
            "max_depth": 10,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "gamma": 0.1,
            "scale_pos_weight": 18,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": -1,
            "tree_method": "hist",
            "target_fbeta": 2.0
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                default_config.update(json.load(f))
        
        return default_config
    
    def _initialize_pipeline(self) -> None:
        model = XGBClassifier(
            n_estimators=self.config["n_estimators"],
            max_depth=self.config["max_depth"],
            learning_rate=self.config["learning_rate"],
            subsample=self.config["subsample"],
            colsample_bytree=self.config["colsample_bytree"],
            min_child_weight=self.config["min_child_weight"],
            gamma=self.config["gamma"],
            scale_pos_weight=self.config["scale_pos_weight"],
            reg_alpha=self.config["reg_alpha"],
            reg_lambda=self.config["reg_lambda"],
            random_state=self.config["random_state"],
            n_jobs=self.config["n_jobs"],
            tree_method=self.config["tree_method"],
            eval_metric='aucpr'
        )
        
        self.pipeline = Pipeline([
            ('scaler', RobustScaler()),
            ('model', model)
        ])
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame, y_val: pd.Series) -> Dict:
        
        print(f"\nEntrenando con {len(X_train)} muestras ({y_train.mean()*100:.2f}% fraude)")
        
        X_train_clean = X_train.select_dtypes(include=[np.number])
        X_val_clean = X_val.select_dtypes(include=[np.number])
        
        self.feature_columns = X_train_clean.columns.tolist()
        
        self.pipeline.fit(X_train_clean, y_train)
        
        train_proba = self.pipeline.predict_proba(X_train_clean)[:, 1]
        self._optimize_threshold(y_train, train_proba)
        self.metrics['train'] = self._compute_metrics(y_train, train_proba)
        
        val_proba = self.pipeline.predict_proba(X_val_clean)[:, 1]
        self.metrics['val'] = self._compute_metrics(y_val, val_proba)
        
        return self.metrics
    
    def _compute_metrics(self, y_true: pd.Series, y_pred_proba: np.ndarray) -> Dict:
        y_pred = (y_pred_proba >= self.optimal_threshold).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        return {
            'auc_pr': average_precision_score(y_true, y_pred_proba),
            'auc_roc': roc_auc_score(y_true, y_pred_proba),
            'precision': precision,
            'recall': recall,
            'f1': 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
        }
    
    def _optimize_threshold(self, y_true: pd.Series, y_pred_proba: np.ndarray) -> None:
        thresholds = np.arange(0.1, 0.9, 0.01)
        best_f2 = 0
        best_thresh = 0.5
        
        for thresh in thresholds:
            y_pred = (y_pred_proba >= thresh).astype(int)
            f2 = fbeta_score(y_true, y_pred, beta=2)
            
            if f2 > best_f2:
                best_f2 = f2
                best_thresh = thresh
        
        self.optimal_threshold = best_thresh
        y_pred_final = (y_pred_proba >= best_thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_final).ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        print(f"Threshold: {best_thresh:.4f} (P={precision:.4f}, R={recall:.4f}, F2={best_f2:.4f})")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_clean = X.select_dtypes(include=[np.number])
        
        if self.feature_columns is not None:
            for col in self.feature_columns:
                if col not in X_clean.columns:
                    X_clean[col] = 0
            X_clean = X_clean[self.feature_columns]
        
        proba = self.pipeline.predict_proba(X_clean)[:, 1]
        return (proba >= self.optimal_threshold).astype(int)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_clean = X.select_dtypes(include=[np.number])
        
        if self.feature_columns is not None:
            for col in self.feature_columns:
                if col not in X_clean.columns:
                    X_clean[col] = 0
            X_clean = X_clean[self.feature_columns]
        
        return self.pipeline.predict_proba(X_clean)[:, 1]
    
    def save_model(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'pipeline': self.pipeline,
                'config': self.config,
                'metrics': self.metrics,
                'optimal_threshold': self.optimal_threshold,
                'feature_columns': self.feature_columns
            }, f)
        
        print(f"Modelo guardado: {filepath}")
    
    def load_model(self, filepath: str) -> None:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.pipeline = data['pipeline']
        self.config = data['config']
        self.metrics = data['metrics']
        self.optimal_threshold = data['optimal_threshold']
        self.feature_columns = data.get('feature_columns', None)


def import_dataloader():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(current_dir)
    
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    
    try:
        from data.loader import DataLoader
        return DataLoader
    except ImportError:
        try:
            sys.path.insert(0, os.path.dirname(src_dir))
            from src.data.loader import DataLoader
            return DataLoader
        except ImportError as e:
            print(f"Error al importar DataLoader: {e}")
            return None

def train_fraud_model(config_path=None, save_path='models/xgboost_fraud_model.pkl'):
    DataLoader = import_dataloader()
    if not DataLoader:
        return None
    
    loader = DataLoader()
    
    try:
        train_df = loader.load_data('data/processed/fraudTrain_processed.parquet')
        test_df = loader.load_data('data/processed/fraudTest_processed.parquet')
        print(f"Datos cargados - Train: {train_df.shape}, Test: {test_df.shape}")
    except Exception as e:
        print(f"Error al cargar los datos: {e}")
        return None
    
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    X = full_df.drop(columns=['is_fraud'])
    y = full_df['is_fraud']
    
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.3, stratify=y_temp, random_state=42)
    
    model = XGBoostFraudModel(config_path)
    model.train(X_train, y_train, X_val, y_val)
    
    test_proba = model.predict_proba(X_test)
    model.metrics['test'] = model._compute_metrics(y_test, test_proba)
    
    print(f"\nTest: AUC-PR={model.metrics['test']['auc_pr']:.4f}, "
          f"Recall={model.metrics['test']['recall']:.4f}, "
          f"Precision={model.metrics['test']['precision']:.4f}")
    
    model.save_model(save_path)
    
    return model

def predict_with_model(model_path, data_path, output_dir='predictions'):
    model = XGBoostFraudModel()
    model.load_model(model_path)
    
    DataLoader = import_dataloader()
    if not DataLoader:
        return None
    
    try:
        data = DataLoader().load_data(data_path)
    except Exception as e:
        print(f"Error al cargar datos: {e}")
        return None
    
    predictions = model.predict(data)
    probabilities = model.predict_proba(data)
    
    os.makedirs(output_dir, exist_ok=True)
    output = data.copy()
    output['prediction'] = predictions
    output['probability'] = probabilities
    
    output_file = os.path.join(output_dir, 'predictions.csv')
    output.to_csv(output_file, index=False)
    
    print(f"\nPredicciones: {sum(predictions)} fraudes de {len(predictions)} transacciones")
    print(f"Guardado en: {output_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', choices=['train', 'predict'])
    parser.add_argument('--config', default=None)
    parser.add_argument('--model_path', default='models/xgboost_fraud_model.pkl')
    parser.add_argument('--data_path', default=None)
    parser.add_argument('--output_dir', default='predictions')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_fraud_model(args.config, args.model_path)
    elif args.mode == 'predict' and args.data_path:
        predict_with_model(args.model_path, args.data_path, args.output_dir)
    else:
        print("Error: modo 'predict' requiere --data_path")

if __name__ == "__main__":
    main()