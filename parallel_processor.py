"""
Jリーグ試合予想システム - 並列処理エンジン
CPU集約的処理の並列化による大幅な高速化
"""

import multiprocessing as mp
import concurrent.futures
import pandas as pd
import numpy as np
import streamlit as st
from typing import List, Callable, Any, Dict, Tuple
import os
import time
from functools import partial


class ParallelProcessor:
    """
    並列処理エンジン
    - CPU集約的な処理を並列化
    - プロセスプール、スレッドプールの適切な選択
    - チャンク分割による効率的な負荷分散
    """
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or min(8, mp.cpu_count())
        
    def parallel_feature_calculation(self, df: pd.DataFrame, chunk_size: int = 100) -> pd.DataFrame:
        """
        特徴量計算の並列処理
        データを分割して並列計算し、結果をマージ
        """
        # 日付列を確実にdatetimeに変換
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        if df.empty or len(df) < chunk_size:
            # 小さなデータセットは直接処理
            from feature_engine import FastFeatureEngine
            engine = FastFeatureEngine()
            return engine.create_features_fast(df)
        
        try:
            # データを時系列順にチャンクに分割
            chunks = self._split_dataframe_chunks(df, chunk_size)
            
            # 各チャンクの累積データを計算
            chunk_data = []
            for i, chunk in enumerate(chunks):
                # 前のチャンクまでの履歴データを含める
                historical_data = df.iloc[:chunk['end_idx']]
                chunk_data.append({
                    'chunk_df': chunk['data'],
                    'historical_df': historical_data,
                    'chunk_id': i
                })
            
            # 並列処理実行
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # 各チャンクを並列処理
                futures = [
                    executor.submit(self._process_feature_chunk, chunk_info)
                    for chunk_info in chunk_data
                ]
                
                # 結果を収集
                results = []
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result(timeout=300)  # 5分タイムアウト
                        if result is not None:
                            results.append(result)
                    except Exception as e:
                        st.warning(f"並列処理チャンクでエラー: {e}")
            
            # 結果をマージ
            if results:
                # チャンクIDでソートして結合
                results.sort(key=lambda x: x['chunk_id'])
                combined_df = pd.concat([r['features'] for r in results], ignore_index=True)
                return combined_df
            else:
                # エラー時はフォールバック
                return self._fallback_processing(df)
                
        except Exception as e:
            st.warning(f"並列処理エラー、順次処理にフォールバック: {e}")
            return self._fallback_processing(df)
    
    def _split_dataframe_chunks(self, df: pd.DataFrame, chunk_size: int) -> List[Dict]:
        """
        DataFrameを時系列順にチャンク分割
        各チャンクは履歴データへのアクセスが可能
        """
        df_sorted = df.sort_values('date').reset_index(drop=True)
        chunks = []
        
        for start_idx in range(0, len(df_sorted), chunk_size):
            end_idx = min(start_idx + chunk_size, len(df_sorted))
            chunk_df = df_sorted.iloc[start_idx:end_idx].copy()
            
            chunks.append({
                'data': chunk_df,
                'start_idx': start_idx,
                'end_idx': end_idx
            })
        
        return chunks
    
    def _process_feature_chunk(self, chunk_info: Dict) -> Dict:
        """
        単一チャンクの特徴量計算（並列処理用）
        """
        try:
            chunk_df = chunk_info['chunk_df']
            historical_df = chunk_info['historical_df']
            chunk_id = chunk_info['chunk_id']
            
            # 独立したプロセスで特徴量エンジンを実行
            from feature_engine import FastFeatureEngine
            engine = FastFeatureEngine()
            
            # チャンクの特徴量を計算
            features_df = engine.create_features_fast(historical_df)
            
            # 対象チャンクの部分のみを抽出
            start_idx = len(historical_df) - len(chunk_df)
            chunk_features = features_df.iloc[start_idx:].copy()
            
            return {
                'features': chunk_features,
                'chunk_id': chunk_id
            }
            
        except Exception as e:
            print(f"チャンク処理エラー: {e}")
            return None
    
    def _fallback_processing(self, df: pd.DataFrame) -> pd.DataFrame:
        """エラー時のフォールバック処理"""
        from feature_engine import FastFeatureEngine
        engine = FastFeatureEngine()
        return engine.create_features_fast(df)
    
    def parallel_model_training(self, df: pd.DataFrame, cv_folds: int = 5) -> Dict:
        """
        モデル訓練の並列処理
        クロスバリデーションを並列実行
        """
        try:
            from ml_models import train_prediction_model
            
            # データを分割
            fold_data = self._create_cv_folds(df, cv_folds)
            
            # 並列でクロスバリデーション実行
            with concurrent.futures.ProcessPoolExecutor(max_workers=min(cv_folds, self.max_workers)) as executor:
                futures = [
                    executor.submit(self._train_fold, fold_info)
                    for fold_info in fold_data
                ]
                
                fold_results = []
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result(timeout=600)  # 10分タイムアウト
                        if result:
                            fold_results.append(result)
                    except Exception as e:
                        st.warning(f"フォールド訓練エラー: {e}")
            
            # 結果を集計
            if fold_results:
                avg_accuracy = np.mean([r['accuracy'] for r in fold_results])
                return {
                    'parallel_cv_accuracy': avg_accuracy,
                    'fold_results': fold_results,
                    'success': True
                }
            else:
                return {'success': False, 'error': '並列処理に失敗しました'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _create_cv_folds(self, df: pd.DataFrame, cv_folds: int) -> List[Dict]:
        """クロスバリデーション用のデータ分割"""
        fold_size = len(df) // cv_folds
        folds = []
        
        for i in range(cv_folds):
            start_idx = i * fold_size
            end_idx = (i + 1) * fold_size if i < cv_folds - 1 else len(df)
            
            test_indices = list(range(start_idx, end_idx))
            train_indices = list(range(0, start_idx)) + list(range(end_idx, len(df)))
            
            folds.append({
                'train_data': df.iloc[train_indices],
                'test_data': df.iloc[test_indices],
                'fold_id': i
            })
        
        return folds
    
    def _train_fold(self, fold_info: Dict) -> Dict:
        """単一フォールドの訓練（並列処理用）"""
        try:
            train_data = fold_info['train_data']
            test_data = fold_info['test_data']
            fold_id = fold_info['fold_id']
            
            # 簡易訓練評価（実際のモデル訓練の代替）
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score
            
            # 特徴量作成
            from feature_engine import FastFeatureEngine
            engine = FastFeatureEngine()
            
            train_features = engine.create_features_fast(train_data)
            test_features = engine.create_features_fast(test_data)
            
            if train_features.empty or test_features.empty:
                return None
            
            # 特徴量選択
            feature_cols = [col for col in train_features.columns 
                          if col.startswith(('home_', 'away_', 'head_to_head', 'elo_'))]
            
            if not feature_cols:
                return None
            
            X_train = train_features[feature_cols].fillna(0)
            y_train = train_features['result']
            X_test = test_features[feature_cols].fillna(0)
            y_test = test_features['result']
            
            # モデル訓練
            model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1)
            model.fit(X_train, y_train)
            
            # 評価
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            return {
                'fold_id': fold_id,
                'accuracy': accuracy,
                'test_samples': len(y_test)
            }
            
        except Exception as e:
            print(f"フォールド訓練エラー: {e}")
            return None
    
    def parallel_team_analysis(self, df: pd.DataFrame, teams: List[str]) -> Dict:
        """
        チーム分析の並列処理
        複数チームの統計を並列計算
        """
        try:
            # チームを並列処理用に分割
            team_chunks = [teams[i:i+3] for i in range(0, len(teams), 3)]
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [
                    executor.submit(self._analyze_team_chunk, df, chunk)
                    for chunk in team_chunks
                ]
                
                results = {}
                for future in concurrent.futures.as_completed(futures):
                    try:
                        chunk_results = future.result(timeout=60)
                        if chunk_results:
                            results.update(chunk_results)
                    except Exception as e:
                        st.warning(f"チーム分析エラー: {e}")
            
            return results
            
        except Exception as e:
            st.error(f"並列チーム分析エラー: {e}")
            return {}
    
    def _analyze_team_chunk(self, df: pd.DataFrame, teams: List[str]) -> Dict:
        """チームチャンクの分析"""
        from team_statistics import calculate_team_statistics
        
        results = {}
        for team in teams:
            try:
                stats = calculate_team_statistics(team)
                if stats:
                    results[team] = stats
            except Exception as e:
                print(f"チーム {team} 分析エラー: {e}")
        
        return results


# グローバル並列プロセッサ
_parallel_processor = ParallelProcessor()


def enable_parallel_processing(max_workers: int = None):
    """並列処理の有効化"""
    global _parallel_processor
    _parallel_processor = ParallelProcessor(max_workers)


def parallel_feature_calculation(df: pd.DataFrame) -> pd.DataFrame:
    """特徴量計算の並列処理（便利関数）"""
    return _parallel_processor.parallel_feature_calculation(df)


def parallel_model_training(df: pd.DataFrame, cv_folds: int = 5) -> Dict:
    """モデル訓練の並列処理（便利関数）"""
    return _parallel_processor.parallel_model_training(df, cv_folds)


def parallel_team_analysis(df: pd.DataFrame, teams: List[str]) -> Dict:
    """チーム分析の並列処理（便利関数）"""
    return _parallel_processor.parallel_team_analysis(df, teams)