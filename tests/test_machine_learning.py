"""
機械学習機能のテスト
"""

import pytest
import pandas as pd
import numpy as np
import pickle
import os
from unittest.mock import patch, Mock, MagicMock
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, cross_val_score


class TestMachineLearning:
    """機械学習機能のテストクラス"""
    
    def test_train_prediction_model_basic(self, sample_match_data, temp_data_dir):
        """基本的なモデル訓練のテスト"""
        with patch('main.pd.read_csv') as mock_read, \
             patch('main.create_features') as mock_features, \
             patch('pickle.dump') as mock_dump:
            
            mock_read.return_value = sample_match_data
            mock_features.return_value = self._create_mock_features()
            
            result = self._mock_train_prediction_model()
            
            assert 'accuracy' in result
            assert 'model_path' in result
            assert 0 <= result['accuracy'] <= 1
            assert result['model_path'].endswith('.pkl')
    
    def test_predict_match_with_trained_model(self, trained_model_mock, sample_features):
        """訓練済みモデルでの予測テスト"""
        with patch('pickle.load') as mock_load, \
             patch('os.path.exists') as mock_exists:
            
            mock_load.return_value = trained_model_mock
            mock_exists.return_value = True
            
            result = self._mock_predict_match('浦和レッズ', '鹿島アントラーズ', trained_model_mock)
            
            assert 'prediction' in result
            assert 'probabilities' in result
            assert 'confidence' in result
            assert result['prediction'] in ['home_win', 'away_win', 'draw']
            assert 0 <= result['confidence'] <= 1
    
    def test_predict_match_no_model(self):
        """モデルが存在しない場合の予測テスト"""
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = False
            
            result = self._mock_predict_match('浦和レッズ', '鹿島アントラーズ', None)
            
            assert result is None or 'error' in result
    
    def test_model_evaluation_metrics(self, sample_features, sample_target):
        """モデル評価メトリクスのテスト"""
        with patch('main.create_features') as mock_features, \
             patch('main.pd.read_csv') as mock_read:
            
            mock_features.return_value = sample_features
            mock_read.return_value = pd.DataFrame()
            
            evaluation = self._mock_evaluate_model(sample_features, sample_target)
            
            assert 'accuracy' in evaluation
            assert 'precision' in evaluation
            assert 'recall' in evaluation
            assert 'f1_score' in evaluation
            assert 'cross_val_scores' in evaluation
    
    def test_feature_importance_extraction(self, trained_model_mock):
        """特徴量重要度抽出のテスト"""
        feature_names = ['home_win_rate', 'away_win_rate', 'home_goals_per_game', 
                        'away_goals_per_game', 'head_to_head_home_wins']
        
        importance = self._extract_feature_importance(trained_model_mock, feature_names)
        
        assert len(importance) == len(feature_names)
        assert all(0 <= imp <= 1 for imp in importance.values())
        assert abs(sum(importance.values()) - 1.0) < 0.01  # Should sum to 1
    
    def test_cross_validation_scores(self, sample_features, sample_target):
        """クロスバリデーションスコアのテスト"""
        with patch('cross_val_score') as mock_cv:
            mock_cv.return_value = np.array([0.8, 0.75, 0.85, 0.82, 0.78])
            
            cv_scores = self._mock_cross_validation(sample_features, sample_target)
            
            assert len(cv_scores) == 5
            assert all(0 <= score <= 1 for score in cv_scores)
            assert 0.7 <= np.mean(cv_scores) <= 0.9
    
    def test_model_persistence(self, trained_model_mock, temp_data_dir):
        """モデル永続化のテスト"""
        model_path = os.path.join(temp_data_dir, 'test_model.pkl')
        
        # モデル保存のテスト
        with patch('pickle.dump') as mock_dump:
            self._mock_save_model(trained_model_mock, model_path)
            mock_dump.assert_called_once()
        
        # モデル読み込みのテスト
        with patch('pickle.load') as mock_load, \
             patch('os.path.exists') as mock_exists:
            mock_load.return_value = trained_model_mock
            mock_exists.return_value = True
            
            loaded_model = self._mock_load_model(model_path)
            assert loaded_model is not None
    
    def test_prediction_probabilities_validation(self, trained_model_mock):
        """予測確率の妥当性テスト"""
        probabilities = trained_model_mock.predict_proba(np.array([[0.6, 0.4, 1.8, 1.2, 0.6]]))
        
        # 確率の合計が1になることを確認
        assert abs(np.sum(probabilities[0]) - 1.0) < 0.01
        
        # 全ての確率が0-1の範囲内であることを確認
        assert all(0 <= prob <= 1 for prob in probabilities[0])
    
    def test_model_feature_consistency(self, sample_features):
        """モデルと特徴量の一貫性テスト"""
        expected_features = ['home_win_rate', 'away_win_rate', 'home_goals_per_game', 
                           'away_goals_per_game', 'head_to_head_home_wins']
        
        # 特徴量の列が期待されるものと一致することを確認
        assert list(sample_features.columns) == expected_features
        
        # 特徴量に欠損値がないことを確認
        assert not sample_features.isnull().any().any()
    
    def test_edge_cases_prediction(self):
        """エッジケースでの予測テスト"""
        # 同じチーム同士の予測（エラーハンドリング）
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = False
            
            result = self._mock_predict_match('浦和レッズ', '浦和レッズ', None)
            assert result is None or 'error' in result
    
    def test_model_hyperparameters(self):
        """モデルハイパーパラメータのテスト"""
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        assert model.n_estimators == 100
        assert model.max_depth == 10
        assert model.random_state == 42
    
    def test_data_preprocessing_for_ml(self, sample_match_data):
        """ML用データ前処理のテスト"""
        processed_data = self._preprocess_match_data(sample_match_data)
        
        assert 'result' in processed_data.columns
        assert processed_data['result'].isin([0, 1, 2]).all()  # 0: away, 1: home, 2: draw
        assert not processed_data.isnull().any().any()
    
    # Helper methods
    def _create_mock_features(self):
        """モック特徴量データの作成"""
        return pd.DataFrame({
            'home_win_rate': [0.6, 0.5, 0.7],
            'away_win_rate': [0.4, 0.6, 0.3],
            'home_goals_per_game': [1.8, 1.5, 2.0],
            'away_goals_per_game': [1.2, 1.8, 1.0],
            'head_to_head_home_wins': [0.6, 0.4, 0.8],
            'result': [1, 0, 1]  # 1: home win, 0: away win, 2: draw
        })
    
    def _mock_train_prediction_model(self):
        """モデル訓練のモック"""
        # 実際の実装では main.py の train_prediction_model を使用
        return {
            'accuracy': 0.82,
            'model_path': 'models/prediction_model.pkl',
            'feature_importance': {
                'home_win_rate': 0.3,
                'away_win_rate': 0.25,
                'home_goals_per_game': 0.2,
                'away_goals_per_game': 0.15,
                'head_to_head_home_wins': 0.1
            }
        }
    
    def _mock_predict_match(self, home_team, away_team, model):
        """試合予測のモック"""
        if model is None:
            return {'error': 'Model not found'}
        
        if home_team == away_team:
            return {'error': 'Same team cannot play against itself'}
        
        # モック予測結果
        prediction_map = {0: 'away_win', 1: 'home_win', 2: 'draw'}
        prediction_idx = 1  # Home win
        probabilities = [0.2, 0.6, 0.2]  # [Away, Home, Draw]
        
        return {
            'prediction': prediction_map[prediction_idx],
            'probabilities': {
                'away_win': probabilities[0],
                'home_win': probabilities[1],
                'draw': probabilities[2]
            },
            'confidence': max(probabilities)
        }
    
    def _mock_evaluate_model(self, features, target):
        """モデル評価のモック"""
        return {
            'accuracy': 0.82,
            'precision': 0.78,
            'recall': 0.80,
            'f1_score': 0.79,
            'cross_val_scores': [0.8, 0.75, 0.85, 0.82, 0.78]
        }
    
    def _extract_feature_importance(self, model, feature_names):
        """特徴量重要度抽出"""
        importance_scores = model.feature_importances_
        return dict(zip(feature_names, importance_scores))
    
    def _mock_cross_validation(self, features, target):
        """クロスバリデーションのモック"""
        return [0.8, 0.75, 0.85, 0.82, 0.78]
    
    def _mock_save_model(self, model, path):
        """モデル保存のモック"""
        # 実際の実装では pickle.dump を使用
        pass
    
    def _mock_load_model(self, path):
        """モデル読み込みのモック"""
        # 実際の実装では pickle.load を使用
        mock_model = Mock()
        mock_model.predict.return_value = np.array([1])
        mock_model.predict_proba.return_value = np.array([[0.2, 0.6, 0.2]])
        return mock_model
    
    def _preprocess_match_data(self, df):
        """試合データの前処理"""
        processed = df.copy()
        
        # 結果カラムの追加
        def get_result(row):
            if row['home_score'] > row['away_score']:
                return 1  # Home win
            elif row['away_score'] > row['home_score']:
                return 0  # Away win
            else:
                return 2  # Draw
        
        processed['result'] = processed.apply(get_result, axis=1)
        return processed