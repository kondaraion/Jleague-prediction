"""
Pytest configuration and fixtures for Jリーグ試合予想システム
"""

import pytest
import pandas as pd
import numpy as np
import os
import tempfile
import shutil
from unittest.mock import Mock, patch
from datetime import datetime, timedelta


@pytest.fixture
def sample_match_data():
    """Sample match data for testing"""
    data = {
        'date': ['2024-01-15', '2024-01-20', '2024-01-25', '2024-02-01', '2024-02-05'],
        'home_team': ['浦和レッズ', '鹿島アントラーズ', '浦和レッズ', 'FC東京', '横浜F・マリノス'],
        'away_team': ['鹿島アントラーズ', 'FC東京', '横浜F・マリノス', '浦和レッズ', '鹿島アントラーズ'],
        'home_score': [2, 1, 0, 3, 2],
        'away_score': [1, 2, 1, 1, 2],
        'stadium': ['埼玉スタジアム', 'カシマスタジアム', '埼玉スタジアム', '味の素スタジアム', '日産スタジアム']
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_team_stats():
    """Sample team statistics for testing"""
    return {
        'win_rate': 0.65,
        'home_win_rate': 0.72,
        'away_win_rate': 0.58,
        'goals_per_game': 1.8,
        'goals_against_per_game': 1.2,
        'recent_form': 0.6,
        'elo_rating': 1650
    }


@pytest.fixture
def temp_data_dir():
    """Temporary data directory for testing"""
    temp_dir = tempfile.mkdtemp()
    data_dir = os.path.join(temp_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    yield data_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_match_csv(temp_data_dir, sample_match_data):
    """Mock matches.csv file"""
    csv_path = os.path.join(temp_data_dir, 'matches.csv')
    sample_match_data.to_csv(csv_path, index=False, encoding='utf-8')
    return csv_path


@pytest.fixture
def mock_requests_get():
    """Mock requests.get for web scraping tests"""
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = """
        <html>
        <body>
        <div class="match-data">
            <span class="home-team">浦和レッズ</span>
            <span class="away-team">鹿島アントラーズ</span>
            <span class="score">2-1</span>
        </div>
        </body>
        </html>
        """
        mock_get.return_value = mock_response
        yield mock_get


@pytest.fixture
def mock_streamlit():
    """Mock Streamlit components for testing"""
    with patch('streamlit.cache_data') as mock_cache:
        mock_cache.side_effect = lambda func: func  # Pass through without caching
        yield mock_cache


@pytest.fixture
def trained_model_mock():
    """Mock trained model for testing"""
    mock_model = Mock()
    mock_model.predict.return_value = np.array([1])  # Home win prediction
    mock_model.predict_proba.return_value = np.array([[0.2, 0.6, 0.2]])  # [Away, Home, Draw]
    mock_model.feature_importances_ = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
    return mock_model


@pytest.fixture
def sample_features():
    """Sample features for ML testing"""
    return pd.DataFrame({
        'home_win_rate': [0.65],
        'away_win_rate': [0.45],
        'home_goals_per_game': [1.8],
        'away_goals_per_game': [1.2],
        'head_to_head_home_wins': [0.6]
    })


@pytest.fixture
def sample_target():
    """Sample target variable for ML testing"""
    return pd.Series([1])  # Home win


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment before each test"""
    # Mock os.path.exists for common file checks
    with patch('os.path.exists') as mock_exists:
        mock_exists.return_value = True
        yield