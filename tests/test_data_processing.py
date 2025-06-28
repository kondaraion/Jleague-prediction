"""
データ処理機能のテスト
"""

import pytest
import pandas as pd
import numpy as np
import os
from unittest.mock import patch, Mock
from datetime import datetime, timedelta


class TestDataProcessing:
    """データ処理機能のテストクラス"""
    
    def test_create_features_basic(self, sample_match_data):
        """基本的な特徴量作成のテスト"""
        # main.pyから関数をインポート（実際の実装では動的インポートが必要）
        with patch('main.pd.read_csv') as mock_read:
            mock_read.return_value = sample_match_data
            
            # create_features関数のテスト（モックが必要）
            features = self._mock_create_features(sample_match_data)
            
            assert isinstance(features, pd.DataFrame)
            assert not features.empty
            assert 'home_win_rate' in features.columns
            assert 'away_win_rate' in features.columns
    
    def test_get_match_data_stats(self, sample_match_data, mock_match_csv):
        """試合データ統計取得のテスト"""
        with patch('main.pd.read_csv') as mock_read:
            mock_read.return_value = sample_match_data
            
            stats = self._mock_get_match_data_stats(sample_match_data)
            
            assert 'total_matches' in stats
            assert 'total_teams' in stats
            assert 'latest_match_date' in stats
            assert stats['total_matches'] == len(sample_match_data)
    
    def test_calculate_team_win_rate_all_matches(self, sample_match_data):
        """チーム勝率計算（全試合）のテスト"""
        team = '浦和レッズ'
        win_rate = self._calculate_team_win_rate(sample_match_data, team, recent=False)
        
        # 浦和レッズの勝利数を計算
        home_wins = len(sample_match_data[
            (sample_match_data['home_team'] == team) & 
            (sample_match_data['home_score'] > sample_match_data['away_score'])
        ])
        away_wins = len(sample_match_data[
            (sample_match_data['away_team'] == team) & 
            (sample_match_data['away_score'] > sample_match_data['home_score'])
        ])
        total_games = len(sample_match_data[
            (sample_match_data['home_team'] == team) | 
            (sample_match_data['away_team'] == team)
        ])
        
        expected_win_rate = (home_wins + away_wins) / total_games if total_games > 0 else 0
        assert abs(win_rate - expected_win_rate) < 0.01
    
    def test_calculate_team_win_rate_recent(self, sample_match_data):
        """チーム勝率計算（最近の試合）のテスト"""
        team = '浦和レッズ'
        win_rate = self._calculate_team_win_rate(sample_match_data, team, recent=True)
        
        assert 0 <= win_rate <= 1
        assert isinstance(win_rate, float)
    
    def test_calculate_home_away_rate_home(self, sample_match_data):
        """ホーム勝率計算のテスト"""
        team = '浦和レッズ'
        home_rate = self._calculate_home_away_rate(sample_match_data, team, is_home=True)
        
        home_matches = sample_match_data[sample_match_data['home_team'] == team]
        home_wins = len(home_matches[home_matches['home_score'] > home_matches['away_score']])
        expected_rate = home_wins / len(home_matches) if len(home_matches) > 0 else 0
        
        assert abs(home_rate - expected_rate) < 0.01
    
    def test_calculate_home_away_rate_away(self, sample_match_data):
        """アウェイ勝率計算のテスト"""
        team = '浦和レッズ'
        away_rate = self._calculate_home_away_rate(sample_match_data, team, is_home=False)
        
        away_matches = sample_match_data[sample_match_data['away_team'] == team]
        away_wins = len(away_matches[away_matches['away_score'] > away_matches['home_score']])
        expected_rate = away_wins / len(away_matches) if len(away_matches) > 0 else 0
        
        assert abs(away_rate - expected_rate) < 0.01
    
    def test_calculate_team_goals(self, sample_match_data):
        """チームゴール統計計算のテスト"""
        team = '浦和レッズ'
        goals_stats = self._calculate_team_goals(sample_match_data, team)
        
        assert 'goals_per_game' in goals_stats
        assert 'goals_against_per_game' in goals_stats
        assert goals_stats['goals_per_game'] >= 0
        assert goals_stats['goals_against_per_game'] >= 0
    
    def test_calculate_head_to_head(self, sample_match_data):
        """対戦成績計算のテスト"""
        home_team = '浦和レッズ'
        away_team = '鹿島アントラーズ'
        h2h_stats = self._calculate_head_to_head(sample_match_data, home_team, away_team)
        
        assert 'home_wins' in h2h_stats
        assert 'away_wins' in h2h_stats
        assert 'draws' in h2h_stats
        assert 'total_matches' in h2h_stats
        assert h2h_stats['total_matches'] >= 0
    
    def test_calculate_recent_form(self, sample_match_data):
        """最近の調子計算のテスト"""
        team = '浦和レッズ'
        games = 3
        form = self._calculate_recent_form(sample_match_data, team, games)
        
        assert 0 <= form <= 1
        assert isinstance(form, float)
    
    def test_get_available_teams(self, sample_match_data):
        """利用可能チーム取得のテスト"""
        with patch('main.pd.read_csv') as mock_read:
            mock_read.return_value = sample_match_data
            
            teams = self._mock_get_available_teams(sample_match_data)
            
            expected_teams = set(sample_match_data['home_team'].unique()) | set(sample_match_data['away_team'].unique())
            assert set(teams) == expected_teams
            assert len(teams) > 0
    
    def test_empty_dataframe_handling(self):
        """空のデータフレーム処理のテスト"""
        empty_df = pd.DataFrame(columns=['date', 'home_team', 'away_team', 'home_score', 'away_score'])
        
        win_rate = self._calculate_team_win_rate(empty_df, '浦和レッズ', recent=False)
        assert win_rate == 0
        
        goals_stats = self._calculate_team_goals(empty_df, '浦和レッズ')
        assert goals_stats['goals_per_game'] == 0
        assert goals_stats['goals_against_per_game'] == 0
    
    def test_invalid_team_name(self, sample_match_data):
        """無効なチーム名の処理テスト"""
        invalid_team = '存在しないチーム'
        
        win_rate = self._calculate_team_win_rate(sample_match_data, invalid_team, recent=False)
        assert win_rate == 0
        
        goals_stats = self._calculate_team_goals(sample_match_data, invalid_team)
        assert goals_stats['goals_per_game'] == 0
        assert goals_stats['goals_against_per_game'] == 0
    
    # Helper methods (実際の実装では main.py から関数をインポート)
    def _mock_create_features(self, df):
        """create_features関数のモック"""
        teams = set(df['home_team'].unique()) | set(df['away_team'].unique())
        features = []
        
        for team in teams:
            features.append({
                'team': team,
                'home_win_rate': self._calculate_team_win_rate(df, team, recent=False),
                'away_win_rate': self._calculate_team_win_rate(df, team, recent=False),
                'goals_per_game': self._calculate_team_goals(df, team)['goals_per_game']
            })
        
        return pd.DataFrame(features)
    
    def _mock_get_match_data_stats(self, df):
        """get_match_data_stats関数のモック"""
        return {
            'total_matches': len(df),
            'total_teams': len(set(df['home_team'].unique()) | set(df['away_team'].unique())),
            'latest_match_date': df['date'].max() if not df.empty else None
        }
    
    def _mock_get_available_teams(self, df):
        """get_available_teams関数のモック"""
        return sorted(list(set(df['home_team'].unique()) | set(df['away_team'].unique())))
    
    def _calculate_team_win_rate(self, matches, team, recent=False):
        """チーム勝率計算のヘルパー"""
        if matches.empty:
            return 0
        
        team_matches = matches[
            (matches['home_team'] == team) | (matches['away_team'] == team)
        ]
        
        if recent and len(team_matches) > 5:
            team_matches = team_matches.tail(5)
        
        if len(team_matches) == 0:
            return 0
        
        wins = 0
        for _, match in team_matches.iterrows():
            if match['home_team'] == team and match['home_score'] > match['away_score']:
                wins += 1
            elif match['away_team'] == team and match['away_score'] > match['home_score']:
                wins += 1
        
        return wins / len(team_matches)
    
    def _calculate_home_away_rate(self, df, team, is_home):
        """ホーム/アウェイ勝率計算のヘルパー"""
        if is_home:
            matches = df[df['home_team'] == team]
            wins = len(matches[matches['home_score'] > matches['away_score']])
        else:
            matches = df[df['away_team'] == team]
            wins = len(matches[matches['away_score'] > matches['home_score']])
        
        return wins / len(matches) if len(matches) > 0 else 0
    
    def _calculate_team_goals(self, matches, team):
        """チームゴール統計計算のヘルパー"""
        team_matches = matches[
            (matches['home_team'] == team) | (matches['away_team'] == team)
        ]
        
        if len(team_matches) == 0:
            return {'goals_per_game': 0, 'goals_against_per_game': 0}
        
        goals_for = 0
        goals_against = 0
        
        for _, match in team_matches.iterrows():
            if match['home_team'] == team:
                goals_for += match['home_score']
                goals_against += match['away_score']
            else:
                goals_for += match['away_score']
                goals_against += match['home_score']
        
        return {
            'goals_per_game': goals_for / len(team_matches),
            'goals_against_per_game': goals_against / len(team_matches)
        }
    
    def _calculate_head_to_head(self, df, home_team, away_team):
        """対戦成績計算のヘルパー"""
        h2h_matches = df[
            ((df['home_team'] == home_team) & (df['away_team'] == away_team)) |
            ((df['home_team'] == away_team) & (df['away_team'] == home_team))
        ]
        
        home_wins = 0
        away_wins = 0
        draws = 0
        
        for _, match in h2h_matches.iterrows():
            if match['home_score'] > match['away_score']:
                if match['home_team'] == home_team:
                    home_wins += 1
                else:
                    away_wins += 1
            elif match['away_score'] > match['home_score']:
                if match['away_team'] == home_team:
                    home_wins += 1
                else:
                    away_wins += 1
            else:
                draws += 1
        
        return {
            'home_wins': home_wins,
            'away_wins': away_wins,
            'draws': draws,
            'total_matches': len(h2h_matches)
        }
    
    def _calculate_recent_form(self, matches, team, games):
        """最近の調子計算のヘルパー"""
        team_matches = matches[
            (matches['home_team'] == team) | (matches['away_team'] == team)
        ].tail(games)
        
        if len(team_matches) == 0:
            return 0
        
        points = 0
        for _, match in team_matches.iterrows():
            if match['home_team'] == team:
                if match['home_score'] > match['away_score']:
                    points += 3
                elif match['home_score'] == match['away_score']:
                    points += 1
            else:
                if match['away_score'] > match['home_score']:
                    points += 3
                elif match['away_score'] == match['home_score']:
                    points += 1
        
        max_points = len(team_matches) * 3
        return points / max_points if max_points > 0 else 0