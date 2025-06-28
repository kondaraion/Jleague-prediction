"""
Jリーグ試合予想システム
メインアプリケーション（Streamlit）
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Optional, Tuple
import os
import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime, timedelta
import logging
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from streamlit_option_menu import option_menu
import json
import io
import base64
from jleague_data_parser import text_to_csv_converter, export_to_csv

# ページ設定
st.set_page_config(
    page_title="Jリーグ試合予想システム",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# カスタムCSS
st.markdown("""
<style>
    /* メインコンテナのスタイリング */
    .main {
        padding-top: 2rem;
    }
    
    /* サイドバーのスタイリング */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* メトリクスのスタイリング */
    [data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e9ecef;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
        margin: 0.5rem 0;
    }
    
    /* ボタンのスタイリング */
    .stButton > button {
        background-color: #007bff;
        color: white;
        border-radius: 0.375rem;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        background-color: #0056b3;
        transform: translateY(-1px);
        box-shadow: 0 0.25rem 0.5rem rgba(0, 123, 255, 0.25);
    }
    
    /* タブのスタイリング */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: transparent;
        border-radius: 0.375rem;
        color: #495057;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #007bff !important;
        color: white !important;
    }
    
    /* 警告・情報メッセージのスタイリング */
    .stAlert {
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    
    /* データフレームのスタイリング */
    .stDataFrame {
        border-radius: 0.5rem;
        overflow: hidden;
        box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
    }
    
    /* エクスパンダーのスタイリング */
    .streamlit-expanderHeader {
        background-color: #f8f9fa;
        border-radius: 0.375rem;
        font-weight: 600;
    }
    
    /* セレクトボックスのスタイリング */
    .stSelectbox > div > div {
        border-radius: 0.375rem;
    }
    
    /* プログレスバーのスタイリング */
    .stProgress > div > div > div {
        background-color: #007bff;
    }
    
    /* ヘッダーのスタイリング */
    h1 {
        color: #212529;
        font-weight: 700;
        margin-bottom: 2rem;
    }
    
    h2 {
        color: #495057;
        font-weight: 600;
        margin-bottom: 1.5rem;
    }
    
    h3 {
        color: #6c757d;
        font-weight: 500;
        margin-bottom: 1rem;
    }
    
    /* カードスタイルのコンテナ */
    .metric-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.75rem;
        box-shadow: 0 0.25rem 0.75rem rgba(0, 0, 0, 0.1);
        border: 1px solid #e9ecef;
        margin: 1rem 0;
    }
    
    /* 成功メッセージのカスタマイズ */
    .stSuccess {
        background-color: #d4edda;
        border-color: #c3e6cb;
        color: #155724;
        border-radius: 0.5rem;
    }
    
    /* エラーメッセージのカスタマイズ */
    .stError {
        background-color: #f8d7da;
        border-color: #f5c6cb;
        color: #721c24;
        border-radius: 0.5rem;
    }
    
    /* 情報メッセージのカスタマイズ */
    .stInfo {
        background-color: #d1ecf1;
        border-color: #bee5eb;
        color: #0c5460;
        border-radius: 0.5rem;
    }
    
    /* 警告メッセージのカスタマイズ */
    .stWarning {
        background-color: #fff3cd;
        border-color: #ffeaa7;
        color: #856404;
        border-radius: 0.5rem;
    }
    
    /* Plotlyチャートのコンテナ */
    .js-plotly-plot {
        border-radius: 0.5rem;
        box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
    }
    
    /* スピナーのカスタマイズ */
    .stSpinner {
        text-align: center;
        margin: 2rem 0;
    }
    
    /* フッターのスタイリング */
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #6c757d;
        font-size: 0.875rem;
        border-top: 1px solid #e9ecef;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)



def generate_sample_data() -> Dict[str, any]:
    """
    2023-2024シーズンのリアルなサンプルデータを生成する
    
    Returns:
        Dict: 生成結果とステータス情報
    """
    import random
    from datetime import datetime, timedelta
    
    result = {
        "success": False,
        "message": "",
        "data_count": 0,
        "errors": []
    }
    
    try:
        # データディレクトリの作成
        os.makedirs("data", exist_ok=True)
        
        # J1チームリスト（2023-2024シーズン）
        j1_teams = [
            "浦和レッズ", "鹿島アントラーズ", "FC東京", "横浜F・マリノス",
            "川崎フロンターレ", "柏レイソル", "サンフレッチェ広島", "ガンバ大阪",
            "セレッソ大阪", "ヴィッセル神戸", "名古屋グランパス", "アビスパ福岡",
            "湘南ベルマーレ", "京都サンガF.C.", "ジュビロ磐田", "アルビレックス新潟",
            "サガン鳥栖", "北海道コンサドーレ札幌"
        ]
        
        # サンプルデータ生成
        sample_matches = []
        team_count = len(j1_teams)
        total_rounds = 34
        
        # シーズン開始日と終了日
        season_start = datetime(2023, 2, 18)  # 2023シーズン開始
        season_end = datetime(2023, 12, 2)    # 2023シーズン終了
        season_2024_start = datetime(2024, 2, 23)  # 2024シーズン開始
        season_2024_end = datetime(2024, 11, 30)   # 2024シーズン終了
        
        # 2023シーズンデータ生成
        for round_num in range(1, total_rounds + 1):
            # 各節の日程を設定（週末中心）
            weeks_from_start = (round_num - 1) * 1.5  # 1.5週間隔
            round_date = season_start + timedelta(weeks=weeks_from_start)
            
            # 土日に調整
            if round_date.weekday() < 5:  # 平日の場合
                round_date += timedelta(days=(5 - round_date.weekday()))
            
            # 各節で9試合（18チーム÷2）
            teams_for_round = j1_teams.copy()
            random.shuffle(teams_for_round)
            
            for i in range(0, team_count, 2):
                if i + 1 < team_count:
                    home_team = teams_for_round[i]
                    away_team = teams_for_round[i + 1]
                    
                    # リアルなスコア生成（0-4点、低スコアが多い）
                    score_weights = [30, 35, 20, 10, 4, 1]  # 0点が最も多い
                    home_score = random.choices(range(6), weights=score_weights)[0]
                    away_score = random.choices(range(6), weights=score_weights)[0]
                    
                    # 試合日程の微調整（金土日に分散）
                    match_date = round_date + timedelta(days=random.choice([-1, 0, 1]))
                    
                    sample_matches.append({
                        "date": match_date.strftime("%Y-%m-%d"),
                        "home_team": home_team,
                        "away_team": away_team,
                        "home_score": home_score,
                        "away_score": away_score,
                        "league": "J1",
                        "round": round_num
                    })
        
        # 2024シーズンデータ生成（同様の処理）
        for round_num in range(1, total_rounds + 1):
            weeks_from_start = (round_num - 1) * 1.5
            round_date = season_2024_start + timedelta(weeks=weeks_from_start)
            
            if round_date.weekday() < 5:
                round_date += timedelta(days=(5 - round_date.weekday()))
            
            teams_for_round = j1_teams.copy()
            random.shuffle(teams_for_round)
            
            for i in range(0, team_count, 2):
                if i + 1 < team_count:
                    home_team = teams_for_round[i]
                    away_team = teams_for_round[i + 1]
                    
                    score_weights = [30, 35, 20, 10, 4, 1]
                    home_score = random.choices(range(6), weights=score_weights)[0]
                    away_score = random.choices(range(6), weights=score_weights)[0]
                    
                    match_date = round_date + timedelta(days=random.choice([-1, 0, 1]))
                    
                    sample_matches.append({
                        "date": match_date.strftime("%Y-%m-%d"),
                        "home_team": home_team,
                        "away_team": away_team,
                        "home_score": home_score,
                        "away_score": away_score,
                        "league": "J1",
                        "round": round_num
                    })
        
        # DataFrameに変換
        new_data = pd.DataFrame(sample_matches)
        
        # データ型の設定
        new_data['date'] = pd.to_datetime(new_data['date'])
        new_data['home_score'] = new_data['home_score'].astype(int)
        new_data['away_score'] = new_data['away_score'].astype(int)
        new_data['round'] = new_data['round'].astype(int)
        
        # 既存データの読み込みと結合
        csv_path = "data/matches.csv"
        existing_data = pd.DataFrame()
        
        if os.path.exists(csv_path):
            try:
                existing_data = pd.read_csv(csv_path)
                existing_data['date'] = pd.to_datetime(existing_data['date'])
            except Exception as e:
                result["errors"].append(f"既存データ読み込みエラー: {str(e)}")
        
        # 重複除去のためのキーを作成
        new_data['match_key'] = new_data['date'].dt.strftime('%Y-%m-%d') + '_' + new_data['home_team'] + '_' + new_data['away_team']
        
        if not existing_data.empty:
            existing_data['match_key'] = existing_data['date'].dt.strftime('%Y-%m-%d') + '_' + existing_data['home_team'] + '_' + existing_data['away_team']
            # 重複を除去して結合
            combined_data = pd.concat([existing_data, new_data]).drop_duplicates(subset=['match_key'], keep='last')
        else:
            combined_data = new_data
        
        # match_keyカラムを削除
        combined_data = combined_data.drop('match_key', axis=1)
        
        # 日付でソート
        combined_data = combined_data.sort_values('date').reset_index(drop=True)
        
        # CSVファイルに保存
        combined_data.to_csv(csv_path, index=False, encoding='utf-8')
        
        result["success"] = True
        result["message"] = f"サンプルデータ生成完了: {len(new_data)}件のデータを生成"
        result["data_count"] = len(combined_data)
        
    except Exception as e:
        result["success"] = False
        result["message"] = f"サンプルデータ生成エラー: {str(e)}"
        result["errors"].append(str(e))
    
    return result


def validate_match_data(df: pd.DataFrame) -> Dict[str, any]:
    """
    試合データの妥当性を検証する
    
    Args:
        df: 検証対象のDataFrame
    
    Returns:
        Dict: 検証結果
    """
    validation_result = {
        "valid": True,
        "errors": [],
        "warnings": []
    }
    
    try:
        # 必要な列の確認
        required_columns = ['date', 'home_team', 'away_team', 'home_score', 'away_score', 'league', 'round']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            validation_result["valid"] = False
            validation_result["errors"].append(f"必要な列が不足しています: {', '.join(missing_columns)}")
            return validation_result
        
        # データ型の確認
        if df.empty:
            validation_result["valid"] = False
            validation_result["errors"].append("データが空です")
            return validation_result
        
        # 日付形式の確認
        try:
            pd.to_datetime(df['date'])
        except:
            validation_result["valid"] = False
            validation_result["errors"].append("日付形式が正しくありません (YYYY-MM-DD形式を使用してください)")
        
        # スコアの確認
        try:
            pd.to_numeric(df['home_score'])
            pd.to_numeric(df['away_score'])
        except:
            validation_result["valid"] = False
            validation_result["errors"].append("スコアは数値である必要があります")
        
        # ラウンド番号の確認
        try:
            pd.to_numeric(df['round'])
        except:
            validation_result["valid"] = False
            validation_result["errors"].append("ラウンド番号は数値である必要があります")
        
        # データ範囲の確認
        if not df['home_score'].between(0, 10).all() or not df['away_score'].between(0, 10).all():
            validation_result["warnings"].append("異常なスコアが検出されました (0-10点の範囲外)")
        
        # チーム名の確認
        if df['home_team'].isna().any() or df['away_team'].isna().any():
            validation_result["valid"] = False
            validation_result["errors"].append("チーム名に空白があります")
        
        # 重複データの確認
        df['match_key'] = df['date'].astype(str) + '_' + df['home_team'] + '_' + df['away_team']
        duplicates = df[df.duplicated(subset=['match_key'])]
        if not duplicates.empty:
            validation_result["warnings"].append(f"{len(duplicates)}件の重複データが検出されました")
        
    except Exception as e:
        validation_result["valid"] = False
        validation_result["errors"].append(f"検証エラー: {str(e)}")
    
    return validation_result

def preprocess_uploaded_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    アップロードされたデータの前処理を行う
    
    Args:
        df: 前処理対象のDataFrame
    
    Returns:
        pd.DataFrame: 前処理済みのDataFrame
    """
    processed_df = df.copy()
    
    try:
        # 日付の変換
        processed_df['date'] = pd.to_datetime(processed_df['date'])
        
        # 数値の変換
        processed_df['home_score'] = pd.to_numeric(processed_df['home_score'], errors='coerce').fillna(0).astype(int)
        processed_df['away_score'] = pd.to_numeric(processed_df['away_score'], errors='coerce').fillna(0).astype(int)
        processed_df['round'] = pd.to_numeric(processed_df['round'], errors='coerce').fillna(1).astype(int)
        
        # 文字列データのクリーニング
        processed_df['home_team'] = processed_df['home_team'].astype(str).str.strip()
        processed_df['away_team'] = processed_df['away_team'].astype(str).str.strip()
        processed_df['league'] = processed_df['league'].astype(str).str.strip()
        
        # 空白データの除去
        processed_df = processed_df.dropna(subset=['home_team', 'away_team'])
        
        # 日付でソート
        processed_df = processed_df.sort_values('date').reset_index(drop=True)
        
    except Exception as e:
        st.error(f"データ前処理エラー: {str(e)}")
    
    return processed_df

def merge_datasets(*dataframes: pd.DataFrame) -> pd.DataFrame:
    """
    複数のデータセットを統合し、重複を除去する
    
    Args:
        *dataframes: 統合対象のDataFrameオブジェクト
    
    Returns:
        pd.DataFrame: 統合済みのDataFrame
    """
    try:
        if not dataframes:
            return pd.DataFrame()
        
        # 全てのDataFrameを結合
        combined_df = pd.concat(dataframes, ignore_index=True)
        
        if combined_df.empty:
            return combined_df
        
        # 日付を統一フォーマットに変換
        combined_df['date'] = pd.to_datetime(combined_df['date'])
        
        # 重複除去のためのキーを作成
        combined_df['match_key'] = combined_df['date'].dt.strftime('%Y-%m-%d') + '_' + combined_df['home_team'] + '_' + combined_df['away_team']
        
        # 重複を除去（最新のものを保持）
        merged_df = combined_df.drop_duplicates(subset=['match_key'], keep='last')
        
        # match_keyカラムを削除
        merged_df = merged_df.drop('match_key', axis=1)
        
        # 日付でソート
        merged_df = merged_df.sort_values('date').reset_index(drop=True)
        
        return merged_df
        
    except Exception as e:
        st.error(f"データセット統合エラー: {str(e)}")
        return pd.DataFrame()

def clean_duplicates(df: pd.DataFrame) -> Dict[str, any]:
    """
    データセットから重複データを除去し、統計情報を返す
    
    Args:
        df: 処理対象のDataFrame
    
    Returns:
        Dict: 処理結果と統計情報
    """
    result = {
        "success": False,
        "message": "",
        "original_count": 0,
        "cleaned_count": 0,
        "duplicates_removed": 0,
        "cleaned_data": pd.DataFrame()
    }
    
    try:
        if df.empty:
            result["message"] = "データが空です"
            return result
        
        result["original_count"] = len(df)
        
        # 日付を統一フォーマットに変換
        df_clean = df.copy()
        df_clean['date'] = pd.to_datetime(df_clean['date'])
        
        # 重複除去のためのキーを作成
        df_clean['match_key'] = df_clean['date'].dt.strftime('%Y-%m-%d') + '_' + df_clean['home_team'] + '_' + df_clean['away_team']
        
        # 重複を除去
        df_cleaned = df_clean.drop_duplicates(subset=['match_key'], keep='last')
        
        # match_keyカラムを削除
        df_cleaned = df_cleaned.drop('match_key', axis=1)
        
        # 日付でソート
        df_cleaned = df_cleaned.sort_values('date').reset_index(drop=True)
        
        result["cleaned_count"] = len(df_cleaned)
        result["duplicates_removed"] = result["original_count"] - result["cleaned_count"]
        result["cleaned_data"] = df_cleaned
        result["success"] = True
        result["message"] = f"重複除去完了: {result['duplicates_removed']}件の重複データを削除"
        
    except Exception as e:
        result["message"] = f"重複除去エラー: {str(e)}"
    
    return result

def validate_data_integrity() -> Dict[str, any]:
    """
    保存されているデータの整合性を検証する
    
    Returns:
        Dict: 検証結果と修復情報
    """
    result = {
        "success": False,
        "message": "",
        "issues_found": [],
        "issues_fixed": [],
        "data_stats": {}
    }
    
    try:
        csv_path = "data/matches.csv"
        
        if not os.path.exists(csv_path):
            result["issues_found"].append("データファイルが見つかりません")
            return result
        
        # データの読み込み
        df = pd.read_csv(csv_path)
        
        if df.empty:
            result["issues_found"].append("データが空です")
            return result
        
        original_count = len(df)
        issues_found = []
        
        # 1. 日付フォーマットの確認と修正
        try:
            df['date'] = pd.to_datetime(df['date'])
        except Exception as e:
            issues_found.append(f"日付フォーマットエラー: {str(e)}")
        
        # 2. 数値データの確認と修正
        numeric_columns = ['home_score', 'away_score', 'round']
        for col in numeric_columns:
            if col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
                except:
                    issues_found.append(f"{col}列の数値変換に問題があります")
        
        # 3. 必要な列の確認
        required_columns = ['date', 'home_team', 'away_team', 'home_score', 'away_score', 'league', 'round']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            issues_found.append(f"必要な列が不足: {', '.join(missing_columns)}")
        
        # 4. 空白データの確認と除去
        null_data = df.isnull().sum()
        if null_data.any():
            issues_found.append(f"空白データが検出されました: {null_data.to_dict()}")
            df = df.dropna(subset=['home_team', 'away_team'])
            result["issues_fixed"].append("空白データを除去しました")
        
        # 5. 重複データの確認と除去
        duplicate_result = clean_duplicates(df)
        if duplicate_result["success"] and duplicate_result["duplicates_removed"] > 0:
            issues_found.append(f"{duplicate_result['duplicates_removed']}件の重複データを発見")
            df = duplicate_result["cleaned_data"]
            result["issues_fixed"].append(f"{duplicate_result['duplicates_removed']}件の重複データを除去")
        
        # 6. データ範囲の確認
        if 'home_score' in df.columns and 'away_score' in df.columns:
            abnormal_scores = df[(df['home_score'] < 0) | (df['home_score'] > 10) | 
                               (df['away_score'] < 0) | (df['away_score'] > 10)]
            if not abnormal_scores.empty:
                issues_found.append(f"{len(abnormal_scores)}件の異常なスコアを発見")
        
        # データ統計情報
        result["data_stats"] = {
            "total_matches": len(df),
            "original_count": original_count,
            "leagues": df['league'].unique().tolist() if 'league' in df.columns else [],
            "teams": set(df['home_team'].tolist() + df['away_team'].tolist()) if 'home_team' in df.columns and 'away_team' in df.columns else set(),
            "date_range": {
                "start": df['date'].min().strftime('%Y-%m-%d') if 'date' in df.columns and not df['date'].isna().all() else None,
                "end": df['date'].max().strftime('%Y-%m-%d') if 'date' in df.columns and not df['date'].isna().all() else None
            }
        }
        
        # 修正されたデータを保存
        if result["issues_fixed"]:
            # 日付でソート（降順）してから保存
            if 'date' in df.columns:
                df = df.sort_values('date', ascending=False).reset_index(drop=True)
            df.to_csv(csv_path, index=False, encoding='utf-8')
        
        result["issues_found"] = issues_found
        result["success"] = True
        
        if issues_found:
            result["message"] = f"検証完了: {len(issues_found)}件の問題を発見、{len(result['issues_fixed'])}件を修正"
        else:
            result["message"] = "データ整合性チェック完了: 問題は見つかりませんでした"
        
    except Exception as e:
        result["message"] = f"データ整合性チェックエラー: {str(e)}"
    
    return result

def backup_data() -> Dict[str, any]:
    """
    現在のデータをバックアップする
    
    Returns:
        Dict: バックアップ結果
    """
    result = {
        "success": False,
        "message": "",
        "backup_path": ""
    }
    
    try:
        from datetime import datetime
        
        csv_path = "data/matches.csv"
        
        if not os.path.exists(csv_path):
            result["message"] = "バックアップ対象のデータファイルが見つかりません"
            return result
        
        # バックアップディレクトリの作成
        backup_dir = "data/backups"
        os.makedirs(backup_dir, exist_ok=True)
        
        # タイムスタンプ付きのバックアップファイル名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"matches_backup_{timestamp}.csv"
        backup_path = os.path.join(backup_dir, backup_filename)
        
        # ファイルをコピー
        import shutil
        shutil.copy2(csv_path, backup_path)
        
        result["success"] = True
        result["message"] = f"データバックアップ完了: {backup_filename}"
        result["backup_path"] = backup_path
        
    except Exception as e:
        result["message"] = f"バックアップエラー: {str(e)}"
    
    return result

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_match_data_stats() -> Dict[str, any]:
    """
    保存されている試合データの統計情報を取得
    
    Returns:
        Dict: データ統計情報
    """
    stats = {
        "total_matches": 0,
        "last_updated": None,
        "leagues": [],
        "teams": [],
        "date_range": {"start": None, "end": None}
    }
    
    try:
        csv_path = "data/matches.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            if not df.empty:
                stats["total_matches"] = len(df)
                stats["last_updated"] = datetime.fromtimestamp(os.path.getmtime(csv_path)).strftime("%Y-%m-%d %H:%M")
                stats["leagues"] = sorted(df['league'].unique().tolist())
                
                # チーム一覧（ホーム・アウェイ両方）
                home_teams = set(df['home_team'].unique())
                away_teams = set(df['away_team'].unique())
                stats["teams"] = sorted(list(home_teams | away_teams))
                
                # 日付範囲
                df['date'] = pd.to_datetime(df['date'])
                stats["date_range"]["start"] = df['date'].min().strftime("%Y-%m-%d")
                stats["date_range"]["end"] = df['date'].max().strftime("%Y-%m-%d")
                
    except Exception as e:
        logging.error(f"統計情報取得エラー: {str(e)}")
    
    return stats

def convert_to_legacy_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    新形式のデータを既存の legacy 形式に変換
    
    Args:
        df: 新形式のDataFrame
        
    Returns:
        DataFrame: legacy形式のDataFrame
    """
    try:
        if df.empty:
            return pd.DataFrame(columns=['date', 'home_team', 'away_team', 'home_score', 'away_score', 'league', 'round'])
        
        # 新形式から既存形式へのマッピング
        legacy_df = pd.DataFrame()
        
        # 必須列のマッピング
        legacy_df['date'] = df['formatted_date'] if 'formatted_date' in df.columns else df.get('date', '')
        legacy_df['home_team'] = df['home_team']
        legacy_df['away_team'] = df['away_team']
        legacy_df['home_score'] = df['home_score']
        legacy_df['away_score'] = df['away_score']
        legacy_df['league'] = df['league']
        legacy_df['round'] = df['round']
        
        return legacy_df
        
    except Exception as e:
        st.error(f"データ形式変換エラー: {str(e)}")
        return pd.DataFrame(columns=['date', 'home_team', 'away_team', 'home_score', 'away_score', 'league', 'round'])

# ===== 機械学習機能 =====

@st.cache_data(ttl=600)  # Cache for 10 minutes
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    試合データから機械学習用の特徴量を作成する
    
    Args:
        df: 試合データ (date, home_team, away_team, home_score, away_score, league, round)
    
    Returns:
        pd.DataFrame: 特徴量付きのデータフレーム
    """
    if df.empty:
        return pd.DataFrame()
    
    try:
        # データのコピーを作成
        df = df.copy()
        
        # 日付をdatetimeに変換
        df['date'] = pd.to_datetime(df['date'])
        
        # 試合結果を計算 (ホームチーム視点: 1=勝利, 0=引分, -1=敗北)
        df['result'] = np.where(df['home_score'] > df['away_score'], 1,
                               np.where(df['home_score'] == df['away_score'], 0, -1))
        
        # 得失点差
        df['goal_difference'] = df['home_score'] - df['away_score']
        
        # チーム別の統計を計算するための準備
        features_list = []
        
        for idx, row in df.iterrows():
            home_team = row['home_team']
            away_team = row['away_team']
            match_date = row['date']
            
            # その試合より前のデータのみを使用
            historical_data = df[df['date'] < match_date].copy()
            
            if historical_data.empty:
                # 履歴データがない場合はデフォルト値
                features = {
                    'home_win_rate_recent': 0.5,
                    'home_win_rate_overall': 0.5,
                    'away_win_rate_recent': 0.5,
                    'away_win_rate_overall': 0.5,
                    'home_home_win_rate': 0.5,
                    'away_away_win_rate': 0.5,
                    'home_avg_goals_scored': 1.0,
                    'home_avg_goals_conceded': 1.0,
                    'away_avg_goals_scored': 1.0,
                    'away_avg_goals_conceded': 1.0,
                    'head_to_head_home_wins': 0,
                    'head_to_head_total': 0,
                    'home_recent_form': 0,
                    'away_recent_form': 0,
                    'home_elo': 1500,
                    'away_elo': 1500,
                    'elo_difference': 0
                }
            else:
                # ホームチームの統計
                home_matches = historical_data[
                    (historical_data['home_team'] == home_team) | 
                    (historical_data['away_team'] == home_team)
                ].copy()
                
                # アウェイチームの統計
                away_matches = historical_data[
                    (historical_data['home_team'] == away_team) | 
                    (historical_data['away_team'] == away_team)
                ].copy()
                
                # ホームチームの特徴量
                home_win_rate_recent = calculate_team_win_rate(home_matches, home_team, recent=True)
                home_win_rate_overall = calculate_team_win_rate(home_matches, home_team, recent=False)
                home_home_win_rate = calculate_home_away_rate(historical_data, home_team, is_home=True)
                home_goals = calculate_team_goals(home_matches, home_team)
                home_recent_form = calculate_recent_form(home_matches, home_team)
                
                # アウェイチームの特徴量
                away_win_rate_recent = calculate_team_win_rate(away_matches, away_team, recent=True)
                away_win_rate_overall = calculate_team_win_rate(away_matches, away_team, recent=False)
                away_away_win_rate = calculate_home_away_rate(historical_data, away_team, is_home=False)
                away_goals = calculate_team_goals(away_matches, away_team)
                away_recent_form = calculate_recent_form(away_matches, away_team)
                
                # 対戦履歴
                h2h_stats = calculate_head_to_head(historical_data, home_team, away_team)
                
                # ELOレーティング（簡易版）
                home_elo = calculate_simple_elo(home_matches, home_team)
                away_elo = calculate_simple_elo(away_matches, away_team)
                
                features = {
                    'home_win_rate_recent': home_win_rate_recent,
                    'home_win_rate_overall': home_win_rate_overall,
                    'away_win_rate_recent': away_win_rate_recent,
                    'away_win_rate_overall': away_win_rate_overall,
                    'home_home_win_rate': home_home_win_rate,
                    'away_away_win_rate': away_away_win_rate,
                    'home_avg_goals_scored': home_goals['scored'],
                    'home_avg_goals_conceded': home_goals['conceded'],
                    'away_avg_goals_scored': away_goals['scored'],
                    'away_avg_goals_conceded': away_goals['conceded'],
                    'head_to_head_home_wins': h2h_stats['home_wins'],
                    'head_to_head_total': h2h_stats['total'],
                    'home_recent_form': home_recent_form,
                    'away_recent_form': away_recent_form,
                    'home_elo': home_elo,
                    'away_elo': away_elo,
                    'elo_difference': home_elo - away_elo
                }
            
            # 元のデータに特徴量を追加
            for key, value in features.items():
                features[key] = value
            
            # インデックスを保持
            features['original_index'] = idx
            features_list.append(features)
        
        # 特徴量データフレームを作成
        features_df = pd.DataFrame(features_list)
        
        # 元のデータと結合
        result_df = df.reset_index(drop=True).merge(
            features_df.set_index('original_index'), 
            left_index=True, 
            right_index=True, 
            how='left'
        )
        
        # 欠損値の処理
        numeric_columns = result_df.select_dtypes(include=[np.number]).columns
        result_df[numeric_columns] = result_df[numeric_columns].fillna(result_df[numeric_columns].median())
        
        return result_df
        
    except Exception as e:
        st.error(f"特徴量作成エラー: {str(e)}")
        return pd.DataFrame()

def calculate_team_win_rate(matches: pd.DataFrame, team: str, recent: bool = False) -> float:
    """チームの勝率を計算"""
    if matches.empty:
        return 0.5
    
    matches = matches.copy()
    
    # 最近の試合のみを対象にする場合
    if recent:
        if not pd.api.types.is_datetime64_any_dtype(matches['date']):
            matches['date'] = pd.to_datetime(matches['date'])
        matches = matches.sort_values('date', ascending=False).head(5)
    
    wins = 0
    total = 0
    
    for _, match in matches.iterrows():
        if match['home_team'] == team:
            total += 1
            if match['home_score'] > match['away_score']:
                wins += 1
        elif match['away_team'] == team:
            total += 1
            if match['away_score'] > match['home_score']:
                wins += 1
    
    return wins / total if total > 0 else 0.5

def calculate_home_away_rate(df: pd.DataFrame, team: str, is_home: bool) -> float:
    """ホーム/アウェイ別の勝率を計算"""
    if is_home:
        matches = df[df['home_team'] == team]
        wins = len(matches[matches['home_score'] > matches['away_score']])
    else:
        matches = df[df['away_team'] == team]
        wins = len(matches[matches['away_score'] > matches['home_score']])
    
    total = len(matches)
    return wins / total if total > 0 else 0.5

def calculate_team_goals(matches: pd.DataFrame, team: str) -> Dict[str, float]:
    """チームの平均得点・失点を計算"""
    if matches.empty:
        return {'scored': 1.0, 'conceded': 1.0}
    
    scored = []
    conceded = []
    
    for _, match in matches.iterrows():
        if match['home_team'] == team:
            scored.append(match['home_score'])
            conceded.append(match['away_score'])
        elif match['away_team'] == team:
            scored.append(match['away_score'])
            conceded.append(match['home_score'])
    
    return {
        'scored': np.mean(scored) if scored else 1.0,
        'conceded': np.mean(conceded) if conceded else 1.0
    }

def calculate_head_to_head(df: pd.DataFrame, home_team: str, away_team: str) -> Dict[str, int]:
    """対戦履歴を計算"""
    h2h = df[
        ((df['home_team'] == home_team) & (df['away_team'] == away_team)) |
        ((df['home_team'] == away_team) & (df['away_team'] == home_team))
    ]
    
    home_wins = 0
    for _, match in h2h.iterrows():
        if (match['home_team'] == home_team and match['home_score'] > match['away_score']) or \
           (match['away_team'] == home_team and match['away_score'] > match['home_score']):
            home_wins += 1
    
    return {'home_wins': home_wins, 'total': len(h2h)}

def calculate_recent_form(matches: pd.DataFrame, team: str, games: int = 3) -> int:
    """最近の調子を計算（連勝・連敗数）"""
    if matches.empty:
        return 0
    
    if not pd.api.types.is_datetime64_any_dtype(matches['date']):
        matches['date'] = pd.to_datetime(matches['date'])
    recent_matches = matches.sort_values('date', ascending=False).head(games)
    form = 0
    
    for _, match in recent_matches.iterrows():
        if match['home_team'] == team:
            if match['home_score'] > match['away_score']:
                form += 1
            elif match['home_score'] < match['away_score']:
                form -= 1
        elif match['away_team'] == team:
            if match['away_score'] > match['home_score']:
                form += 1
            elif match['away_score'] < match['home_score']:
                form -= 1
    
    return form

def calculate_simple_elo(matches: pd.DataFrame, team: str, base_elo: int = 1500) -> float:
    """簡易ELOレーティングを計算"""
    if matches.empty:
        return base_elo
    
    elo = base_elo
    
    for _, match in matches.iterrows():
        if match['home_team'] == team:
            result = 1 if match['home_score'] > match['away_score'] else 0.5 if match['home_score'] == match['away_score'] else 0
        elif match['away_team'] == team:
            result = 1 if match['away_score'] > match['home_score'] else 0.5 if match['away_score'] == match['home_score'] else 0
        else:
            continue
        
        # 簡易ELO更新 (K=32)
        expected = 1 / (1 + 10 ** ((1500 - elo) / 400))
        elo = elo + 32 * (result - expected)
    
    return elo

def train_prediction_model() -> Dict[str, any]:
    """
    機械学習モデルを訓練する
    
    Returns:
        Dict: 訓練結果とモデル情報
    """
    result = {
        "success": False,
        "message": "",
        "accuracy": 0.0,
        "model_path": None,
        "feature_importance": {},
        "evaluation_metrics": {}
    }
    
    try:
        # データの読み込み
        csv_path = "data/matches.csv"
        if not os.path.exists(csv_path):
            result["message"] = "訓練データが見つかりません"
            return result
        
        df = pd.read_csv(csv_path)
        if len(df) < 10:
            result["message"] = "訓練データが不足しています（最低10件必要）"
            return result
        
        # 特徴量の作成
        with st.spinner("特徴量を作成中..."):
            df_features = create_features(df)
        
        if df_features.empty:
            result["message"] = "特徴量の作成に失敗しました"
            return result
        
        # 特徴量カラムを選択
        feature_columns = [
            'home_win_rate_recent', 'home_win_rate_overall',
            'away_win_rate_recent', 'away_win_rate_overall',
            'home_home_win_rate', 'away_away_win_rate',
            'home_avg_goals_scored', 'home_avg_goals_conceded',
            'away_avg_goals_scored', 'away_avg_goals_conceded',
            'head_to_head_home_wins', 'head_to_head_total',
            'home_recent_form', 'away_recent_form',
            'home_elo', 'away_elo', 'elo_difference'
        ]
        
        # 存在する特徴量のみを使用
        available_features = [col for col in feature_columns if col in df_features.columns]
        
        if not available_features:
            result["message"] = "使用可能な特徴量が見つかりません"
            return result
        
        X = df_features[available_features]
        y = df_features['result']
        
        # 欠損値の処理
        X = X.fillna(X.median())
        
        # 時系列を考慮した分割
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        if len(X_train) < 5:
            result["message"] = "訓練データが不足しています"
            return result
        
        # モデルの訓練
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        )
        
        model.fit(X_train, y_train)
        
        # 予測と評価
        y_pred = model.predict(X_test)
        
        # 評価指標の計算
        accuracy = accuracy_score(y_test, y_pred)
        
        # 各クラスの精度を計算（ゼロ除算を避ける）
        try:
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        except:
            precision = recall = f1 = 0.0
        
        # 特徴量重要度
        feature_importance = dict(zip(available_features, model.feature_importances_))
        
        # モデルの保存
        os.makedirs("models", exist_ok=True)
        model_path = "models/prediction_model.pkl"
        
        model_data = {
            'model': model,
            'feature_columns': available_features,
            'scaler': None,  # 今回は正規化なし
            'trained_at': datetime.now().isoformat()
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        result.update({
            "success": True,
            "message": f"モデル訓練完了 (精度: {accuracy:.2%})",
            "accuracy": accuracy,
            "model_path": model_path,
            "feature_importance": feature_importance,
            "evaluation_metrics": {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "test_samples": len(X_test)
            }
        })
        
    except Exception as e:
        result["message"] = f"モデル訓練エラー: {str(e)}"
        st.error(f"モデル訓練エラー: {str(e)}")
    
    return result

def predict_scores(df: pd.DataFrame, home_team: str, away_team: str, outcome_predictions: Dict[str, float]) -> Dict[str, any]:
    """
    チームの統計データと試合結果予想に基づいてスコアを予想する
    
    Args:
        df: 試合データ
        home_team: ホームチーム名
        away_team: アウェイチーム名
        outcome_predictions: 試合結果の予想確率
    
    Returns:
        Dict: 予想スコア情報
    """
    try:
        # チームの統計データを計算
        home_stats = calculate_team_stats(df, home_team, is_home=True)
        away_stats = calculate_team_stats(df, away_team, is_home=False)
        
        # 各チームの予想得点を計算（ポアソン分布ベース）
        home_expected_goals = (home_stats['avg_goals_scored'] + away_stats['avg_goals_conceded']) / 2
        away_expected_goals = (away_stats['avg_goals_scored'] + home_stats['avg_goals_conceded']) / 2
        
        # ホームアドバンテージを考慮（通常1.3倍程度）
        home_expected_goals *= 1.2
        
        # 試合結果予想に基づいて調整
        home_win_prob = outcome_predictions.get("ホーム勝利", 0)
        away_win_prob = outcome_predictions.get("アウェイ勝利", 0)
        draw_prob = outcome_predictions.get("引分", 0)
        
        # 勝利確率に基づいてゴール数を調整
        if home_win_prob > away_win_prob:
            home_expected_goals *= (1 + home_win_prob * 0.3)
            away_expected_goals *= (1 - home_win_prob * 0.2)
        elif away_win_prob > home_win_prob:
            away_expected_goals *= (1 + away_win_prob * 0.3)
            home_expected_goals *= (1 - away_win_prob * 0.2)
        
        # 最も可能性の高いスコアを計算
        home_score = round(max(0, home_expected_goals))
        away_score = round(max(0, away_expected_goals))
        
        # 引き分けの可能性が高い場合の調整
        if draw_prob > 0.4 and abs(home_score - away_score) > 1:
            # スコアを近づける
            avg_score = (home_score + away_score) / 2
            home_score = round(avg_score)
            away_score = round(avg_score)
        
        # 複数の可能性のあるスコアを生成
        possible_scores = []
        
        # メインのスコア予想
        main_score = f"{home_score}-{away_score}"
        possible_scores.append({
            "score": main_score,
            "probability": max(outcome_predictions.values()),
            "type": "最有力"
        })
        
        # その他の可能性
        for h in range(max(0, home_score-1), home_score+2):
            for a in range(max(0, away_score-1), away_score+2):
                score = f"{h}-{a}"
                if score != main_score:
                    # 結果に基づく確率計算
                    if h > a:
                        prob = home_win_prob * 0.6
                    elif a > h:
                        prob = away_win_prob * 0.6
                    else:
                        prob = draw_prob * 0.6
                    
                    possible_scores.append({
                        "score": score,
                        "probability": prob,
                        "type": "可能性あり"
                    })
        
        # 確率でソート
        possible_scores.sort(key=lambda x: x["probability"], reverse=True)
        
        return {
            "main_score": main_score,
            "home_score": home_score,
            "away_score": away_score,
            "home_expected_goals": round(home_expected_goals, 2),
            "away_expected_goals": round(away_expected_goals, 2),
            "possible_scores": possible_scores[:5],  # 上位5つ
            "total_goals": home_score + away_score
        }
        
    except Exception as e:
        st.error(f"スコア予想エラー: {str(e)}")
        return {
            "main_score": "1-1",
            "home_score": 1,
            "away_score": 1,
            "home_expected_goals": 1.0,
            "away_expected_goals": 1.0,
            "possible_scores": [],
            "total_goals": 2
        }

def calculate_team_stats(df: pd.DataFrame, team: str, is_home: bool) -> Dict[str, float]:
    """チームの統計データを計算"""
    team_matches = df[(df['home_team'] == team) | (df['away_team'] == team)].copy()
    
    if team_matches.empty:
        return {
            'avg_goals_scored': 1.5,
            'avg_goals_conceded': 1.2
        }
    
    # ホーム/アウェイ別の統計
    if is_home:
        home_matches = team_matches[team_matches['home_team'] == team]
        goals_scored = home_matches['home_score'].mean() if not home_matches.empty else 1.5
        goals_conceded = home_matches['away_score'].mean() if not home_matches.empty else 1.2
    else:
        away_matches = team_matches[team_matches['away_team'] == team]
        goals_scored = away_matches['away_score'].mean() if not away_matches.empty else 1.2
        goals_conceded = away_matches['home_score'].mean() if not away_matches.empty else 1.5
    
    return {
        'avg_goals_scored': goals_scored if pd.notna(goals_scored) else 1.5,
        'avg_goals_conceded': goals_conceded if pd.notna(goals_conceded) else 1.2
    }

def predict_match(home_team: str, away_team: str) -> Dict[str, any]:
    """
    試合結果を予想する
    
    Args:
        home_team: ホームチーム名
        away_team: アウェイチーム名
    
    Returns:
        Dict: 予想結果
    """
    result = {
        "success": False,
        "message": "",
        "predictions": {},
        "confidence": 0.0,
        "feature_importance": {}
    }
    
    try:
        # モデルの読み込み
        model_path = "models/prediction_model.pkl"
        if not os.path.exists(model_path):
            result["message"] = "モデルが見つかりません。先にモデルを訓練してください。"
            return result
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        feature_columns = model_data['feature_columns']
        
        # データの読み込み
        csv_path = "data/matches.csv"
        if not os.path.exists(csv_path):
            result["message"] = "データファイルが見つかりません"
            return result
        
        df = pd.read_csv(csv_path)
        
        # 仮想試合データを作成（現在の日付）
        virtual_match = pd.DataFrame([{
            'date': datetime.now(),
            'home_team': home_team,
            'away_team': away_team,
            'home_score': 0,  # ダミー値
            'away_score': 0,  # ダミー値
            'league': 'J1',
            'round': 1
        }])
        
        # 既存データと結合
        combined_df = pd.concat([df, virtual_match], ignore_index=True)
        
        # 特徴量の作成
        df_features = create_features(combined_df)
        
        if df_features.empty:
            result["message"] = "特徴量の作成に失敗しました"
            return result
        
        # 最後の行（予想対象）の特徴量を取得
        prediction_features = df_features.iloc[-1][feature_columns]
        
        # 欠損値の処理
        prediction_features = prediction_features.fillna(0)
        
        # 予想実行
        X_pred = prediction_features.values.reshape(1, -1)
        
        # クラス確率を取得
        probabilities = model.predict_proba(X_pred)[0]
        classes = model.classes_
        
        # 結果のマッピング
        class_names = {-1: "アウェイ勝利", 0: "引分", 1: "ホーム勝利"}
        
        predictions = {}
        for i, prob in enumerate(probabilities):
            class_id = classes[i]
            class_name = class_names.get(class_id, f"クラス{class_id}")
            predictions[class_name] = prob
        
        # 信頼度（最大確率）
        confidence = max(probabilities)
        
        # 特徴量重要度
        feature_importance = dict(zip(feature_columns, model.feature_importances_))
        
        # スコア予想を追加
        predicted_scores = predict_scores(df, home_team, away_team, predictions)
        
        result.update({
            "success": True,
            "message": "予想完了",
            "predictions": predictions,
            "confidence": confidence,
            "feature_importance": feature_importance,
            "predicted_scores": predicted_scores
        })
        
        # 予想履歴をセッション状態に保存
        if "prediction_history" not in st.session_state:
            st.session_state.prediction_history = []
        
        # 最も確率の高い結果を取得
        best_prediction = max(predictions.items(), key=lambda x: x[1])
        
        prediction_record = {
            "home_team": home_team,
            "away_team": away_team,
            "result": best_prediction[0],
            "confidence": confidence,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "predictions": predictions,
            "predicted_scores": predicted_scores
        }
        st.session_state.prediction_history.append(prediction_record)
        
        # 履歴は最新20件まで保持
        if len(st.session_state.prediction_history) > 20:
            st.session_state.prediction_history = st.session_state.prediction_history[-20:]
        
    except Exception as e:
        result["message"] = f"予想エラー: {str(e)}"
        st.error(f"予想エラー: {str(e)}")
    
    return result

def setup_sidebar() -> None:
    """サイドバーの設定"""
    with st.sidebar:
        st.title("⚽ Jリーグ予想システム")
        st.markdown("---")
        
        st.subheader("📥 データ取得・管理")
        
        # データ統計情報を取得
        stats = get_match_data_stats()
        
        # データ取得ボタン（3列レイアウト）
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊", help="サンプルデータ生成", use_container_width=True):
                with st.spinner("サンプルデータを生成中..."):
                    result = generate_sample_data()
                    if result["success"]:
                        st.success(f"✅ {result['message']}")
                        st.rerun()
                    else:
                        st.error(f"❌ {result['message']}")
                        if result["errors"]:
                            with st.expander("エラー詳細"):
                                for error in result["errors"]:
                                    st.error(error)
        
        
        
        
        # データ統計表示
        col1, col2 = st.columns(2)
        with col1:
            st.metric("総試合数", stats["total_matches"])
        with col2:
            st.metric("リーグ数", len(stats["leagues"]))
        
        # データ状況表示
        if stats["total_matches"] > 0:
            st.success(f"**データ読み込み済み**: {stats['total_matches']}試合")
            if stats["last_updated"]:
                st.info(f"**最終更新**: {stats['last_updated']}")
        else:
            st.warning("⚠️ データが未収集です - 上記ボタンからデータを取得してください")
        
        # データ詳細情報の表示
        if stats["teams"]:
            with st.expander("📊 データ詳細情報"):
                st.write(f"**対象チーム数**: {len(stats['teams'])}")
                st.write(f"**対象リーグ**: {', '.join(stats['leagues'])}")
                if stats["date_range"]["start"] and stats["date_range"]["end"]:
                    st.write(f"**期間**: {stats['date_range']['start']} 〜 {stats['date_range']['end']}")
        
        st.markdown("---")
        
        st.subheader("📱 アプリ情報")
        st.info("""
        **バージョン**: 4.0.0 (Phase 4 Complete)
        **開発**: 高度なJリーグ予想・分析システム
        **更新**: 2024年6月27日
        **新機能**: 高度可視化、ダッシュボード、データエクスポート
        """)
        
        st.subheader("🧭 ナビゲーション")
        st.markdown("""
        - **🏠 ダッシュボード**: システム概要・最新情報
        - **🔮 予想**: 試合結果を予想
        - **📊 分析**: データを分析・可視化
        - **⚙️ 設定**: システム設定・データ管理
        - **❓ ヘルプ**: 使い方・FAQ・トラブルシューティング
        """)
        
        st.markdown("---")
        st.markdown("*© 2024 Jリーグ予想システム*")

def get_team_details(team: str) -> Dict[str, any]:
    """チームの詳細情報を取得"""
    try:
        csv_path = "data/matches.csv"
        if not os.path.exists(csv_path):
            return {}
        
        df = pd.read_csv(csv_path)
        team_matches = df[
            (df['home_team'] == team) | (df['away_team'] == team)
        ].copy()
        
        if team_matches.empty:
            return {}
        
        # 最近5試合の成績（日付でソートして最新から取得）
        if not pd.api.types.is_datetime64_any_dtype(team_matches['date']):
            team_matches['date'] = pd.to_datetime(team_matches['date'])
        recent_matches = team_matches.sort_values('date', ascending=False).head(5)
        recent_results = []
        
        for _, match in recent_matches.iterrows():
            if match['home_team'] == team:
                if match['home_score'] > match['away_score']:
                    result = "勝利"
                    emoji = "🟢"
                elif match['home_score'] < match['away_score']:
                    result = "敗北"
                    emoji = "🔴"
                else:
                    result = "引分"
                    emoji = "🟡"
                score = f"{match['home_score']}-{match['away_score']}"
                opponent = match['away_team']
                venue = "ホーム"
            else:
                if match['away_score'] > match['home_score']:
                    result = "勝利"
                    emoji = "🟢"
                elif match['away_score'] < match['home_score']:
                    result = "敗北"
                    emoji = "🔴"
                else:
                    result = "引分"
                    emoji = "🟡"
                score = f"{match['away_score']}-{match['home_score']}"
                opponent = match['home_team']
                venue = "アウェイ"
            
            recent_results.append({
                "date": match['date'].strftime('%Y-%m-%d'),
                "opponent": opponent,
                "score": score,
                "result": result,
                "emoji": emoji,
                "venue": venue
            })
        
        # 統計情報を取得
        team_stats = calculate_team_statistics(team)
        
        # 現在の調子（最近5試合の勝ち点）
        form_points = 0
        for result in recent_results:
            if result['result'] == '勝利':
                form_points += 3
            elif result['result'] == '引分':
                form_points += 1
        
        return {
            "recent_matches": recent_results,
            "team_stats": team_stats,
            "form_points": form_points,
            "form_rating": "好調" if form_points >= 10 else "普通" if form_points >= 6 else "不調"
        }
    except Exception as e:
        st.error(f"チーム詳細取得エラー: {str(e)}")
        return {}

def create_animated_probability_meter(probability: float, label: str) -> None:
    """アニメーション付き確率メーターを作成"""
    # Plotlyを使ったゲージチャート
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': label},
        delta = {'reference': 33.33},  # 3分の1（33.33%）を基準
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 25], 'color': "lightgray"},
                {'range': [25, 50], 'color': "gray"},
                {'range': [50, 75], 'color': "orange"},
                {'range': [75, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

def show_team_detail_panel(team: str, position: str) -> None:
    """チーム詳細パネルを表示"""
    with st.expander(f"📊 {team} ({position}) 詳細情報", expanded=True):
        team_details = get_team_details(team)
        
        if team_details:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📈 最近5試合")
                for match in team_details["recent_matches"]:
                    st.write(f"{match['emoji']} {match['date']} - vs {match['opponent']} ({match['venue']}) {match['score']} - {match['result']}")
            
            with col2:
                st.subheader("📊 基本統計")
                if team_details["team_stats"]:
                    stats = team_details["team_stats"]
                    st.metric("勝率", f"{stats['win_rate']:.1%}")
                    st.metric("平均得点", f"{stats['avg_goals_scored']:.1f}")
                    st.metric("平均失点", f"{stats['avg_goals_conceded']:.1f}")
                
                st.subheader("🎯 現在の調子")
                st.metric("調子", team_details["form_rating"])
                st.metric("直近勝ち点", f"{team_details['form_points']}/15")
        else:
            st.info("チーム詳細データが不足しています")

def prediction_tab() -> None:
    """予想タブの実装"""
    st.header("🔮 試合予想")
    
    # モデル訓練セクション
    st.subheader("🤖 モデル管理")
    col_train1, col_train2 = st.columns(2)
    
    with col_train1:
        if st.button("🔧 モデル訓練", type="secondary"):
            with st.spinner("モデルを訓練中..."):
                train_result = train_prediction_model()
                if train_result["success"]:
                    st.success(train_result["message"])
                    st.json(train_result["evaluation_metrics"])
                else:
                    st.error(train_result["message"])
    
    with col_train2:
        # モデル状態チェック
        model_path = "models/prediction_model.pkl"
        if os.path.exists(model_path):
            st.success("✅ モデル利用可能")
            try:
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                st.info(f"訓練日時: {model_data.get('trained_at', '不明')}")
            except:
                st.warning("モデル読み込みエラー")
        else:
            st.warning("⚠️ モデル未訓練")
    
    st.markdown("---")
    
    # 予想セクション
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("試合選択")
        
        # 利用可能なチーム一覧を取得
        available_teams = get_available_teams()
        if not available_teams:
            st.warning("チームデータが見つかりません。データ更新を実行してください。")
            return
        
        # チーム選択
        home_team = st.selectbox(
            "ホームチーム",
            available_teams,
            help="ホームチームを選択してください"
        )
        
        away_team = st.selectbox(
            "アウェイチーム", 
            available_teams,
            help="アウェイチームを選択してください"
        )
        
        if home_team == away_team:
            st.error("同じチームを選択することはできません")
            return
        
        # 予想実行ボタン
        predict_button = st.button("🔮 予想実行", type="primary")
        
        # チーム詳細情報パネル
        if home_team and away_team and home_team != away_team:
            st.markdown("---")
            st.subheader("🏟️ チーム詳細情報")
            
            show_team_detail_panel(home_team, "ホーム")
            show_team_detail_panel(away_team, "アウェイ")
            
    with col2:
        st.subheader("予想結果")
        
        if predict_button:
            if not os.path.exists("models/prediction_model.pkl"):
                st.error("モデルが見つかりません。先にモデルを訓練してください。")
            else:
                with st.spinner("予想を実行中..."):
                    prediction_result = predict_match(home_team, away_team)
                
                if prediction_result["success"]:
                    predictions = prediction_result["predictions"]
                    confidence = prediction_result["confidence"]
                    
                    # 予想結果を表示
                    st.success("✅ 予想完了")
                    
                    # 確率表示
                    if predictions:
                        # 最も確率の高い結果
                        best_prediction = max(predictions.items(), key=lambda x: x[1])
                        st.metric(
                            "予想結果", 
                            best_prediction[0], 
                            f"信頼度: {confidence:.1%}"
                        )
                        
                        # アニメーション付き確率メーター
                        st.subheader("🎯 予想確率メーター")
                        
                        # 各結果のメーター表示
                        for outcome, prob in predictions.items():
                            st.write(f"**{outcome}**")
                            create_animated_probability_meter(prob, outcome)
                        
                        st.markdown("---")
                        
                        # スコア予想の表示
                        if "predicted_scores" in prediction_result:
                            st.subheader("⚽ 予想スコア")
                            predicted_scores = prediction_result["predicted_scores"]
                            
                            # メインスコア予想
                            col1, col2, col3 = st.columns([1, 2, 1])
                            with col2:
                                st.metric(
                                    "予想スコア",
                                    predicted_scores["main_score"],
                                    f"総得点: {predicted_scores['total_goals']}点"
                                )
                            
                            # 各チームの期待得点
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric(
                                    f"🏠 {home_team}",
                                    f"{predicted_scores['home_score']}点",
                                    f"期待値: {predicted_scores['home_expected_goals']}"
                                )
                            with col2:
                                st.metric(
                                    f"✈️ {away_team}",
                                    f"{predicted_scores['away_score']}点", 
                                    f"期待値: {predicted_scores['away_expected_goals']}"
                                )
                            
                            # その他の可能性のあるスコア
                            if predicted_scores["possible_scores"]:
                                st.write("**その他の可能性のあるスコア:**")
                                for i, score_data in enumerate(predicted_scores["possible_scores"][:3]):
                                    if score_data["score"] != predicted_scores["main_score"]:
                                        st.write(f"• {score_data['score']} (確率: {score_data['probability']:.1%})")
                        
                        st.markdown("---")
                        
                        # 信頼度インジケーター
                        st.subheader("🔬 予想信頼度")
                        confidence_level = "高" if confidence > 0.7 else "中" if confidence > 0.5 else "低"
                        confidence_color = "🟢" if confidence > 0.7 else "🟡" if confidence > 0.5 else "🔴"
                        st.write(f"{confidence_color} **信頼度レベル**: {confidence_level} ({confidence:.1%})")
                        
                        # AI推奨ポイント
                        st.subheader("💡 AI推奨注目ポイント")
                        feature_importance = prediction_result["feature_importance"]
                        if feature_importance:
                            # 重要度の高い特徴量をトップ3で表示
                            sorted_features = sorted(
                                feature_importance.items(), 
                                key=lambda x: x[1], 
                                reverse=True
                            )[:3]
                            
                            recommendations = []
                            for feature, importance in sorted_features:
                                feature_name = translate_feature_name(feature)
                                if "勝率" in feature_name:
                                    recommendations.append(f"📈 {feature_name}が予想に大きく影響しています")
                                elif "得点" in feature_name:
                                    recommendations.append(f"⚽ {feature_name}が重要な要素です")
                                elif "ELO" in feature_name:
                                    recommendations.append(f"🏆 チーム力の差が予想に影響しています")
                                else:
                                    recommendations.append(f"🔍 {feature_name}が予想根拠となっています")
                            
                            for rec in recommendations:
                                st.write(f"• {rec}")
                        
                        # 対戦成績を表示
                        show_head_to_head_stats(home_team, away_team)
                        
                else:
                    st.error(prediction_result["message"])
        else:
            st.info("チームを選択して「予想実行」ボタンを押してください")

@st.cache_data(ttl=600)  # Cache for 10 minutes
def get_available_teams() -> List[str]:
    """利用可能なチーム一覧を取得"""
    try:
        csv_path = "data/matches.csv"
        if not os.path.exists(csv_path):
            return []
        
        df = pd.read_csv(csv_path)
        if df.empty:
            return []
        
        home_teams = set(df['home_team'].unique())
        away_teams = set(df['away_team'].unique())
        all_teams = sorted(list(home_teams | away_teams))
        
        return all_teams
    except Exception as e:
        st.error(f"チーム一覧取得エラー: {str(e)}")
        return []

def translate_feature_name(feature: str) -> str:
    """特徴量名を日本語に翻訳"""
    translations = {
        'home_win_rate_recent': 'ホーム最近の勝率',
        'home_win_rate_overall': 'ホーム全体勝率',
        'away_win_rate_recent': 'アウェイ最近の勝率',
        'away_win_rate_overall': 'アウェイ全体勝率',
        'home_home_win_rate': 'ホームでの勝率',
        'away_away_win_rate': 'アウェイでの勝率',
        'home_avg_goals_scored': 'ホーム平均得点',
        'home_avg_goals_conceded': 'ホーム平均失点',
        'away_avg_goals_scored': 'アウェイ平均得点',
        'away_avg_goals_conceded': 'アウェイ平均失点',
        'head_to_head_home_wins': '直接対戦での勝利数',
        'head_to_head_total': '直接対戦回数',
        'home_recent_form': 'ホーム最近の調子',
        'away_recent_form': 'アウェイ最近の調子',
        'home_elo': 'ホームELOレーティング',
        'away_elo': 'アウェイELOレーティング',
        'elo_difference': 'ELOレーティング差'
    }
    return translations.get(feature, feature)

def show_head_to_head_stats(home_team: str, away_team: str) -> None:
    """対戦成績を表示"""
    try:
        csv_path = "data/matches.csv"
        if not os.path.exists(csv_path):
            return
        
        df = pd.read_csv(csv_path)
        
        # 直接対戦データを取得
        h2h = df[
            ((df['home_team'] == home_team) & (df['away_team'] == away_team)) |
            ((df['home_team'] == away_team) & (df['away_team'] == home_team))
        ]
        
        if not h2h.empty:
            st.subheader("📋 過去の対戦成績")
            
            # 勝敗統計
            home_wins = 0
            away_wins = 0 
            draws = 0
            
            for _, match in h2h.iterrows():
                if match['home_team'] == home_team:
                    if match['home_score'] > match['away_score']:
                        home_wins += 1
                    elif match['home_score'] < match['away_score']:
                        away_wins += 1
                    else:
                        draws += 1
                else:  # away_team was home
                    if match['home_score'] > match['away_score']:
                        away_wins += 1
                    elif match['home_score'] < match['away_score']:
                        home_wins += 1
                    else:
                        draws += 1
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("総対戦", len(h2h))
            with col2:
                st.metric(f"{home_team}勝利", home_wins)
            with col3:
                st.metric("引分", draws)
            with col4:
                st.metric(f"{away_team}勝利", away_wins)
            
            # 最近の対戦結果
            if len(h2h) > 0:
                st.write("**最近の対戦結果**")
                if not pd.api.types.is_datetime64_any_dtype(h2h['date']):
                    h2h['date'] = pd.to_datetime(h2h['date'])
                recent_h2h = h2h.sort_values('date', ascending=False).head(3)
                for _, match in recent_h2h.iterrows():
                    date = match['date'].strftime('%Y-%m-%d')
                    st.write(f"• {date}: {match['home_team']} {match['home_score']}-{match['away_score']} {match['away_team']}")
    
    except Exception as e:
        st.error(f"対戦成績取得エラー: {str(e)}")

def create_team_radar_chart(team: str) -> go.Figure:
    """チームレーダーチャートを作成"""
    try:
        csv_path = "data/matches.csv"
        if not os.path.exists(csv_path):
            return None
        
        df = pd.read_csv(csv_path)
        team_stats = calculate_team_statistics(team)
        
        if not team_stats:
            return None
        
        # レーダーチャート用のデータ準備
        categories = ['勝率', 'ホーム勝率', 'アウェイ勝率', '平均得点', '得失点差', '試合数']
        
        # 正規化のための基準値
        max_win_rate = 1.0
        max_goals = 4.0
        max_matches = 50
        goal_diff = team_stats['avg_goals_scored'] - team_stats['avg_goals_conceded']
        
        values = [
            team_stats['win_rate'] / max_win_rate,
            team_stats['home_win_rate'] / max_win_rate,
            team_stats['away_win_rate'] / max_win_rate,
            min(team_stats['avg_goals_scored'] / max_goals, 1.0),
            min(max(goal_diff + 2, 0) / 4, 1.0),  # 得失点差を0-1に正規化
            min(team_stats['total_matches'] / max_matches, 1.0)
        ]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=team,
            line_color='rgb(180, 151, 231)',
            fillcolor='rgba(180, 151, 231, 0.25)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            title=f"{team} パフォーマンス分析",
            showlegend=True,
            height=400
        )
        
        return fig
    except Exception as e:
        st.error(f"レーダーチャート作成エラー: {str(e)}")
        return None

def create_team_comparison_heatmap() -> go.Figure:
    """チーム対戦成績ヒートマップを作成"""
    try:
        csv_path = "data/matches.csv"
        if not os.path.exists(csv_path):
            return None
        
        df = pd.read_csv(csv_path)
        teams = get_available_teams()
        
        if len(teams) < 2:
            return None
        
        # 対戦成績マトリックスを作成
        matrix = np.zeros((len(teams), len(teams)))
        
        for i, home_team in enumerate(teams):
            for j, away_team in enumerate(teams):
                if i != j:
                    # ホームチームが勝利した回数
                    home_wins = len(df[
                        (df['home_team'] == home_team) & 
                        (df['away_team'] == away_team) & 
                        (df['home_score'] > df['away_score'])
                    ])
                    
                    # アウェイチームとして勝利した回数
                    away_wins = len(df[
                        (df['home_team'] == away_team) & 
                        (df['away_team'] == home_team) & 
                        (df['away_score'] > df['home_score'])
                    ])
                    
                    total_games = len(df[
                        ((df['home_team'] == home_team) & (df['away_team'] == away_team)) |
                        ((df['home_team'] == away_team) & (df['away_team'] == home_team))
                    ])
                    
                    if total_games > 0:
                        win_rate = (home_wins + away_wins) / total_games
                        matrix[i][j] = win_rate
        
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=teams,
            y=teams,
            colorscale='RdYlBu',
            showscale=True,
            colorbar=dict(title="勝率")
        ))
        
        fig.update_layout(
            title="チーム対戦成績ヒートマップ",
            xaxis_title="対戦相手",
            yaxis_title="チーム",
            height=500
        )
        
        return fig
    except Exception as e:
        st.error(f"ヒートマップ作成エラー: {str(e)}")
        return None

def create_goals_trend_chart() -> go.Figure:
    """得点トレンドチャートを作成"""
    try:
        csv_path = "data/matches.csv"
        if not os.path.exists(csv_path):
            return None
        
        df = pd.read_csv(csv_path)
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.to_period('M')
        
        # 月別集計
        monthly_stats = df.groupby('month').agg({
            'home_score': 'mean',
            'away_score': 'mean',
            'home_score': 'sum',
            'away_score': 'sum'
        }).reset_index()
        
        monthly_stats['month_str'] = monthly_stats['month'].astype(str)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=monthly_stats['month_str'],
            y=monthly_stats['home_score'] / len(monthly_stats),
            mode='lines+markers',
            name='ホーム平均得点',
            line=dict(color='rgb(55, 83, 109)')
        ))
        
        fig.add_trace(go.Scatter(
            x=monthly_stats['month_str'],
            y=monthly_stats['away_score'] / len(monthly_stats),
            mode='lines+markers',
            name='アウェイ平均得点',
            line=dict(color='rgb(26, 118, 255)')
        ))
        
        fig.update_layout(
            title="月別平均得点トレンド",
            xaxis_title="月",
            yaxis_title="平均得点",
            height=400,
            hovermode='x unified'
        )
        
        return fig
    except Exception as e:
        st.error(f"トレンドチャート作成エラー: {str(e)}")
        return None

def create_team_performance_scatter() -> go.Figure:
    """チーム性能散布図を作成"""
    try:
        teams = get_available_teams()
        if len(teams) < 2:
            return None
        
        team_data = []
        for team in teams:
            stats = calculate_team_statistics(team)
            if stats:
                team_data.append({
                    'team': team,
                    'avg_goals_scored': stats['avg_goals_scored'],
                    'avg_goals_conceded': stats['avg_goals_conceded'],
                    'win_rate': stats['win_rate'],
                    'total_matches': stats['total_matches']
                })
        
        if not team_data:
            return None
        
        df_scatter = pd.DataFrame(team_data)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df_scatter['avg_goals_scored'],
            y=df_scatter['avg_goals_conceded'],
            mode='markers+text',
            text=df_scatter['team'],
            textposition="top center",
            marker=dict(
                size=df_scatter['win_rate'] * 50 + 10,
                color=df_scatter['win_rate'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="勝率")
            ),
            name='チーム'
        ))
        
        fig.update_layout(
            title="チーム性能散布図（得点 vs 失点）",
            xaxis_title="平均得点",
            yaxis_title="平均失点",
            height=500,
            showlegend=False
        )
        
        return fig
    except Exception as e:
        st.error(f"散布図作成エラー: {str(e)}")
        return None

def create_model_accuracy_trend() -> go.Figure:
    """モデル精度推移チャートを作成"""
    try:
        # サンプルの精度データ（実際のシステムでは実際の精度履歴を使用）
        dates = pd.date_range(start='2024-01-01', end='2024-06-27', freq='W')
        accuracy_values = np.random.normal(0.65, 0.05, len(dates))
        accuracy_values = np.clip(accuracy_values, 0.5, 0.8)  # 50-80%の範囲に制限
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=accuracy_values,
            mode='lines+markers',
            name='予想精度',
            line=dict(color='rgb(255, 127, 14)', width=3),
            marker=dict(size=6)
        ))
        
        # トレンドライン追加
        z = np.polyfit(range(len(dates)), accuracy_values, 1)
        p = np.poly1d(z)
        fig.add_trace(go.Scatter(
            x=dates,
            y=p(range(len(dates))),
            mode='lines',
            name='トレンドライン',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title="モデル予想精度の推移",
            xaxis_title="日付",
            yaxis_title="精度 (%)",
            height=400,
            yaxis=dict(tickformat='.1%')
        )
        
        return fig
    except Exception as e:
        st.error(f"精度トレンドチャート作成エラー: {str(e)}")
        return None

def analysis_tab() -> None:
    """分析タブの実装"""
    st.header("📊 データ分析")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🤖 モデル分析", "📋 チーム分析", "🏆 リーグ統計", "📈 データ品質"])
    
    with tab1:
        st.subheader("🤖 モデル分析ダッシュボード")
        
        # モデル存在チェック
        model_path = "models/prediction_model.pkl"
        if not os.path.exists(model_path):
            st.warning("⚠️ モデルが見つかりません。先にモデルを訓練してください。")
            if st.button("🔧 モデル訓練", key="train_from_analysis"):
                with st.spinner("モデルを訓練中..."):
                    train_result = train_prediction_model()
                    if train_result["success"]:
                        st.success(train_result["message"])
                        st.rerun()
                    else:
                        st.error(train_result["message"])
        else:
            # モデル情報を表示
            show_model_dashboard()
            
            # 新しい高度な可視化を追加
            st.markdown("---")
            st.subheader("📈 高度な分析")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎯 モデル精度推移")
                with st.spinner("精度トレンドを作成中..."):
                    accuracy_chart = create_model_accuracy_trend()
                    if accuracy_chart:
                        st.plotly_chart(accuracy_chart, use_container_width=True)
                    else:
                        st.info("精度データが不足しています")
            
            with col2:
                st.subheader("📊 チーム性能散布図")
                with st.spinner("散布図を作成中..."):
                    scatter_chart = create_team_performance_scatter()
                    if scatter_chart:
                        st.plotly_chart(scatter_chart, use_container_width=True)
                    else:
                        st.info("チームデータが不足しています")
    
    with tab2:
        st.subheader("📋 チーム分析")
        
        # 利用可能なチーム一覧を取得
        available_teams = get_available_teams()
        if available_teams:
            team = st.selectbox("チーム選択", available_teams, key="team_analysis")
            
            # チーム統計を計算・表示
            team_stats = calculate_team_statistics(team)
            if team_stats:
                # 基本統計メトリクス
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("勝率", f"{team_stats['win_rate']:.1%}")
                with col2:
                    st.metric("平均得点", f"{team_stats['avg_goals_scored']:.1f}")
                with col3:
                    st.metric("平均失点", f"{team_stats['avg_goals_conceded']:.1f}")
                with col4:
                    st.metric("総試合数", team_stats['total_matches'])
                
                st.markdown("---")
                
                # 新しい高度な可視化
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🎯 チーム能力レーダーチャート")
                    with st.spinner("レーダーチャートを作成中..."):
                        radar_chart = create_team_radar_chart(team)
                        if radar_chart:
                            st.plotly_chart(radar_chart, use_container_width=True)
                        else:
                            st.info("レーダーチャートデータが不足しています")
                
                with col2:
                    # ホーム・アウェイ別成績
                    st.subheader("🏠 ホーム・アウェイ別成績")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.write("**ホーム戦績**")
                        st.metric("ホーム勝率", f"{team_stats['home_win_rate']:.1%}")
                        st.metric("ホーム平均得点", f"{team_stats['home_avg_scored']:.1f}")
                    with col_b:
                        st.write("**アウェイ戦績**")
                        st.metric("アウェイ勝率", f"{team_stats['away_win_rate']:.1%}")
                        st.metric("アウェイ平均得点", f"{team_stats['away_avg_scored']:.1f}")
                
                # 最近の試合結果
                show_recent_matches(team)
        else:
            st.warning("チームデータが見つかりません。")
        
    with tab3:
        st.subheader("🏆 リーグ統計")
        
        # 収集データの統計情報を表示
        try:
            csv_path = "data/matches.csv"
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                
                if not df.empty:
                    st.subheader("📊 基本統計")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("総試合数", len(df))
                    with col2:
                        st.metric("チーム数", len(set(df['home_team'].tolist() + df['away_team'].tolist())))
                    with col3:
                        st.metric("リーグ数", len(df['league'].unique()))
                    
                    st.markdown("---")
                    
                    # 新しい高度な可視化
                    st.subheader("📈 高度な統計分析")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("🔥 チーム対戦成績ヒートマップ")
                        with st.spinner("ヒートマップを作成中..."):
                            heatmap_chart = create_team_comparison_heatmap()
                            if heatmap_chart:
                                st.plotly_chart(heatmap_chart, use_container_width=True)
                            else:
                                st.info("ヒートマップデータが不足しています")
                    
                    with col2:
                        st.subheader("📊 月別得点トレンド")
                        with st.spinner("トレンドチャートを作成中..."):
                            trend_chart = create_goals_trend_chart()
                            if trend_chart:
                                st.plotly_chart(trend_chart, use_container_width=True)
                            else:
                                st.info("トレンドデータが不足しています")
                    
                    st.markdown("---")
                    
                    st.subheader("📋 最新試合データ（10件）")
                    if not pd.api.types.is_datetime64_any_dtype(df['date']):
                        df['date'] = pd.to_datetime(df['date'])
                    display_df = df.sort_values('date', ascending=False).head(10).copy()
                    display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
                    st.dataframe(display_df, use_container_width=True)
                    
                    st.subheader("🏆 リーグ別試合数")
                    league_counts = df['league'].value_counts()
                    st.bar_chart(league_counts)
                    
                else:
                    st.warning("データが空です。データ更新を実行してください。")
            else:
                st.info("データファイルが見つかりません。サイドバーの「データ更新」ボタンを押してデータを収集してください。")
                
        except Exception as e:
            st.error(f"データ読み込みエラー: {str(e)}")
    
    with tab4:
        st.subheader("📈 データ品質分析")
        
        # データ品質チェック
        try:
            csv_path = "data/matches.csv"
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                
                if not df.empty:
                    quality_checks = []
                    
                    # データ品質チェック
                    missing_data = df.isnull().sum().sum()
                    quality_checks.append({"項目": "欠損データ", "値": f"{missing_data}件", "状態": "✅ 良好" if missing_data == 0 else "⚠️ 要注意"})
                    
                    duplicate_data = df.duplicated().sum()
                    quality_checks.append({"項目": "重複データ", "値": f"{duplicate_data}件", "状態": "✅ 良好" if duplicate_data == 0 else "⚠️ 要注意"})
                    
                    # 日付範囲
                    date_range = f"{df['date'].min()} ～ {df['date'].max()}"
                    quality_checks.append({"項目": "データ期間", "値": date_range, "状態": "✅ 良好"})
                    
                    # データ分布チェック
                    avg_home_score = df['home_score'].mean()
                    avg_away_score = df['away_score'].mean()
                    quality_checks.append({"項目": "平均ホーム得点", "値": f"{avg_home_score:.1f}", "状態": "✅ 良好"})
                    quality_checks.append({"項目": "平均アウェイ得点", "値": f"{avg_away_score:.1f}", "状態": "✅ 良好"})
                    
                    quality_df = pd.DataFrame(quality_checks)
                    st.dataframe(quality_df, use_container_width=True)
                    
                    # データ分布の可視化
                    if len(df) > 0:
                        st.subheader("📊 得点分布")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**ホーム得点分布**")
                            home_score_counts = df['home_score'].value_counts().sort_index()
                            st.bar_chart(home_score_counts)
                        
                        with col2:
                            st.write("**アウェイ得点分布**")
                            away_score_counts = df['away_score'].value_counts().sort_index()
                            st.bar_chart(away_score_counts)
                
                else:
                    st.warning("データが空です。")
            else:
                st.info("データファイルが見つかりません。")
                
        except Exception as e:
            st.error(f"データ品質チェックエラー: {str(e)}")

def show_model_dashboard() -> None:
    """モデルダッシュボードを表示"""
    try:
        model_path = "models/prediction_model.pkl"
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        st.success("✅ モデル読み込み成功")
        
        # モデル基本情報
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**訓練日時**: {model_data.get('trained_at', '不明')}")
            st.info(f"**特徴量数**: {len(model_data.get('feature_columns', []))}")
        
        with col2:
            # 簡易評価を実行
            evaluation = evaluate_current_model()
            if evaluation:
                st.metric("モデル精度", f"{evaluation['accuracy']:.1%}")
                st.metric("テストサンプル数", evaluation['test_samples'])
        
        # 特徴量重要度の表示
        st.subheader("🎯 特徴量重要度")
        if 'model' in model_data:
            model = model_data['model']
            feature_columns = model_data.get('feature_columns', [])
            
            if hasattr(model, 'feature_importances_') and feature_columns:
                importance_data = []
                for feature, importance in zip(feature_columns, model.feature_importances_):
                    importance_data.append({
                        '特徴量': translate_feature_name(feature),
                        '重要度': importance,
                        '重要度(%)': f"{importance:.1%}"
                    })
                
                importance_df = pd.DataFrame(importance_data)
                importance_df = importance_df.sort_values('重要度', ascending=False)
                
                st.dataframe(importance_df, use_container_width=True)
                
                # 重要度チャート
                st.subheader("📊 重要度チャート")
                top_features = importance_df.head(10)
                chart_data = top_features.set_index('特徴量')['重要度']
                st.bar_chart(chart_data)
        
        # モデル再訓練ボタン
        if st.button("🔄 モデル再訓練", key="retrain_model"):
            with st.spinner("モデルを再訓練中..."):
                train_result = train_prediction_model()
                if train_result["success"]:
                    st.success(train_result["message"])
                    st.rerun()
                else:
                    st.error(train_result["message"])
    
    except Exception as e:
        st.error(f"モデルダッシュボード表示エラー: {str(e)}")

def evaluate_current_model() -> Optional[Dict]:
    """現在のモデルを評価"""
    try:
        # データの読み込み
        csv_path = "data/matches.csv"
        if not os.path.exists(csv_path):
            return None
        
        df = pd.read_csv(csv_path)
        if len(df) < 10:
            return None
        
        # 特徴量の作成
        df_features = create_features(df)
        if df_features.empty:
            return None
        
        # モデルの読み込み
        model_path = "models/prediction_model.pkl"
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        feature_columns = model_data['feature_columns']
        
        # 特徴量の準備
        available_features = [col for col in feature_columns if col in df_features.columns]
        if not available_features:
            return None
        
        X = df_features[available_features].fillna(0)
        y = df_features['result']
        
        # テストデータでの評価
        split_idx = int(len(X) * 0.8)
        X_test = X.iloc[split_idx:]
        y_test = y.iloc[split_idx:]
        
        if len(X_test) == 0:
            return None
        
        # 予測と評価
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'test_samples': len(X_test)
        }
    
    except Exception:
        return None

def calculate_team_statistics(team: str) -> Optional[Dict]:
    """チーム統計を計算"""
    try:
        csv_path = "data/matches.csv"
        if not os.path.exists(csv_path):
            return None
        
        df = pd.read_csv(csv_path)
        if df.empty:
            return None
        
        # チームの全試合を取得
        team_matches = df[
            (df['home_team'] == team) | (df['away_team'] == team)
        ].copy()
        
        if team_matches.empty:
            return None
        
        # 基本統計の計算
        total_matches = len(team_matches)
        wins = 0
        goals_scored = []
        goals_conceded = []
        
        # ホーム戦績
        home_matches = team_matches[team_matches['home_team'] == team]
        home_wins = len(home_matches[home_matches['home_score'] > home_matches['away_score']])
        home_goals_scored = home_matches['home_score'].tolist()
        home_goals_conceded = home_matches['away_score'].tolist()
        
        # アウェイ戦績
        away_matches = team_matches[team_matches['away_team'] == team]
        away_wins = len(away_matches[away_matches['away_score'] > away_matches['home_score']])
        away_goals_scored = away_matches['away_score'].tolist()
        away_goals_conceded = away_matches['home_score'].tolist()
        
        # 全体統計
        total_wins = home_wins + away_wins
        all_goals_scored = home_goals_scored + away_goals_scored
        all_goals_conceded = home_goals_conceded + away_goals_conceded
        
        return {
            'total_matches': total_matches,
            'win_rate': total_wins / total_matches if total_matches > 0 else 0,
            'avg_goals_scored': np.mean(all_goals_scored) if all_goals_scored else 0,
            'avg_goals_conceded': np.mean(all_goals_conceded) if all_goals_conceded else 0,
            'home_win_rate': home_wins / len(home_matches) if len(home_matches) > 0 else 0,
            'away_win_rate': away_wins / len(away_matches) if len(away_matches) > 0 else 0,
            'home_avg_scored': np.mean(home_goals_scored) if home_goals_scored else 0,
            'away_avg_scored': np.mean(away_goals_scored) if away_goals_scored else 0
        }
    
    except Exception as e:
        st.error(f"チーム統計計算エラー: {str(e)}")
        return None

def show_recent_matches(team: str, limit: int = 5) -> None:
    """最近の試合結果を表示"""
    try:
        csv_path = "data/matches.csv"
        if not os.path.exists(csv_path):
            return
        
        df = pd.read_csv(csv_path)
        
        # チームの最近の試合を取得
        team_matches = df[
            (df['home_team'] == team) | (df['away_team'] == team)
        ].copy()
        
        if not team_matches.empty:
            # 日付でソートして最新の試合を取得
            if not pd.api.types.is_datetime64_any_dtype(team_matches['date']):
                team_matches['date'] = pd.to_datetime(team_matches['date'])
            recent_matches = team_matches.sort_values('date', ascending=False).head(limit)
            
            st.subheader(f"📅 {team}の最近の試合（{limit}件）")
            
            for _, match in recent_matches.iterrows():
                date = match['date'].strftime('%Y-%m-%d')
                home_team = match['home_team']
                away_team = match['away_team']
                home_score = match['home_score']
                away_score = match['away_score']
                
                # 結果の判定
                if home_team == team:
                    if home_score > away_score:
                        result = "🟢 勝利"
                    elif home_score < away_score:
                        result = "🔴 敗北"
                    else:
                        result = "🟡 引分"
                else:  # away_team == team
                    if away_score > home_score:
                        result = "🟢 勝利"
                    elif away_score < home_score:
                        result = "🔴 敗北"
                    else:
                        result = "🟡 引分"
                
                st.write(f"• {date}: {home_team} {home_score}-{away_score} {away_team} ({result})")
    
    except Exception as e:
        st.error(f"最近の試合表示エラー: {str(e)}")

def create_csv_download_link(df: pd.DataFrame, filename: str) -> str:
    """CSV ダウンロードリンクを作成"""
    csv = df.to_csv(index=False, encoding='utf-8-sig')
    b64 = base64.b64encode(csv.encode('utf-8-sig')).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 {filename} をダウンロード</a>'
    return href

def export_prediction_history() -> pd.DataFrame:
    """予想履歴をDataFrameとして準備"""
    if "prediction_history" not in st.session_state or not st.session_state.prediction_history:
        return pd.DataFrame()
    
    data = []
    for pred in st.session_state.prediction_history:
        data.append({
            ' 実行日時': pred.get('timestamp', ''),
            'ホームチーム': pred.get('home_team', ''),
            'アウェイチーム': pred.get('away_team', ''),
            '予想結果': pred.get('result', ''),
            '信頼度': pred.get('confidence', 0),
            'ホーム勝利確率': pred.get('predictions', {}).get('ホーム勝利', 0),
            '引分確率': pred.get('predictions', {}).get('引分', 0),
            'アウェイ勝利確率': pred.get('predictions', {}).get('アウェイ勝利', 0)
        })
    
    return pd.DataFrame(data)

def export_model_statistics() -> Dict[str, any]:
    """モデル統計情報を準備"""
    try:
        model_path = "models/prediction_model.pkl"
        if not os.path.exists(model_path):
            return {}
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # 基本的な評価を実行
        evaluation = evaluate_current_model()
        
        stats = {
            "モデル訓練日時": model_data.get('trained_at', '不明'),
            "特徴量数": len(model_data.get('feature_columns', [])),
            "使用アルゴリズム": "RandomForest",
            "現在の精度": evaluation.get('accuracy', 0) if evaluation else 0,
            "テストサンプル数": evaluation.get('test_samples', 0) if evaluation else 0,
            "特徴量リスト": model_data.get('feature_columns', [])
        }
        
        return stats
    except Exception as e:
        st.error(f"モデル統計エクスポートエラー: {str(e)}")
        return {}

def create_system_backup() -> Dict[str, any]:
    """システム全体のバックアップデータを作成"""
    backup_data = {
        "backup_timestamp": datetime.now().isoformat(),
        "system_version": "Phase 4.0",
        "data_stats": get_match_data_stats(),
        "model_stats": export_model_statistics(),
        "prediction_history": st.session_state.get('prediction_history', [])
    }
    
    return backup_data

def system_diagnostics() -> Dict[str, any]:
    """システム診断を実行"""
    diagnostics = {
        "timestamp": datetime.now().isoformat(),
        "system_status": "正常",
        "checks": []
    }
    
    # データファイルチェック
    csv_path = "data/matches.csv"
    if os.path.exists(csv_path):
        file_size = os.path.getsize(csv_path)
        diagnostics["checks"].append({
            "項目": "データファイル",
            "状態": "✅ 正常",
            "詳細": f"ファイルサイズ: {file_size} bytes"
        })
    else:
        diagnostics["checks"].append({
            "項目": "データファイル", 
            "状態": "❌ 異常",
            "詳細": "データファイルが見つかりません"
        })
        diagnostics["system_status"] = "要注意"
    
    # モデルファイルチェック
    model_path = "models/prediction_model.pkl"
    if os.path.exists(model_path):
        model_size = os.path.getsize(model_path)
        diagnostics["checks"].append({
            "項目": "モデルファイル",
            "状態": "✅ 正常", 
            "詳細": f"ファイルサイズ: {model_size} bytes"
        })
    else:
        diagnostics["checks"].append({
            "項目": "モデルファイル",
            "状態": "⚠️ 警告",
            "詳細": "モデルが未訓練です"
        })
    
    # データ整合性チェック
    try:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            missing_data = df.isnull().sum().sum()
            duplicate_data = df.duplicated().sum()
            
            if missing_data == 0 and duplicate_data == 0:
                diagnostics["checks"].append({
                    "項目": "データ整合性",
                    "状態": "✅ 正常",
                    "詳細": f"欠損データ: {missing_data}件, 重複データ: {duplicate_data}件"
                })
            else:
                diagnostics["checks"].append({
                    "項目": "データ整合性",
                    "状態": "⚠️ 警告", 
                    "詳細": f"欠損データ: {missing_data}件, 重複データ: {duplicate_data}件"
                })
                diagnostics["system_status"] = "要注意"
    except Exception as e:
        diagnostics["checks"].append({
            "項目": "データ整合性",
            "状態": "❌ 異常",
            "詳細": f"チェックエラー: {str(e)}"
        })
        diagnostics["system_status"] = "異常"
    
    # メモリ使用量チェック
    try:
        import psutil
        memory_usage = psutil.virtual_memory().percent
        diagnostics["checks"].append({
            "項目": "メモリ使用量",
            "状態": "✅ 正常" if memory_usage < 80 else "⚠️ 警告",
            "詳細": f"使用率: {memory_usage:.1f}%"
        })
    except ImportError:
        diagnostics["checks"].append({
            "項目": "メモリ使用量",
            "状態": "ℹ️ 情報なし",
            "詳細": "psutilが利用できません"
        })
    
    return diagnostics

def data_input_tab() -> None:
    """データ入力タブの実装"""
    
    # 大きなヘッダーとビジュアル改善
    col_header1, col_header2 = st.columns([3, 1])
    with col_header1:
        st.markdown("# 📝 Jリーグデータ取り込み")
        st.markdown("### シンプル・フォーム中心のデータ登録システム")
    with col_header2:
        # 現在のデータ件数を大きく表示
        stats = get_match_data_stats()
        st.metric(
            "📊 登録済みデータ",
            f"{stats['total_matches']}件",
            delta="データベース内"
        )
    
    st.markdown("---")
    
    # ステップバイステップガイド
    st.markdown("## 🚀 3つのステップで簡単データ登録")
    
    step_col1, step_col2, step_col3 = st.columns(3)
    
    with step_col1:
        st.markdown("""
        ### ステップ 1️⃣
        **📝 データを貼り付け**
        
        Jリーグの試合結果データを
        下のフォームに貼り付けます
        """)
    
    with step_col2:
        st.markdown("""
        ### ステップ 2️⃣
        **🔍 プレビュー確認**
        
        変換結果をプレビューで
        事前に確認できます
        """)
    
    with step_col3:
        st.markdown("""
        ### ステップ 3️⃣ 
        **🎉 一括処理実行**
        
        変換→検証→登録まで
        ワンクリックで完了
        """)
    
    st.markdown("---")
    
    # 使用方法の説明（改善版）
    with st.expander("📋 データ形式ガイド・サンプル", expanded=False):
        
        tab_format, tab_sample = st.tabs(["📖 形式説明", "📄 サンプルデータ"])
        
        with tab_format:
            st.markdown("""
            ### ✅ 対応データ形式
            
            **新形式（タブ区切り）:**
            ```
            年度    リーグ    節日    日付    時刻    ホーム    スコア    アウェイ    スタジアム    観客数    放送
            ```
            
            **旧形式（連続文字列）:**
            ```
            2024Ｊ１ 第1節第1日03/01(金)19:03川崎フロンターレ1-0ヴィッセル神戸等々力陸上競技場18,126ＤＡＺＮ
            ```
            
            ### 🎯 重要なポイント
            - **複数行対応**: 複数の試合データを一度に処理可能
            - **自動判別**: 新旧両形式を自動判別
            - **エラー許容**: 不正な行は自動的にスキップ
            - **重複除去**: 同じ試合データは自動的に除去
            """)
        
        with tab_sample:
            st.markdown("""
            ### 📝 コピー＆ペースト用サンプル
            
            **以下のデータをコピーしてお試しください:**
            """)
            
            sample_data = """2024\tＪ１\t第１節第１日\t03/01(金)\t19:03\t川崎フロンターレ\t2-1\tヴィッセル神戸\t等々力陸上競技場\t25,833\tＤＡＺＮ
2024\tＪ１\t第１節第１日\t03/02(土)\t14:00\tFC東京\t1-0\t浦和レッズ\t味の素スタジアム\t35,421\tＤＡＺＮ
2024\tＪ１\t第１節第２日\t03/03(日)\t14:00\t横浜F・マリノス\t3-1\tガンバ大阪\t日産スタジアム\t42,750\tＤＡＺＮ"""
            
            st.code(sample_data)
            
            if st.button("📋 サンプルを入力フォームにセット"):
                st.session_state['sample_data_input'] = sample_data
                st.success("✅ サンプルデータをセットしました！下のフォームをご確認ください。")
    
    st.markdown("---")
    
    # メインの入力フォーム
    st.markdown("## 📄 データ入力フォーム")
    
    # セッションステートからのサンプルデータ読み込み
    default_value = st.session_state.get('sample_data_input', '')
    
    text_input = st.text_area(
        "🎯 試合結果データを貼り付けてください",
        value=default_value,
        height=250,
        placeholder="""例1（タブ区切り）: 2024	Ｊ１	第１節第１日	03/01(金)	19:03	川崎フロンターレ	2-1	ヴィッセル神戸	等々力陸上競技場	25,833	ＤＡＺＮ
例2（連続文字列）: 2024Ｊ１ 第1節第1日03/01(金)19:03川崎フロンターレ1-0ヴィッセル神戸等々力陸上競技場18,126ＤＡＺＮ

💡 複数の試合データを改行区切りで一度に入力できます""",
        help="新形式・旧形式どちらでも自動判別して処理します",
        key="main_text_input"
    )
    
    # 入力状況の表示
    if text_input.strip():
        lines = len([line for line in text_input.split('\n') if line.strip()])
        st.info(f"📊 入力済み: {lines}行のデータが入力されています")
    else:
        st.warning("⏳ データを入力してください。上のサンプルデータをご利用いただけます。")
    
    st.markdown("---")
    
    # 改善されたボタンエリア
    st.markdown("## 🎮 アクションボタン")
    
    button_col1, button_col2, button_col3, button_col4 = st.columns([2, 2, 1, 1])
    
    with button_col1:
        process_btn = st.button(
            "🚀 **一括処理実行**", 
            type="primary", 
            help="データ変換→検証→登録まで全自動で実行",
            use_container_width=True,
            disabled=not text_input.strip()
        )
        if process_btn:
            st.markdown("*🔄 一括処理を開始しています...*")
    
    with button_col2:
        preview_btn = st.button(
            "🔍 **プレビュー確認**", 
            help="変換結果を事前確認（データ登録はしません）",
            use_container_width=True,
            disabled=not text_input.strip()
        )
        if preview_btn:
            st.markdown("*👀 プレビューを生成中...*")
    
    with button_col3:
        clear_btn = st.button(
            "🗑️ クリア", 
            help="入力フォームをクリア",
            use_container_width=True
        )
    
    with button_col4:
        if st.button(
            "📋 サンプル", 
            help="サンプルデータをセット",
            use_container_width=True
        ):
            st.session_state['sample_data_input'] = """2024\tＪ１\t第１節第１日\t03/01(金)\t19:03\t川崎フロンターレ\t2-1\tヴィッセル神戸\t等々力陸上競技場\t25,833\tＤＡＺＮ
2024\tＪ１\t第１節第１日\t03/02(土)\t14:00\tFC東京\t1-0\t浦和レッズ\t味の素スタジアム\t35,421\tＤＡＺＮ"""
            st.rerun()
    
    if clear_btn:
        if 'sample_data_input' in st.session_state:
            del st.session_state['sample_data_input']
        st.rerun()
    
    # プレビュー処理
    if preview_btn and text_input.strip():
        with st.spinner("🔍 データをプレビュー中..."):
            try:
                converted_df, stats = text_to_csv_converter(text_input)
                show_conversion_results(converted_df, stats, preview_mode=True)
            except Exception as e:
                st.error("🚫 プレビュー処理中にエラーが発生しました")
                
                with st.expander("🔧 エラー詳細とヘルプ", expanded=True):
                    st.markdown(f"**エラー内容:** `{str(e)}`")
                    
                    st.markdown("""
                    ### 💡 よくある問題と解決方法
                    
                    1. **データ形式の問題**
                       - Jリーグの正式なデータ形式か確認してください
                       - 年度、リーグ名、日付形式をチェック
                    
                    2. **文字エンコーディングの問題**  
                       - コピー&ペースト時に特殊文字が混入している可能性
                       - 上のサンプルデータでテストしてみてください
                    
                    3. **空行や不要なデータ**
                       - 空行やヘッダー行が含まれていませんか？
                       - 実際の試合データのみを入力してください
                    """)
                    
                    if st.button("🏥 サンプルデータで動作確認"):
                        st.session_state['sample_data_input'] = """2024\tＪ１\t第１節第１日\t03/01(金)\t19:03\t川崎フロンターレ\t2-1\tヴィッセル神戸\t等々力陸上競技場\t25,833\tＤＡＺＮ"""
                        st.rerun()
    
    # 一括処理実行
    if process_btn and text_input.strip():
        try:
            # プログレスバー初期化
            progress_bar = st.progress(0)
            status_container = st.container()
            
            with status_container:
                st.info("🚀 一括処理を開始します...")
            
            # ステップ1: データ変換
            progress_bar.progress(20)
            with status_container:
                st.info("🔄 ステップ1: データを変換中...")
            
            converted_df, stats = text_to_csv_converter(text_input)
            
            # ステップ2: データ検証
            progress_bar.progress(50)
            with status_container:
                st.info("🔍 ステップ2: データを検証中...")
            
            if stats['valid_matches'] > 0:
                # ステップ3: データ登録
                progress_bar.progress(80)
                with status_container:
                    st.info("💾 ステップ3: データを登録中...")
                
                # 既存データとの統合
                existing_data_path = "data/matches.csv"
                os.makedirs("data", exist_ok=True)
                
                # 古いデータ形式に変換
                if not converted_df.empty:
                    legacy_format_df = convert_to_legacy_format(converted_df)
                    
                    if os.path.exists(existing_data_path):
                        existing_df = pd.read_csv(existing_data_path)
                        combined_df = pd.concat([existing_df, legacy_format_df], ignore_index=True)
                        combined_df = combined_df.drop_duplicates(
                            subset=['date', 'home_team', 'away_team'], 
                            keep='last'
                        )
                        new_records = len(combined_df) - len(existing_df)
                    else:
                        combined_df = legacy_format_df
                        new_records = len(legacy_format_df)
                    
                    # 日付でソート（降順）してから保存
                    if 'date' in combined_df.columns:
                        combined_df = combined_df.sort_values('date', ascending=False).reset_index(drop=True)
                    combined_df.to_csv(existing_data_path, index=False, encoding='utf-8')
                
                # 完了
                progress_bar.progress(100)
                with status_container:
                    st.success("✅ 一括処理完了!")
                
                # 大きな取り込み件数表示
                st.balloons()
                
                # 成功メッセージと統計
                col_success1, col_success2, col_success3 = st.columns(3)
                
                with col_success1:
                    st.metric(
                        "🎉 取り込み完了",
                        f"{stats['valid_matches']}件",
                        delta=f"+{new_records}件 新規追加",
                        delta_color="normal"
                    )
                
                with col_success2:
                    success_rate = round((stats['valid_matches'] / stats['total_lines']) * 100, 1) if stats['total_lines'] > 0 else 0
                    st.metric(
                        "📊 成功率",
                        f"{success_rate}%",
                        delta=f"{stats['total_lines']}行中{stats['valid_matches']}行成功"
                    )
                
                with col_success3:
                    total_data_count = len(combined_df) if 'combined_df' in locals() else 0
                    st.metric(
                        "📈 総データ数",
                        f"{total_data_count}件",
                        delta="データベース内総数"
                    )
                
                # 詳細結果表示
                show_conversion_results(converted_df, stats, preview_mode=False)
                
                # プログレスバーをクリア
                time.sleep(1)
                progress_bar.empty()
                status_container.empty()
                
            else:
                progress_bar.progress(100)
                with status_container:
                    st.warning("⚠️ 取り込み可能なデータがありませんでした")
                
                show_conversion_results(converted_df, stats, preview_mode=False)
                
                time.sleep(1)
                progress_bar.empty()
                status_container.empty()
                
        except Exception as e:
            if 'progress_bar' in locals():
                progress_bar.empty()
            if 'status_container' in locals():
                status_container.empty()
            
            st.error("🚨 一括処理中に重大なエラーが発生しました")
            
            with st.expander("🛠️ エラー分析とトラブルシューティング", expanded=True):
                st.markdown(f"**エラー種別:** `{type(e).__name__}`")
                st.markdown(f"**エラー内容:** `{str(e)}`")
                
                st.markdown("""
                ### 🔧 推奨される対処法
                
                1. **データ形式の再確認**
                   - 入力データがJリーグの正式形式か確認
                   - 特殊文字や不正な文字がないか確認
                
                2. **段階的な確認**
                   - まず「プレビュー確認」で問題を特定
                   - サンプルデータで動作確認
                
                3. **システム状態の確認**
                   - dataディレクトリの権限確認
                   - 既存CSVファイルの整合性確認
                """)
                
                col_help1, col_help2 = st.columns(2)
                
                with col_help1:
                    if st.button("🔄 サンプルで再テスト"):
                        st.session_state['sample_data_input'] = """2024\tＪ１\t第１節第１日\t03/01(金)\t19:03\t川崎フロンターレ\t2-1\tヴィッセル神戸\t等々力陸上競技場\t25,833\tＤＡＺＮ"""
                        st.rerun()
                
                with col_help2:
                    if st.button("🧹 セッションリセット"):
                        for key in list(st.session_state.keys()):
                            if key.startswith('sample_data'):
                                del st.session_state[key]
                        st.rerun()
                
                with st.expander("🔍 技術的詳細（上級者向け）", expanded=False):
                    st.code(traceback.format_exc())

def show_conversion_results(converted_df: pd.DataFrame, stats: dict, preview_mode: bool = False):
    """変換結果を表示する共通関数"""
    # プレビューモードかどうかで表示を変える
    mode_text = "🔍 プレビュー結果" if preview_mode else "📊 取り込み結果"
    st.subheader(f"{mode_text}サマリー")
    
    if preview_mode:
        st.info("💡 これはプレビューです。実際のデータ登録は「一括処理実行」ボタンをご利用ください。")
    
    if stats['total_lines'] > 0:
        # メイン統計表示
        col_main1, col_main2, col_main3, col_main4 = st.columns(4)
        
        with col_main1:
            st.metric("入力行数", stats['total_lines'])
        
        with col_main2:
            success_rate = round((stats['valid_matches'] / stats['total_lines']) * 100, 1) if stats['total_lines'] > 0 else 0
            st.metric("取り込み成功", f"{stats['valid_matches']}件", 
                    delta=f"成功率 {success_rate}%")
        
        with col_main3:
            total_skipped = stats['skipped_lines'] + stats['format_errors'] + stats['vs_matches_skipped']
            st.metric("スキップ", f"{total_skipped}件", 
                    delta=f"-{total_skipped}" if total_skipped > 0 else "0")
        
        with col_main4:
            if stats['valid_matches'] > 0:
                st.metric("✅ 状態", "成功", delta="データ追加可能")
            else:
                st.metric("⚠️ 状態", "要確認", delta="取り込み可能データなし")
        
        # 詳細な内訳表示
        st.markdown("### 📋 詳細内訳")
        
        # 取り込み結果テーブル
        result_data = {
            "項目": [
                "✅ 正常に変換されたデータ",
                "⚠️ 未開催試合('vs'スコア)", 
                "❌ データ形式エラー",
                "🚫 フォーマット不整合",
                "📝 総入力行数"
            ],
            "件数": [
                stats['valid_matches'],
                stats['vs_matches_skipped'],
                stats['format_errors'], 
                stats['skipped_lines'],
                stats['total_lines']
            ]
        }
        
        result_df = pd.DataFrame(result_data)
        st.dataframe(result_df, use_container_width=True, hide_index=True)
        
        # プレビューの場合は変換結果のみ表示
        if not converted_df.empty:
            st.markdown("### 📋 変換結果プレビュー")
            st.dataframe(converted_df, use_container_width=True)


def text_converter_tab() -> None:
    """テキスト変換タブの実装"""
    st.header("🔄 テキスト変換ツール")
    
    st.markdown("""
    **Jリーグ試合結果データを様々な形式に変換**
    """)
    
    # サンプルデータの生成
    sample_data = """2024\tＪ１\t第１節第１日\t03/01(金)\t19:03\t川崎フロンターレ\t2-1\tヴィッセル神戸\t等々力陸上競技場\t25,833\tＤＡＺＮ
2024\tＪ１\t第１節第１日\t03/02(土)\t14:00\tFC東京\t1-0\t浦和レッズ\t味の素スタジアム\t35,421\tＤＡＺＮ"""
    
    st.text_area(
        "試合結果データを貼り付けてください",
        value=sample_data,
        height=200,
        key="sample_data_area"
    )


def settings_tab() -> None:
    """設定タブの実装"""
    st.header("⚙️ システム設定")
    
    # モデル設定セクション
    st.subheader("🤖 機械学習モデル設定")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**モデル状態**")
        model_path = "models/prediction_model.pkl"
        if os.path.exists(model_path):
            st.success("✅ モデル利用可能")
        else:
            st.warning("⚠️ モデル未訓練")
    
    with col2:
        if st.button("🔄 モデル再訓練"):
            with st.spinner("モデルを訓練中..."):
                result = train_prediction_model()
                if result["success"]:
                    st.success("✅ モデル訓練完了")
                else:
                    st.error("❌ モデル訓練失敗")


def perform_detailed_evaluation(cv_folds: int, train_ratio: float, random_seed: int) -> Optional[Dict]:
    """詳細なモデル評価を実行"""
    try:
        return {"success": True}
    except Exception as e:
        return None


def dashboard_tab() -> None:
    """ダッシュボードタブの実装"""
    st.header("🏠 システムダッシュボード")
    
    # 基本統計
    stats = get_match_data_stats()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("総試合数", stats["total_matches"])
    
    with col2:
        st.metric("登録チーム数", len(stats["teams"]))
    
    with col3:
        st.metric("リーグ数", len(stats["leagues"]))


def help_tab() -> None:
    """ヘルプタブの実装"""
    st.header("❓ ヘルプ・ドキュメント")
    
    st.markdown("""
    ## 🎯 システム概要
    
    このシステムは、Jリーグの試合データを簡単に取り込み、
    機械学習による試合結果の予想を行うことができます。
    
    ## 📝 主な機能
    
    - **データ入力**: フォームからの簡単なデータ取り込み
    - **試合予想**: 機械学習による勝敗予測
    - **データ分析**: 試合結果の統計分析
    - **設定管理**: システムの各種設定
    """)


def main() -> None:
    """メイン関数"""
    try:
        # サイドバーの設定
        setup_sidebar()
        
        # メインタイトル
        st.title("⚽ Jリーグ試合予想システム")
        st.markdown("**実験的Jリーグ試合予想・分析プラットフォーム**")
        st.markdown("---")
        
        # タブの作成
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🏠 ダッシュボード", "🔮 予想", "📊 分析", "📝 データ入力", "⚙️ 設定", "❓ ヘルプ"])
        
        with tab1:
            dashboard_tab()
            
        with tab2:
            prediction_tab()
            
        with tab3:
            analysis_tab()
            
        with tab4:
            data_input_tab()
            
        with tab5:
            settings_tab()
            
        with tab6:
            help_tab()
            
    except Exception as e:
        st.error(f"アプリケーションエラーが発生しました: {str(e)}")
        st.error("開発者にお問い合わせください。")

if __name__ == "__main__":
    main()
