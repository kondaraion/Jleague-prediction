"""
Jリーグ試合結果テキストデータをCSV形式に変換する機能
"""

import re
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional


def parse_jleague_text_data(text_data: str) -> tuple[pd.DataFrame, dict]:
    """
    貼り付けられたテキストデータをCSV形式に変換
    
    Args:
        text_data (str): 貼り付けられた試合結果テキスト
    
    Returns:
        tuple: (DataFrame: 整理されたデータフレーム, dict: 解析統計情報)
    """
    
    # データ行の抽出と解析
    matches = []
    lines = text_data.split('\n')
    
    # 統計情報
    stats = {
        'total_lines': 0,
        'processed_lines': 0,
        'skipped_lines': 0,
        'valid_matches': 0,
        'vs_matches_skipped': 0,
        'format_errors': 0
    }
    
    for line in lines:
        line = line.strip()
        if not line:  # 空行はスキップ
            continue
            
        stats['total_lines'] += 1
        
        # Jリーグデータらしい行かチェック
        if is_potential_match_line(line):
            stats['processed_lines'] += 1
            parsed_data = parse_match_line(line)
            
            if parsed_data:
                if 'vs_skipped' in parsed_data:
                    stats['vs_matches_skipped'] += 1
                else:
                    matches.append(parsed_data)
                    stats['valid_matches'] += 1
            else:
                stats['format_errors'] += 1
        else:
            stats['skipped_lines'] += 1
    
    return pd.DataFrame(matches), stats


def is_potential_match_line(line: str) -> bool:
    """
    Jリーグの試合データらしい行かどうかを判定
    """
    # 基本的な条件をチェック
    if len(line) < 20:  # 最低限の長さ
        return False
    
    # タブ区切りの場合
    if '\t' in line:
        parts = line.split('\t')
        return (len(parts) >= 8 and 
                any(char.isdigit() for char in parts[0]) and  # 年度が含まれる
                'Ｊ' in line and '節' in line)  # Jリーグの節情報
    
    # 旧形式の場合
    return ('第' in line and '節' in line and 
            any(char.isdigit() for char in line) and
            'Ｊ' in line)


def parse_match_line(line: str) -> Optional[Dict]:
    """
    単一の試合データ行を解析（新旧両形式対応）
    """
    try:
        # タブ区切りの新形式かチェック
        if '\t' in line:
            return parse_tab_separated_line(line)
        else:
            return parse_old_format_line(line)
        
    except Exception:
        return None


def parse_tab_separated_line(line: str) -> Optional[Dict]:
    """
    タブ区切りの新形式データを解析
    """
    try:
        parts = line.split('\t')
        if len(parts) < 11:
            return None
        
        year = parts[0].strip()
        league = parts[1].strip()
        round_day = parts[2].strip()  # "第１節第１日"
        date = parts[3].strip()
        kickoff = parts[4].strip()
        home_team = parts[5].strip()
        original_score = parts[6].strip()
        away_team = parts[7].strip()
        stadium = parts[8].strip()
        attendance = parts[9].strip()
        broadcast = parts[10].strip()
        
        # データ検証
        if not validate_basic_data(year, league, round_day, date, kickoff, home_team, away_team):
            return None
        
        # 'vs'スコア（未開催試合）の場合は特別なマーカーを返す
        if 'vs' in original_score.lower():
            return {'vs_skipped': True}
        
        score = normalize_score(original_score)
        
        # 節と日を分離
        round_match = re.search(r'第(\d+)節第(\d+)日', round_day)
        if round_match:
            round_num, day_num = round_match.groups()
        else:
            round_num, day_num = "1", "1"
        
        return {
            'year': year,
            'league': league,
            'round': f"第{round_num}節",
            'day': f"第{day_num}日",
            'date': date,
            'kickoff': kickoff,
            'home_team': home_team,
            'score': score,
            'away_team': away_team,
            'stadium': stadium,
            'attendance': attendance,
            'broadcast': broadcast
        }
        
    except Exception:
        return None


def parse_old_format_line(line: str) -> Optional[Dict]:
    """
    旧形式（連続文字列）データを解析
    """
    try:
        # 基本パターンの抽出
        base_pattern = r'(\d{4})(Ｊ[１２３])\s+第(\d+)節第(\d+)日(\d{2}/\d{2}\([^)]+\))(\d{2}:\d{2})'
        base_match = re.search(base_pattern, line)
        
        if not base_match:
            return None
            
        year, league, round_num, day, date, kickoff = base_match.groups()
        
        # 基本情報の後の部分を抽出
        remaining = line[base_match.end():]
        
        # スコアパターンを探す（'vs'や'－'なども対応）
        score_pattern = r'(\d+-\d+|\d+－\d+|vs|\d+\s*vs\s*\d+)'
        score_match = re.search(score_pattern, remaining)
        
        if not score_match:
            return None
        
        original_score = score_match.group(1)
        
        # 'vs'スコア（未開催試合）の場合は特別なマーカーを返す
        if 'vs' in original_score.lower():
            return {'vs_skipped': True}
            
        score = normalize_score(original_score)
        
        # ホームチーム（キックオフ時刻からスコアまで）
        home_team = remaining[:score_match.start()].strip()
        
        # アウェイチーム、スタジアム、観客数、放送局（スコア後）
        after_score = remaining[score_match.end():]
        
        # 観客数パターン（数字とカンマ）
        attendance_pattern = r'(\d{1,3}(?:,\d{3})*)'
        attendance_match = re.search(attendance_pattern, after_score)
        
        if attendance_match:
            attendance = attendance_match.group(1)
            before_attendance = after_score[:attendance_match.start()]
            broadcast = after_score[attendance_match.end():]
            
            # アウェイチームとスタジアムを分離するための改良されたロジック
            away_team, stadium = separate_team_and_stadium(before_attendance.strip())
        else:
            return None
        
        return {
            'year': year,
            'league': league,
            'round': f"第{round_num}節",
            'day': f"第{day}日",
            'date': date,
            'kickoff': kickoff,
            'home_team': home_team,
            'score': score,
            'away_team': away_team,
            'stadium': stadium,
            'attendance': attendance,
            'broadcast': broadcast.strip()
        }
        
    except Exception:
        return None


def separate_team_and_stadium(text: str) -> tuple:
    """
    アウェイチーム名とスタジアム名を分離
    """
    # 一般的なJリーグチーム名のパターン
    team_patterns = [
        r'([^競場館]+)(.*(?:競技場|スタジアム|球場|ドーム|パーク).*)',
        r'(ヴィッセル神戸|川崎フロンターレ|FC東京|浦和レッズ|横浜F・マリノス|ガンバ大阪|セレッソ大阪|鹿島アントラーズ|柏レイソル|名古屋グランパス|湘南ベルマーレ|サガン鳥栖|アビスパ福岡|京都サンガ|ジュビロ磐田|清水エスパルス|コンサドーレ札幌|アルビレックス新潟|サンフレッチェ広島|横浜FC)(.*)',
        r'([^ー・]+(?:ー|・)[^競場館]*)(.*)',
        r'([^競場館]+)(.*)'
    ]
    
    for pattern in team_patterns:
        match = re.match(pattern, text)
        if match:
            team_name = match.group(1).strip()
            stadium_name = match.group(2).strip() if len(match.groups()) > 1 else ""
            
            # チーム名が長すぎる場合は調整
            if len(team_name) > 15:
                # スタジアム名の一部が含まれている可能性
                continue
                
            return team_name, stadium_name
    
    # デフォルト: 全体をチーム名として扱う
    return text, ""


def validate_basic_data(year: str, league: str, round_day: str, date: str, kickoff: str, home_team: str, away_team: str) -> bool:
    """
    基本的なデータの妥当性をチェック
    """
    try:
        # 年度チェック（4桁の数字）
        if not year.isdigit() or len(year) != 4:
            return False
        
        year_int = int(year)
        if year_int < 1990 or year_int > 2030:  # 妥当な範囲
            return False
        
        # リーグチェック
        if not ('Ｊ' in league and any(c in league for c in ['１', '２', '３', '1', '2', '3'])):
            return False
        
        # 節情報チェック
        if not ('節' in round_day and '第' in round_day):
            return False
        
        # 日付形式チェック（MM/DD形式）
        if not re.match(r'\d{2}/\d{2}\([^)]+\)', date):
            return False
        
        # キックオフ時刻チェック（HH:MM形式）
        if not re.match(r'\d{2}:\d{2}', kickoff):
            return False
        
        # チーム名チェック（空でない）
        if not home_team or not away_team:
            return False
        
        return True
        
    except Exception:
        return False


def normalize_score(score_str: str) -> str:
    """
    スコア文字列を正規化（全角文字などを標準形式に変換）
    """
    if not score_str:
        return "0-0"
    
    score_str = score_str.strip()
    
    # 全角ハイフンを半角に変換
    score_str = score_str.replace('－', '-')
    
    # 数字とハイフンのみの形式に正規化
    score_match = re.search(r'(\d+)\s*[-－]\s*(\d+)', score_str)
    if score_match:
        home_score, away_score = score_match.groups()
        return f"{home_score}-{away_score}"
    
    # パターンが見つからない場合は0-0
    return "0-0"


def clean_and_validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """データクリーニングと検証（エラーハンドリング強化）"""
    
    if df.empty:
        return df
    
    # スコア分割（安全な変換）
    try:
        score_split = df['score'].str.split('-', expand=True)
        if score_split.shape[1] >= 2:
            df['home_score'] = pd.to_numeric(score_split[0], errors='coerce').fillna(0).astype(int)
            df['away_score'] = pd.to_numeric(score_split[1], errors='coerce').fillna(0).astype(int)
        else:
            df['home_score'] = 0
            df['away_score'] = 0
    except Exception:
        df['home_score'] = 0
        df['away_score'] = 0
    
    # 日付正規化
    df['formatted_date'] = df.apply(lambda row: format_date(row['year'], row['date']), axis=1)
    
    # チーム名正規化
    df['home_team'] = df['home_team'].apply(normalize_team_name)
    df['away_team'] = df['away_team'].apply(normalize_team_name)
    
    # 入場者数数値化（安全な変換）
    try:
        df['attendance'] = df['attendance'].str.replace(',', '').str.replace(' ', '')
        df['attendance'] = pd.to_numeric(df['attendance'], errors='coerce').fillna(0).astype(int)
    except Exception:
        df['attendance'] = 0
    
    return df


def format_date(year: str, date_str: str) -> str:
    """日付を標準形式に変換"""
    try:
        # 日付部分を抽出 (MM/DD形式)
        date_match = re.search(r'(\d{2})/(\d{2})', date_str)
        if date_match:
            month, day = date_match.groups()
            formatted_date = f"{year}-{month}-{day}"
            # 日付の妥当性チェック
            datetime.strptime(formatted_date, '%Y-%m-%d')
            return formatted_date
    except (ValueError, AttributeError):
        pass
    
    return f"{year}-01-01"  # デフォルト値


def normalize_team_name(team_name: str) -> str:
    """チーム名の正規化"""
    # チーム名の前後の空白を削除
    team_name = team_name.strip()
    
    # 一般的なチーム名の正規化マッピング
    team_mapping = {
        'FC東京': 'FC東京',
        'サンフレッチェ広島': 'サンフレッチェ広島',
        'ヴィッセル神戸': 'ヴィッセル神戸',
        'ガンバ大阪': 'ガンバ大阪',
        'セレッソ大阪': 'セレッソ大阪',
        '横浜F・マリノス': '横浜F・マリノス',
        '横浜FC': '横浜FC',
        '川崎フロンターレ': '川崎フロンターレ',
        '鹿島アントラーズ': '鹿島アントラーズ',
        '浦和レッズ': '浦和レッズ',
        '柏レイソル': '柏レイソル',
        'アルビレックス新潟': 'アルビレックス新潟',
        '名古屋グランパス': '名古屋グランパス',
        '湘南ベルマーレ': '湘南ベルマーレ',
        'サガン鳥栖': 'サガン鳥栖',
        'アビスパ福岡': 'アビスパ福岡',
        '京都サンガF.C.': '京都サンガ',
        'ジュビロ磐田': 'ジュビロ磐田',
        '清水エスパルス': '清水エスパルス',
        '北海道コンサドーレ札幌': 'コンサドーレ札幌'
    }
    
    return team_mapping.get(team_name, team_name)


def text_to_csv_converter(text_data: str) -> tuple[pd.DataFrame, dict]:
    """
    メイン変換関数：テキストデータをCSV形式のDataFrameに変換
    
    Args:
        text_data (str): 貼り付けられた試合結果テキスト
    
    Returns:
        tuple: (DataFrame: 変換されたデータフレーム, dict: 解析統計情報)
    """
    # データ解析
    raw_df, stats = parse_jleague_text_data(text_data)
    
    if raw_df.empty:
        return pd.DataFrame(), stats
    
    # データクリーニング
    cleaned_df = clean_and_validate_data(raw_df)
    
    return cleaned_df, stats


def export_to_csv(df: pd.DataFrame, filename: str = "jleague_matches.csv") -> str:
    """
    DataFrameをCSVファイルとして出力
    
    Args:
        df (DataFrame): 出力するデータフレーム
        filename (str): 出力ファイル名
    
    Returns:
        str: 出力されたファイルのパス
    """
    try:
        # 日付でソート（降順）してから保存
        if 'date' in df.columns:
            df = df.sort_values('date', ascending=False).reset_index(drop=True)
        df.to_csv(filename, index=False, encoding='utf-8')
        return filename
    except Exception as e:
        raise Exception(f"CSV出力エラー: {str(e)}")