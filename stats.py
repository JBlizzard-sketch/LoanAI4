import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from utils import db
import sqlite3
from typing import Dict, List, Tuple, Any, Optional
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def safe_str(value: Any) -> str:
    """Safely convert value to string"""
    return str(value) if value is not None else 'N/A'

def safe_float(value: Any) -> float:
    """Safely convert value to float"""
    try:
        return float(value) if value is not None else 0.0
    except (ValueError, TypeError):
        return 0.0

def safe_int(value: Any) -> int:
    """Safely convert value to int"""
    try:
        return int(value) if value is not None else 0
    except (ValueError, TypeError):
        return 0

def get_loan_portfolio_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate comprehensive loan portfolio statistics"""
    if df.empty:
        return {}
    
    stats = {
        'total_loans': len(df),
        'total_amount': safe_float(df['loan_amount'].sum()),
        'avg_loan_amount': safe_float(df['loan_amount'].mean()),
        'median_loan_amount': safe_float(df['loan_amount'].median()),
        'min_loan_amount': safe_float(df['loan_amount'].min()),
        'max_loan_amount': safe_float(df['loan_amount'].max()),
        'std_loan_amount': safe_float(df['loan_amount'].std()),
        
        # Repayment statistics
        'repayment_rate': safe_float(df['repay_good'].mean() * 100) if 'repay_good' in df.columns else 0,
        'default_rate': safe_float((1 - df['repay_good'].mean()) * 100) if 'repay_good' in df.columns else 0,
        'total_good_loans': safe_int(df['repay_good'].sum()) if 'repay_good' in df.columns else 0,
        'total_default_loans': len(df) - safe_int(df['repay_good'].sum()) if 'repay_good' in df.columns else 0,
        
        # Fraud statistics
        'fraud_rate': safe_float(df['is_fraud'].mean() * 100) if 'is_fraud' in df.columns else 0,
        'total_fraud_cases': safe_int(df['is_fraud'].sum()) if 'is_fraud' in df.columns else 0,
        'fraud_amount': safe_float(df[df['is_fraud'] == 1]['loan_amount'].sum()) if 'is_fraud' in df.columns else 0,
        
        # Credit score statistics
        'avg_credit_score': safe_float(df['credit_score'].mean()) if 'credit_score' in df.columns else 0,
        'median_credit_score': safe_float(df['credit_score'].median()) if 'credit_score' in df.columns else 0,
        'high_credit_score_rate': safe_float((df['credit_score'] >= 700).mean() * 100) if 'credit_score' in df.columns else 0,
        'low_credit_score_rate': safe_float((df['credit_score'] < 500).mean() * 100) if 'credit_score' in df.columns else 0,
        
        # Eligibility statistics
        'eligibility_rate': safe_float(df['eligible'].mean() * 100) if 'eligible' in df.columns else 0,
        'total_eligible': safe_int(df['eligible'].sum()) if 'eligible' in df.columns else 0,
        'total_ineligible': len(df) - safe_int(df['eligible'].sum()) if 'eligible' in df.columns else 0,
        
        # Risk statistics
        'avg_default_risk': safe_float(df['default_risk'].mean() * 100) if 'default_risk' in df.columns else 0,
        'high_risk_rate': safe_float((df['default_risk'] > 0.7).mean() * 100) if 'default_risk' in df.columns else 0,
        'low_risk_rate': safe_float((df['default_risk'] < 0.3).mean() * 100) if 'default_risk' in df.columns else 0,
    }
    
    return stats

def get_demographic_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate demographic-based statistics"""
    if df.empty:
        return {}
    
    stats = {
        # Gender statistics
        'female_percentage': safe_float((df['gender'] == 'F').mean() * 100) if 'gender' in df.columns else 0,
        'male_percentage': safe_float((df['gender'] == 'M').mean() * 100) if 'gender' in df.columns else 0,
        'female_avg_loan': safe_float(df[df['gender'] == 'F']['loan_amount'].mean()) if 'gender' in df.columns else 0,
        'male_avg_loan': safe_float(df[df['gender'] == 'M']['loan_amount'].mean()) if 'gender' in df.columns else 0,
        'female_repayment_rate': safe_float(df[df['gender'] == 'F']['repay_good'].mean() * 100) if 'gender' in df.columns and 'repay_good' in df.columns else 0,
        'male_repayment_rate': safe_float(df[df['gender'] == 'M']['repay_good'].mean() * 100) if 'gender' in df.columns and 'repay_good' in df.columns else 0,
        
        # Age statistics
        'avg_age': safe_float(df['age'].mean()) if 'age' in df.columns else 0,
        'median_age': safe_float(df['age'].median()) if 'age' in df.columns else 0,
        'young_borrowers_rate': safe_float((df['age'] < 30).mean() * 100) if 'age' in df.columns else 0,
        'senior_borrowers_rate': safe_float((df['age'] >= 50).mean() * 100) if 'age' in df.columns else 0,
        
        # Income statistics
        'avg_income': safe_float(df['income'].mean()) if 'income' in df.columns else 0,
        'median_income': safe_float(df['income'].median()) if 'income' in df.columns else 0,
        'low_income_rate': safe_float((df['income'] < 20000).mean() * 100) if 'income' in df.columns else 0,
        'high_income_rate': safe_float((df['income'] >= 50000).mean() * 100) if 'income' in df.columns else 0,
        
        # Dependents statistics
        'avg_dependents': safe_float(df['dependents'].mean()) if 'dependents' in df.columns else 0,
        'no_dependents_rate': safe_float((df['dependents'] == 0).mean() * 100) if 'dependents' in df.columns else 0,
        'many_dependents_rate': safe_float((df['dependents'] >= 4).mean() * 100) if 'dependents' in df.columns else 0,
    }
    
    return stats

def get_geographic_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate geographic/branch-based statistics"""
    if df.empty or 'branch' not in df.columns:
        return {}
    
    try:
        branch_stats = df.groupby('branch').agg({
            'loan_amount': ['count', 'sum', 'mean'],
            'repay_good': 'mean' if 'repay_good' in df.columns else 'count',
            'is_fraud': 'sum' if 'is_fraud' in df.columns else 'count'
        }).round(2)
        
        if len(branch_stats) == 0:
            return {}
        
        stats = {
            'total_branches': df['branch'].nunique(),
            'top_branch_by_volume': safe_str(branch_stats[('loan_amount', 'sum')].idxmax()),
            'top_branch_volume': safe_float(branch_stats[('loan_amount', 'sum')].max()),
            'top_branch_by_count': safe_str(branch_stats[('loan_amount', 'count')].idxmax()),
            'top_branch_count': safe_int(branch_stats[('loan_amount', 'count')].max()),
            'avg_loans_per_branch': safe_float(branch_stats[('loan_amount', 'count')].mean()),
            'branch_concentration': safe_float((branch_stats[('loan_amount', 'count')].max() / branch_stats[('loan_amount', 'count')].sum()) * 100),
        }
        
        if 'repay_good' in df.columns:
            stats['best_performing_branch'] = safe_str(branch_stats[('repay_good', 'mean')].idxmax())
            stats['best_branch_repayment_rate'] = safe_float(branch_stats[('repay_good', 'mean')].max() * 100)
            stats['worst_performing_branch'] = safe_str(branch_stats[('repay_good', 'mean')].idxmin())
            stats['worst_branch_repayment_rate'] = safe_float(branch_stats[('repay_good', 'mean')].min() * 100)
        
        return stats
    except Exception:
        return {}

def get_product_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate product-based statistics"""
    if df.empty or 'product' not in df.columns:
        return {}
    
    try:
        product_stats = df.groupby('product').agg({
            'loan_amount': ['count', 'sum', 'mean'],
            'repay_good': 'mean' if 'repay_good' in df.columns else 'count',
            'is_fraud': 'sum' if 'is_fraud' in df.columns else 'count'
        }).round(2)
        
        if len(product_stats) == 0:
            return {}
        
        stats = {
            'total_products': df['product'].nunique(),
            'most_popular_product': safe_str(product_stats[('loan_amount', 'count')].idxmax()),
            'most_popular_count': safe_int(product_stats[('loan_amount', 'count')].max()),
            'highest_value_product': safe_str(product_stats[('loan_amount', 'sum')].idxmax()),
            'highest_value_amount': safe_float(product_stats[('loan_amount', 'sum')].max()),
            'avg_amount_per_product': safe_float(product_stats[('loan_amount', 'mean')].mean()),
        }
        
        if 'repay_good' in df.columns:
            stats['safest_product'] = safe_str(product_stats[('repay_good', 'mean')].idxmax())
            stats['safest_product_rate'] = safe_float(product_stats[('repay_good', 'mean')].max() * 100)
            stats['riskiest_product'] = safe_str(product_stats[('repay_good', 'mean')].idxmin())
            stats['riskiest_product_rate'] = safe_float(product_stats[('repay_good', 'mean')].min() * 100)
        
        return stats
    except Exception:
        return {}

def get_model_performance_stats() -> Dict[str, Any]:
    """Get ML model performance statistics from database"""
    try:
        models = db.list_models()
        if not models:
            return {}
        
        stats = {
            'total_models': len(models),
            'deployed_models': sum(1 for model in models if model[4] == 1),
            'model_families': len(set(model[0] for model in models)),
        }
        
        # Get performance metrics for deployed models
        deployed_metrics = []
        all_metrics = []
        
        for family, version, metrics_str, path, deployed, created in models:
            try:
                import json
                metrics = json.loads(metrics_str)
                all_metrics.append(metrics)
                if deployed == 1:
                    deployed_metrics.append(metrics)
            except:
                continue
        
        if deployed_metrics:
            deployed_metric = deployed_metrics[0]  # Assuming one deployed model
            stats['deployed_model_auc'] = safe_float(deployed_metric.get('AUC', 0))
            stats['deployed_model_accuracy'] = safe_float(deployed_metric.get('accuracy', 0))
            stats['deployed_model_recall'] = safe_float(deployed_metric.get('recall', 0))
        
        if all_metrics:
            auc_values = [m.get('AUC', 0) for m in all_metrics if 'AUC' in m and m.get('AUC') is not None]
            if auc_values:
                stats['best_model_auc'] = max(auc_values)
                stats['avg_model_auc'] = sum(auc_values) / len(auc_values)
                stats['worst_model_auc'] = min(auc_values)
        
        return stats
    except Exception as e:
        return {'error': str(e)}

def get_system_health_stats() -> Dict[str, Any]:
    """Get system health and usage statistics"""
    try:
        conn = db.get_conn()
        cur = conn.cursor()
        
        # User statistics
        cur.execute("SELECT COUNT(*) FROM users")
        total_users = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM users WHERE role='client'")
        client_users = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM users WHERE role='admin'")
        admin_users = cur.fetchone()[0]
        
        # Audit statistics
        cur.execute("SELECT COUNT(*) FROM audit")
        total_actions = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM audit WHERE action='login'")
        total_logins = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM audit WHERE action='register'")
        total_registrations = cur.fetchone()[0]
        
        # Recent activity (last 7 days)
        week_ago = (datetime.now() - timedelta(days=7)).isoformat()
        cur.execute("SELECT COUNT(*) FROM audit WHERE created_at > ?", (week_ago,))
        recent_activity = cur.fetchone()[0]
        
        conn.close()
        
        stats = {
            'total_users': total_users,
            'client_users': client_users,
            'admin_users': admin_users,
            'user_growth_rate': safe_float((client_users / max(total_users, 1)) * 100),
            'total_system_actions': total_actions,
            'total_logins': total_logins,
            'total_registrations': total_registrations,
            'recent_activity_week': recent_activity,
            'avg_actions_per_user': safe_float(total_actions / max(total_users, 1)),
            'login_to_registration_ratio': safe_float(total_logins / max(total_registrations, 1)),
        }
        
        return stats
    except Exception as e:
        return {'error': str(e)}

def create_advanced_charts(df: pd.DataFrame) -> Dict[str, go.Figure]:
    """Create advanced visualization charts"""
    charts = {}
    
    if df.empty:
        return charts
    
    try:
        # 1. Risk vs Credit Score Scatter with Size by Loan Amount
        if all(col in df.columns for col in ['credit_score', 'default_risk', 'loan_amount']):
            fig_risk_score = px.scatter(
                df, 
                x='credit_score', 
                y='default_risk',
                size='loan_amount',
                color='eligible' if 'eligible' in df.columns else None,
                title='Risk vs Credit Score Analysis',
                labels={'default_risk': 'Default Risk (%)', 'credit_score': 'Credit Score'}
            )
            fig_risk_score.update_layout(height=400)
            charts['risk_vs_score'] = fig_risk_score
        
        # 2. Geographic Distribution
        if 'branch' in df.columns:
            branch_summary = df.groupby('branch').agg({
                'loan_amount': ['count', 'sum'],
                'repay_good': 'mean' if 'repay_good' in df.columns else 'count'
            }).reset_index()
            branch_summary.columns = ['branch', 'loan_count', 'total_amount', 'repayment_rate']
            
            fig_geo = px.bar(
                branch_summary.head(15), 
                x='branch', 
                y='loan_count',
                title='Top 15 Branches by Loan Volume',
                color='repayment_rate' if 'repay_good' in df.columns else None
            )
            fig_geo.update_xaxes(tickangle=45)
            fig_geo.update_layout(height=400)
            charts['geographic_distribution'] = fig_geo
        
        # 3. Age vs Income Analysis
        if all(col in df.columns for col in ['age', 'income', 'loan_amount']):
            fig_age_income = px.scatter(
                df,
                x='age',
                y='income',
                size='loan_amount',
                color='gender' if 'gender' in df.columns else None,
                title='Age vs Income Distribution',
                labels={'income': 'Monthly Income (KES)', 'age': 'Age (years)'}
            )
            fig_age_income.update_layout(height=400)
            charts['age_income_analysis'] = fig_age_income
        
        # 4. Product Performance Matrix
        if all(col in df.columns for col in ['product', 'loan_amount', 'repay_good']):
            product_perf = df.groupby('product').agg({
                'loan_amount': ['count', 'mean'],
                'repay_good': 'mean'
            }).reset_index()
            product_perf.columns = ['product', 'count', 'avg_amount', 'repayment_rate']
            
            fig_product = px.scatter(
                product_perf,
                x='avg_amount',
                y='repayment_rate',
                size='count',
                text='product',
                title='Product Performance Matrix',
                labels={'avg_amount': 'Average Loan Amount (KES)', 'repayment_rate': 'Repayment Rate'}
            )
            fig_product.update_traces(textposition="top center")
            fig_product.update_layout(height=400)
            charts['product_performance'] = fig_product
    except Exception:
        pass
    
    return charts

def calculate_business_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate business intelligence metrics"""
    if df.empty:
        return {}
    
    # Assume average interest rate and loan terms for calculations
    avg_interest_rate = 0.15  # 15% annual
    avg_term_months = 12
    
    metrics = {}
    
    if 'loan_amount' in df.columns:
        total_portfolio = safe_float(df['loan_amount'].sum())
        metrics['total_portfolio_value'] = total_portfolio
        metrics['estimated_annual_revenue'] = total_portfolio * avg_interest_rate
        
        if 'repay_good' in df.columns:
            good_loans_value = safe_float(df[df['repay_good'] == 1]['loan_amount'].sum())
            bad_loans_value = safe_float(df[df['repay_good'] == 0]['loan_amount'].sum())
            
            metrics['performing_portfolio_value'] = good_loans_value
            metrics['non_performing_portfolio_value'] = bad_loans_value
            metrics['portfolio_at_risk'] = safe_float((bad_loans_value / total_portfolio) * 100) if total_portfolio > 0 else 0
            metrics['expected_loss'] = bad_loans_value
            metrics['net_portfolio_value'] = good_loans_value - (bad_loans_value * 0.5)  # Assuming 50% recovery
            
        if 'is_fraud' in df.columns:
            fraud_loss = safe_float(df[df['is_fraud'] == 1]['loan_amount'].sum())
            metrics['fraud_loss_amount'] = fraud_loss
            metrics['fraud_loss_percentage'] = safe_float((fraud_loss / total_portfolio) * 100) if total_portfolio > 0 else 0
    
    # Calculate risk-adjusted returns
    if all(col in df.columns for col in ['loan_amount', 'default_risk']):
        df_calc = df.copy()
        df_calc['risk_adjusted_value'] = df_calc['loan_amount'] * (1 - df_calc['default_risk'])
        metrics['risk_adjusted_portfolio_value'] = safe_float(df_calc['risk_adjusted_value'].sum())
        metrics['portfolio_risk_score'] = safe_float(df_calc['default_risk'].mean() * 100)
    
    return metrics