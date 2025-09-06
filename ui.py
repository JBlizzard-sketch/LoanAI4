import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, Any, List

def app_header():
    st.markdown("""
    <style>
      .block-container{padding-top:1.2rem;}
      .stButton>button { 
        border-radius:12px; 
        padding:0.7rem 1.2rem; 
        font-weight:600;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
      }
      .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
      }
      .stDownloadButton>button { 
        border-radius:12px;
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        border: none;
        color: white;
        font-weight: 600;
      }
      .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
      }
      .stat-container {
        background: rgba(255,255,255,0.9);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        border-left: 5px solid #667eea;
        margin: 1rem 0;
      }
      .admin-badge {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        padding: 0.8rem 1.5rem;
        border-radius: 25px;
        display: inline-block;
        font-weight: 600;
        color: #333;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
      }
      .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
      }
      .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
      }
      .stTabs [data-baseweb="tab"] {
        padding: 12px 20px;
        border-radius: 10px;
        background: rgba(102, 126, 234, 0.1);
        border: 2px solid transparent;
      }
      .stTabs [data-baseweb="tab"]:hover {
        background: rgba(102, 126, 234, 0.2);
        border-color: #667eea;
      }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="
            font-size: 3rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        ">📊 LoanIQ</h1>
        <p style="
            font-size: 1.2rem;
            color: #666;
            margin: 0;
            font-weight: 300;
        ">Advanced Credit Risk & Loan Analytics Platform</p>
    </div>
    """, unsafe_allow_html=True)

def require_auth():
    if "user" not in st.session_state or st.session_state.user is None:
        st.warning("Please login to continue.")
        st.stop()

def pill(text, color="#eef6ff"):
    st.markdown(f"<span style='background:{color};padding:8px 16px;border-radius:20px;font-weight:600;display:inline-block;margin:4px;box-shadow:0 2px 10px rgba(0,0,0,0.1)'>{text}</span>", unsafe_allow_html=True)

def metric_card(title: str, value: str, subtitle: str = "", color: str = "#667eea"):
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {color} 0%, {color}aa 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        margin: 0.5rem 0;
    ">
        <h3 style="margin: 0; font-size: 2rem; font-weight: bold;">{value}</h3>
        <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; opacity: 0.9;">{title}</p>
        {f'<p style="margin: 0.2rem 0 0 0; font-size: 0.8rem; opacity: 0.7;">{subtitle}</p>' if subtitle else ''}
    </div>
    """, unsafe_allow_html=True)

def stats_grid(stats_data: List[Dict[str, str]], cols: int = 4):
    """Display statistics in a responsive grid"""
    colors = ["#667eea", "#11998e", "#ff6b6b", "#4ecdc4", "#45b7d1", "#f9ca24", "#f0932b", "#eb4d4b"]
    
    col_layout = st.columns(cols)
    for i, stat in enumerate(stats_data):
        with col_layout[i % cols]:
            color = colors[i % len(colors)]
            metric_card(
                title=stat.get('title', ''),
                value=stat.get('value', ''),
                subtitle=stat.get('subtitle', ''),
                color=color
            )

def info_row(items):
    cols = st.columns(len(items))
    for (label, value), c in zip(items, cols):
        with c: 
            metric_card(title=label, value=str(value))

def role_tag(role):
    if role == "client":
        color = "linear-gradient(135deg, #11998e 0%, #38ef7d 100%)"
        icon = "👤"
    else:
        color = "linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%)"
        icon = "🔐"
    
    st.markdown(f"""
    <div style="
        background: {color};
        padding: 0.8rem 1.5rem;
        border-radius: 25px;
        display: inline-block;
        font-weight: 600;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
    ">
        {icon} {role.title()} Access
    </div>
    """, unsafe_allow_html=True)

def create_gauge_chart(value: float, title: str, max_val: float = 100):
    """Create a gauge chart for KPIs"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        delta = {'reference': max_val * 0.8},
        gauge = {
            'axis': {'range': [None, max_val]},
            'bar': {'color': "#667eea"},
            'steps': [
                {'range': [0, max_val * 0.5], 'color': "lightgray"},
                {'range': [max_val * 0.5, max_val * 0.8], 'color': "gray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': max_val * 0.9
            }
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def create_trend_chart(data: pd.DataFrame, x_col: str, y_col: str, title: str):
    """Create a trend line chart"""
    fig = px.line(data, x=x_col, y=y_col, title=title)
    fig.update_traces(line_color='#667eea', line_width=3)
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#333',
        title_font_size=16,
        title_font_color='#333'
    )
    return fig

def create_distribution_chart(data: pd.DataFrame, column: str, title: str):
    """Create a distribution chart with enhanced styling"""
    fig = px.histogram(data, x=column, title=title, nbins=30)
    fig.update_traces(marker_color='#667eea', opacity=0.7)
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#333',
        title_font_size=16,
        title_font_color='#333'
    )
    return fig

def section_header(title: str, subtitle: str = ""):
    """Create a styled section header"""
    st.markdown(f"""
    <div style="
        padding: 1rem 0;
        border-bottom: 3px solid #667eea;
        margin: 2rem 0 1rem 0;
    ">
        <h2 style="
            color: #333;
            margin: 0;
            font-size: 1.8rem;
            font-weight: 600;
        ">{title}</h2>
        {f'<p style="color: #666; margin: 0.5rem 0 0 0; font-size: 1rem;">{subtitle}</p>' if subtitle else ''}
    </div>
    """, unsafe_allow_html=True)

def admin_badge():
    st.markdown("""
    <div class="admin-badge">
        🔐 Admin Sandbox - Advanced Analytics & Control Panel
    </div>
    """, unsafe_allow_html=True)

def success_alert(message: str):
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    ">
        ✅ {message}
    </div>
    """, unsafe_allow_html=True)

def error_alert(message: str):
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    ">
        ❌ {message}
    </div>
    """, unsafe_allow_html=True)

def info_alert(message: str):
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #45b7d1 0%, #96c93f 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    ">
        ℹ️ {message}
    </div>
    """, unsafe_allow_html=True)