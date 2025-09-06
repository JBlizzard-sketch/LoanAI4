import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from utils import db, auth, synth, ml, report, ui, stats

st.set_page_config(page_title="LoanIQ", layout="wide")
ui.app_header()
auth.ensure_admin()
db.init()

if "user" not in st.session_state:
    st.session_state.user = None
if "client_df" not in st.session_state:
    st.session_state.client_df = None
if "predictions" not in st.session_state:
    st.session_state.predictions = None

def login_panel():
    tab_login, tab_register = st.tabs(["Login", "Register"])
    with tab_login:
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.button("Login"):
            ok, msg, user = auth.login(u, p)
            if ok and user:
                st.session_state.user = user
                st.success("Welcome, " + user["username"])
                st.rerun()
            else:
                st.error(msg)
    with tab_register:
        u = st.text_input("New Username")
        p = st.text_input("New Password", type="password")
        if st.button("Create Account"):
            ok, msg, user = auth.register(u, p)
            if ok and user:
                st.session_state.user = user
                st.success("Registered & logged in as " + u)
                st.rerun()
            else:
                st.error(msg)

def predict_and_score(df: pd.DataFrame):
    # load deployed model; if none, quick-train on provided df
    deployed = ml.load_deployed()
    if not deployed:
        with st.spinner("No deployed model found. Training quick baseline..."):
            ml.train_and_version(df, families=["LogReg","RF","GB"])
        deployed = ml.load_deployed()
    
    if not deployed:
        st.error("Could not train or load a model. Please check the data format.")
        return df, {"model": "None", "AUC": 0}
        
    fam, ver, metrics, path = deployed
    import joblib
    model = joblib.load(path)
    X = df.copy()
    
    # Ensure we have the target column for transformer shape
    if "repay_good" not in X.columns:
        X["repay_good"] = 1  # dummy for transformer shape; will be ignored
    
    # Standardize column names for compatibility
    if "status" in X.columns and "loan_status" not in X.columns:
        X["loan_status"] = X["status"]  # Create loan_status from status for compatibility
    
    # Try prediction with error handling
    try:
        # Remove target column before prediction
        X_pred = X.drop(columns=["repay_good"], errors="ignore")
        proba = model.predict_proba(X_pred)[:,1]
        eligibility = (proba >= 0.55).astype(int)
        # simple credit score scaled 300-900
        score = (proba*600 + 300).astype(int)
        out = df.copy()
        out["default_risk"] = 1 - proba
        out["credit_score"] = score
        out["eligible"] = eligibility
        return out, {"model": f"{fam} v{ver}", **metrics}
    except Exception as e:
        # If prediction fails, retrain model with current data
        st.warning(f"Model compatibility issue. Retraining with current data...")
        ml.train_and_version(df, families=["LogReg","RF","GB"])
        deployed = ml.load_deployed()
        
        if deployed:
            fam, ver, metrics, path = deployed
            model = joblib.load(path)
            X_pred = X.drop(columns=["repay_good"], errors="ignore")
            proba = model.predict_proba(X_pred)[:,1]
            eligibility = (proba >= 0.55).astype(int)
            score = (proba*600 + 300).astype(int)
            out = df.copy()
            out["default_risk"] = 1 - proba
            out["credit_score"] = score
            out["eligible"] = eligibility
            return out, {"model": f"{fam} v{ver} (retrained)", **metrics}
        else:
            # Fallback: return data with dummy predictions
            out = df.copy()
            out["default_risk"] = 0.3  # Default risk
            out["credit_score"] = 650  # Default score
            out["eligible"] = 1  # Default eligible
            return out, {"model": "Fallback", "AUC": 0}

def client_overview_page():
    """Basic overview and data upload - Page 1"""
    ui.section_header("📊 Portfolio Overview", "Quick summary and data management")
    
    # Data upload section
    col1, col2 = st.columns([3, 2])
    
    with col1:
        ui.section_header("📤 Data Upload")
        up = st.file_uploader("Upload your loan portfolio (CSV/Excel)", type=["csv","xlsx"], help="Upload your loan book data for analysis")
        if up is not None:
            df = pd.read_csv(up) if up.name.endswith(".csv") else pd.read_excel(up)
            st.session_state.client_df = df
            # Automatically predict and score after upload
            with st.spinner("Processing your data and generating insights..."):
                preds, model_meta = predict_and_score(df)
                st.session_state.predictions = preds
            ui.success_alert(f"Successfully loaded and analyzed {df.shape[0]} loan records!")
            st.rerun()  # Navigate to insights immediately
    
    with col2:
        ui.section_header("🧪 Quick Test Data")
        if st.button("🚀 Generate Sample Data", help="Create sample loan data for testing"):
            with st.spinner("Generating realistic sample data..."):
                df = synth.generate(n=500, female_bias=0.64, fraud_rate=0.03, seed=42)
                st.session_state.client_df = df
                # Auto-predict for sample data too
                preds, model_meta = predict_and_score(df)
                st.session_state.predictions = preds
            ui.success_alert("Sample data generated and analyzed!")
            st.rerun()

    if st.session_state.client_df is None:
        ui.info_alert("📋 No data loaded yet. Upload your loan portfolio or generate sample data to get started.")
        return

    df = st.session_state.client_df
    
    # Basic stats summary for overview page
    if st.session_state.predictions is not None:
        preds = st.session_state.predictions
        # Quick KPI cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            ui.metric_card("Total Loans", f"{len(preds):,}", color="#11998e")
        with col2:
            avg_amount = preds['loan_amount'].mean() if 'loan_amount' in preds.columns else 0
            ui.metric_card("Avg Loan Amount", f"KES {avg_amount:,.0f}", color="#667eea")
        with col3:
            eligible_rate = preds['eligible'].mean() * 100 if 'eligible' in preds.columns else 0
            ui.metric_card("Eligibility Rate", f"{eligible_rate:.1f}%", color="#ff6b6b")
        with col4:
            avg_score = preds['credit_score'].mean() if 'credit_score' in preds.columns else 0
            ui.metric_card("Avg Credit Score", f"{avg_score:.0f}", color="#f9ca24")
    
    # Show data preview
    st.markdown("### 👀 Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    if st.session_state.predictions is None and st.button("🧠 Analyze Portfolio"):
        with st.spinner("Running credit scoring analysis..."):
            preds, model_meta = predict_and_score(df)
            st.session_state.predictions = preds
            ui.success_alert("Analysis complete! Check other tabs for detailed insights.")
            st.rerun()

def client_analytics_page():
    """Advanced analytics and visualizations - Page 2"""
    ui.section_header("🔍 Advanced Portfolio Analytics", "Deep insights into your loan portfolio")
    
    if st.session_state.predictions is None:
        ui.info_alert("Please upload and analyze data in the Overview tab first.")
        return

    preds = st.session_state.predictions
    
    # Calculate comprehensive statistics
    portfolio_stats = stats.get_loan_portfolio_stats(preds)
    demographic_stats = stats.get_demographic_stats(preds)
    geographic_stats = stats.get_geographic_stats(preds)
    product_stats = stats.get_product_stats(preds)
    business_metrics = stats.calculate_business_metrics(preds)
    
    # Key Performance Indicators
    kpi_data = [
        {"title": "Average Credit Score", "value": f"{portfolio_stats.get('avg_credit_score', 0):.0f}", "subtitle": "Portfolio Average"},
        {"title": "Eligibility Rate", "value": f"{portfolio_stats.get('eligibility_rate', 0):.1f}%", "subtitle": "Approved Applications"},
        {"title": "Default Risk", "value": f"{portfolio_stats.get('avg_default_risk', 0):.1f}%", "subtitle": "Average Risk Level"},
        {"title": "Portfolio Value", "value": f"KES {business_metrics.get('total_portfolio_value', 0):,.0f}", "subtitle": "Total Loan Amount"},
        {"title": "Repayment Rate", "value": f"{portfolio_stats.get('repayment_rate', 0):.1f}%", "subtitle": "Success Rate"},
        {"title": "Fraud Detection", "value": f"{portfolio_stats.get('total_fraud_cases', 0)}", "subtitle": "Cases Identified"},
        {"title": "High Credit Score", "value": f"{portfolio_stats.get('high_credit_score_rate', 0):.1f}%", "subtitle": "Score ≥ 700"},
        {"title": "Portfolio at Risk", "value": f"{business_metrics.get('portfolio_at_risk', 0):.1f}%", "subtitle": "Non-performing"}
    ]
    ui.stats_grid(kpi_data, cols=4)

    # Advanced Analytics Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Portfolio", "👥 Demographics", "🌍 Geographic", "📦 Products"])
    
    with tab1:
        ui.section_header("Portfolio Performance Analysis")
        
        # Portfolio metrics in columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### 📊 Loan Distribution")
            dist_data = [
                {"title": "Total Loans", "value": f"{portfolio_stats.get('total_loans', 0):,}"},
                {"title": "Average Amount", "value": f"KES {portfolio_stats.get('avg_loan_amount', 0):,.0f}"},
                {"title": "Median Amount", "value": f"KES {portfolio_stats.get('median_loan_amount', 0):,.0f}"},
                {"title": "Standard Deviation", "value": f"KES {portfolio_stats.get('std_loan_amount', 0):,.0f}"}
            ]
            for item in dist_data:
                ui.metric_card(item["title"], item["value"], color="#667eea")
        
        with col2:
            st.markdown("### ⚖️ Risk Assessment")
            risk_data = [
                {"title": "High Risk Rate", "value": f"{portfolio_stats.get('high_risk_rate', 0):.1f}%"},
                {"title": "Low Risk Rate", "value": f"{portfolio_stats.get('low_risk_rate', 0):.1f}%"},
                {"title": "Default Rate", "value": f"{portfolio_stats.get('default_rate', 0):.1f}%"},
                {"title": "Fraud Rate", "value": f"{portfolio_stats.get('fraud_rate', 0):.1f}%"}
            ]
            for item in risk_data:
                ui.metric_card(item["title"], item["value"], color="#ff6b6b")
        
        with col3:
            st.markdown("### 💳 Credit Scores")
            credit_data = [
                {"title": "Average Score", "value": f"{portfolio_stats.get('avg_credit_score', 0):.0f}"},
                {"title": "Median Score", "value": f"{portfolio_stats.get('median_credit_score', 0):.0f}"},
                {"title": "High Score Rate", "value": f"{portfolio_stats.get('high_credit_score_rate', 0):.1f}%"},
                {"title": "Low Score Rate", "value": f"{portfolio_stats.get('low_credit_score_rate', 0):.1f}%"}
            ]
            for item in credit_data:
                ui.metric_card(item["title"], item["value"], color="#11998e")
        
        # Charts
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            if 'credit_score' in preds.columns:
                fig_credit = ui.create_distribution_chart(preds, 'credit_score', 'Credit Score Distribution')
                st.plotly_chart(fig_credit, use_container_width=True)
        
        with col_chart2:
            if 'default_risk' in preds.columns:
                fig_risk = ui.create_distribution_chart(preds, 'default_risk', 'Default Risk Distribution')
                st.plotly_chart(fig_risk, use_container_width=True)
    
    with tab2:
        ui.section_header("Demographic Analysis")
        
        # Gender Analysis
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 👫 Gender Distribution")
            gender_data = [
                {"title": "Female Borrowers", "value": f"{demographic_stats.get('female_percentage', 0):.1f}%"},
                {"title": "Male Borrowers", "value": f"{demographic_stats.get('male_percentage', 0):.1f}%"},
                {"title": "Female Avg Loan", "value": f"KES {demographic_stats.get('female_avg_loan', 0):,.0f}"},
                {"title": "Male Avg Loan", "value": f"KES {demographic_stats.get('male_avg_loan', 0):,.0f}"}
            ]
            for item in gender_data:
                ui.metric_card(item["title"], item["value"], color="#4ecdc4")
        
        with col2:
            st.markdown("### 👶 Age Demographics")
            age_data = [
                {"title": "Average Age", "value": f"{demographic_stats.get('avg_age', 0):.1f} years"},
                {"title": "Median Age", "value": f"{demographic_stats.get('median_age', 0):.0f} years"},
                {"title": "Young Borrowers", "value": f"{demographic_stats.get('young_borrowers_rate', 0):.1f}%"},
                {"title": "Senior Borrowers", "value": f"{demographic_stats.get('senior_borrowers_rate', 0):.1f}%"}
            ]
            for item in age_data:
                ui.metric_card(item["title"], item["value"], color="#45b7d1")
    
    with tab3:
        ui.section_header("Geographic Distribution")
        
        if geographic_stats:
            geo_data = [
                {"title": "Total Branches", "value": f"{geographic_stats.get('total_branches', 0)}"},
                {"title": "Top Branch (Volume)", "value": geographic_stats.get('top_branch_by_volume', 'N/A')},
                {"title": "Branch Concentration", "value": f"{geographic_stats.get('branch_concentration', 0):.1f}%"},
                {"title": "Avg Loans/Branch", "value": f"{geographic_stats.get('avg_loans_per_branch', 0):.0f}"}
            ]
            ui.stats_grid(geo_data, cols=4)
    
    with tab4:
        ui.section_header("Product Performance")
        
        if product_stats:
            product_data = [
                {"title": "Total Products", "value": f"{product_stats.get('total_products', 0)}"},
                {"title": "Most Popular", "value": product_stats.get('most_popular_product', 'N/A')},
                {"title": "Highest Value", "value": product_stats.get('highest_value_product', 'N/A')},
                {"title": "Safest Product", "value": product_stats.get('safest_product', 'N/A')}
            ]
            ui.stats_grid(product_data, cols=4)

def client_risk_page():
    """Risk assessment and fraud detection - Page 3"""
    ui.section_header("💰 Risk Assessment & Fraud Detection", "Advanced risk analysis and anomaly detection")
    
    if st.session_state.predictions is None:
        ui.info_alert("Please upload and analyze data in the Overview tab first.")
        return
    
    preds = st.session_state.predictions
    
    # Risk overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    avg_risk = preds['default_risk'].mean() * 100 if 'default_risk' in preds.columns else 0
    high_risk_count = ((preds['default_risk'] > 0.7).sum()) if 'default_risk' in preds.columns else 0
    fraud_count = preds['is_fraud'].sum() if 'is_fraud' in preds.columns else 0
    low_score_count = ((preds['credit_score'] < 500).sum()) if 'credit_score' in preds.columns else 0
    
    with col1:
        ui.metric_card("Avg Default Risk", f"{avg_risk:.1f}%", color="#ff6b6b")
    with col2:
        ui.metric_card("High Risk Loans", f"{high_risk_count}", color="#e74c3c")
    with col3:
        ui.metric_card("Fraud Cases", f"{fraud_count}", color="#c0392b")
    with col4:
        ui.metric_card("Low Credit Scores", f"{low_score_count}", color="#f39c12")
    
    # Risk analysis charts
    risk_tab1, risk_tab2 = st.tabs(["⚠️ Risk Analysis", "🔍 Fraud Detection"])
    
    with risk_tab1:
        if 'default_risk' in preds.columns and 'credit_score' in preds.columns:
            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                fig_risk_dist = ui.create_distribution_chart(preds, 'default_risk', 'Default Risk Distribution (%)')
                st.plotly_chart(fig_risk_dist, use_container_width=True)
            with col_chart2:
                fig_score_dist = ui.create_distribution_chart(preds, 'credit_score', 'Credit Score Distribution')
                st.plotly_chart(fig_score_dist, use_container_width=True)
    
    with risk_tab2:
        if 'is_fraud' in preds.columns:
            fraud_data = preds[preds['is_fraud'] == 1]
            if not fraud_data.empty:
                st.markdown(f"### 🚨 Detected {len(fraud_data)} Potential Fraud Cases")
                
                # Show fraud cases with key info
                fraud_display = fraud_data[['customer_name', 'loan_amount', 'credit_score', 'default_risk', 'branch']].head(20) if 'customer_name' in fraud_data.columns else fraud_data.head(20)
                st.dataframe(fraud_display, use_container_width=True)

def client_reports_page():
    """Reports and exports - Page 4"""
    ui.section_header("📈 Reports & Data Export", "Generate comprehensive reports and download data")
    
    if st.session_state.predictions is None:
        ui.info_alert("Please upload and analyze data in the Overview tab first.")
        return
    
    preds = st.session_state.predictions
    
    # Export options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.download_button(
            "📊 Download Scored Data",
            data=preds.to_csv(index=False).encode("utf-8"),
            file_name=f"loaniq_scored_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            help="Download complete dataset with predictions and scores"
        )
    
    with col2:
        # Create summary statistics CSV
        portfolio_stats = stats.get_loan_portfolio_stats(preds)
        demographic_stats = stats.get_demographic_stats(preds)
        geographic_stats = stats.get_geographic_stats(preds)
        product_stats = stats.get_product_stats(preds)
        business_metrics = stats.calculate_business_metrics(preds)
        
        summary_data = {
            'Portfolio Statistics': portfolio_stats,
            'Demographic Statistics': demographic_stats,
            'Geographic Statistics': geographic_stats,
            'Product Statistics': product_stats,
            'Business Metrics': business_metrics
        }
        import json
        summary_json = json.dumps(summary_data, indent=2, default=str)
        st.download_button(
            "📈 Download Analytics Report",
            data=summary_json.encode("utf-8"),
            file_name=f"loaniq_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            help="Download comprehensive analytics summary"
        )
    
    with col3:
        # Enhanced PDF report
        if st.button("📄 Generate Executive Report", help="Create comprehensive PDF report with all analytics"):
            enhanced_summary = {
                "date": datetime.today().date().isoformat(),
                "total_records": preds.shape[0],
                "portfolio_value": f"KES {business_metrics.get('total_portfolio_value', 0):,.0f}",
                "avg_credit_score": f"{portfolio_stats.get('avg_credit_score', 0):.0f}",
                "eligibility_rate": f"{portfolio_stats.get('eligibility_rate', 0):.1f}%",
                "repayment_rate": f"{portfolio_stats.get('repayment_rate', 0):.1f}%",
                "default_risk": f"{portfolio_stats.get('avg_default_risk', 0):.1f}%",
                "fraud_cases": portfolio_stats.get('total_fraud_cases', 0),
                "total_branches": geographic_stats.get('total_branches', 0),
                "portfolio_at_risk": f"{business_metrics.get('portfolio_at_risk', 0):.1f}%",
            }
            pdf_bytes = report.build_credit_report(st.session_state.user, enhanced_summary)
            st.download_button(
                "📄 Download Executive Report", 
                data=pdf_bytes, 
                file_name=f"loaniq_executive_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf", 
                mime="application/pdf",
                help="Comprehensive PDF report for executives"
            )

def client_lookup_page():
    """Individual client lookup and analysis - Page 5"""
    ui.section_header("👥 Client Lookup & Analysis", "Search and analyze individual loan clients")
    
    if st.session_state.predictions is None:
        ui.info_alert("Please upload and analyze data in the Overview tab first.")
        return
    
    preds = st.session_state.predictions
    
    # Client search interface
    search_col1, search_col2 = st.columns([2, 1])
    
    with search_col1:
        search_type = st.radio("Search by:", ["Customer Name", "ID/Reg Number", "Reference Number"], horizontal=True)
        
        if search_type == "Customer Name":
            search_key = 'customer_name'
            available_values = preds['customer_name'].tolist() if 'customer_name' in preds.columns else []
        elif search_type == "ID/Reg Number":
            search_key = 'id_reg_number'
            available_values = preds['id_reg_number'].tolist() if 'id_reg_number' in preds.columns else []
        else:
            search_key = 'ref_number'
            available_values = preds['ref_number'].tolist() if 'ref_number' in preds.columns else []
        
        if available_values:
            selected_client = st.selectbox(f"Select {search_type}:", available_values)
        else:
            selected_client = st.text_input(f"Enter {search_type}:")
    
    with search_col2:
        st.markdown("### 🔍 Quick Stats")
        ui.metric_card("Total Clients", f"{len(preds):,}", color="#11998e")
        if 'customer_name' in preds.columns:
            unique_clients = preds['customer_name'].nunique()
            ui.metric_card("Unique Clients", f"{unique_clients:,}", color="#667eea")
    
    # Client analysis
    if selected_client and search_key in preds.columns:
        client_data = preds[preds[search_key] == selected_client]
        
        if not client_data.empty:
            client_record = client_data.iloc[0]
            
            # Client overview
            st.markdown(f"### 💼 Client Profile: {client_record.get('customer_name', 'N/A')}")
            
            # Client details in columns
            detail_col1, detail_col2, detail_col3, detail_col4 = st.columns(4)
            
            with detail_col1:
                ui.metric_card("Credit Score", f"{client_record.get('credit_score', 0):.0f}", color="#11998e")
                ui.metric_card("Loan Amount", f"KES {client_record.get('loan_amount', 0):,.0f}", color="#667eea")
            
            with detail_col2:
                ui.metric_card("Default Risk", f"{client_record.get('default_risk', 0)*100:.1f}%", color="#ff6b6b")
                ui.metric_card("Eligibility", "Eligible" if client_record.get('eligible', 0) else "Not Eligible", color="#f9ca24")
            
            with detail_col3:
                ui.metric_card("Branch", f"{client_record.get('branch', 'N/A')}", color="#4ecdc4")
                ui.metric_card("Product", f"{client_record.get('product', 'N/A')}", color="#45b7d1")
            
            with detail_col4:
                ui.metric_card("Loan Health", f"{client_record.get('loan_health', 'N/A')}", color="#96ceb4")
                ui.metric_card("Status", f"{client_record.get('status', 'N/A')}", color="#ffeaa7")
            
            # Loan recommendation
            st.markdown("### 🎯 Loan Recommendation")
            
            credit_score = client_record.get('credit_score', 0)
            current_amount = client_record.get('loan_amount', 0)
            risk_level = client_record.get('default_risk', 1)
            
            # Calculate recommended loan limit based on credit score and risk
            if credit_score >= 750:
                max_multiplier = 3.0
                risk_category = "🟢 Low Risk"
            elif credit_score >= 650:
                max_multiplier = 2.0
                risk_category = "🟡 Medium Risk"
            elif credit_score >= 500:
                max_multiplier = 1.2
                risk_category = "🟠 High Risk"
            else:
                max_multiplier = 0.8
                risk_category = "🔴 Very High Risk"
            
            # Adjust for risk level
            risk_adjustment = max(0.5, 1 - risk_level)
            recommended_limit = current_amount * max_multiplier * risk_adjustment
            
            rec_col1, rec_col2, rec_col3 = st.columns(3)
            
            with rec_col1:
                ui.metric_card("Risk Category", risk_category, color="#e17055")
            with rec_col2:
                ui.metric_card("Current Limit", f"KES {current_amount:,.0f}", color="#74b9ff")
            with rec_col3:
                ui.metric_card("Recommended Limit", f"KES {recommended_limit:,.0f}", color="#55a3ff")
            
            # Additional recommendations
            st.markdown("#### 📄 Recommendations:")
            
            recommendations = []
            
            if credit_score < 500:
                recommendations.append("⚠️ Consider requiring additional collateral or guarantor")
            if risk_level > 0.7:
                recommendations.append("🚨 High default risk - recommend closer monitoring")
            if client_record.get('is_fraud', 0):
                recommendations.append("🚨 FRAUD ALERT - Requires manual review")
            if credit_score >= 700:
                recommendations.append("🎆 Excellent candidate for premium products")
            if recommended_limit > current_amount * 1.5:
                recommendations.append(f"📈 Client qualifies for loan increase to KES {recommended_limit:,.0f}")
            
            if recommendations:
                for rec in recommendations:
                    st.markdown(f"- {rec}")
            else:
                st.markdown("- ✅ Standard lending terms apply")
            
            # Show full client record
            with st.expander("📁 Full Client Record"):
                st.dataframe(client_data.T, use_container_width=True)
        
        else:
            ui.error_alert(f"No client found with {search_type}: {selected_client}")

def client_dashboard():
    """Main client dashboard with 5 focused pages"""
    if st.session_state.user:
        ui.role_tag(st.session_state.user["role"])
    
    # Create 5 focused pages for client experience
    page1, page2, page3, page4, page5 = st.tabs(["📊 Overview", "🔍 Analytics", "💰 Risk Assessment", "📈 Reports", "👥 Client Lookup"])
    
    with page1:
        client_overview_page()
    
    with page2:
        client_analytics_page()
    
    with page3:
        client_risk_page()
    
    with page4:
        client_reports_page()
    
    with page5:
        client_lookup_page()

def admin_only():
    return st.session_state.user and st.session_state.user["role"] == "admin"

def admin_sandbox():
    if not admin_only():
        st.error("Access denied")
    else:
        ui.admin_badge()
        ui.section_header("🔧 Advanced Administration & Analytics Hub", "Complete system monitoring, model management, and business intelligence")
        
        # System Health Overview
        system_stats = stats.get_system_health_stats()
        model_stats = stats.get_model_performance_stats()
        
        if system_stats:
            ui.section_header("📊 System Health Dashboard")
            health_data = [
                {"title": "Total Users", "value": f"{system_stats.get('total_users', 0)}", "subtitle": "Registered"},
                {"title": "Client Users", "value": f"{system_stats.get('client_users', 0)}", "subtitle": "Active Clients"},
                {"title": "Total Logins", "value": f"{system_stats.get('total_logins', 0)}", "subtitle": "All Time"},
                {"title": "Recent Activity", "value": f"{system_stats.get('recent_activity_week', 0)}", "subtitle": "Last 7 Days"},
                {"title": "Total Models", "value": f"{model_stats.get('total_models', 0)}", "subtitle": "Trained"},
                {"title": "Deployed Models", "value": f"{model_stats.get('deployed_models', 0)}", "subtitle": "Active"},
                {"title": "Model Families", "value": f"{model_stats.get('model_families', 0)}", "subtitle": "Available"},
                {"title": "Best Model AUC", "value": f"{model_stats.get('best_model_auc', 0):.3f}", "subtitle": "Performance"}
            ]
            ui.stats_grid(health_data, cols=4)
        
        # Basic admin functionality for now
        with st.expander("🏭 Data Generation Laboratory", expanded=True):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("### 🎛️ Generation Parameters")
                n = st.slider("Number of Records", 1000, 25000, 5000, 1000)
                female_bias = st.slider("Women Borrower Percentage", 0.3, 0.9, 0.64, 0.01)
                fraud = st.slider("Fraud Injection Rate", 0.0, 0.3, 0.03, 0.005)
                
                # Business configuration
                st.markdown("### 🏢 Business Configuration")
                col_biz1, col_biz2 = st.columns(2)
                with col_biz1:
                    num_branches = st.number_input("Number of Branches", min_value=1, max_value=100, value=25)
                    daily_disbursement = st.number_input("Daily Loan Disbursement Target", min_value=1, max_value=1000, value=50)
                
                with col_biz2:
                    avg_loan_size = st.number_input("Average Loan Size (KES)", min_value=1000, max_value=200000, value=25000)
                    business_days_per_month = st.number_input("Business Days/Month", min_value=15, max_value=30, value=22)
                
                # Calculate realistic dataset size based on business params
                monthly_loans = daily_disbursement * business_days_per_month
                recommended_n = min(n, monthly_loans * 12)  # One year of data
                
                if recommended_n != n:
                    st.info(f"Adjusted dataset size to {recommended_n} based on business parameters (1 year of loans)")
                    n = recommended_n
            
            with col2:
                st.markdown("### 📋 Dataset Specifications")
                spec_data = [
                    {"title": "Est. File Size", "value": f"{(n * 0.5 / 1000):.1f} KB"},
                    {"title": "Generation Time", "value": f"~{(n / 5000):.1f}s"},
                    {"title": "Expected Frauds", "value": f"{int(n * fraud)}"},
                    {"title": "Female Ratio", "value": f"{female_bias:.1%}"}
                ]
                for item in spec_data:
                    ui.metric_card(item["title"], item["value"], color="#4ecdc4")
            
            if st.button("🚀 Generate Advanced Dataset", help="Generate synthetic dataset with specified parameters"):
                with st.spinner("Generating comprehensive dataset..."):
                    df = synth.generate(n=n, female_bias=female_bias, fraud_rate=fraud, seed=42)
                    st.session_state.admin_df = df
                    
                    # Calculate immediate statistics
                    quick_stats = stats.get_loan_portfolio_stats(df)
                    demographic_stats = stats.get_demographic_stats(df)
                    
                    ui.success_alert(f"Successfully generated {df.shape[0]} records with {quick_stats.get('total_fraud_cases', 0)} fraud cases")
                    
                    # Show generation summary
                    col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
                    with col_sum1:
                        ui.metric_card("Total Records", f"{len(df):,}", color="#11998e")
                    with col_sum2:
                        ui.metric_card("Avg Loan Amount", f"KES {quick_stats.get('avg_loan_amount', 0):,.0f}", color="#667eea")
                    with col_sum3:
                        ui.metric_card("Female Borrowers", f"{demographic_stats.get('female_percentage', 0):.1f}%", color="#ff6b6b")
                    with col_sum4:
                        ui.metric_card("Fraud Rate", f"{quick_stats.get('fraud_rate', 0):.2f}%", color="#f9ca24")
                    
                    # Preview with enhanced display
                    st.markdown("### 👀 Dataset Preview")
                    st.dataframe(df.head(10), use_container_width=True)
                    
                    # Download options
                    col_dl1, col_dl2, col_dl3 = st.columns(3)
                    with col_dl1:
                        st.download_button(
                            "💾 Download Full Dataset", 
                            df.to_csv(index=False).encode("utf-8"), 
                            f"loaniq_synthetic_{n}rows_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            "text/csv"
                        )
                    with col_dl2:
                        # Create sample dataset
                        sample_df = df.sample(min(1000, len(df)))
                        st.download_button(
                            "📄 Download Sample (1K)", 
                            sample_df.to_csv(index=False).encode("utf-8"), 
                            f"loaniq_sample_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            "text/csv"
                        )
                    with col_dl3:
                        # Create fraud-only dataset
                        fraud_df = df[df['is_fraud'] == 1] if 'is_fraud' in df.columns else pd.DataFrame()
                        if not fraud_df.empty:
                            st.download_button(
                                "⚠️ Download Fraud Cases", 
                                fraud_df.to_csv(index=False).encode("utf-8"), 
                                f"loaniq_fraud_cases_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                "text/csv"
                            )
        
        with st.expander("🤖 ML Engine Control", expanded=False):
            if st.button("🏃‍♂️ Train Model Suite"):
                if "admin_df" not in st.session_state or st.session_state.admin_df is None:
                    st.warning("Generate dataset first; using quick default 3k rows.")
                    st.session_state.admin_df = synth.generate(n=3000, seed=11)
                res = ml.train_and_version(st.session_state.admin_df)
                st.success("Training complete.")
                st.json([{ "family": r[0], "version": r[1], "metrics": r[2], "path": r[3] } for r in res])
            
            st.write("Deployed model:")
            dep = ml.load_deployed()
            if dep:
                st.code(f"{dep[0]} v{dep[1]} | metrics={dep[2]}")
            else:
                st.info("None deployed yet.")
        
        with st.expander("🛠️ System Tools", expanded=False):
            c1,c2,c3,c4 = st.columns(4)
            with c1:
                if st.button("🔒 Backup DB"):
                    import shutil
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    shutil.copy("data/loaniq.sqlite", f"data/backup_{ts}.sqlite")
                    st.success("DB backed up.")
            with c2:
                if st.button("📋 List Models"):
                    st.json([{"family":r[0], "version": r[1], "metrics": r[2], "path": r[3], "deployed": r[4]} for r in db.list_models()])
            with c3:
                user_to_imp = st.text_input("Impersonate username")
                if st.button("🔄 Impersonate"):
                    urow = db.get_user(user_to_imp)
                    if urow:
                        st.session_state.user = {"id":urow[0],"username":urow[1],"role":urow[3]}
                        db.record_audit("admin","impersonate",user_to_imp)
                        st.success("Now impersonating: "+user_to_imp)
                        st.rerun()
                    else:
                        st.error("User not found")
            with c4:
                if st.button("🚨 Fraud Stress Test"):
                    stress_df = synth.generate(n=3000, seed=99, fraud_rate=0.2)
                    st.dataframe(stress_df[stress_df["is_fraud"]==1].head(50))
                    st.info("Use Train tab to see how models perform under higher fraud rates.")

# Main app routing
if not st.session_state.user:
    login_panel()
else:
    # Auto-redirect admin to admin panel, hide from regular users
    if st.session_state.user["role"] == "admin":
        # Admin gets admin panel by default
        choice = st.sidebar.selectbox("Navigate", ["🔧 Admin Sandbox", "📊 Client Dashboard"])
        
        if choice == "🔧 Admin Sandbox":
            admin_sandbox()
        elif choice == "📊 Client Dashboard":
            client_dashboard()
    else:
        # Regular users only see client dashboard
        client_dashboard()