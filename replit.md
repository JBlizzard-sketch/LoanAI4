# LoanIQ - Credit Scoring & Loan Risk Platform

## Overview

LoanIQ is a Streamlit-based microfinance credit scoring and loan risk analysis platform specifically designed for Kenyan microfinance institutions. The application serves two primary user roles: clients who can upload/generate loan data and receive credit scoring insights, and admins who have access to advanced ML model management and synthetic data generation capabilities. The platform focuses on the Kenyan microfinance market with culturally relevant data generation including local names, occupations, and geographic distributions.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web application with wide layout configuration
- **UI Components**: Tab-based navigation for login/registration, interactive dashboards with rich visualizations using Plotly
- **Session Management**: Browser session state management for user authentication and data persistence
- **Styling**: Custom CSS with gradient backgrounds, hover effects, and responsive design elements

### Backend Architecture
- **Application Structure**: Modular utility-based architecture with separate modules for authentication, database operations, machine learning, reporting, synthetic data generation, statistics, and UI components
- **Data Processing**: Pandas-based data manipulation with NumPy for numerical operations
- **ML Pipeline**: Scikit-learn based machine learning pipeline with support for multiple algorithms (Logistic Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM)
- **Model Management**: Joblib serialization for model persistence with versioning and deployment tracking

### Data Storage Solutions
- **Primary Database**: SQLite database with four main tables:
  - Users (authentication and role management)
  - Datasets (user data storage with metadata)
  - Models (ML model versioning and deployment status)
  - Audit (activity logging and compliance tracking)
- **File Storage**: Local file system storage for model artifacts and temporary data files
- **Data Processing**: In-memory processing with Pandas DataFrames for real-time analytics

### Authentication & Authorization
- **Password Security**: SHA-256 hashing with salt/pepper for password storage
- **Role-Based Access**: Two-tier system with 'client' and 'admin' roles
- **Session Management**: Streamlit session state for maintaining user context
- **Auto-Admin Creation**: Automatic admin account creation with default credentials
- **Audit Trail**: Comprehensive logging of user actions and system events

### Machine Learning Architecture
- **Model Training**: Automated training pipeline with cross-validation and multiple algorithm support
- **Feature Engineering**: ColumnTransformer with OneHotEncoder for categorical variables
- **Model Evaluation**: ROC-AUC, accuracy, and recall metrics for performance assessment
- **Model Deployment**: Version-controlled deployment system with rollback capabilities
- **Scoring Engine**: Real-time credit scoring (300-900 scale) with risk categorization

### Data Generation & Processing
- **Synthetic Data**: Contextually aware synthetic data generation reflecting Kenyan microfinance patterns
- **Geographic Distribution**: Coverage across 70+ Kenyan counties and towns
- **Demographic Modeling**: Gender bias modeling (62% female) and occupation-specific distributions
- **Risk Modeling**: Configurable fraud rates and repayment behavior simulation
- **Statistical Analysis**: Comprehensive portfolio analytics with safe type conversion utilities

## External Dependencies

### Core Python Libraries
- **Streamlit**: Web application framework for the entire frontend
- **Pandas**: Data manipulation and analysis backbone
- **NumPy**: Numerical computing foundation
- **Scikit-learn**: Primary machine learning library for preprocessing, modeling, and evaluation

### Machine Learning Libraries
- **XGBoost**: Advanced gradient boosting (optional dependency)
- **LightGBM**: Microsoft's gradient boosting framework (optional dependency)
- **Joblib**: Model serialization and persistence

### Visualization & Reporting
- **Plotly**: Interactive data visualizations and dashboard components
- **Matplotlib**: Static plotting capabilities
- **ReportLab**: PDF report generation for credit reports

### Database
- **SQLite3**: Built-in Python database engine for data persistence
- **No external database dependencies**: Self-contained data storage solution

### Infrastructure
- **Local File System**: Model storage and temporary file handling
- **Environment Variables**: Configuration management (LOANIQ_DB path)
- **No external APIs**: Fully self-contained application without third-party service dependencies