# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Customer Churn Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)    

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #374151;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
    }
    .positive-change {
        color: #10B981;
        font-weight: 600;
    }
    .negative-change {
        color: #EF4444;
        font-weight: 600;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2c3e50 0%, #1a1a2e 100%);
    }
    .sidebar-header {
        color: white;
        font-size: 1.5rem;
        font-weight: 600;
        padding: 1rem;
        text-align: center;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Sample data generation function
def generate_sample_data():
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    
    data = {
        'date': np.random.choice(dates, 5000),
        'customer_id': np.arange(1000, 6000),
        'tenure': np.random.randint(1, 72, 5000),
        'monthly_charges': np.random.uniform(20, 120, 5000),
        'total_charges': np.random.uniform(100, 8000, 5000),
        'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], 5000, p=[0.55, 0.30, 0.15]),
        'payment_method': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], 5000),
        'paperless_billing': np.random.choice(['Yes', 'No'], 5000, p=[0.7, 0.3]),
        'churn': np.random.choice(['Yes', 'No'], 5000, p=[0.27, 0.73]),
        'gender': np.random.choice(['Male', 'Female'], 5000),
        'senior_citizen': np.random.choice([0, 1], 5000, p=[0.8, 0.2]),
        'partner': np.random.choice(['Yes', 'No'], 5000, p=[0.5, 0.5]),
        'dependents': np.random.choice(['Yes', 'No'], 5000, p=[0.3, 0.7]),
        'phone_service': np.random.choice(['Yes', 'No'], 5000, p=[0.9, 0.1]),
        'multiple_lines': np.random.choice(['Yes', 'No', 'No phone service'], 5000, p=[0.4, 0.5, 0.1]),
        'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], 5000, p=[0.35, 0.45, 0.2]),
        'online_security': np.random.choice(['Yes', 'No', 'No internet service'], 5000, p=[0.3, 0.6, 0.1]),
        'online_backup': np.random.choice(['Yes', 'No', 'No internet service'], 5000, p=[0.3, 0.6, 0.1]),
        'device_protection': np.random.choice(['Yes', 'No', 'No internet service'], 5000, p=[0.3, 0.6, 0.1]),
        'tech_support': np.random.choice(['Yes', 'No', 'No internet service'], 5000, p=[0.3, 0.6, 0.1]),
        'streaming_tv': np.random.choice(['Yes', 'No', 'No internet service'], 5000, p=[0.4, 0.5, 0.1]),
        'streaming_movies': np.random.choice(['Yes', 'No', 'No internet service'], 5000, p=[0.4, 0.5, 0.1]),
    }
    
    df = pd.DataFrame(data)
    df['month'] = df['date'].dt.to_period('M').astype(str)
    return df

# Sidebar Navigation
with st.sidebar:
    st.markdown('<div class="sidebar-header">📊 Churn Analytics</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Navigation buttons with icons
    page = st.radio(
        "Navigation",
        ["Dashboard Overview", "Customer Analysis", "Churn Predictions", "Retention Strategies", "Data Management"],
        key="navigation"
    )
    
    st.markdown("---")
    
    # Filters for the entire app
    st.markdown("### 🔍 Filters")
    
    df = generate_sample_data()
    
    # Date range filter
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()
    date_range = st.date_input(
        "Select Date Range",
        [min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )
    
    # Contract type filter
    contract_types = st.multiselect(
        "Contract Type",
        options=df['contract_type'].unique(),
        default=df['contract_type'].unique()
    )
    
    # Payment method filter
    payment_methods = st.multiselect(
        "Payment Method",
        options=df['payment_method'].unique(),
        default=df['payment_method'].unique()
    )
    
    st.markdown("---")
    
    # Info section
    st.markdown("### ℹ️ Information")
    st.info("""
    This dashboard provides:
    - Real-time churn analytics
    - Customer segmentation
    - Predictive insights
    - Retention strategies
    """)
    
    st.markdown("---")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# Apply filters
if len(date_range) == 2:
    mask = (
        (df['date'].dt.date >= date_range[0]) &
        (df['date'].dt.date <= date_range[1]) &
        (df['contract_type'].isin(contract_types)) &
        (df['payment_method'].isin(payment_methods))
    )
    df_filtered = df[mask]
else:
    df_filtered = df.copy()

# Main content based on selected page
if page == "Dashboard Overview":
    # Dashboard Overview Page
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown('<div class="main-header">Customer Churn Analytics Dashboard</div>', unsafe_allow_html=True)
    
    with col2:
        st.metric("Total Customers", f"{len(df_filtered):,}")
    
    with col3:
        churn_rate = (df_filtered['churn'] == 'Yes').mean() * 100
        st.metric("Churn Rate", f"{churn_rate:.1f}%")
    
    # Key Metrics Row
    st.markdown("---")
    st.markdown('<div class="sub-header">📈 Key Performance Indicators</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_revenue = df_filtered['total_charges'].sum()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Revenue</div>
            <div class="metric-value">${total_revenue:,.0f}</div>
            <div style="margin-top: 0.5rem; font-size: 0.9rem;">Lifetime value</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_tenure = df_filtered['tenure'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg Customer Tenure</div>
            <div class="metric-value">{avg_tenure:.1f} months</div>
            <div style="margin-top: 0.5rem; font-size: 0.9rem;">Retention indicator</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        churned_customers = (df_filtered['churn'] == 'Yes').sum()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Churned Customers</div>
            <div class="metric-value">{churned_customers:,}</div>
            <div style="margin-top: 0.5rem; font-size: 0.9rem;">This period</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        retention_rate = 100 - churn_rate
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Retention Rate</div>
            <div class="metric-value">{retention_rate:.1f}%</div>
            <div style="margin-top: 0.5rem; font-size: 0.9rem;">Customer loyalty</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Charts Row
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="sub-header">📅 Monthly Churn Trend</div>', unsafe_allow_html=True)
        
        # Monthly churn trend
        monthly_data = df_filtered.groupby('month').agg({
            'customer_id': 'count',
            'churn': lambda x: (x == 'Yes').sum()
        }).rename(columns={'customer_id': 'total_customers', 'churn': 'churned'})
        monthly_data['churn_rate'] = (monthly_data['churned'] / monthly_data['total_customers'] * 100)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly_data.index,
            y=monthly_data['churn_rate'],
            mode='lines+markers',
            name='Churn Rate',
            line=dict(color='#FF6B6B', width=3),
            fill='tozeroy',
            fillcolor='rgba(255, 107, 107, 0.1)'
        ))
        
        fig.update_layout(
            height=400,
            plot_bgcolor='white',
            xaxis_title='Month',
            yaxis_title='Churn Rate (%)',
            showlegend=True,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<div class="sub-header">📊 Churn by Contract Type</div>', unsafe_allow_html=True)
        
        # Churn by contract type
        contract_churn = df_filtered.groupby('contract_type').agg({
            'customer_id': 'count',
            'churn': lambda x: (x == 'Yes').sum()
        })
        contract_churn['churn_rate'] = (contract_churn['churn'] / contract_churn['customer_id'] * 100)
        contract_churn = contract_churn.sort_values('churn_rate', ascending=False)
        
        fig = px.bar(
            contract_churn,
            x=contract_churn.index,
            y='churn_rate',
            color='churn_rate',
            color_continuous_scale='Reds',
            text=contract_churn['churn_rate'].round(1).astype(str) + '%'
        )
        
        fig.update_layout(
            height=400,
            plot_bgcolor='white',
            xaxis_title='Contract Type',
            yaxis_title='Churn Rate (%)',
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Bottom Row
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="sub-header">👥 Customer Segmentation</div>', unsafe_allow_html=True)
        
        # Customer demographics
        demo_data = pd.DataFrame({
            'Segment': ['New Customers', 'At Risk', 'Loyal', 'VIP'],
            'Count': [1500, 800, 2500, 200],
            'Churn Risk': ['Low', 'High', 'Low', 'Very Low']
        })
        
        fig = px.pie(
            demo_data,
            values='Count',
            names='Segment',
            color='Churn Risk',
            color_discrete_map={
                'Low': '#10B981',
                'High': '#EF4444',
                'Very Low': '#3B82F6'
            },
            hole=0.4
        )
        
        fig.update_layout(
            height=400,
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<div class="sub-header">📋 Recent Churn Activity</div>', unsafe_allow_html=True)
        
        # Recent churned customers table
        recent_churned = df_filtered[df_filtered['churn'] == 'Yes'].sort_values('date', ascending=False).head(10)
        
        st.dataframe(
            recent_churned[['customer_id', 'tenure', 'monthly_charges', 'contract_type', 'payment_method']]
            .rename(columns={
                'customer_id': 'Customer ID',
                'tenure': 'Tenure (months)',
                'monthly_charges': 'Monthly Charge',
                'contract_type': 'Contract Type',
                'payment_method': 'Payment Method'
            }),
            height=400,
            use_container_width=True
        )

elif page == "Customer Analysis":
    # Customer Analysis Page
    st.markdown('<div class="main-header">👥 Customer Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="sub-header">Customer Demographics</div>', unsafe_allow_html=True)
    
    with col2:
        # Additional filter for customer analysis
        churn_status = st.multiselect(
            "Churn Status",
            options=['All', 'Churned', 'Retained'],
            default=['All']
        )
    
    # Demographic charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Gender distribution
        gender_data = df_filtered.groupby('gender').size()
        fig = px.pie(
            values=gender_data.values,
            names=gender_data.index,
            title='Gender Distribution',
            hole=0.3,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Senior citizen distribution
        senior_data = df_filtered.groupby('senior_citizen').size()
        senior_data.index = ['Non-Senior', 'Senior']
        fig = px.bar(
            x=senior_data.index,
            y=senior_data.values,
            title='Senior Citizen Distribution',
            color=senior_data.index,
            color_discrete_sequence=['#3B82F6', '#10B981']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Service usage analysis
    st.markdown("---")
    st.markdown('<div class="sub-header">📱 Service Usage Analysis</div>', unsafe_allow_html=True)
    
    services = ['phone_service', 'multiple_lines', 'internet_service', 'online_security']
    
    for service in services:
        col1, col2 = st.columns(2)
        
        with col1:
            service_dist = df_filtered[service].value_counts()
            fig = px.pie(
                values=service_dist.values,
                names=service_dist.index,
                title=f'{service.replace("_", " ").title()} Distribution',
                hole=0.3
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Churn rate by service
            service_churn = df_filtered.groupby(service)['churn'].apply(
                lambda x: (x == 'Yes').mean() * 100
            ).reset_index()
            service_churn.columns = ['Service', 'Churn Rate']
            
            fig = px.bar(
                service_churn,
                x='Service',
                y='Churn Rate',
                title=f'Churn Rate by {service.replace("_", " ").title()}',
                color='Churn Rate',
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig, use_container_width=True)

elif page == "Churn Predictions":
    # Churn Predictions Page
    st.markdown('<div class="main-header">🔮 Churn Predictions</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Simulated prediction model results
        high_risk = np.random.randint(100, 300)
        st.metric("High Risk Customers", f"{high_risk}", "12%")
    
    with col2:
        medium_risk = np.random.randint(300, 600)
        st.metric("Medium Risk Customers", f"{medium_risk}", "24%")
    
    with col3:
        low_risk = len(df_filtered) - high_risk - medium_risk
        st.metric("Low Risk Customers", f"{low_risk}", "64%")
    
    # Risk factors visualization
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="sub-header">📊 Top Churn Risk Factors</div>', unsafe_allow_html=True)
        
        risk_factors = pd.DataFrame({
            'Factor': ['Contract Type', 'Tenure < 6 months', 'High Monthly Charges', 
                      'Electronic Check Payment', 'No Online Security'],
            'Impact Score': [85, 78, 72, 65, 58],
            'Customers Affected': [1250, 890, 760, 1100, 520]
        })
        
        fig = px.bar(
            risk_factors,
            x='Impact Score',
            y='Factor',
            orientation='h',
            color='Impact Score',
            color_continuous_scale='Reds',
            text='Impact Score'
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<div class="sub-header">🎯 Customer Risk Segmentation</div>', unsafe_allow_html=True)
        
        # Simulate risk segmentation
        np.random.seed(42)
        risk_data = pd.DataFrame({
            'Risk Score': np.random.randn(1000) * 20 + 50,
            'Monthly Charges': np.random.uniform(20, 120, 1000),
            'Tenure': np.random.randint(1, 72, 1000)
        })
        risk_data['Risk Category'] = pd.cut(risk_data['Risk Score'], 
                                          bins=[0, 30, 70, 100], 
                                          labels=['Low', 'Medium', 'High'])
        
        fig = px.scatter(
            risk_data,
            x='Tenure',
            y='Monthly Charges',
            color='Risk Category',
            color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'},
            title='Customer Risk Distribution',
            size='Risk Score',
            hover_data=['Risk Score']
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Prediction interface
    st.markdown("---")
    st.markdown('<div class="sub-header">🔍 Predict Churn for Individual Customer</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        tenure = st.slider("Tenure (months)", 1, 72, 24)
        monthly_charges = st.slider("Monthly Charges ($)", 20, 120, 65)
    
    with col2:
        contract_type = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        payment_method = st.selectbox("Payment Method", 
                                     ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
    
    with col3:
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    
    # Simulate prediction
    if st.button("Predict Churn Risk", type="primary"):
        # Simple rule-based prediction for demo
        risk_score = 50
        
        if contract_type == "Month-to-month":
            risk_score += 20
        elif contract_type == "One year":
            risk_score += 5
        
        if payment_method == "Electronic check":
            risk_score += 15
        
        if online_security == "No":
            risk_score += 10
        
        if tenure < 6:
            risk_score += 25
        elif tenure < 12:
            risk_score += 10
        
        risk_score = min(100, max(0, risk_score))
        
        if risk_score > 70:
            risk_category = "High"
            color = "red"
        elif risk_score > 40:
            risk_category = "Medium"
            color = "orange"
        else:
            risk_category = "Low"
            color = "green"
        
        col1, col2, col3 = st.columns(3)
        with col2:
            st.markdown(f"""
            <div style="text-align: center; padding: 2rem; border: 2px solid {color}; border-radius: 10px;">
                <h2 style="color: {color};">{risk_category} Risk</h2>
                <h1 style="color: {color};">{risk_score:.0f}/100</h1>
                <p>Probability of churn: {(risk_score/100)*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)

elif page == "Retention Strategies":
    # Retention Strategies Page
    st.markdown('<div class="main-header">🎯 Retention Strategies</div>', unsafe_allow_html=True)
    
    # Strategy cards
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="padding: 1.5rem; background: #F0F9FF; border-radius: 10px; border-left: 4px solid #3B82F6;">
            <h3 style="color: #1E40AF;">🎁 Loyalty Programs</h3>
            <p><strong>Target:</strong> Customers with tenure > 24 months</p>
            <p><strong>Strategy:</strong> Offer exclusive discounts and early access to new features</p>
            <p><strong>Expected Impact:</strong> Reduce churn by 15% in this segment</p>
            <p><strong>Cost:</strong> $25 per customer</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="padding: 1.5rem; background: #FEF2F2; border-radius: 10px; border-left: 4px solid #EF4444;">
            <h3 style="color: #B91C1C;">📞 Proactive Support</h3>
            <p><strong>Target:</strong> Customers with technical service issues</p>
            <p><strong>Strategy:</strong> Proactive tech support calls and personalized training</p>
            <p><strong>Expected Impact:</strong> Reduce churn by 25% in this segment</p>
            <p><strong>Cost:</strong> $40 per customer</p>
        </div>
        """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="padding: 1.5rem; background: #F0FDF4; border-radius: 10px; border-left: 4px solid #10B981;">
            <h3 style="color: #047857;">💰 Contract Incentives</h3>
            <p><strong>Target:</strong> Month-to-month contract customers</p>
            <p><strong>Strategy:</strong> Offer 10% discount for switching to annual contracts</p>
            <p><strong>Expected Impact:</strong> Increase contract length by 8 months average</p>
            <p><strong>Cost:</strong> $15 per customer</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="padding: 1.5rem; background: #FFFBEB; border-radius: 10px; border-left: 4px solid #F59E0B;">
            <h3 style="color: #B45309;">📊 Usage Optimization</h3>
            <p><strong>Target:</strong> Customers with declining usage patterns</p>
            <p><strong>Strategy:</strong> Personalized recommendations and usage tutorials</p>
            <p><strong>Expected Impact:</strong> Increase engagement by 30%</p>
            <p><strong>Cost:</strong> $20 per customer</p>
        </div>
        """, unsafe_allow_html=True)
    
    # ROI Analysis
    st.markdown("---")
    st.markdown('<div class="sub-header">📈 ROI Analysis</div>', unsafe_allow_html=True)
    
    roi_data = pd.DataFrame({
        'Strategy': ['Loyalty Programs', 'Proactive Support', 'Contract Incentives', 'Usage Optimization'],
        'Cost per Customer': [25, 40, 15, 20],
        'Customers Targeted': [800, 500, 1200, 600],
        'Expected Retention Gain': [15, 25, 20, 18],
        'Lifetime Value Saved': [120000, 100000, 180000, 90000]
    })
    
    roi_data['Total Cost'] = roi_data['Cost per Customer'] * roi_data['Customers Targeted']
    roi_data['ROI'] = ((roi_data['Lifetime Value Saved'] - roi_data['Total Cost']) / roi_data['Total Cost']) * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            roi_data,
            x='Strategy',
            y='ROI',
            title='ROI by Strategy (%)',
            color='ROI',
            color_continuous_scale='RdYlGn',
            text=roi_data['ROI'].round(1).astype(str) + '%'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.dataframe(
            roi_data.style.format({
                'Cost per Customer': '${:.0f}',
                'Total Cost': '${:,.0f}',
                'Lifetime Value Saved': '${:,.0f}',
                'ROI': '{:.1f}%',
                'Expected Retention Gain': '{:.0f}%'
            }).background_gradient(subset=['ROI'], cmap='RdYlGn'),
            height=400,
            use_container_width=True
        )

elif page == "Data Management":
    # Data Management Page
    st.markdown('<div class="main-header">🗃️ Data Management</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Records", f"{len(df_filtered):,}")
    
    with col2:
        st.metric("Data Completeness", "98.5%")
    
    with col3:
        st.metric("Last Updated", datetime.now().strftime("%Y-%m-%d"))
    
    # Data preview and management
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["📊 Data Preview", "📈 Data Quality", "⚙️ Data Settings"])
    
    with tab1:
        st.markdown('<div class="sub-header">Data Sample</div>', unsafe_allow_html=True)
        
        # Show data preview
        num_rows = st.slider("Number of rows to display", 10, 100, 20)
        st.dataframe(df_filtered.head(num_rows), use_container_width=True)
        
        # Data export options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Export to CSV", type="secondary"):
                csv = df_filtered.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"churn_data_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("Export to Excel", type="secondary"):
                # In a real app, you would create an Excel file
                st.success("Excel export feature would be implemented here")
        
        with col3:
            if st.button("Generate Report", type="primary"):
                st.success("Report generation started...")
    
    with tab2:
        st.markdown('<div class="sub-header">Data Quality Metrics</div>', unsafe_allow_html=True)
        
        # Data quality metrics
        quality_data = pd.DataFrame({
            'Metric': ['Completeness', 'Accuracy', 'Consistency', 'Timeliness', 'Validity'],
            'Score': [98.5, 96.2, 97.8, 99.1, 95.4],
            'Status': ['Good', 'Good', 'Good', 'Excellent', 'Fair']
        })
        
        fig = px.bar(
            quality_data,
            x='Metric',
            y='Score',
            color='Status',
            color_discrete_map={'Excellent': '#10B981', 'Good': '#3B82F6', 'Fair': '#F59E0B'},
            text=quality_data['Score'].round(1).astype(str) + '%',
            title='Data Quality Metrics'
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Missing values analysis
        st.markdown("#### Missing Values Analysis")
        missing_values = df_filtered.isnull().sum()
        missing_data = pd.DataFrame({
            'Column': missing_values.index,
            'Missing Count': missing_values.values,
            'Missing %': (missing_values.values / len(df_filtered) * 100).round(2)
        }).sort_values('Missing %', ascending=False)
        
        st.dataframe(missing_data[missing_data['Missing Count'] > 0], use_container_width=True)
    
    with tab3:
        st.markdown('<div class="sub-header">Data Configuration</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Data Sources")
            st.checkbox("CRM System", value=True)
            st.checkbox("Billing System", value=True)
            st.checkbox("Usage Analytics", value=True)
            st.checkbox("Customer Support", value=True)
            
            st.markdown("#### Update Frequency")
            update_freq = st.selectbox(
                "Data Refresh Rate",
                ["Real-time", "Hourly", "Daily", "Weekly"]
            )
        
        with col2:
            st.markdown("#### Data Retention")
            retention = st.slider("Retention Period (months)", 1, 60, 24)
            
            st.markdown("#### Backup Settings")
            st.checkbox("Daily Backups", value=True)
            st.checkbox("Weekly Full Backups", value=True)
            backup_location = st.selectbox(
                "Backup Location",
                ["Cloud Storage", "On-premises", "Both"]
            )
        
        if st.button("Save Configuration", type="primary"):
            st.success("Configuration saved successfully!")

# Footer
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)
with footer_col2:
    st.caption("© 2025 Customer Churn Analytics Dashboard | v1.0.0")