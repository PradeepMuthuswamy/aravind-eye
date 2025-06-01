import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Aravind Eye Care - Data Analysis & Prediction", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .tab-header {
        font-size: 1.8rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Load data function
@st.cache_data
def load_data():
    try:
        data = pd.read_csv("AravindEyeCareV2.csv")
        # Basic data cleaning
        data.dropna(subset=['TOT_QTY', 'PRICE'], inplace=True)
        data = data[(data['TOT_QTY'] > 0) & (data['PRICE'] > 0)]
        return data
    except FileNotFoundError:
        st.error("Error: 'AravindEyeCareV2.csv' not found. Please ensure the file is in the same directory as app.py.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading data: {e}")
        return None

# Main title
st.markdown('<h1 class="main-header">🏥 Aravind Eye Care - Data Analysis & Prediction Platform</h1>', unsafe_allow_html=True)

# Load data
data = load_data()

if data is not None:
    # Create tabs
    tab1, tab2 = st.tabs(["📊 Data Exploration", "🔮 Prediction & Analysis"])
    
    with tab1:
        st.markdown('<h2 class="tab-header">Data Exploration Dashboard</h2>', unsafe_allow_html=True)
        
        # Data overview section
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Total Records", f"{len(data):,}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Total Quantity Sold", f"{data['TOT_QTY'].sum():,}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Average Price", f"₹{data['PRICE'].mean():.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Unique Products", f"{data['PRODUCTS'].nunique()}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Data table preview
        st.subheader("📋 Data Preview")
        col1, col2 = st.columns([3, 1])
        
        with col2:
            show_rows = st.selectbox("Rows to display:", [10, 25, 50, 100], index=1)
        
        with col1:
            st.dataframe(data.head(show_rows), use_container_width=True)
        
        st.markdown("---")
        
        # Statistical summary with comprehensive metrics
        st.subheader("📈 Statistical Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Numerical Variables Summary:**")
            numerical_cols = data.select_dtypes(include=[np.number]).columns
            
            # Enhanced statistical summary
            stats_summary = data[numerical_cols].describe()
            
            # Add additional statistics
            additional_stats = pd.DataFrame(index=numerical_cols)
            additional_stats['Median'] = data[numerical_cols].median()
            additional_stats['Mode'] = data[numerical_cols].mode().iloc[0] if len(data[numerical_cols].mode()) > 0 else np.nan
            additional_stats['Skewness'] = data[numerical_cols].skew()
            additional_stats['Kurtosis'] = data[numerical_cols].kurtosis()
            
            # Combine with describe output
            enhanced_stats = pd.concat([stats_summary, additional_stats.T])
            st.dataframe(enhanced_stats, use_container_width=True)
            
            # Statistical explanations
            with st.expander("📖 Statistical Metrics Explained"):
                st.markdown("""
                **Basic Statistics:**
                - **Count**: Number of non-missing values
                - **Mean**: Average value (sum of all values ÷ count)
                - **Std**: Standard deviation (measure of data spread)
                - **Min/Max**: Smallest and largest values
                - **25%/50%/75%**: Quartiles (data divided into 4 equal parts)
                
                **Additional Metrics:**
                - **Median**: Middle value when data is sorted (50th percentile)
                - **Mode**: Most frequently occurring value
                - **Skewness**: Measure of asymmetry (0 = symmetric, >0 = right-skewed, <0 = left-skewed)
                - **Kurtosis**: Measure of tail heaviness (>0 = heavy tails, <0 = light tails)
                """)
        
        with col2:
            st.write("**Categorical Variables Summary:**")
            categorical_cols = data.select_dtypes(include=['object']).columns
            cat_summary = pd.DataFrame({
                'Column': categorical_cols,
                'Unique Values': [data[col].nunique() for col in categorical_cols],
                'Most Frequent': [data[col].mode().iloc[0] if len(data[col].mode()) > 0 else 'N/A' for col in categorical_cols],
                'Frequency': [data[col].value_counts().iloc[0] if len(data[col].value_counts()) > 0 else 0 for col in categorical_cols],
                'Missing %': [round((data[col].isnull().sum() / len(data)) * 100, 2) for col in categorical_cols]
            })
            st.dataframe(cat_summary, use_container_width=True)
            
            # Categorical explanations
            with st.expander("📖 Categorical Metrics Explained"):
                st.markdown("""
                **Categorical Statistics:**
                - **Unique Values**: Number of distinct categories
                - **Most Frequent**: Category that appears most often (mode)
                - **Frequency**: Count of the most frequent category
                - **Missing %**: Percentage of missing/null values
                
                **Interpretation Tips:**
                - High unique values suggest detailed categorization
                - Low frequency of most common category indicates even distribution
                - High missing % may require data cleaning
                """)
        
        st.markdown("---")
        
        # Visualization section
        st.subheader("📊 Data Visualizations")
        
        # Create visualization tabs
        viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
            "Distribution Plots", "Correlation Analysis", "Categorical Analysis", "Time Series"
        ])
        
        with viz_tab1:
            # Distribution analysis with explanations
            with st.expander("📖 Distribution Analysis Explained"):
                st.markdown("""
                **Understanding Distributions:**
                - **Normal Distribution**: Bell-shaped, symmetric around mean
                - **Right-skewed**: Tail extends to the right (mean > median)
                - **Left-skewed**: Tail extends to the left (mean < median)
                - **Multimodal**: Multiple peaks indicating different groups
                
                **Log Transformations:**
                - **Purpose**: Convert skewed data to more normal distribution
                - **Benefit**: Better for statistical modeling and outlier handling
                - **Interpretation**: Log values represent proportional changes
                
                **Business Insights:**
                - **Quantity distribution**: Shows demand patterns and outliers
                - **Price distribution**: Reveals pricing strategy and market segments
                """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Quantity distribution
                fig_qty = px.histogram(
                    data, x='TOT_QTY', 
                    title='Distribution of Total Quantity',
                    nbins=50,
                    color_discrete_sequence=['#1f77b4']
                )
                fig_qty.update_layout(height=400)
                st.plotly_chart(fig_qty, use_container_width=True)
                
                # Quantity statistics
                qty_stats = {
                    'Mean': data['TOT_QTY'].mean(),
                    'Median': data['TOT_QTY'].median(),
                    'Mode': data['TOT_QTY'].mode().iloc[0] if len(data['TOT_QTY'].mode()) > 0 else 'N/A',
                    'Skewness': data['TOT_QTY'].skew()
                }
                
                st.write("**Quantity Statistics:**")
                for stat, value in qty_stats.items():
                    if stat == 'Skewness':
                        if value > 0.5:
                            skew_desc = "Right-skewed (few high values)"
                        elif value < -0.5:
                            skew_desc = "Left-skewed (few low values)"
                        else:
                            skew_desc = "Approximately symmetric"
                        st.write(f"• {stat}: {value:.2f} ({skew_desc})")
                    else:
                        st.write(f"• {stat}: {value:.2f}" if isinstance(value, (int, float)) else f"• {stat}: {value}")
                
                # Log quantity distribution
                fig_log_qty = px.histogram(
                    data, x='Ln_Tot_Qty', 
                    title='Distribution of Log(Total Quantity)',
                    nbins=50,
                    color_discrete_sequence=['#ff7f0e']
                )
                fig_log_qty.update_layout(height=400)
                st.plotly_chart(fig_log_qty, use_container_width=True)
            
            with col2:
                # Price distribution
                fig_price = px.histogram(
                    data, x='PRICE', 
                    title='Distribution of Price',
                    nbins=50,
                    color_discrete_sequence=['#2ca02c']
                )
                fig_price.update_layout(height=400)
                st.plotly_chart(fig_price, use_container_width=True)
                
                # Price statistics
                price_stats = {
                    'Mean': data['PRICE'].mean(),
                    'Median': data['PRICE'].median(),
                    'Mode': data['PRICE'].mode().iloc[0] if len(data['PRICE'].mode()) > 0 else 'N/A',
                    'Skewness': data['PRICE'].skew()
                }
                
                st.write("**Price Statistics:**")
                for stat, value in price_stats.items():
                    if stat == 'Skewness':
                        if value > 0.5:
                            skew_desc = "Right-skewed (few expensive items)"
                        elif value < -0.5:
                            skew_desc = "Left-skewed (few cheap items)"
                        else:
                            skew_desc = "Approximately symmetric"
                        st.write(f"• {stat}: ₹{value:.2f} ({skew_desc})" if isinstance(value, (int, float)) else f"• {stat}: {value} ({skew_desc})")
                    else:
                        st.write(f"• {stat}: ₹{value:.2f}" if isinstance(value, (int, float)) else f"• {stat}: {value}")
                
                # Log price distribution
                fig_log_price = px.histogram(
                    data, x='Ln_Price', 
                    title='Distribution of Log(Price)',
                    nbins=50,
                    color_discrete_sequence=['#d62728']
                )
                fig_log_price.update_layout(height=400)
                st.plotly_chart(fig_log_price, use_container_width=True)
        
        with viz_tab2:
            # Correlation heatmap with enhanced analysis
            numeric_data = data.select_dtypes(include=[np.number])
            correlation_matrix = numeric_data.corr()
            
            # Correlation explanations
            with st.expander("📖 Correlation Analysis Explained"):
                st.markdown("""
                **Understanding Correlation:**
                - **Range**: -1 to +1
                - **+1**: Perfect positive correlation (as one increases, other increases)
                - **-1**: Perfect negative correlation (as one increases, other decreases)
                - **0**: No linear correlation
                
                **Interpretation Guide:**
                - **0.7 to 1.0**: Strong positive correlation
                - **0.3 to 0.7**: Moderate positive correlation
                - **-0.3 to 0.3**: Weak correlation
                - **-0.7 to -0.3**: Moderate negative correlation
                - **-1.0 to -0.7**: Strong negative correlation
                
                **Business Applications:**
                - **Price-Quantity correlation**: Reveals demand elasticity
                - **Product correlations**: Identify substitute/complementary products
                - **Regional correlations**: Understand market similarities
                """)
            
            fig_corr = px.imshow(
                correlation_matrix, 
                text_auto=True, 
                aspect="auto",
                title="Correlation Matrix of Numerical Variables",
                color_continuous_scale='RdBu',
                color_continuous_midpoint=0
            )
            fig_corr.update_layout(height=600)
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Highlight strongest correlations
            st.subheader("🔍 Key Correlation Insights")
            
            # Find strongest positive and negative correlations (excluding diagonal)
            corr_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_pairs.append({
                        'Variable 1': correlation_matrix.columns[i],
                        'Variable 2': correlation_matrix.columns[j],
                        'Correlation': correlation_matrix.iloc[i, j]
                    })
            
            corr_df = pd.DataFrame(corr_pairs)
            corr_df = corr_df.dropna()
            
            if not corr_df.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Strongest Positive Correlations:**")
                    top_positive = corr_df.nlargest(5, 'Correlation')
                    for _, row in top_positive.iterrows():
                        st.write(f"• {row['Variable 1']} ↔ {row['Variable 2']}: **{row['Correlation']:.3f}**")
                
                with col2:
                    st.write("**Strongest Negative Correlations:**")
                    top_negative = corr_df.nsmallest(5, 'Correlation')
                    for _, row in top_negative.iterrows():
                        st.write(f"• {row['Variable 1']} ↔ {row['Variable 2']}: **{row['Correlation']:.3f}**")
            
            # Scatter plot: Price vs Quantity with enhanced analysis
            col1, col2 = st.columns(2)
            
            with col1:
                fig_scatter = px.scatter(
                    data, x='PRICE', y='TOT_QTY',
                    title='Price vs Total Quantity (Linear Scale)',
                    trendline="ols",
                    color_discrete_sequence=['#1f77b4']
                )
                fig_scatter.update_layout(height=400)
                st.plotly_chart(fig_scatter, use_container_width=True)
                
                # Calculate and display correlation
                price_qty_corr = data['PRICE'].corr(data['TOT_QTY'])
                st.metric("Price-Quantity Correlation", f"{price_qty_corr:.4f}")
            
            with col2:
                fig_log_scatter = px.scatter(
                    data, x='Ln_Price', y='Ln_Tot_Qty',
                    title='Log(Price) vs Log(Total Quantity)',
                    trendline="ols",
                    color_discrete_sequence=['#ff7f0e']
                )
                fig_log_scatter.update_layout(height=400)
                st.plotly_chart(fig_log_scatter, use_container_width=True)
                
                # Calculate and display log correlation
                log_price_qty_corr = data['Ln_Price'].corr(data['Ln_Tot_Qty'])
                st.metric("Log Price-Quantity Correlation", f"{log_price_qty_corr:.4f}")
            
            # Interpretation of price-quantity relationship
            st.markdown("**💡 Price-Quantity Relationship Insights:**")
            if price_qty_corr < -0.3:
                st.success("✅ **Normal demand pattern**: Higher prices are associated with lower quantities (negative correlation)")
            elif price_qty_corr > 0.3:
                st.warning("⚠️ **Unusual pattern**: Higher prices are associated with higher quantities (positive correlation) - may indicate premium/luxury goods or supply constraints")
            else:
                st.info("ℹ️ **Weak relationship**: Price and quantity show little linear correlation - other factors may be more influential")
        
        with viz_tab3:
            # Categorical analysis with enhanced explanations
            with st.expander("📖 Categorical Analysis Explained"):
                st.markdown("""
                **Business Intelligence from Categories:**
                - **Product Analysis**: Identifies best-selling and high-value products
                - **Channel Performance**: Shows which distribution channels are most effective
                - **Geographic Insights**: Reveals regional market strength and opportunities
                - **Market Concentration**: Understanding if sales are concentrated or distributed
                
                **Key Metrics:**
                - **Total Quantity**: Overall demand and market size
                - **Average Price**: Price positioning and premium/value segments
                - **Market Share**: Relative importance of categories
                
                **Strategic Applications:**
                - **Inventory Planning**: Focus on high-volume products
                - **Pricing Strategy**: Optimize prices based on regional/channel patterns
                - **Resource Allocation**: Invest in high-performing channels/regions
                """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Sales by product with insights
                product_sales = data.groupby('PRODUCTS')['TOT_QTY'].sum().sort_values(ascending=False).head(10)
                fig_products = px.bar(
                    x=product_sales.values, 
                    y=product_sales.index,
                    orientation='h',
                    title='Top 10 Products by Total Quantity Sold',
                    labels={'x': 'Total Quantity', 'y': 'Products'}
                )
                fig_products.update_layout(height=500)
                st.plotly_chart(fig_products, use_container_width=True)
                
                # Product insights
                st.write("**Product Performance Insights:**")
                total_sales = data['TOT_QTY'].sum()
                top_product = product_sales.index[0]
                top_product_share = (product_sales.iloc[0] / total_sales) * 100
                st.write(f"• **Leading Product**: {top_product}")
                st.write(f"• **Market Share**: {top_product_share:.1f}% of total sales")
                st.write(f"• **Top 10 Products**: {(product_sales.sum() / total_sales) * 100:.1f}% of total sales")
                
                # Sales by channel with insights
                channel_sales = data.groupby('CHANNEL')['TOT_QTY'].sum().sort_values(ascending=False)
                fig_channel = px.pie(
                    values=channel_sales.values,
                    names=channel_sales.index,
                    title='Sales Distribution by Channel'
                )
                fig_channel.update_layout(height=500)
                st.plotly_chart(fig_channel, use_container_width=True)
                
                # Channel insights
                st.write("**Channel Performance Insights:**")
                top_channel = channel_sales.index[0]
                top_channel_share = (channel_sales.iloc[0] / total_sales) * 100
                st.write(f"• **Dominant Channel**: {top_channel}")
                st.write(f"• **Channel Share**: {top_channel_share:.1f}% of total sales")
                st.write(f"• **Channel Diversity**: {len(channel_sales)} different channels")
            
            with col2:
                # Sales by state with insights
                state_sales = data.groupby('STATE')['TOT_QTY'].sum().sort_values(ascending=False).head(10)
                fig_states = px.bar(
                    x=state_sales.values,
                    y=state_sales.index,
                    orientation='h',
                    title='Top 10 States by Total Quantity Sold',
                    labels={'x': 'Total Quantity', 'y': 'States'}
                )
                fig_states.update_layout(height=500)
                st.plotly_chart(fig_states, use_container_width=True)
                
                # Geographic insights
                st.write("**Geographic Performance Insights:**")
                top_state = state_sales.index[0]
                top_state_share = (state_sales.iloc[0] / total_sales) * 100
                st.write(f"• **Leading State**: {top_state}")
                st.write(f"• **State Share**: {top_state_share:.1f}% of total sales")
                st.write(f"• **Geographic Spread**: {data['STATE'].nunique()} states/regions")
                
                # Average price by product (top 10) with insights
                avg_price_product = data.groupby('PRODUCTS')['PRICE'].mean().sort_values(ascending=False).head(10)
                fig_price_product = px.bar(
                    x=avg_price_product.values,
                    y=avg_price_product.index,
                    orientation='h',
                    title='Top 10 Products by Average Price',
                    labels={'x': 'Average Price (₹)', 'y': 'Products'}
                )
                fig_price_product.update_layout(height=500)
                st.plotly_chart(fig_price_product, use_container_width=True)
                
                # Pricing insights
                st.write("**Pricing Strategy Insights:**")
                premium_product = avg_price_product.index[0]
                price_range = avg_price_product.max() - avg_price_product.min()
                st.write(f"• **Premium Product**: {premium_product}")
                st.write(f"• **Price at**: ₹{avg_price_product.iloc[0]:.2f}")
                st.write(f"• **Price Range**: ₹{price_range:.2f} between highest and lowest")
        
        with viz_tab4:
            # Time series analysis with fixed datetime conversion
            if 'YEAR' in data.columns and 'MONTH' in data.columns:
                # Convert year and month to proper datetime - Fixed version
                data_ts = data.copy()
                try:
                    # Extract numeric values using raw strings to avoid SyntaxWarning
                    data_ts['year_num'] = data_ts['YEAR'].str.extract(r'(\d+)').astype(int)
                    data_ts['month_num'] = data_ts['MONTH'].str.extract(r'(\d+)').astype(int)
                    
                    # Create datetime with proper column names
                    date_df = pd.DataFrame({
                        'year': data_ts['year_num'],
                        'month': data_ts['month_num'],
                        'day': 1
                    })
                    data_ts['date'] = pd.to_datetime(date_df)
                    
                    # Monthly sales trend
                    monthly_sales = data_ts.groupby('date').agg({
                        'TOT_QTY': 'sum',
                        'PRICE': 'mean'
                    }).reset_index()
                    
                    # Time series explanations
                    with st.expander("📖 Time Series Analysis Explained"):
                        st.markdown("""
                        **Time Series Components:**
                        - **Monthly Total Quantity**: Sum of all quantities sold per month
                        - **Monthly Average Price**: Mean price across all products per month
                        - **Trend**: Overall direction of data over time
                        - **Seasonality**: Recurring patterns within specific periods
                        
                        **Business Insights:**
                        - **Upward trends** indicate growing demand
                        - **Seasonal patterns** help plan inventory and marketing
                        - **Price vs. quantity relationships** reveal market dynamics
                        """)
                    
                    fig_ts = make_subplots(
                        rows=2, cols=1,
                        subplot_titles=('Monthly Total Quantity Sold', 'Monthly Average Price'),
                        vertical_spacing=0.1
                    )
                    
                    fig_ts.add_trace(
                        go.Scatter(x=monthly_sales['date'], y=monthly_sales['TOT_QTY'],
                                  mode='lines+markers', name='Total Quantity'),
                        row=1, col=1
                    )
                    
                    fig_ts.add_trace(
                        go.Scatter(x=monthly_sales['date'], y=monthly_sales['PRICE'],
                                  mode='lines+markers', name='Average Price', line=dict(color='orange')),
                        row=2, col=1
                    )
                    
                    fig_ts.update_layout(height=600, title_text="Time Series Analysis")
                    st.plotly_chart(fig_ts, use_container_width=True)
                    
                    # Year over year comparison
                    yearly_sales = data_ts.groupby('year_num')['TOT_QTY'].sum()
                    fig_yearly = px.bar(
                        x=yearly_sales.index,
                        y=yearly_sales.values,
                        title='Year over Year Sales Comparison',
                        labels={'x': 'Year', 'y': 'Total Quantity Sold'}
                    )
                    fig_yearly.update_layout(height=400)
                    st.plotly_chart(fig_yearly, use_container_width=True)
                    
                    # Monthly statistics
                    st.subheader("📊 Monthly Sales Statistics")
                    monthly_stats = pd.DataFrame({
                        'Metric': ['Mean Monthly Sales', 'Median Monthly Sales', 'Max Monthly Sales', 'Min Monthly Sales', 'Std Dev'],
                        'Value': [
                            f"{monthly_sales['TOT_QTY'].mean():.0f}",
                            f"{monthly_sales['TOT_QTY'].median():.0f}",
                            f"{monthly_sales['TOT_QTY'].max():.0f}",
                            f"{monthly_sales['TOT_QTY'].min():.0f}",
                            f"{monthly_sales['TOT_QTY'].std():.0f}"
                        ]
                    })
                    st.dataframe(monthly_stats, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error in time series analysis: {e}")
                    st.info("Time series analysis requires proper YEAR and MONTH column formats.")
            else:
                st.info("Time series analysis requires YEAR and MONTH columns in the dataset.")
        
        # Missing values analysis
        st.markdown("---")
        st.subheader("🔍 Data Quality Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Missing values
            missing_data = data.isnull().sum()
            missing_percent = (missing_data / len(data)) * 100
            missing_df = pd.DataFrame({
                'Column': missing_data.index,
                'Missing Count': missing_data.values,
                'Missing Percentage': missing_percent.values
            }).sort_values('Missing Count', ascending=False)
            
            st.write("**Missing Values Analysis:**")
            st.dataframe(missing_df, use_container_width=True)
        
        with col2:
            # Data types
            dtype_df = pd.DataFrame({
                'Column': data.dtypes.index,
                'Data Type': data.dtypes.values,
                'Non-Null Count': data.count().values,
                'Null Count': data.isnull().sum().values
            })
            
            st.write("**Data Types Summary:**")
            st.dataframe(dtype_df, use_container_width=True)
    
    with tab2:
        st.markdown('<h2 class="tab-header">Prediction & Analysis Dashboard</h2>', unsafe_allow_html=True)
        
        # Sidebar for model configuration
        st.sidebar.header("🔧 Model Configuration")
        
        # Target variable selection
        target_options = ['TOT_QTY', 'Ln_Tot_Qty', 'PRICE', 'Ln_Price']
        target_var = st.sidebar.selectbox("Select Target Variable:", target_options, index=0)
        
        # Feature selection
        st.sidebar.subheader("Feature Selection")
        
        # Exclude target variables and identifiers from features
        exclude_cols = ['Obs', target_var] + [col for col in target_options if col != target_var]
        potential_features = [col for col in data.columns if col not in exclude_cols]
        
        # Separate categorical and numerical features
        categorical_features = data[potential_features].select_dtypes(include='object').columns.tolist()
        numerical_features = data[potential_features].select_dtypes(include=np.number).columns.tolist()
        
        selected_categorical = st.sidebar.multiselect(
            "Categorical Features:",
            categorical_features,
            default=categorical_features[:3] if len(categorical_features) >= 3 else categorical_features
        )
        
        selected_numerical = st.sidebar.multiselect(
            "Numerical Features:",
            numerical_features,
            default=numerical_features[:3] if len(numerical_features) >= 3 else numerical_features
        )
        
        # Model selection
        st.sidebar.subheader("Model Selection")
        model_options = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        selected_models = st.sidebar.multiselect(
            "Select Models:",
            list(model_options.keys()),
            default=['Linear Regression', 'Random Forest']
        )
        
        # Test size
        test_size = st.sidebar.slider("Test Size (%):", 10, 40, 20) / 100
        
        # Random state
        random_state = st.sidebar.number_input("Random State:", value=42, min_value=0)
        
        if st.sidebar.button("🚀 Run Analysis", type="primary"):
            if not selected_models:
                st.error("Please select at least one model.")
            elif not (selected_categorical or selected_numerical):
                st.error("Please select at least one feature.")
            else:
                # Prepare data for modeling
                with st.spinner("Preparing data and training models..."):
                    try:
                        # Create feature matrix
                        model_data = data.copy()
                        X_features = []
                        
                        # Handle categorical features (one-hot encoding)
                        if selected_categorical:
                            for cat_col in selected_categorical:
                                if cat_col in model_data.columns:
                                    model_data[cat_col] = model_data[cat_col].astype(str)
                                    dummies = pd.get_dummies(model_data[cat_col], prefix=cat_col, drop_first=True)
                                    model_data = pd.concat([model_data, dummies], axis=1)
                                    X_features.extend(dummies.columns.tolist())
                        
                        # Handle numerical features
                        if selected_numerical:
                            for num_col in selected_numerical:
                                if num_col in model_data.columns:
                                    model_data[num_col] = pd.to_numeric(model_data[num_col], errors='coerce')
                                    X_features.append(num_col)
                        
                        # Prepare X and y
                        X = model_data[X_features].copy()
                        y = model_data[target_var].copy()
                        
                        # Handle missing values
                        initial_rows = len(X)
                        X.dropna(inplace=True)
                        y = y.loc[X.index]
                        
                        if len(X) == 0:
                            st.error("No valid data remaining after preprocessing.")
                        else:
                            if len(X) < initial_rows:
                                st.warning(f"Dropped {initial_rows - len(X)} rows due to missing values. Using {len(X)} observations.")
                            
                            # Split data
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=test_size, random_state=random_state
                            )
                            
                            # Scale features for some models
                            scaler = StandardScaler()
                            X_train_scaled = scaler.fit_transform(X_train)
                            X_test_scaled = scaler.transform(X_test)
                            
                            # Train models and collect results
                            results = {}
                            feature_importance_data = {}
                            
                            for model_name in selected_models:
                                model = model_options[model_name]
                                
                                # Use scaled features for linear models, original for tree-based
                                if model_name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']:
                                    model.fit(X_train_scaled, y_train)
                                    y_pred = model.predict(X_test_scaled)
                                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
                                else:
                                    model.fit(X_train, y_train)
                                    y_pred = model.predict(X_test)
                                    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
                                
                                # Calculate metrics
                                mse = mean_squared_error(y_test, y_pred)
                                rmse = np.sqrt(mse)
                                mae = mean_absolute_error(y_test, y_pred)
                                r2 = r2_score(y_test, y_pred)
                                
                                results[model_name] = {
                                    'MSE': mse,
                                    'RMSE': rmse,
                                    'MAE': mae,
                                    'R²': r2,
                                    'CV R² Mean': cv_scores.mean(),
                                    'CV R² Std': cv_scores.std(),
                                    'y_pred': y_pred,
                                    'model': model
                                }
                                
                                # Feature importance (for tree-based models)
                                if hasattr(model, 'feature_importances_'):
                                    feature_importance_data[model_name] = pd.DataFrame({
                                        'Feature': X.columns,
                                        'Importance': model.feature_importances_
                                    }).sort_values('Importance', ascending=False)
                            
                            # Display results
                            st.success("✅ Models trained successfully!")
                            
                            # Model comparison
                            st.subheader("📊 Model Performance Comparison")
                            
                            comparison_df = pd.DataFrame({
                                model_name: {
                                    'R² Score': f"{results[model_name]['R²']:.4f}",
                                    'RMSE': f"{results[model_name]['RMSE']:.4f}",
                                    'MAE': f"{results[model_name]['MAE']:.4f}",
                                    'CV R² Mean': f"{results[model_name]['CV R² Mean']:.4f}",
                                    'CV R² Std': f"{results[model_name]['CV R² Std']:.4f}"
                                }
                                for model_name in selected_models
                            }).T
                            
                            st.dataframe(comparison_df, use_container_width=True)
                            
                            # Best model highlight
                            best_model = max(results.keys(), key=lambda x: results[x]['R²'])
                            st.success(f"🏆 Best performing model: **{best_model}** (R² = {results[best_model]['R²']:.4f})")
                            
                            # Visualization tabs
                            viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
                                "Model Comparison", "Predictions vs Actual", "Feature Importance", "Residual Analysis"
                            ])
                            
                            with viz_tab1:
                                # Performance metrics comparison
                                metrics_df = pd.DataFrame({
                                    'Model': list(results.keys()),
                                    'R² Score': [results[model]['R²'] for model in results.keys()],
                                    'RMSE': [results[model]['RMSE'] for model in results.keys()],
                                    'MAE': [results[model]['MAE'] for model in results.keys()]
                                })
                                
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    fig_r2 = px.bar(
                                        metrics_df, x='Model', y='R² Score',
                                        title='R² Score Comparison',
                                        color='R² Score',
                                        color_continuous_scale='viridis'
                                    )
                                    st.plotly_chart(fig_r2, use_container_width=True)
                                
                                with col2:
                                    fig_rmse = px.bar(
                                        metrics_df, x='Model', y='RMSE',
                                        title='RMSE Comparison',
                                        color='RMSE',
                                        color_continuous_scale='viridis_r'
                                    )
                                    st.plotly_chart(fig_rmse, use_container_width=True)
                            
                            with viz_tab2:
                                # Predictions vs Actual plots
                                for model_name in selected_models:
                                    fig_pred = px.scatter(
                                        x=y_test, 
                                        y=results[model_name]['y_pred'],
                                        title=f'{model_name}: Predictions vs Actual',
                                        labels={'x': f'Actual {target_var}', 'y': f'Predicted {target_var}'}
                                    )
                                    
                                    # Add perfect prediction line
                                    min_val = min(y_test.min(), results[model_name]['y_pred'].min())
                                    max_val = max(y_test.max(), results[model_name]['y_pred'].max())
                                    fig_pred.add_trace(go.Scatter(
                                        x=[min_val, max_val], 
                                        y=[min_val, max_val],
                                        mode='lines',
                                        name='Perfect Prediction',
                                        line=dict(dash='dash', color='red')
                                    ))
                                    
                                    st.plotly_chart(fig_pred, use_container_width=True)
                            
                            with viz_tab3:
                                # Feature importance for tree-based models
                                if feature_importance_data:
                                    for model_name, importance_df in feature_importance_data.items():
                                        fig_importance = px.bar(
                                            importance_df.head(15), 
                                            x='Importance', 
                                            y='Feature',
                                            orientation='h',
                                            title=f'{model_name}: Top 15 Feature Importances'
                                        )
                                        fig_importance.update_layout(height=500)
                                        st.plotly_chart(fig_importance, use_container_width=True)
                                else:
                                    st.info("Feature importance is only available for tree-based models (Random Forest, Gradient Boosting).")
                            
                            with viz_tab4:
                                # Residual analysis for each model
                                for model_name in selected_models:
                                    residuals = y_test - results[model_name]['y_pred']
                                    
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        fig_residual = px.scatter(
                                            x=results[model_name]['y_pred'], 
                                            y=residuals,
                                            title=f'{model_name}: Residuals vs Predicted',
                                            labels={'x': f'Predicted {target_var}', 'y': 'Residuals'}
                                        )
                                        fig_residual.add_hline(y=0, line_dash="dash", line_color="red")
                                        st.plotly_chart(fig_residual, use_container_width=True)
                                    
                                    with col2:
                                        fig_hist_residual = px.histogram(
                                            x=residuals,
                                            title=f'{model_name}: Residuals Distribution',
                                            nbins=30
                                        )
                                        st.plotly_chart(fig_hist_residual, use_container_width=True)
                            
                            # Statistical analysis for best model
                            st.markdown("---")
                            st.subheader("📋 Detailed Statistical Analysis (Best Model)")
                            
                            if best_model in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']:
                                # Detailed regression analysis using statsmodels
                                if best_model == 'Linear Regression':
                                    X_sm = sm.add_constant(X_train_scaled)
                                    sm_model = sm.OLS(y_train, X_sm).fit()
                                    
                                    col1, col2 = st.columns([2, 1])
                                    
                                    with col1:
                                        st.text("Regression Summary:")
                                        st.text(str(sm_model.summary()))
                                    
                                    with col2:
                                        st.write("**Key Statistics:**")
                                        st.metric("R-squared", f"{sm_model.rsquared:.4f}")
                                        st.metric("Adj. R-squared", f"{sm_model.rsquared_adj:.4f}")
                                        st.metric("F-statistic", f"{sm_model.fvalue:.2f}")
                                        st.metric("Prob (F-statistic)", f"{sm_model.f_pvalue:.4e}")
                                        st.metric("AIC", f"{sm_model.aic:.2f}")
                                        st.metric("BIC", f"{sm_model.bic:.2f}")
                            
                            # Price elasticity analysis (if applicable)
                            if target_var == 'TOT_QTY' and 'PRICE' in selected_numerical:
                                st.markdown("---")
                                st.subheader("💰 Price Elasticity Analysis")
                                
                                # Calculate elasticity for the best model if it's linear
                                if best_model in ['Linear Regression'] and 'PRICE' in X.columns:
                                    best_model_obj = results[best_model]['model']
                                    price_coeff_idx = list(X.columns).index('PRICE')
                                    price_coefficient = best_model_obj.coef_[price_coeff_idx]
                                    
                                    mean_price = X['PRICE'].mean()
                                    mean_qty = y.mean()
                                    
                                    elasticity = price_coefficient * (mean_price / mean_qty)
                                    
                                    col1, col2, col3 = st.columns(3)
                                    
                                    with col1:
                                        st.metric("Price Coefficient", f"{price_coefficient:.6f}")
                                    
                                    with col2:
                                        st.metric("Price Elasticity", f"{elasticity:.4f}")
                                    
                                    with col3:
                                        if elasticity < -1:
                                            elasticity_type = "Elastic"
                                            color = "green"
                                        elif -1 <= elasticity < 0:
                                            elasticity_type = "Inelastic"
                                            color = "orange"
                                        else:
                                            elasticity_type = "Unusual"
                                            color = "red"
                                        
                                        st.markdown(f"**Demand Type:** <span style='color:{color}'>{elasticity_type}</span>", unsafe_allow_html=True)
                                    
                                    if elasticity < -1:
                                        st.success("✅ Demand is elastic - customers are sensitive to price changes.")
                                    elif -1 <= elasticity < 0:
                                        st.warning("⚠️ Demand is inelastic - customers are less sensitive to price changes.")
                                    else:
                                        st.error("❌ Unusual elasticity value - may indicate model issues or special market conditions.")
                    
                    except Exception as e:
                        st.error(f"An error occurred during analysis: {e}")
                        st.error("Please check your feature selections and ensure the data is properly formatted.")

else:
    st.error("Unable to load data. Please ensure 'AravindEyeCareV2.csv' is available in the application directory.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666666;'>
        <p>🏥 Aravind Eye Care - Data Analysis & Prediction Platform</p>
        <p>Built with Streamlit • For internal use only</p>
    </div>
    """, 
    unsafe_allow_html=True
)