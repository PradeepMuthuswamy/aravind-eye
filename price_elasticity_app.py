import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Price Elasticity Analysis - Aravind Eye Care", 
    layout="wide"
)

# Title
st.title("📊 Price Elasticity of Demand Analysis")
st.subheader("Aravind Eye Care - Intraocular Lenses")

# Load data function
@st.cache_data
def load_data():
    """Load and preprocess the data"""
    try:
        # Load the data file
        data = pd.read_csv("AravindEyeCareV2.csv")
        
        # Ensure numeric data types for key columns
        data['TOT_QTY'] = pd.to_numeric(data['TOT_QTY'], errors='coerce')
        data['PRICE'] = pd.to_numeric(data['PRICE'], errors='coerce')
        
        # Remove any rows with missing values in key columns
        data = data.dropna(subset=['TOT_QTY', 'PRICE'])
        
        # Remove any rows with zero or negative values
        data = data[(data['TOT_QTY'] > 0) & (data['PRICE'] > 0)]
        
        # Clean STATE column - remove NaN and convert to string
        data = data.dropna(subset=['STATE'])
        data['STATE'] = data['STATE'].astype(str)
        
        # Clean PRODUCTS column as well
        data = data.dropna(subset=['PRODUCTS'])
        data['PRODUCTS'] = data['PRODUCTS'].astype(str)
        
        # Clean CHANNEL column
        if 'CHANNEL' in data.columns:
            data = data.dropna(subset=['CHANNEL'])
            data['CHANNEL'] = data['CHANNEL'].astype(str)
            
            # Create channel regrouping based on Exhibit 7
            channel_regroup = {
                'DOM_AEH': 'Aravind',
                'DOM_EXP_LAICO': 'Aravind',
                'DOM_MGD_HOSP': 'Aravind',
                'DOM_DEALER': 'Dealer',
                'DOM_EXP_DEALER': 'Dealer',
                'DOM_SUB_DEALER': 'Dealer',
                'DOM_DONOR_CBM': 'NGO',
                'DOM_GOVT_OTHERS': 'NGO',
                'DOM_LARGE_GOVT': 'NGO',
                'DOM_LARGE_NPO': 'NGO',
                'DOM_NGO': 'NGO',
                'DOM_EXP_HOSPITAL': "Aurolab's Sales Team",
                'DOM_OTH': "Aurolab's Sales Team",
                'DOM_PVT_INS': "Aurolab's Sales Team",
                'DOM_PVT_PRAC': "Aurolab's Sales Team"
            }
            
            # Apply regrouping
            data['CHANNEL_GROUP'] = data['CHANNEL'].map(channel_regroup)
            # If any channels don't match, keep original
            data['CHANNEL_GROUP'] = data['CHANNEL_GROUP'].fillna(data['CHANNEL'])
        
        # Create log variables
        data['Ln_Tot_Qty'] = np.log(data['TOT_QTY'])
        data['Ln_Price'] = np.log(data['PRICE'])
        
        return data
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def estimate_linear_model(data):
    """Estimate linear price elasticity model: Quantity = β₀ + β₁ * Price"""
    try:
        # Prepare variables
        X = data[['PRICE']].values
        y = data['TOT_QTY'].values
        
        # Add constant for intercept
        X_with_const = sm.add_constant(X)
        
        # Fit the model
        model = sm.OLS(y, X_with_const).fit()
        
        # Calculate elasticity at mean values
        price_coeff = model.params[1]
        mean_price = data['PRICE'].mean()
        mean_qty = data['TOT_QTY'].mean()
        elasticity = price_coeff * (mean_price / mean_qty)
        
        return {
            'model': model,
            'elasticity': elasticity,
            'price_coefficient': price_coeff,
            'mean_price': mean_price,
            'mean_quantity': mean_qty,
            'r_squared': model.rsquared,
            'n_observations': len(X)
        }
        
    except Exception as e:
        st.error(f"Error in linear model: {e}")
        return None

def estimate_loglog_model(data):
    """Estimate log-log price elasticity model: ln(Quantity) = β₀ + β₁ * ln(Price)"""
    try:
        # Prepare variables
        X = data[['Ln_Price']].values
        y = data['Ln_Tot_Qty'].values
        
        # Add constant for intercept
        X_with_const = sm.add_constant(X)
        
        # Fit the model
        model = sm.OLS(y, X_with_const).fit()
        
        # In log-log model, the coefficient IS the elasticity
        elasticity = model.params[1]
        
        return {
            'model': model,
            'elasticity': elasticity,
            'r_squared': model.rsquared,
            'n_observations': len(X)
        }
        
    except Exception as e:
        st.error(f"Error in log-log model: {e}")
        return None

def plot_regression(data, model_type, selected_products=None, selected_states=None):
    """Create regression plot for the selected model"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Add filter info to title
    title_suffix = ""
    if selected_products and len(selected_products) < len(data['PRODUCTS'].unique()):
        if len(selected_products) <= 3:
            title_suffix += f" - Products: {', '.join(selected_products)}"
        else:
            title_suffix += f" - {len(selected_products)} Products"
    
    if selected_states and len(selected_states) < len(data['STATE'].unique()):
        if len(selected_states) <= 3:
            title_suffix += f" - States: {', '.join(selected_states)}"
        else:
            title_suffix += f" - {len(selected_states)} States"
    
    if model_type == "Linear":
        # Scatter plot
        ax.scatter(data['PRICE'], data['TOT_QTY'], alpha=0.5, s=30)
        
        # Regression line
        price_range = np.linspace(data['PRICE'].min(), data['PRICE'].max(), 100)
        X_pred = sm.add_constant(price_range)
        
        # Get the linear model to make predictions
        linear_results = estimate_linear_model(data)
        if linear_results:
            y_pred = linear_results['model'].predict(X_pred)
            ax.plot(price_range, y_pred, 'r-', linewidth=2, label='Regression Line')
        
        ax.set_xlabel('Price (₹)', fontsize=12)
        ax.set_ylabel('Quantity', fontsize=12)
        ax.set_title(f'Linear Model: Quantity vs Price{title_suffix}', fontsize=14)
        
    else:  # Log-Log
        # Scatter plot
        ax.scatter(data['Ln_Price'], data['Ln_Tot_Qty'], alpha=0.5, s=30)
        
        # Regression line
        ln_price_range = np.linspace(data['Ln_Price'].min(), data['Ln_Price'].max(), 100)
        X_pred = sm.add_constant(ln_price_range)
        
        # Get the log-log model to make predictions
        loglog_results = estimate_loglog_model(data)
        if loglog_results:
            y_pred = loglog_results['model'].predict(X_pred)
            ax.plot(ln_price_range, y_pred, 'r-', linewidth=2, label='Regression Line')
        
        ax.set_xlabel('Log(Price)', fontsize=12)
        ax.set_ylabel('Log(Quantity)', fontsize=12)
        ax.set_title(f'Log-Log Model: Log(Quantity) vs Log(Price){title_suffix}', fontsize=14)
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig

# Main application
def main():
    # Load data
    data = load_data()
    
    if data is None:
        st.error("Unable to load data. Please ensure 'AravindEyeCareV2.csv' is in the current directory.")
        return
    
    # Get unique products and states
    all_products = sorted(data['PRODUCTS'].unique())
    all_states = sorted(data['STATE'].unique())
    
    # Data overview - for all data
    st.markdown("### 📊 Overall Data Overview")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Observations", f"{len(data):,}")
    with col2:
        st.metric("Average Price", f"₹{data['PRICE'].mean():.2f}")
    with col3:
        st.metric("Average Quantity", f"{data['TOT_QTY'].mean():.0f}")
    with col4:
        st.metric("Total Products", f"{len(all_products)}")
    with col5:
        st.metric("Total States", f"{len(all_states)}")
    
    st.markdown("---")
    
    # Filters section
    st.subheader("🎯 Data Filters")
    
    # Create two columns for filters
    filter_col1, filter_col2 = st.columns(2)
    
    with filter_col1:
        st.markdown("#### 📦 Product Selection")
        selected_products = st.multiselect(
            "Choose products to analyze:",
            options=all_products,
            default=[],
            help="Select specific products or leave empty to analyze all products."
        )
    
    with filter_col2:
        st.markdown("#### 📍 State Selection")
        selected_states = st.multiselect(
            "Choose states to analyze:",
            options=all_states,
            default=[],
            help="Select specific states or leave empty to analyze all states."
        )
    
    # Filter data based on selections
    filtered_data = data.copy()
    
    if selected_products:
        filtered_data = filtered_data[filtered_data['PRODUCTS'].isin(selected_products)]
    
    if selected_states:
        filtered_data = filtered_data[filtered_data['STATE'].isin(selected_states)]
    
    # Show filter summary
    filter_summary = []
    if selected_products:
        filter_summary.append(f"{len(selected_products)} product(s)")
    else:
        filter_summary.append("All products")
    
    if selected_states:
        filter_summary.append(f"{len(selected_states)} state(s)")
    else:
        filter_summary.append("All states")
    
    st.info(f"Analyzing: {' and '.join(filter_summary)}")
    
    # Show filtered data statistics
    if len(filtered_data) > 0:
        st.markdown("### 📈 Filtered Data Statistics")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Observations", f"{len(filtered_data):,}")
        with col2:
            st.metric("Avg Price", f"₹{filtered_data['PRICE'].mean():.2f}")
        with col3:
            st.metric("Avg Quantity", f"{filtered_data['TOT_QTY'].mean():.0f}")
        with col4:
            st.metric("Price Range", f"₹{filtered_data['PRICE'].min():.0f} - ₹{filtered_data['PRICE'].max():.0f}")
        with col5:
            # Show unique count of products/states in filtered data
            unique_products = filtered_data['PRODUCTS'].nunique()
            unique_states = filtered_data['STATE'].nunique()
            st.metric("Coverage", f"{unique_products} prod, {unique_states} states")
    else:
        st.error("No data available for the selected filters.")
        return
    
    st.markdown("---")
    
    # Model selection
    st.subheader("📈 Select Regression Model")
    model_type = st.radio(
        "Choose the model type:",
        ["Linear Model", "Log-Log Model"],
        help="Linear: Quantity = β₀ + β₁ × Price | Log-Log: ln(Quantity) = β₀ + β₁ × ln(Price)"
    )
    
    # Estimate button
    if st.button("Estimate Price Elasticity", type="primary"):
        st.markdown("---")
        
        # Check if we have enough data
        if len(filtered_data) < 10:
            st.error("Not enough data points for regression analysis. Please adjust your filters to include more data.")
            return
        
        # Run the selected model
        if model_type == "Linear Model":
            results = estimate_linear_model(filtered_data)
            st.subheader("Linear Model Results")
            
            if results:
                # Display regression output
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.text("Regression Summary:")
                    st.text(str(results['model'].summary()))
                
                with col2:
                    st.markdown("### Key Metrics")
                    st.metric("R-squared", f"{results['r_squared']:.4f}")
                    st.metric("Observations", f"{results['n_observations']:,}")
                    st.metric("Price Coefficient", f"{results['price_coefficient']:.6f}")
                    
                    st.markdown("### Elasticity")
                    st.metric("Price Elasticity", f"{results['elasticity']:.4f}")
                    st.caption("(Calculated at mean values)")
                
        else:  # Log-Log Model
            results = estimate_loglog_model(filtered_data)
            st.subheader("Log-Log Model Results")
            
            if results:
                # Display regression output
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.text("Regression Summary:")
                    st.text(str(results['model'].summary()))
                
                with col2:
                    st.markdown("### Key Metrics")
                    st.metric("R-squared", f"{results['r_squared']:.4f}")
                    st.metric("Observations", f"{results['n_observations']:,}")
                    
                    st.markdown("### Elasticity")
                    st.metric("Price Elasticity", f"{results['elasticity']:.4f}")
                    st.caption("(Constant elasticity)")
        
        if results:
            # Interpretation
            st.markdown("---")
            st.subheader("📊 Interpretation")
            
            elasticity = results['elasticity']
            
            # Add filter-specific context
            context_parts = []
            if selected_products:
                context_parts.append(f"Products: {', '.join(selected_products[:3])}{'...' if len(selected_products) > 3 else ''}")
            if selected_states:
                context_parts.append(f"States: {', '.join(selected_states[:3])}{'...' if len(selected_states) > 3 else ''}")
            
            if context_parts:
                st.markdown(f"**Analysis for: {' | '.join(context_parts)}**")
            
            if elasticity < -1:
                st.success(f"**Elastic Demand**: The price elasticity is {elasticity:.4f}, which means demand is elastic. A 1% increase in price leads to a {abs(elasticity):.2f}% decrease in quantity demanded.")
            elif -1 <= elasticity < 0:
                st.info(f"**Inelastic Demand**: The price elasticity is {elasticity:.4f}, which means demand is inelastic. A 1% increase in price leads to a {abs(elasticity):.2f}% decrease in quantity demanded.")
            else:
                st.warning(f"**Unusual Result**: The price elasticity is {elasticity:.4f}. Positive elasticity is unusual and may indicate data issues or special market conditions.")
            
            # Business implications with variable cost
            st.markdown("---")
            st.subheader("💼 Business Implications")
            
            variable_cost = 1000  # Rs. 1,000 per Toric lens
            avg_price = filtered_data['PRICE'].mean()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Variable Cost (Toric lens)", f"₹{variable_cost:,}")
                st.metric("Average Selling Price", f"₹{avg_price:.2f}")
            with col2:
                margin = avg_price - variable_cost
                margin_pct = (margin / avg_price) * 100 if avg_price > 0 else 0
                st.metric("Average Margin", f"₹{margin:.2f}")
                st.metric("Margin %", f"{margin_pct:.1f}%")
            
            # Pricing recommendations
            st.markdown("### Pricing Strategy Recommendations")
            
            if context_parts:
                st.markdown(f"**For {' | '.join(context_parts)}:**")
            
            if elasticity < -1:
                st.markdown("""
                - ✅ **Demand is elastic** - customers are price sensitive
                - 📉 Consider price reductions to increase total revenue
                - 📈 A small price decrease could lead to a proportionally larger increase in quantity sold
                - ⚠️ Ensure price remains above variable cost of ₹1,000
                """)
            elif -1 <= elasticity < 0:
                st.markdown("""
                - ✅ **Demand is inelastic** - customers are less price sensitive  
                - 📈 You have pricing power - consider strategic price increases
                - 💰 Price increases will lead to higher total revenue
                - 📊 Monitor competitor reactions and market share
                """)
            
            # Visualization
            st.markdown("---")
            st.subheader("📈 Regression Visualization")
            fig = plot_regression(filtered_data, model_type, selected_products, selected_states)
            st.pyplot(fig)
    
    # State comparison section
    if st.checkbox("🗺️ Compare Elasticities Across States"):
        st.markdown("---")
        st.subheader("State-wise Elasticity Comparison")
        
        state_comparison_model = st.radio(
            "Select model for state comparison:",
            ["Linear", "Log-Log"],
            key="state_comparison_model"
        )
        
        # Determine which states to compare
        states_to_compare = selected_states if selected_states else all_states
        
        # If specific states are selected, show only those
        if selected_states:
            st.info(f"Comparing selected states: {', '.join(selected_states)}")
        else:
            st.info("Comparing all states in the dataset")
        
        # If products are selected, use filtered data for comparison
        state_comparison_data = data.copy()
        if selected_products:
            state_comparison_data = state_comparison_data[state_comparison_data['PRODUCTS'].isin(selected_products)]
            st.info(f"Comparing states for selected products: {', '.join(selected_products)}")
        
        # Calculate elasticity for each state
        state_elasticity_results = []
        
        for state in states_to_compare:
            state_data = state_comparison_data[state_comparison_data['STATE'] == state]
            if len(state_data) >= 10:  # Need minimum data points
                if state_comparison_model == "Linear":
                    result = estimate_linear_model(state_data)
                else:
                    result = estimate_loglog_model(state_data)
                
                if result:
                    state_elasticity_results.append({
                        'State': state,
                        'Elasticity': result['elasticity'],
                        'R-squared': result['r_squared'],
                        'Observations': result['n_observations'],
                        'Avg Price': state_data['PRICE'].mean(),
                        'Avg Quantity': state_data['TOT_QTY'].mean()
                    })
        
        if state_elasticity_results:
            # Create DataFrame for display
            state_comparison_df = pd.DataFrame(state_elasticity_results)
            state_comparison_df = state_comparison_df.sort_values('Elasticity')
            
            # Display table
            st.dataframe(
                state_comparison_df.style.format({
                    'Elasticity': '{:.4f}',
                    'R-squared': '{:.4f}',
                    'Avg Price': '₹{:.2f}',
                    'Avg Quantity': '{:.0f}'
                }),
                use_container_width=True
            )
            
            # Bar chart of elasticities
            fig, ax = plt.subplots(figsize=(14, 6))
            colors = ['green' if x < -1 else 'orange' if -1 <= x < 0 else 'red' for x in state_comparison_df['Elasticity']]
            bars = ax.bar(state_comparison_df['State'], state_comparison_df['Elasticity'], color=colors)
            ax.axhline(y=-1, color='black', linestyle='--', alpha=0.5, label='Unit Elastic')
            ax.set_xlabel('State', fontsize=12)
            ax.set_ylabel('Price Elasticity', fontsize=12)
            ax.set_title(f'Price Elasticity by State ({state_comparison_model} Model)', fontsize=14)
            ax.legend()
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Insights
            elastic_states = state_comparison_df[state_comparison_df['Elasticity'] < -1]['State'].tolist()
            inelastic_states = state_comparison_df[(state_comparison_df['Elasticity'] >= -1) & (state_comparison_df['Elasticity'] < 0)]['State'].tolist()
            
            if elastic_states:
                st.success(f"**Price-Sensitive States ({len(elastic_states)})**: {', '.join(elastic_states[:5])}{'...' if len(elastic_states) > 5 else ''}")
            if inelastic_states:
                st.info(f"**Less Price-Sensitive States ({len(inelastic_states)})**: {', '.join(inelastic_states[:5])}{'...' if len(inelastic_states) > 5 else ''}")
            
            # Optimal Pricing Table
            st.markdown("---")
            st.subheader("💰 Optimal Pricing by State")
            st.markdown("Using the formula: **p = mc × e / (e + 1)**")
            st.markdown("Where: p = optimal price, mc = marginal cost (₹1,000), e = elasticity")
            
            # Calculate optimal prices
            marginal_cost = 1000  # Variable cost per Toric lens
            
            # Create pricing table
            pricing_data = []
            for _, row in state_comparison_df.iterrows():
                elasticity = row['Elasticity']
                
                # Calculate optimal price using the formula
                # Note: elasticity is typically negative, so we need to handle this carefully
                if elasticity < -1:  # Elastic demand
                    optimal_price = marginal_cost * elasticity / (elasticity + 1)
                elif -1 < elasticity < 0:  # Inelastic demand
                    optimal_price = marginal_cost * elasticity / (elasticity + 1)
                else:  # Handle edge cases
                    optimal_price = None
                
                # Calculate markup percentage
                if optimal_price and optimal_price > 0:
                    markup_pct = ((optimal_price - marginal_cost) / marginal_cost) * 100
                else:
                    markup_pct = None
                
                pricing_data.append({
                    'State': row['State'],
                    'Elasticity': elasticity,
                    'Current Avg Price': row['Avg Price'],
                    'Optimal Price': optimal_price if optimal_price and optimal_price > 0 else 'N/A',
                    'Markup %': f"{markup_pct:.1f}%" if markup_pct is not None else 'N/A',
                    'Price Adjustment': f"₹{optimal_price - row['Avg Price']:.0f}" if optimal_price and optimal_price > 0 else 'N/A'
                })
            
            pricing_df = pd.DataFrame(pricing_data)
            
            # Display the pricing table
            st.dataframe(
                pricing_df.style.format({
                    'Elasticity': '{:.4f}',
                    'Current Avg Price': '₹{:.2f}',
                    'Optimal Price': lambda x: f'₹{x:.2f}' if isinstance(x, (int, float)) else x
                }),
                use_container_width=True
            )
            
            # Additional insights
            st.markdown("#### 📊 Pricing Insights")
            
            # States where price should be increased
            price_increase_states = []
            price_decrease_states = []
            
            for _, row in pricing_df.iterrows():
                if row['Optimal Price'] != 'N/A':
                    adjustment = float(row['Price Adjustment'].replace('₹', ''))
                    if adjustment > 50:  # Significant increase threshold
                        price_increase_states.append((row['State'], adjustment))
                    elif adjustment < -50:  # Significant decrease threshold
                        price_decrease_states.append((row['State'], adjustment))
            
            col1, col2 = st.columns(2)
            
            with col1:
                if price_increase_states:
                    st.markdown("**📈 Consider Price Increases in:**")
                    for state, adj in sorted(price_increase_states, key=lambda x: x[1], reverse=True)[:5]:
                        st.write(f"• {state}: +₹{adj:.0f}")
            
            with col2:
                if price_decrease_states:
                    st.markdown("**📉 Consider Price Reductions in:**")
                    for state, adj in sorted(price_decrease_states, key=lambda x: x[1])[:5]:
                        st.write(f"• {state}: ₹{adj:.0f}")
            
            # Formula explanation
            with st.expander("📚 Understanding the Optimal Pricing Formula"):
                st.markdown("""
                The formula **p = mc × e / (e + 1)** is derived from profit maximization principles:
                
                - **For elastic demand (e < -1)**: Lower prices can increase total revenue
                - **For inelastic demand (-1 < e < 0)**: Higher prices can increase total revenue
                - **Marginal cost**: ₹1,000 (variable cost per Toric lens)
                
                This formula assumes:
                - Profit maximization objective
                - Constant marginal cost
                - No capacity constraints
                - No competitive reactions
                """)
            
            # Channel Analysis Section
            if 'CHANNEL_GROUP' in data.columns:
                st.markdown("---")
                st.subheader("🏢 Channel-wise Elasticity Analysis")
                
                # If products are selected, filter for channel analysis
                channel_data = data.copy()
                if selected_products:
                    channel_data = channel_data[channel_data['PRODUCTS'].isin(selected_products)]
                if selected_states:
                    channel_data = channel_data[channel_data['STATE'].isin(selected_states)]
                
                # Calculate elasticity for each channel group
                channel_elasticity_results = []
                
                for channel_group in ['Aravind', 'Dealer', 'NGO', "Aurolab's Sales Team"]:
                    channel_group_data = channel_data[channel_data['CHANNEL_GROUP'] == channel_group]
                    if len(channel_group_data) >= 10:
                        if state_comparison_model == "Linear":
                            result = estimate_linear_model(channel_group_data)
                        else:
                            result = estimate_loglog_model(channel_group_data)
                        
                        if result:
                            channel_elasticity_results.append({
                                'Channel Group': channel_group,
                                'Elasticity': result['elasticity'],
                                'R-squared': result['r_squared'],
                                'Observations': result['n_observations'],
                                'Avg Price': channel_group_data['PRICE'].mean(),
                                'Avg Quantity': channel_group_data['TOT_QTY'].mean()
                            })
                
                if channel_elasticity_results:
                    # Create DataFrame
                    channel_df = pd.DataFrame(channel_elasticity_results)
                    channel_df = channel_df.sort_values('Elasticity')
                    
                    # Display channel comparison table
                    st.markdown("#### 📊 Elasticity by Distribution Channel")
                    st.dataframe(
                        channel_df.style.format({
                            'Elasticity': '{:.4f}',
                            'R-squared': '{:.4f}',
                            'Avg Price': '₹{:.2f}',
                            'Avg Quantity': '{:.0f}'
                        }),
                        use_container_width=True
                    )
                    
                    # Bar chart for channels
                    fig, ax = plt.subplots(figsize=(10, 6))
                    colors = ['green' if x < -1 else 'orange' if -1 <= x < 0 else 'red' for x in channel_df['Elasticity']]
                    bars = ax.bar(channel_df['Channel Group'], channel_df['Elasticity'], color=colors)
                    ax.axhline(y=-1, color='black', linestyle='--', alpha=0.5, label='Unit Elastic')
                    ax.set_xlabel('Channel Group', fontsize=12)
                    ax.set_ylabel('Price Elasticity', fontsize=12)
                    ax.set_title(f'Price Elasticity by Distribution Channel ({state_comparison_model} Model)', fontsize=14)
                    ax.legend()
                    plt.xticks(rotation=0)
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Channel-specific optimal pricing
                    st.markdown("#### 💰 Optimal Pricing by Channel")
                    
                    channel_pricing_data = []
                    for _, row in channel_df.iterrows():
                        elasticity = row['Elasticity']
                        
                        if elasticity < -1 or (-1 < elasticity < 0):
                            optimal_price = marginal_cost * elasticity / (elasticity + 1)
                        else:
                            optimal_price = None
                        
                        if optimal_price and optimal_price > 0:
                            markup_pct = ((optimal_price - marginal_cost) / marginal_cost) * 100
                        else:
                            markup_pct = None
                        
                        channel_pricing_data.append({
                            'Channel Group': row['Channel Group'],
                            'Elasticity': elasticity,
                            'Current Avg Price': row['Avg Price'],
                            'Optimal Price': optimal_price if optimal_price and optimal_price > 0 else 'N/A',
                            'Markup %': f"{markup_pct:.1f}%" if markup_pct is not None else 'N/A',
                            'Price Adjustment': f"₹{optimal_price - row['Avg Price']:.0f}" if optimal_price and optimal_price > 0 else 'N/A'
                        })
                    
                    channel_pricing_df = pd.DataFrame(channel_pricing_data)
                    
                    st.dataframe(
                        channel_pricing_df.style.format({
                            'Elasticity': '{:.4f}',
                            'Current Avg Price': '₹{:.2f}',
                            'Optimal Price': lambda x: f'₹{x:.2f}' if isinstance(x, (int, float)) else x
                        }),
                        use_container_width=True
                    )
                    
                    # Channel insights
                    st.markdown("#### 🎯 Channel Strategy Insights")
                    
                    for _, row in channel_df.iterrows():
                        channel = row['Channel Group']
                        elasticity = row['Elasticity']
                        
                        if elasticity < -1:
                            st.success(f"**{channel}**: Elastic demand ({elasticity:.3f}) - Price-sensitive channel, consider competitive pricing")
                        else:
                            st.info(f"**{channel}**: Inelastic demand ({elasticity:.3f}) - Less price-sensitive, potential for premium pricing")
                    
                    # Channel details expander
                    with st.expander("📋 Channel Grouping Details"):
                        st.markdown("""
                        **Channel Regrouping (Based on Exhibit 7):**
                        
                        **Aravind:**
                        - DOM_AEH: Aravind Eye Hospital
                        - DOM_EXP_LAICO: Purchase by Trainees who come to LAICO
                        - DOM_MGD_HOSP: Managed by Aravind
                        
                        **Dealer:**
                        - DOM_DEALER: Dealer
                        - DOM_EXP_DEALER: International buyers who buy in India to supply in their markets
                        - DOM_SUB_DEALER: Sub Dealer
                        
                        **NGO:**
                        - DOM_DONOR_CBM: International NGOs who donate products free in India
                        - DOM_GOVT_OTHERS: Small Volume Govt Hospitals
                        - DOM_LARGE_GOVT: Large Volume Govt Hospitals
                        - DOM_LARGE_NPO: Large Volume Nonprofit Hospitals
                        - DOM_NGO: Non Profit Hospitals
                        
                        **Aurolab's Sales Team:**
                        - DOM_EXP_HOSPITAL: International buyers who buy in India for their Hospital
                        - DOM_OTH: Any other
                        - DOM_PVT_INS: Private Hospitals
                        - DOM_PVT_PRAC: Individual Doctor Practice
                        """)
                else:
                    st.info("Not enough data to analyze channels with current filters.")
        else:
            st.warning("Not enough data points to calculate elasticity for the selected states. Each state needs at least 10 observations.")
    
    # Data preview
    with st.expander("View Raw Data Sample"):
        preview_info = []
        if selected_products:
            preview_info.append(f"Products: {', '.join(selected_products[:3])}{'...' if len(selected_products) > 3 else ''}")
        if selected_states:
            preview_info.append(f"States: {', '.join(selected_states[:3])}{'...' if len(selected_states) > 3 else ''}")
        
        if preview_info:
            st.markdown(f"**Showing data for: {' | '.join(preview_info)}**")
            display_data = filtered_data[['PRODUCTS', 'STATE', 'TOT_QTY', 'PRICE', 'Ln_Tot_Qty', 'Ln_Price']].head(20)
        else:
            st.markdown("**Showing data for all products and states**")
            display_data = data[['PRODUCTS', 'STATE', 'TOT_QTY', 'PRICE', 'Ln_Tot_Qty', 'Ln_Price']].head(20)
        st.dataframe(display_data)

if __name__ == "__main__":
    main() 