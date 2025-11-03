"""
Streamlit UI for the Sales Agent System.
Interactive interface for multi-layered sales analytics.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from datetime import datetime, timedelta
import os

from sample_data import load_sales_data, generate_sales_data, get_data_summary
from orchestrator import build_agent_system


# Page configuration
st.set_page_config(
    page_title="Sales Agent System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .json-output {
        background-color: #282c34;
        color: #abb2bf;
        padding: 1rem;
        border-radius: 0.5rem;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
        overflow-x: auto;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load or generate sales data"""
    return load_sales_data("sales_sample.csv")


@st.cache_resource
def initialize_agent_system(_df, api_key):
    """Initialize the orchestrator (cached)"""
    return build_agent_system(_df, api_key=api_key)


def create_time_series_chart(data: dict):
    """Create Plotly time series chart from descriptive data"""
    if not data or 'time_series' not in data:
        return None
    
    time_series = data['time_series']
    
    # Check if time_series is empty
    if not time_series or len(time_series) == 0:
        return None
    
    df_ts = pd.DataFrame(time_series)
    
    # Check if dataframe has required columns
    if df_ts.empty or 'date' not in df_ts.columns or 'sales' not in df_ts.columns:
        return None
    
    fig = px.line(
        df_ts,
        x='date',
        y='sales',
        title='Daily Sales Trend',
        labels={'date': 'Date', 'sales': 'Sales ($)'}
    )
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Sales ($)",
        hovermode='x unified',
        height=400
    )
    
    return fig


def create_region_chart(data: dict):
    """Create bar chart for regional sales with performance-based colors"""
    if not data or 'by_region' not in data:
        return None
    
    by_region = data['by_region']
    
    # Check if by_region is empty
    if not by_region or len(by_region) == 0:
        return None
    
    regions = list(by_region.keys())
    sales = list(by_region.values())
    
    # Calculate average for color coding
    avg_sales = sum(sales) / len(sales)
    
    # Assign colors based on performance (above/below average)
    colors = []
    for sale in sales:
        if sale >= avg_sales * 1.1:  # 10% above average
            colors.append('#2ecc71')  # Green (high performance)
        elif sale <= avg_sales * 0.9:  # 10% below average
            colors.append('#e74c3c')  # Red (low performance)
        else:
            colors.append('#f39c12')  # Orange (average performance)
    
    # Format values for display
    text_values = [f'${s:,.0f}' for s in sales]
    
    fig = go.Figure(data=[
        go.Bar(
            x=regions, 
            y=sales, 
            marker_color=colors,
            text=text_values,
            textposition='outside',
            textfont=dict(size=12, color='#2c3e50'),
            hovertemplate='<b>%{x}</b><br>Sales: $%{y:,.2f}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title='Sales by Region',
        xaxis_title='Region',
        yaxis_title='Sales ($)',
        height=350,
        showlegend=False,
        yaxis=dict(tickformat='$,.0f')
    )
    
    return fig


def create_forecast_chart(descriptive: dict, predictive: dict):
    """Create combined historical + forecast chart"""
    if not descriptive or not predictive:
        return None
    
    # Historical data
    hist_df = pd.DataFrame(descriptive['time_series'])
    hist_df['type'] = 'Historical'
    
    # Forecast data
    forecast_df = pd.DataFrame({
        'date': predictive['forecast_dates'],
        'sales': predictive['forecast_values'],
        'type': 'Forecast'
    })
    
    # Combine
    combined_df = pd.concat([hist_df, forecast_df], ignore_index=True)
    
    fig = px.line(
        combined_df,
        x='date',
        y='sales',
        color='type',
        title='Historical Sales + 14-Day Forecast',
        labels={'date': 'Date', 'sales': 'Sales ($)', 'type': 'Type'}
    )
    
    # Add confidence interval
    if 'lower_bound' in predictive and 'upper_bound' in predictive:
        fig.add_trace(go.Scatter(
            x=predictive['forecast_dates'],
            y=predictive['upper_bound'],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=predictive['forecast_dates'],
            y=predictive['lower_bound'],
            mode='lines',
            line=dict(width=0),
            fillcolor='rgba(68, 68, 68, 0.2)',
            fill='tonexty',
            showlegend=True,
            name='Confidence Interval'
        ))
    
    fig.update_layout(height=450)
    
    return fig


def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<div class="main-header">🧠 Sales Agent System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">AI-Powered Multi-Layer Analytics: Descriptive → Diagnostic → Predictive → Prescriptive</div>',
        unsafe_allow_html=True
    )
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input(
            "Groq API Key (optional)",
            type="password",
            help="Enter your Groq API key for LLM features. System works without it using mock responses."
        )
        
        if api_key:
            os.environ["GROQ_API_KEY"] = api_key
        
        st.divider()
        
        # Data controls
        st.subheader("📊 Data Options")
        
        if st.button("🔄 Regenerate Sample Data"):
            with st.spinner("Generating new data..."):
                generate_sales_data(days=120, save_to_file=True)
                st.cache_data.clear()
                st.success("Data regenerated!")
                st.rerun()
        
        # Date range filter
        st.subheader("📅 Date Range")
        date_from = st.date_input(
            "From",
            value=datetime.now() - timedelta(days=30),
            max_value=datetime.now()
        )
        date_to = st.date_input(
            "To",
            value=datetime.now(),
            max_value=datetime.now()
        )
        
        # Region filter
        st.subheader("🗺️ Filters")
        region_filter = st.selectbox(
            "Region",
            ["All", "East", "West", "North", "South"]
        )
        
        st.divider()
        
        # System info
        st.subheader("ℹ️ System Info")
        st.caption("Developed by: Basudab Chowdhury")
        st.caption("basudab.chowdhory@gmail.com")
        st.caption("Analytics Layers: 4")
        st.caption("Status: 🟢 Online")
    
    # Load data
    df = load_data()
    
    # Display data summary
    with st.expander("📋 Dataset Summary", expanded=False):
        summary = get_data_summary(df)
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", f"{summary['total_records']:,}")
        with col2:
            st.metric("Total Sales", f"${summary['total_sales']:,.0f}")
        with col3:
            st.metric("Avg per Record", f"${summary['avg_sales_per_record']:,.2f}")
        with col4:
            st.metric("Date Range", f"{(pd.to_datetime(summary['date_range']['end']) - pd.to_datetime(summary['date_range']['start'])).days} days")
    
    # Main analysis section
    st.header("💬 Chat Interface")
    
    # Query input
    query = st.text_input(
        "Ask a question about your sales data:",
        placeholder="e.g., Analyze last 30 days for all regions",
        label_visibility="collapsed"
    )
    
    # Run analysis button
    if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
        if not query:
            st.warning("Please enter a query.")
        else:
            with st.spinner("🔄 Running multi-layer analysis..."):
                # Initialize agent system
                agent_system = initialize_agent_system(df, api_key if api_key else None)
                
                # Prepare filters
                filters = None
                if region_filter != "All":
                    filters = {"region": region_filter}
                
                # Execute analysis
                try:
                    result = agent_system.analyze(
                        query=query,
                        date_from=date_from.strftime("%Y-%m-%d"),
                        date_to=date_to.strftime("%Y-%m-%d"),
                        filters=filters
                    )
                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")
                    st.exception(e)  # Show full traceback
                    result = None
                
                # Display results
                if result and result.get('descriptive'):
                    st.success("✅ Analysis complete!")
                    
                    # Tabs for different views
                    tab1, tab2, tab3, tab4, tab5 = st.tabs([
                        "📊 Descriptive",
                        "🔍 Diagnostic",
                        "🔮 Predictive",
                        "💡 Prescriptive",
                        "📝 Summary"
                    ])
                    
                    # Descriptive tab
                    with tab1:
                        st.subheader("Descriptive Analytics")
                        desc_data = result.get('descriptive', {})
                        
                        if desc_data and desc_data.get('num_transactions', 0) > 0:
                            # Metrics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Sales", f"${desc_data.get('total_sales', 0):,.2f}")
                            with col2:
                                st.metric("Avg Daily Sales", f"${desc_data.get('avg_daily_sales', 0):,.2f}")
                            with col3:
                                st.metric("Transactions", f"{desc_data.get('num_transactions', 0):,}")
                            
                            # Charts
                            col1, col2 = st.columns(2)
                            with col1:
                                fig = create_time_series_chart(desc_data)
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.warning("⚠️ No time series data available for the selected date range.")
                            
                            with col2:
                                fig = create_region_chart(desc_data)
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.warning("⚠️ No regional data available for the selected filters.")
                            
                            # Raw JSON
                            with st.expander("View Raw Data"):
                                st.json(desc_data)
                        else:
                            st.error("❌ No data found for the selected date range and filters. Please try:")
                            st.markdown("""
                            - **Expanding the date range**
                            - **Removing region filters**
                            - **Checking if data exists in `sales_sample.csv`**
                            """)
                            if st.button("🔄 Regenerate Sample Data"):
                                with st.spinner("Generating fresh data..."):
                                    generate_sales_data(days=120, save_to_file=True)
                                    st.cache_data.clear()
                                    st.success("✅ Data regenerated! Please rerun your analysis.")
                                    st.rerun()
                    
                    # Diagnostic tab
                    with tab2:
                        st.subheader("Diagnostic Analytics")
                        diag_data = result.get('diagnostic', {})
                        
                        if diag_data:
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Best Region", diag_data.get('best_region', 'N/A'), delta="Top Performer", delta_color="normal")
                                st.metric("Best Product", diag_data.get('best_product', 'N/A'))
                            with col2:
                                st.metric("Worst Region", diag_data.get('worst_region', 'N/A'), delta="Needs Attention", delta_color="inverse")
                                st.metric("Volatility", f"{diag_data.get('volatility_coefficient', 0):.1f}%")
                            
                            st.subheader("Key Insights")
                            for insight in diag_data.get('insights', []):
                                with st.container():
                                    st.markdown(f"**{insight['entity']}** ({insight['dimension']})")
                                    st.write(f"🔍 {insight['root_cause']}")
                                    st.progress(insight['confidence'], text=f"Confidence: {insight['confidence']*100:.0f}%")
                                    st.divider()
                            
                            with st.expander("View Raw Data"):
                                st.json(diag_data)
                    
                    # Predictive tab
                    with tab3:
                        st.subheader("Predictive Analytics")
                        pred_data = result.get('predictive', {})
                        
                        if pred_data:
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Forecast Period", f"{pred_data.get('forecast_days', 0)} days")
                                st.metric("Total Forecast", f"${pred_data.get('total_forecast', 0):,.2f}")
                            with col2:
                                st.metric("Avg Daily Forecast", f"${pred_data.get('avg_daily_forecast', 0):,.2f}")
                            
                            # Forecast chart
                            fig = create_forecast_chart(result.get('descriptive', {}), pred_data)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with st.expander("View Raw Data"):
                                st.json(pred_data)
                    
                    # Prescriptive tab
                    with tab4:
                        st.subheader("Prescriptive Analytics")
                        presc_data = result.get('prescriptive', {})
                        
                        if presc_data:
                            st.metric("Total Recommendations", presc_data.get('total_recommendations', 0))
                            
                            st.subheader("Recommended Actions")
                            for i, action in enumerate(presc_data.get('actions', []), 1):
                                priority_colors = {
                                    'high': '🔴',
                                    'medium': '🟡',
                                    'low': '🟢'
                                }
                                priority_icon = priority_colors.get(action['priority'], '⚪')
                                
                                with st.container():
                                    st.markdown(f"### {priority_icon} Action {i}")
                                    st.markdown(f"**{action['action']}**")
                                    st.write(f"💭 {action['rationale']}")
                                    st.write(f"Priority: **{action['priority'].upper()}**")
                                    st.progress(action['confidence'], text=f"Confidence: {action['confidence']*100:.0f}%")
                                    st.divider()
                            
                            with st.expander("View Raw Data"):
                                st.json(presc_data)
                    
                    # Summary tab
                    with tab5:
                        st.subheader("Executive Summary")
                        
                        # LLM Summary
                        if 'llm_summary' in result:
                            st.markdown("### 🤖 AI Analysis")
                            st.info(result['llm_summary'])
                        
                        # Explainability
                        st.markdown("### 📊 Explainability")
                        explainability = result.get('explainability', [])
                        if explainability:
                            for item in explainability:
                                st.markdown(f"**{item['layer'].title()} Layer**")
                                st.write(f"Entity: {item['entity']}")
                                st.write(f"Finding: {item['finding']}")
                                st.progress(item['confidence'], text=f"Confidence: {item['confidence']*100:.0f}%")
                                st.divider()
                        
                        # Export options
                        st.subheader("📥 Export Results")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            json_str = json.dumps(result, indent=2)
                            st.download_button(
                                "Download JSON",
                                data=json_str,
                                file_name=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json"
                            )
                        
                        with col2:
                            # Create summary CSV
                            summary_data = {
                                'Metric': ['Total Sales', 'Avg Daily Sales', 'Best Region', 'Worst Region', 'Forecast Total'],
                                'Value': [
                                    result.get('descriptive', {}).get('total_sales', 0),
                                    result.get('descriptive', {}).get('avg_daily_sales', 0),
                                    result.get('diagnostic', {}).get('best_region', 'N/A'),
                                    result.get('diagnostic', {}).get('worst_region', 'N/A'),
                                    result.get('predictive', {}).get('total_forecast', 0)
                                ]
                            }
                            summary_df = pd.DataFrame(summary_data)
                            csv = summary_df.to_csv(index=False)
                            st.download_button(
                                "Download CSV Summary",
                                data=csv,
                                file_name=f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                else:
                    st.error("Analysis failed. Please check your inputs and try again.")
    
    # Footer
    st.divider()
    st.markdown(
        '<div style="text-align: center; color: #888;">Built with LangChain • LangGraph • Groq • Streamlit</div>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()