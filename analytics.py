"""
Core analytics functions for the Sales Agent System.
Implements descriptive, diagnostic, predictive, and prescriptive analysis layers.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from datetime import datetime, timedelta
import json


def descriptive_summary(df: pd.DataFrame, date_from=None, date_to=None, filters=None) -> dict:
    """
    Descriptive analytics: summarize "what happened"
    
    Args:
        df: Sales DataFrame with columns [date, product, segment, region, channel, sales]
        date_from: Start date filter
        date_to: End date filter
        filters: Dict of column filters (e.g., {"region": "West"})
    
    Returns:
        Dict with KPIs and aggregations
    """
    # Apply date filters
    if date_from:
        df = df[df['date'] >= pd.to_datetime(date_from)]
    if date_to:
        df = df[df['date'] <= pd.to_datetime(date_to)]
    
    # Apply dimension filters
    if filters:
        for col, val in filters.items():
            if col in df.columns:
                df = df[df[col] == val]
    
    # Calculate KPIs
    total_sales = float(df['sales'].sum())
    avg_daily_sales = float(df.groupby('date')['sales'].sum().mean())
    
    # Aggregations
    by_region = df.groupby('region')['sales'].sum().to_dict()
    by_product = df.groupby('product')['sales'].sum().to_dict()
    by_segment = df.groupby('segment')['sales'].sum().to_dict()
    by_channel = df.groupby('channel')['sales'].sum().to_dict()
    
    # Time series
    daily_sales = df.groupby('date')['sales'].sum().reset_index()
    daily_sales['date'] = daily_sales['date'].astype(str)
    time_series = daily_sales.to_dict('records')
    
    return {
        "total_sales": round(total_sales, 2),
        "avg_daily_sales": round(avg_daily_sales, 2),
        "num_transactions": len(df),
        "date_range": {
            "start": str(df['date'].min()),
            "end": str(df['date'].max())
        },
        "by_region": {k: round(v, 2) for k, v in by_region.items()},
        "by_product": {k: round(v, 2) for k, v in by_product.items()},
        "by_segment": {k: round(v, 2) for k, v in by_segment.items()},
        "by_channel": {k: round(v, 2) for k, v in by_channel.items()},
        "time_series": time_series
    }


def diagnostic_analysis(df: pd.DataFrame, descriptive_output: dict = None) -> dict:
    """
    Diagnostic analytics: explain "why it happened"
    
    Analyzes correlations, variances, and identifies anomalies or root causes.
    
    Args:
        df: Sales DataFrame
        descriptive_output: Output from descriptive_summary (optional)
    
    Returns:
        Dict with diagnostic insights and root causes
    """
    insights = []
    
    # Identify best and worst performing dimensions
    by_region = df.groupby('region')['sales'].sum().sort_values(ascending=False)
    best_region = by_region.index[0]
    worst_region = by_region.index[-1]
    
    by_product = df.groupby('product')['sales'].sum().sort_values(ascending=False)
    best_product = by_product.index[0]
    worst_product = by_product.index[-1]
    
    # Analyze worst region's channel mix
    worst_region_data = df[df['region'] == worst_region]
    channel_distribution = worst_region_data.groupby('channel')['sales'].sum()
    dominant_channel = channel_distribution.idxmax()
    channel_pct = (channel_distribution[dominant_channel] / channel_distribution.sum()) * 100
    
    insights.append({
        "dimension": "region",
        "entity": worst_region,
        "issue": "underperformance",
        "root_cause": f"Heavy reliance on {dominant_channel} channel ({channel_pct:.1f}%)",
        "confidence": 0.78
    })
    
    # Analyze variance in sales
    daily_variance = df.groupby('date')['sales'].sum().var()
    daily_mean = df.groupby('date')['sales'].sum().mean()
    cv = (np.sqrt(daily_variance) / daily_mean) * 100  # Coefficient of variation
    
    if cv > 30:
        insights.append({
            "dimension": "time",
            "entity": "daily_sales",
            "issue": "high_volatility",
            "root_cause": f"Sales variance is high (CV={cv:.1f}%), indicating inconsistent demand",
            "confidence": 0.85
        })
    
    # Segment analysis
    by_segment = df.groupby('segment')['sales'].sum().sort_values(ascending=False)
    top_segment = by_segment.index[0]
    top_segment_pct = (by_segment.iloc[0] / by_segment.sum()) * 100
    
    insights.append({
        "dimension": "segment",
        "entity": top_segment,
        "issue": "concentration",
        "root_cause": f"{top_segment} segment dominates with {top_segment_pct:.1f}% of sales",
        "confidence": 0.92
    })
    
    return {
        "best_region": best_region,
        "worst_region": worst_region,
        "best_product": best_product,
        "worst_product": worst_product,
        "insights": insights,
        "volatility_coefficient": round(cv, 2)
    }


def predictive_forecast(df: pd.DataFrame, forecast_days: int = 14) -> dict:
    """
    Predictive analytics: forecast "what will happen"
    
    Uses RandomForest to predict future sales based on historical patterns.
    
    Args:
        df: Sales DataFrame
        forecast_days: Number of days to forecast
    
    Returns:
        Dict with forecast dates, values, and confidence intervals
    """
    # Prepare time series data
    daily_sales = df.groupby('date')['sales'].sum().reset_index().sort_values('date')
    daily_sales['day_index'] = range(len(daily_sales))
    
    # Extract temporal features
    daily_sales['day_of_week'] = pd.to_datetime(daily_sales['date']).dt.dayofweek
    daily_sales['day_of_month'] = pd.to_datetime(daily_sales['date']).dt.day
    
    # Create lag features
    daily_sales['lag_1'] = daily_sales['sales'].shift(1)
    daily_sales['lag_7'] = daily_sales['sales'].shift(7)
    daily_sales['rolling_mean_7'] = daily_sales['sales'].rolling(window=7, min_periods=1).mean()
    
    # Fill NaN values
    daily_sales = daily_sales.bfill().ffill()
    
    # Prepare training data
    features = ['day_index', 'day_of_week', 'day_of_month', 'lag_1', 'lag_7', 'rolling_mean_7']
    X = daily_sales[features].values
    y = daily_sales['sales'].values
    
    # Train RandomForest model
    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X, y)
    
    # Generate future predictions
    last_date = pd.to_datetime(daily_sales['date'].max())
    last_day_index = daily_sales['day_index'].max()
    
    future_dates = []
    predictions = []
    
    # Iteratively predict
    last_values = daily_sales[['sales']].tail(7).values.flatten()
    
    for i in range(1, forecast_days + 1):
        future_date = last_date + timedelta(days=i)
        future_day_index = last_day_index + i
        
        # Create features for prediction
        day_of_week = future_date.dayofweek
        day_of_month = future_date.day
        lag_1 = last_values[-1] if len(last_values) > 0 else daily_sales['sales'].mean()
        lag_7 = last_values[-7] if len(last_values) >= 7 else daily_sales['sales'].mean()
        rolling_mean_7 = np.mean(last_values[-7:]) if len(last_values) >= 7 else daily_sales['sales'].mean()
        
        X_future = np.array([[future_day_index, day_of_week, day_of_month, lag_1, lag_7, rolling_mean_7]])
        pred = model.predict(X_future)[0]
        
        future_dates.append(str(future_date.date()))
        predictions.append(float(pred))
        
        # Update last_values for next iteration
        last_values = np.append(last_values, pred)[-7:]
    
    # Calculate confidence intervals (using standard deviation)
    std_dev = daily_sales['sales'].std()
    lower_bound = [max(0, p - 1.96 * std_dev) for p in predictions]
    upper_bound = [p + 1.96 * std_dev for p in predictions]
    
    return {
        "forecast_days": forecast_days,
        "forecast_dates": future_dates,
        "forecast_values": [round(p, 2) for p in predictions],
        "lower_bound": [round(l, 2) for l in lower_bound],
        "upper_bound": [round(u, 2) for u in upper_bound],
        "total_forecast": round(sum(predictions), 2),
        "avg_daily_forecast": round(np.mean(predictions), 2)
    }


def prescriptive_action(diagnostic: dict, forecast: dict, descriptive: dict = None) -> dict:
    """
    Prescriptive analytics: suggest "what should be done"
    
    Generates actionable recommendations based on diagnostic and predictive insights.
    
    Args:
        diagnostic: Output from diagnostic_analysis
        forecast: Output from predictive_forecast
        descriptive: Output from descriptive_summary (optional)
    
    Returns:
        Dict with actionable strategies and rationales
    """
    actions = []
    
    # Address worst performing region
    worst_region = diagnostic.get('worst_region')
    if worst_region:
        actions.append({
            "action": f"Increase marketing budget in {worst_region} region by 25%",
            "rationale": f"{worst_region} is underperforming. Boost visibility through targeted campaigns.",
            "priority": "high",
            "confidence": 0.82
        })
    
    # Channel diversification
    for insight in diagnostic.get('insights', []):
        if insight.get('issue') == 'underperformance':
            actions.append({
                "action": f"Diversify sales channels in {insight['entity']}",
                "rationale": insight['root_cause'],
                "priority": "high",
                "confidence": insight['confidence']
            })
    
    # Forecast-based inventory planning
    avg_forecast = forecast.get('avg_daily_forecast', 0)
    if descriptive:
        current_avg = descriptive.get('avg_daily_sales', 0)
        if avg_forecast > current_avg * 1.1:
            actions.append({
                "action": "Increase inventory by 15% to meet projected demand",
                "rationale": f"Forecast shows {((avg_forecast/current_avg - 1) * 100):.1f}% increase in daily sales",
                "priority": "medium",
                "confidence": 0.75
            })
    
    # Product focus
    best_product = diagnostic.get('best_product')
    if best_product:
        actions.append({
            "action": f"Create bundle promotions featuring {best_product}",
            "rationale": f"{best_product} is the top performer. Leverage it to boost cross-sales.",
            "priority": "medium",
            "confidence": 0.88
        })
    
    # Volatility management
    if diagnostic.get('volatility_coefficient', 0) > 30:
        actions.append({
            "action": "Implement dynamic pricing to smooth demand fluctuations",
            "rationale": f"High sales volatility detected (CV={diagnostic['volatility_coefficient']:.1f}%)",
            "priority": "low",
            "confidence": 0.68
        })
    
    return {
        "actions": actions,
        "total_recommendations": len(actions),
        "high_priority_count": sum(1 for a in actions if a['priority'] == 'high')
    }