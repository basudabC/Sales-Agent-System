"""
Synthetic sales dataset generator.
Creates realistic hierarchical sales data for testing.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random


def generate_sales_data(days: int = 120, save_to_file: bool = True, 
                       filename: str = "sales_sample.csv") -> pd.DataFrame:
    """
    Generate synthetic sales dataset
    
    Args:
        days: Number of days of data to generate
        save_to_file: Whether to save to CSV
        filename: Output filename
    
    Returns:
        DataFrame with sales data
    """
    
    # Define dimensions
    products = ["Product_A", "Product_B", "Product_C", "Product_D"]
    segments = ["Consumer", "Corporate", "Home_Office"]
    regions = ["East", "West", "North", "South"]
    channels = ["Online", "Retail", "Direct"]
    
    # Base sales ranges by product (to create realistic hierarchy)
    product_base_sales = {
        "Product_A": (500, 1500),
        "Product_B": (300, 1000),
        "Product_C": (400, 1200),
        "Product_D": (200, 800)
    }
    
    # Regional multipliers
    region_multipliers = {
        "East": 0.85,
        "West": 1.15,
        "North": 1.0,
        "South": 0.95
    }
    
    # Channel multipliers
    channel_multipliers = {
        "Online": 1.2,
        "Retail": 1.0,
        "Direct": 0.9
    }
    
    # Segment multipliers
    segment_multipliers = {
        "Consumer": 1.1,
        "Corporate": 1.3,
        "Home_Office": 0.9
    }
    
    # Generate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate records
    records = []
    
    for date in date_range:
        # Day of week effect (weekends lower)
        day_of_week = date.dayofweek
        if day_of_week >= 5:  # Weekend
            dow_multiplier = 0.7
        else:
            dow_multiplier = 1.0
        
        # Seasonal trend (slight upward)
        days_since_start = (date - start_date).days
        trend_multiplier = 1.0 + (days_since_start / days) * 0.3
        
        # Random noise
        noise = np.random.normal(1.0, 0.15)
        
        # Generate transactions for each combination
        for product in products:
            for segment in segments:
                for region in regions:
                    for channel in channels:
                        # Probability of transaction occurring
                        if random.random() > 0.3:  # 70% chance of transaction
                            
                            # Calculate base sales
                            base_min, base_max = product_base_sales[product]
                            base_sales = random.uniform(base_min, base_max)
                            
                            # Apply all multipliers
                            final_sales = (
                                base_sales *
                                region_multipliers[region] *
                                channel_multipliers[channel] *
                                segment_multipliers[segment] *
                                dow_multiplier *
                                trend_multiplier *
                                noise
                            )
                            
                            # Ensure non-negative
                            final_sales = max(0, final_sales)
                            
                            records.append({
                                "date": date.strftime("%Y-%m-%d"),
                                "product": product,
                                "segment": segment,
                                "region": region,
                                "channel": channel,
                                "sales": round(final_sales, 2)
                            })
    
    # Create DataFrame
    df = pd.DataFrame(records)
    
    # Sort by date
    df = df.sort_values("date").reset_index(drop=True)
    
    # Save to file if requested
    if save_to_file:
        df.to_csv(filename, index=False)
        print(f"✓ Generated {len(df):,} records and saved to {filename}")
        print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"  Total sales: ${df['sales'].sum():,.2f}")
    
    return df


def load_sales_data(filename: str = "sales_sample.csv") -> pd.DataFrame:
    """
    Load sales data from CSV
    
    Args:
        filename: CSV filename
    
    Returns:
        DataFrame with sales data
    """
    try:
        df = pd.read_csv(filename)
        df['date'] = pd.to_datetime(df['date'])
        return df
    except FileNotFoundError:
        print(f"File {filename} not found. Generating new data...")
        return generate_sales_data(save_to_file=True, filename=filename)


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Get quick summary statistics of the dataset
    
    Args:
        df: Sales DataFrame
    
    Returns:
        Dict with summary statistics
    """
    return {
        "total_records": len(df),
        "date_range": {
            "start": str(df['date'].min()),
            "end": str(df['date'].max())
        },
        "total_sales": round(df['sales'].sum(), 2),
        "avg_sales_per_record": round(df['sales'].mean(), 2),
        "dimensions": {
            "products": df['product'].nunique(),
            "segments": df['segment'].nunique(),
            "regions": df['region'].nunique(),
            "channels": df['channel'].nunique()
        }
    }


if __name__ == "__main__":
    # Generate sample data when run directly
    df = generate_sales_data(days=120, save_to_file=True)
    
    # Display summary
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    summary = get_data_summary(df)
    print(f"Total Records: {summary['total_records']:,}")
    print(f"Date Range: {summary['date_range']['start']} to {summary['date_range']['end']}")
    print(f"Total Sales: ${summary['total_sales']:,.2f}")
    print(f"Average per Record: ${summary['avg_sales_per_record']:,.2f}")
    print(f"\nDimensions:")
    for dim, count in summary['dimensions'].items():
        print(f"  {dim}: {count}")
    print("="*60)