"""
LangChain Tool wrappers for each analytics agent.
Each tool encapsulates one layer of the analytics pipeline.
"""

from langchain.tools import BaseTool
from typing import Optional, Type
from pydantic import BaseModel, Field
import pandas as pd
import json
from analytics import (
    descriptive_summary,
    diagnostic_analysis,
    predictive_forecast,
    prescriptive_action
)


class DescriptiveInput(BaseModel):
    """Input schema for Descriptive Agent"""
    date_from: Optional[str] = Field(None, description="Start date in YYYY-MM-DD format")
    date_to: Optional[str] = Field(None, description="End date in YYYY-MM-DD format")
    filters: Optional[str] = Field(None, description="JSON string of filters, e.g., '{\"region\": \"West\"}'")


class DescriptiveTool(BaseTool):
    """
    Descriptive Analytics Tool: Summarizes "what happened"
    Returns KPIs, aggregations, and time series data
    """
    name: str = "descriptive_analysis"
    description: str = """
    Use this tool to get descriptive statistics and summaries of sales data.
    Returns total sales, averages, and breakdowns by region, product, segment, and channel.
    Input should include optional date_from, date_to, and filters as JSON.
    """
    args_schema: Type[BaseModel] = DescriptiveInput
    
    df: pd.DataFrame = None
    
    def __init__(self, dataframe: pd.DataFrame):
        super().__init__()
        self.df = dataframe
    
    def _run(self, date_from: Optional[str] = None, date_to: Optional[str] = None, 
             filters: Optional[str] = None) -> str:
        """Execute descriptive analysis"""
        try:
            # Parse filters if provided
            filter_dict = json.loads(filters) if filters else None
            
            # Run analysis
            result = descriptive_summary(
                self.df.copy(),
                date_from=date_from,
                date_to=date_to,
                filters=filter_dict
            )
            
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    async def _arun(self, *args, **kwargs):
        """Async version"""
        return self._run(*args, **kwargs)


class DiagnosticInput(BaseModel):
    """Input schema for Diagnostic Agent"""
    descriptive_output: Optional[str] = Field(None, description="JSON output from descriptive analysis")


class DiagnosticTool(BaseTool):
    """
    Diagnostic Analytics Tool: Explains "why it happened"
    Identifies anomalies, correlations, and root causes
    """
    name: str = "diagnostic_analysis"
    description: str = """
    Use this tool to understand why sales patterns occurred.
    Analyzes correlations, identifies underperforming dimensions, and explains root causes.
    Optionally accepts descriptive_output as JSON string.
    """
    args_schema: Type[BaseModel] = DiagnosticInput
    
    df: pd.DataFrame = None
    
    def __init__(self, dataframe: pd.DataFrame):
        super().__init__()
        self.df = dataframe
    
    def _run(self, descriptive_output: Optional[str] = None) -> str:
        """Execute diagnostic analysis"""
        try:
            # Parse descriptive output if provided
            desc_dict = json.loads(descriptive_output) if descriptive_output else None
            
            # Run analysis
            result = diagnostic_analysis(
                self.df.copy(),
                descriptive_output=desc_dict
            )
            
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    async def _arun(self, *args, **kwargs):
        """Async version"""
        return self._run(*args, **kwargs)


class PredictiveInput(BaseModel):
    """Input schema for Predictive Agent"""
    forecast_days: int = Field(14, description="Number of days to forecast (default: 14)")


class PredictiveTool(BaseTool):
    """
    Predictive Analytics Tool: Forecasts "what will happen"
    Uses RandomForest to predict future sales
    """
    name: str = "predictive_forecast"
    description: str = """
    Use this tool to forecast future sales.
    Predicts sales for the specified number of days using machine learning.
    Input: forecast_days (default 14)
    """
    args_schema: Type[BaseModel] = PredictiveInput
    
    df: pd.DataFrame = None
    
    def __init__(self, dataframe: pd.DataFrame):
        super().__init__()
        self.df = dataframe
    
    def _run(self, forecast_days: int = 14) -> str:
        """Execute predictive forecast"""
        try:
            result = predictive_forecast(
                self.df.copy(),
                forecast_days=forecast_days
            )
            
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    async def _arun(self, *args, **kwargs):
        """Async version"""
        return self._run(*args, **kwargs)


class PrescriptiveInput(BaseModel):
    """Input schema for Prescriptive Agent"""
    diagnostic_output: str = Field(..., description="JSON output from diagnostic analysis")
    forecast_output: str = Field(..., description="JSON output from predictive forecast")
    descriptive_output: Optional[str] = Field(None, description="JSON output from descriptive analysis")


class PrescriptiveTool(BaseTool):
    """
    Prescriptive Analytics Tool: Suggests "what should be done"
    Generates actionable recommendations based on insights
    """
    name: str = "prescriptive_recommendations"
    description: str = """
    Use this tool to get actionable recommendations.
    Requires diagnostic_output and forecast_output as JSON strings.
    Returns strategic actions with priority levels and confidence scores.
    """
    args_schema: Type[BaseModel] = PrescriptiveInput
    
    def _run(self, diagnostic_output: str, forecast_output: str, 
             descriptive_output: Optional[str] = None) -> str:
        """Execute prescriptive analysis"""
        try:
            # Parse inputs
            diagnostic_dict = json.loads(diagnostic_output)
            forecast_dict = json.loads(forecast_output)
            descriptive_dict = json.loads(descriptive_output) if descriptive_output else None
            
            # Run analysis
            result = prescriptive_action(
                diagnostic=diagnostic_dict,
                forecast=forecast_dict,
                descriptive=descriptive_dict
            )
            
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    async def _arun(self, *args, **kwargs):
        """Async version"""
        return self._run(*args, **kwargs)


def create_agent_tools(dataframe: pd.DataFrame) -> list:
    """
    Factory function to create all agent tools with the provided dataframe
    
    Args:
        dataframe: Sales DataFrame to be analyzed
    
    Returns:
        List of LangChain tool instances
    """
    return [
        DescriptiveTool(dataframe),
        DiagnosticTool(dataframe),
        PredictiveTool(dataframe),
        PrescriptiveTool()
    ]