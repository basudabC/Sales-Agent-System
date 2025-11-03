"""
Orchestrator using LangGraph for multi-agent coordination.
Manages the sequential execution of analytics agents.
"""

from typing import TypedDict, Annotated, Sequence
import operator
from langgraph.graph import StateGraph, END
import pandas as pd
import json
import os

from groq_llm import get_groq_llm
from tools_agents import create_agent_tools
from analytics import (
    descriptive_summary,
    diagnostic_analysis,
    predictive_forecast,
    prescriptive_action
)


class AgentState(TypedDict):
    """State object passed between nodes in the graph"""
    query: str
    date_from: str | None
    date_to: str | None
    filters: dict | None
    descriptive_result: dict | None
    diagnostic_result: dict | None
    predictive_result: dict | None
    prescriptive_result: dict | None
    final_output: dict | None
    explainability: list | None
    error: str | None


class SalesOrchestrator:
    """
    Main orchestrator class using LangGraph to coordinate analytics agents
    """
    
    def __init__(self, dataframe: pd.DataFrame, api_key: str = None):
        """
        Initialize the orchestrator
        
        Args:
            dataframe: Sales data to analyze
            api_key: Optional Groq API key
        """
        self.df = dataframe
        self.llm = get_groq_llm(api_key=api_key)
        self.tools = create_agent_tools(dataframe)
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph execution flow
        
        Returns:
            Compiled StateGraph
        """
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("parse_query", self.parse_query_node)
        workflow.add_node("descriptive", self.descriptive_node)
        workflow.add_node("diagnostic", self.diagnostic_node)
        workflow.add_node("predictive", self.predictive_node)
        workflow.add_node("prescriptive", self.prescriptive_node)
        workflow.add_node("explain", self.explain_node)
        
        # Define edges (execution flow)
        workflow.set_entry_point("parse_query")
        workflow.add_edge("parse_query", "descriptive")
        workflow.add_edge("descriptive", "diagnostic")
        workflow.add_edge("diagnostic", "predictive")
        workflow.add_edge("predictive", "prescriptive")
        workflow.add_edge("prescriptive", "explain")
        workflow.add_edge("explain", END)
        
        return workflow.compile()
    
    def parse_query_node(self, state: AgentState) -> AgentState:
        """
        Parse user query and extract parameters
        """
        query = state.get("query", "")
        
        # Simple keyword extraction (can be enhanced with LLM)
        date_from = state.get("date_from")
        date_to = state.get("date_to")
        filters = state.get("filters")
        
        # Use LLM to understand intent if needed
        if not date_from and not date_to and not filters:
            # Default to last 30 days
            import datetime
            date_to = datetime.datetime.now().strftime("%Y-%m-%d")
            date_from = (datetime.datetime.now() - datetime.timedelta(days=30)).strftime("%Y-%m-%d")
        
        state["date_from"] = date_from
        state["date_to"] = date_to
        state["filters"] = filters or {}
        
        return state
    
    def descriptive_node(self, state: AgentState) -> AgentState:
        """
        Execute descriptive analysis
        """
        try:
            result = descriptive_summary(
                self.df.copy(),
                date_from=state.get("date_from"),
                date_to=state.get("date_to"),
                filters=state.get("filters")
            )
            state["descriptive_result"] = result
        except Exception as e:
            state["error"] = f"Descriptive analysis error: {str(e)}"
        
        return state
    
    def diagnostic_node(self, state: AgentState) -> AgentState:
        """
        Execute diagnostic analysis
        """
        try:
            result = diagnostic_analysis(
                self.df.copy(),
                descriptive_output=state.get("descriptive_result")
            )
            state["diagnostic_result"] = result
        except Exception as e:
            state["error"] = f"Diagnostic analysis error: {str(e)}"
        
        return state
    
    def predictive_node(self, state: AgentState) -> AgentState:
        """
        Execute predictive forecast
        """
        try:
            result = predictive_forecast(
                self.df.copy(),
                forecast_days=14
            )
            state["predictive_result"] = result
        except Exception as e:
            state["error"] = f"Predictive analysis error: {str(e)}"
        
        return state
    
    def prescriptive_node(self, state: AgentState) -> AgentState:
        """
        Execute prescriptive recommendations
        """
        try:
            result = prescriptive_action(
                diagnostic=state.get("diagnostic_result", {}),
                forecast=state.get("predictive_result", {}),
                descriptive=state.get("descriptive_result")
            )
            state["prescriptive_result"] = result
        except Exception as e:
            state["error"] = f"Prescriptive analysis error: {str(e)}"
        
        return state
    
    def explain_node(self, state: AgentState) -> AgentState:
        """
        Generate explainability summary using LLM
        """
        try:
            # Build context for LLM
            context = f"""
Analyze the following results and provide clear explanations:

Descriptive Summary:
{json.dumps(state.get('descriptive_result', {}), indent=2)}

Diagnostic Insights:
{json.dumps(state.get('diagnostic_result', {}), indent=2)}

Predictive Forecast:
{json.dumps(state.get('predictive_result', {}), indent=2)}

Prescriptive Actions:
{json.dumps(state.get('prescriptive_result', {}), indent=2)}

Provide a brief, actionable summary of key findings and recommendations.
"""
            
            explanation = self.llm(context)
            
            # Build explainability array
            explainability = []
            
            # Add insights from diagnostic
            for insight in state.get("diagnostic_result", {}).get("insights", []):
                explainability.append({
                    "layer": "diagnostic",
                    "entity": insight.get("entity"),
                    "finding": insight.get("root_cause"),
                    "confidence": insight.get("confidence")
                })
            
            # Add forecast summary
            explainability.append({
                "layer": "predictive",
                "entity": "future_sales",
                "finding": f"Forecast shows average daily sales of {state.get('predictive_result', {}).get('avg_daily_forecast', 0):.2f}",
                "confidence": 0.75
            })
            
            # Add top action
            actions = state.get("prescriptive_result", {}).get("actions", [])
            if actions:
                top_action = actions[0]
                explainability.append({
                    "layer": "prescriptive",
                    "entity": "recommendation",
                    "finding": top_action.get("action"),
                    "confidence": top_action.get("confidence")
                })
            
            state["explainability"] = explainability
            
            # Build final output
            state["final_output"] = {
                "descriptive": state.get("descriptive_result"),
                "diagnostic": state.get("diagnostic_result"),
                "predictive": state.get("predictive_result"),
                "prescriptive": state.get("prescriptive_result"),
                "explainability": explainability,
                "llm_summary": explanation
            }
            
        except Exception as e:
            state["error"] = f"Explanation error: {str(e)}"
        
        return state
    
    def analyze(self, query: str, date_from: str = None, date_to: str = None, 
                filters: dict = None) -> dict:
        """
        Main entry point for analysis
        
        Args:
            query: Natural language query
            date_from: Start date (YYYY-MM-DD)
            date_to: End date (YYYY-MM-DD)
            filters: Dict of filters
        
        Returns:
            Complete analysis results
        """
        initial_state = {
            "query": query,
            "date_from": date_from,
            "date_to": date_to,
            "filters": filters,
            "descriptive_result": None,
            "diagnostic_result": None,
            "predictive_result": None,
            "prescriptive_result": None,
            "final_output": None,
            "explainability": None,
            "error": None
        }
        
        try:
            # Execute graph
            final_state = self.graph.invoke(initial_state)
            
            # Check for errors
            if final_state.get("error"):
                print(f"Error in analysis: {final_state['error']}")
                return {
                    "error": final_state["error"],
                    "descriptive": final_state.get("descriptive_result"),
                    "diagnostic": final_state.get("diagnostic_result"),
                    "predictive": final_state.get("predictive_result"),
                    "prescriptive": final_state.get("prescriptive_result"),
                }
            
            return final_state.get("final_output", {})
        except Exception as e:
            print(f"Exception in analyze: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "error": str(e),
                "descriptive": None,
                "diagnostic": None,
                "predictive": None,
                "prescriptive": None,
            }


def build_agent_system(dataframe: pd.DataFrame, api_key: str = None) -> SalesOrchestrator:
    """
    Factory function to build the complete agent system
    
    Args:
        dataframe: Sales DataFrame
        api_key: Optional Groq API key
    
    Returns:
        Configured SalesOrchestrator instance
    """
    return SalesOrchestrator(dataframe, api_key=api_key)