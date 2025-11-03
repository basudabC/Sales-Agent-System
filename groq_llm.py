"""
Groq LLM wrapper for LangChain integration.
Provides a LangChain-compatible interface to the Groq API.
"""

from langchain_core.language_models.llms import LLM
from typing import Optional, List, Any
from pydantic import Field
import os


class GroqLLM(LLM):
    """
    Custom LangChain LLM wrapper for Groq API
    
    Attributes:
        api_key: Groq API key (reads from GROQ_API_KEY env var)
        model: Model name (default: mixtral-8x7b-32768)
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
    """
    
    api_key: str = Field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    model: str = "mixtral-8x7b-32768"
    temperature: float = 0.1
    max_tokens: int = 2000
    
    @property
    def _llm_type(self) -> str:
        """Return identifier for this LLM"""
        return "groq"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs: Any
    ) -> str:
        """
        Execute the LLM call
        
        Args:
            prompt: Input prompt
            stop: Stop sequences
            **kwargs: Additional parameters
        
        Returns:
            Generated text response
        """
        try:
            # Try importing groq
            try:
                from groq import Groq
                
                if not self.api_key:
                    return "Error: GROQ_API_KEY environment variable not set"
                
                client = Groq(api_key=self.api_key)
                
                # Make API call
                chat_completion = client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert data analyst coordinating multi-layered analytics."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stop=stop
                )
                
                return chat_completion.choices[0].message.content
                
            except ImportError:
                # Fallback: Use mock responses for development without API key
                return self._mock_response(prompt)
                
        except Exception as e:
            return f"Error calling Groq API: {str(e)}"
    
    def _mock_response(self, prompt: str) -> str:
        """
        Generate mock response for development/testing
        Used when Groq API is unavailable
        """
        if "descriptive" in prompt.lower():
            return """Based on the descriptive analysis:
- Total sales show strong performance across regions
- Regional distribution indicates opportunities for optimization
- Time series data reveals seasonal patterns
            
Next, I'll analyze the diagnostic insights to understand root causes."""
        
        elif "diagnostic" in prompt.lower():
            return """The diagnostic analysis reveals:
- Performance gaps between regions require attention
- Channel mix imbalances affecting certain markets
- Volatility patterns suggest need for demand smoothing

I'll now forecast future trends."""
        
        elif "predictive" in prompt.lower():
            return """The forecast indicates:
- Expected growth trend over the next 14 days
- Confidence intervals suggest stable demand
- Inventory planning should account for projected increase

Moving to prescriptive recommendations."""
        
        elif "prescriptive" in prompt.lower():
            return """Strategic recommendations:
1. Optimize channel distribution in underperforming regions
2. Adjust inventory levels based on forecast
3. Implement targeted marketing campaigns
4. Bundle top-performing products for cross-sell opportunities

These actions should improve overall sales performance."""
        
        else:
            return """Analysis complete. I've coordinated all four analytics layers:
1. Descriptive: Summarized current sales performance
2. Diagnostic: Identified root causes of patterns
3. Predictive: Forecasted future trends
4. Prescriptive: Generated actionable recommendations

The comprehensive results are available in the structured output."""
    
    @property
    def _identifying_params(self) -> dict:
        """Return identifying parameters"""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }


def get_groq_llm(
    api_key: Optional[str] = None,
    model: str = "mixtral-8x7b-32768",
    temperature: float = 0.1,
    max_tokens: int = 2000
) -> GroqLLM:
    """
    Factory function to create a GroqLLM instance
    
    Args:
        api_key: Optional API key (uses env var if not provided)
        model: Groq model name
        temperature: Sampling temperature
        max_tokens: Max tokens to generate
    
    Returns:
        Configured GroqLLM instance
    """
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key
    
    return GroqLLM(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens
    )