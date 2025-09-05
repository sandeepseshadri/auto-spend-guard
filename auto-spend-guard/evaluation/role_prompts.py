"""
Role-based prompting system for Auto-Spend-Guard
Provides role-specific prompts and question type detection
"""

from enum import Enum
from typing import Dict, List

class UserRole(Enum):
    EXECUTIVE = "executive"
    FINANCE_MANAGER = "finance_manager"
    ENGINEER = "engineer"
    OPERATIONS = "operations"
    ANALYST = "analyst"
    GENERAL = "general"

class QuestionType(Enum):
    COST_ANALYSIS = "cost_analysis"
    ANOMALY_DETECTION = "anomaly_detection"
    BUDGET_TRACKING = "budget_tracking"
    VENDOR_MANAGEMENT = "vendor_management"
    OPTIMIZATION = "optimization"
    FORECASTING = "forecasting"

# Role detection patterns
role_patterns = {
    "executive": [
        "executive", "ceo", "cto", "cfo", "vp", "director", "head of",
        "strategic", "business impact", "roi", "investment", "growth",
        "quarterly", "annual", "board", "stakeholder"
    ],
    "finance_manager": [
        "finance", "financial", "accounting", "budget", "cost control",
        "spending", "expenses", "revenue", "profit", "loss", "p&l",
        "fianance", "fiancial", "finacial", "finace", "finacial manager"
    ],
    "engineer": [
        "engineer", "engineering", "technical", "developer", "devops",
        "infrastructure", "aws", "azure", "gcp", "cloud", "code",
        "engineering manager", "tech lead", "architect", "sre"
    ],
    "operations": [
        "operations", "operational", "process", "efficiency", "workflow",
        "day to day", "daily", "management", "team lead", "supervisor"
    ],
    "analyst": [
        "analyst", "analysis", "data", "reporting", "insights", "trends",
        "metrics", "kpi", "dashboard", "business intelligence"
    ]
}

# Role-specific prompts
role_prompts = {
    "executive": {
        "default": "You are a strategic advisor providing high-level insights for executives. Focus on business impact, ROI, strategic implications, and executive-level summaries. Keep responses concise but comprehensive.",
        "cost_analysis": "Provide executive summary of cost analysis with business impact and strategic recommendations.",
        "budget_tracking": "Focus on budget performance against business objectives and strategic implications.",
        "vendor_analysis": "Emphasize vendor relationships, strategic partnerships, and business value.",
        "anomaly_detection": "Highlight business risks and strategic implications of cost anomalies.",
        "optimization": "Focus on strategic cost optimization opportunities and business value.",
        "forecasting": "Provide strategic outlook and business planning implications."
    },
    "finance_manager": {
        "default": "You are a financial expert providing detailed financial analysis. Focus on numbers, metrics, budget analysis, cost breakdowns, and financial reporting. Include specific figures and financial insights.",
        "cost_analysis": "Provide detailed cost breakdown with specific figures, trends, and financial metrics.",
        "budget_tracking": "Focus on budget vs actuals, variance analysis, and financial performance.",
        "vendor_analysis": "Emphasize vendor cost analysis, contract terms, and financial optimization.",
        "anomaly_detection": "Provide detailed financial impact analysis and risk assessment.",
        "optimization": "Focus on specific cost savings opportunities and financial ROI.",
        "forecasting": "Provide detailed financial projections and budget planning insights."
    },
    "engineer": {
        "default": "You are a technical expert providing detailed technical analysis. Focus on technical details, infrastructure costs, optimization opportunities, and engineering best practices. Include technical specifications and implementation details.",
        "cost_analysis": "Provide technical cost analysis with infrastructure details and optimization opportunities.",
        "budget_tracking": "Focus on technical budget allocation and resource optimization.",
        "vendor_analysis": "Emphasize technical vendor capabilities and integration considerations.",
        "anomaly_detection": "Provide technical root cause analysis and engineering solutions.",
        "optimization": "Focus on technical optimization strategies and implementation details.",
        "forecasting": "Provide technical capacity planning and infrastructure scaling insights."
    },
    "operations": {
        "default": "You are an operations expert focusing on process efficiency and operational insights. Focus on day-to-day operations, process optimization, and operational metrics. Provide actionable operational recommendations.",
        "cost_analysis": "Focus on operational cost drivers and process efficiency improvements.",
        "budget_tracking": "Emphasize operational budget utilization and process optimization.",
        "vendor_analysis": "Focus on operational vendor performance and service delivery.",
        "anomaly_detection": "Provide operational impact analysis and process improvement recommendations.",
        "optimization": "Focus on operational efficiency improvements and process optimization.",
        "forecasting": "Provide operational capacity planning and process scaling insights."
    },
    "analyst": {
        "default": "You are a data analyst providing comprehensive analytical insights. Focus on data patterns, trends, comprehensive analysis, and data-driven recommendations. Include detailed analysis and multiple perspectives.",
        "cost_analysis": "Provide comprehensive cost analysis with multiple perspectives and detailed insights.",
        "budget_tracking": "Focus on comprehensive budget analysis with trend analysis and forecasting.",
        "vendor_analysis": "Provide comprehensive vendor analysis with performance metrics and trends.",
        "anomaly_detection": "Provide comprehensive anomaly analysis with root cause investigation.",
        "optimization": "Focus on comprehensive optimization analysis with multiple scenarios.",
        "forecasting": "Provide comprehensive forecasting analysis with multiple models and scenarios."
    },
    "general": {
        "default": "You are a helpful financial advisor providing accessible insights for all users. Focus on clear explanations, practical insights, and actionable recommendations. Balance technical and business perspectives.",
        "cost_analysis": "Provide clear cost analysis with practical insights and recommendations.",
        "budget_tracking": "Focus on clear budget insights and practical financial guidance.",
        "vendor_analysis": "Provide clear vendor insights and practical vendor management advice.",
        "anomaly_detection": "Provide clear anomaly insights and practical risk management advice.",
        "optimization": "Focus on clear optimization opportunities and practical cost savings advice.",
        "forecasting": "Provide clear forecasting insights and practical planning advice."
    }
}

def detect_user_role(question: str) -> str:
    """
    Detect user role from question text
    Returns the detected role or 'general' if no specific role is detected
    """
    question_lower = question.lower()
    
    # Check for exact matches first
    for role, patterns in role_patterns.items():
        for pattern in patterns:
            if pattern.lower() in question_lower:
                return role
    
    # Check for partial matches
    for role, patterns in role_patterns.items():
        for pattern in patterns:
            if any(word in question_lower for word in pattern.lower().split()):
                return role
    
    # Check for fuzzy matches (common misspellings)
    for role, patterns in role_patterns.items():
        for pattern in patterns:
            if _fuzzy_match(pattern.lower(), question_lower):
                return role
    
    return "general"

def _fuzzy_match(pattern: str, text: str) -> bool:
    """
    Fuzzy matching for common misspellings and variations
    """
    # Common misspellings and variations
    variations = {
        "finance": ["fianance", "fiancial", "finacial", "finace"],
        "engineering": ["engineerin", "enginering", "engeneering"],
        "manager": ["manger", "maneger", "mangager"],
        "executive": ["executve", "executiv", "executie"],
        "technical": ["techical", "techncal", "techincal"]
    }
    
    for correct, misspellings in variations.items():
        if pattern == correct:
            return any(misspelling in text for misspelling in misspellings)
    
    return False

def get_role_prompt(role: str, question_type: str) -> str:
    """
    Get role-specific prompt for a given question type
    """
    if role in role_prompts:
        return role_prompts[role].get(question_type, role_prompts[role]["default"])
    return role_prompts["general"]["default"]

def get_available_roles() -> List[str]:
    """
    Get list of available user roles
    """
    return list(role_prompts.keys())

def get_available_question_types() -> List[str]:
    """
    Get list of available question types
    """
    return [
        "cost_analysis",
        "anomaly_detection", 
        "budget_tracking",
        "vendor_management",
        "optimization",
        "forecasting"
    ]

if __name__ == "__main__":
    # Test role detection
    test_questions = [
        "I am a finance manager and want to understand our budget",
        "As an engineer, how can we optimize our AWS costs?",
        "I need executive summary of our spending",
        "i am fianance guy want to understand if there is any anomoly in july cost",
        "I AM AN ENGINEERIN manager looking for cost optimization",
        "What are our operational costs this quarter?"
    ]
    
    print("Testing Role Detection:")
    print("=" * 50)
    
    for question in test_questions:
        detected_role = detect_user_role(question)
        print(f"Question: {question}")
        print(f"Detected Role: {detected_role}")
        print(f"Role Prompt: {get_role_prompt(detected_role, 'cost_analysis')[:100]}...")
        print("-" * 50)



