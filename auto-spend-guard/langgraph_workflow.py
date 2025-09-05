"""
Auto-Spend-Guard LangGraph Workflow
AI-powered FinOps system for cloud cost analysis and optimization
"""

import os
import sys
from typing import Dict, Any, List
from datetime import datetime
import json
import yaml
import csv
from pathlib import Path

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

# Add evaluation directory to path
sys.path.append('evaluation')
from role_prompts import role_prompts, UserRole, QuestionType

# Initialize OpenAI client
os.environ.setdefault("OPENAI_API_KEY", "your-api-key-here")
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

class WorkflowState:
    """State for the workflow"""
    def __init__(self):
        self.question: str = ""
        self.classification: str = ""
        self.detected_role: str = "general"
        self.question_type: str = "cost_analysis"
        self.retrieved_data: Dict[str, Any] = {}
        self.final_answer: str = ""
        self.relevance_score: float = 0.0
        self.relevance_validated: bool = False
        self.latency_metrics: Dict[str, Any] = {}

class SpendAnalyzerWorkflow:
    """Main workflow for spend analysis"""
    
    def __init__(self):
        self.llm = llm
        self.role_prompts = role_prompts
        self.workflow = self._build_workflow()
        
    def _build_workflow(self):
        """Build the LangGraph workflow"""
        workflow = StateGraph(WorkflowState)
        
        # Add nodes
        workflow.add_node("classify_question", self._classify_question)
        workflow.add_node("retrieve_data", self._retrieve_data)
        workflow.add_node("generate_answer", self._generate_answer)
        workflow.add_node("validate_relevance", self._validate_relevance)
        
        # Add edges
        workflow.set_entry_point("classify_question")
        workflow.add_edge("classify_question", "retrieve_data")
        workflow.add_edge("retrieve_data", "generate_answer")
        workflow.add_edge("generate_answer", "validate_relevance")
        workflow.add_edge("validate_relevance", END)
        
        return workflow.compile()
    
    def _classify_question(self, state: WorkflowState) -> WorkflowState:
        """Classify the question and detect user role"""
        start_time = datetime.now()
        
        # Detect user role
        detected_role = self._detect_user_role(state.question)
        question_type = self._detect_question_type(state.question)
        
        # Generate role-aware prompt
        role_prompt = self._get_role_prompt(detected_role, question_type)
        
        # Classify the question
        classification_prompt = f"""
        {role_prompt}
        
        Classify the following question into one of these categories:
        - aws_costs: Questions about AWS service costs
        - budget_tracking: Questions about budget vs actuals
        - vendor_analysis: Questions about vendor spending
        - anomaly_detection: Questions about cost anomalies
        - cost_optimization: Questions about cost savings
        - forecasting: Questions about future spending
        
        Question: {state.question}
        
        Respond with just the category name.
        """
        
        response = self.llm.invoke([HumanMessage(content=classification_prompt)])
        classification = response.content.strip().lower()
        
        # Update state
        state.classification = classification
        state.detected_role = detected_role
        state.question_type = question_type
        
        # Record metrics
        duration = (datetime.now() - start_time).total_seconds()
        state.latency_metrics["question_classification"] = {
            "total_duration_seconds": duration,
            "input_tokens": len(state.question.split()),
            "output_tokens": len(response.content.split())
        }
        
        return state
    
    def _detect_user_role(self, question: str) -> str:
        """Detect user role from question"""
        question_lower = question.lower()
        
        # Check for role indicators
        for role, patterns in self.role_prompts.items():
            for pattern in patterns:
                if pattern.lower() in question_lower:
                    return role
        
        return "general"
    
    def _detect_question_type(self, question: str) -> str:
        """Detect question type from question"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ["cost", "spend", "expense"]):
            return "cost_analysis"
        elif any(word in question_lower for word in ["budget", "forecast", "plan"]):
            return "budget_tracking"
        elif any(word in question_lower for word in ["vendor", "saas", "subscription"]):
            return "vendor_analysis"
        elif any(word in question_lower for word in ["anomaly", "unusual", "spike"]):
            return "anomaly_detection"
        elif any(word in question_lower for word in ["optimize", "save", "reduce"]):
            return "cost_optimization"
        else:
            return "cost_analysis"
    
    def _get_role_prompt(self, role: str, question_type: str) -> str:
        """Get role-specific prompt"""
        if role in self.role_prompts:
            return self.role_prompts[role].get(question_type, self.role_prompts[role]["default"])
        return self.role_prompts["general"]["default"]
    
    def _retrieve_data(self, state: WorkflowState) -> WorkflowState:
        """Retrieve relevant data based on classification"""
        start_time = datetime.now()
        
        # Load sample data
        data = self._load_sample_data()
        
        # Filter data based on classification
        if state.classification == "aws_costs":
            state.retrieved_data = self._get_aws_cost_data(data)
        elif state.classification == "budget_tracking":
            state.retrieved_data = self._get_budget_data(data)
        elif state.classification == "vendor_analysis":
            state.retrieved_data = self._get_vendor_data(data)
        else:
            state.retrieved_data = data
        
        # Record metrics
        duration = (datetime.now() - start_time).total_seconds()
        state.latency_metrics["data_retrieval"] = {
            "total_duration_seconds": duration,
            "data_sources": list(state.retrieved_data.keys())
        }
        
        return state
    
    def _load_sample_data(self) -> Dict[str, Any]:
        """Load sample data from docs folder"""
        data = {}
        
        # Load cloud costs
        cloud_costs_path = Path("../docs/sample-cloud-costs.json")
        if cloud_costs_path.exists():
            with open(cloud_costs_path, 'r') as f:
                data["cloud_costs"] = json.load(f)
        
        # Load vendor data
        vendor_data_path = Path("../docs/sample-vendor-data.json")
        if vendor_data_path.exists():
            with open(vendor_data_path, 'r') as f:
                data["vendor_data"] = json.load(f)
        
        # Load budget data
        budget_path = Path("../docs/sample-budget-tracking.csv")
        if budget_path.exists():
            data["budget_data"] = self._load_csv_data(budget_path)
        
        return data
    
    def _load_csv_data(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load CSV data"""
        data = []
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
        return data
    
    def _get_aws_cost_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract AWS cost data"""
        if "cloud_costs" in data:
            return {"aws_costs": data["cloud_costs"]}
        return {"aws_costs": []}
    
    def _get_budget_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract budget data"""
        if "budget_data" in data:
            return {"budget_data": data["budget_data"]}
        return {"budget_data": []}
    
    def _get_vendor_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract vendor data"""
        if "vendor_data" in data:
            return {"vendor_data": data["vendor_data"]}
        return {"vendor_data": []}
    
    def _generate_answer(self, state: WorkflowState) -> WorkflowState:
        """Generate answer based on retrieved data"""
        start_time = datetime.now()
        
        # Get role-specific prompt
        role_prompt = self._get_role_prompt(state.detected_role, state.question_type)
        
        # Generate answer
        answer_prompt = f"""
        {role_prompt}
        
        Based on the following data, answer this question: {state.question}
        
        Data: {json.dumps(state.retrieved_data, indent=2)}
        
        Provide a comprehensive, role-appropriate answer.
        """
        
        response = self.llm.invoke([HumanMessage(content=answer_prompt)])
        state.final_answer = response.content
        
        # Record metrics
        duration = (datetime.now() - start_time).total_seconds()
        state.latency_metrics["answer_generation"] = {
            "total_duration_seconds": duration,
            "input_tokens": len(answer_prompt.split()),
            "output_tokens": len(response.content.split())
        }
        
        return state
    
    def _validate_relevance(self, state: WorkflowState) -> WorkflowState:
        """Validate answer relevance"""
        start_time = datetime.now()
        
        # Simple relevance check (you can enhance this)
        question_words = set(state.question.lower().split())
        answer_words = set(state.final_answer.lower().split())
        
        # Calculate relevance score
        common_words = question_words.intersection(answer_words)
        if len(question_words) > 0:
            relevance_score = len(common_words) / len(question_words)
        else:
            relevance_score = 0.0
        
        state.relevance_score = relevance_score
        state.relevance_validated = relevance_score > 0.3
        
        # Record metrics
        duration = (datetime.now() - start_time).total_seconds()
        state.latency_metrics["relevance_validation"] = {
            "total_duration_seconds": duration,
            "relevance_score": relevance_score
        }
        
        return state
    
    def run(self, question: str) -> Dict[str, Any]:
        """Run the workflow"""
        # Initialize state
        state = WorkflowState()
        state.question = question
        
        # Run workflow
        result = self.workflow.invoke(state)
        
        # Return results
        return {
            "classification": result.classification,
            "detected_role": result.detected_role,
            "question_type": result.question_type,
            "final_answer": result.final_answer,
            "relevance_score": result.relevance_score,
            "relevance_validated": result.relevance_validated,
            "latency_metrics": result.latency_metrics
        }

if __name__ == "__main__":
    # Test the workflow
    workflow = SpendAnalyzerWorkflow()
    result = workflow.run("How much did we spend on S3 last month?")
    print("Workflow Result:", json.dumps(result, indent=2))



