#!/usr/bin/env python3
"""
Base Agent Class for Market Pulse AGI
Provides the foundation for all specialized AI agents in the system.
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

# AEGIS Framework imports
from aegis_framework import AEGISNode, SecureMessage
from crypto_framework import initialize_crypto

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    """Status of an AI agent"""
    IDLE = "idle"
    PROCESSING = "processing"
    ERROR = "error"
    OFFLINE = "offline"


class AgentCapability(Enum):
    """Capabilities that agents can have"""
    FINANCIAL_ANALYSIS = "financial_analysis"
    NEWS_INTELLIGENCE = "news_intelligence"
    SOCIAL_MEDIA_TRENDS = "social_media_trends"
    COMPETITIVE_INTELLIGENCE = "competitive_intelligence"
    SYNTHESIS = "synthesis"


@dataclass
class AgentTask:
    """Task assigned to an agent"""
    task_id: str
    query_id: str
    task_type: str
    parameters: Dict[str, Any]
    assigned_at: float
    deadline: float


@dataclass
class TaskResult:
    """Result from agent task processing"""
    task_id: str
    agent_id: str
    result_data: Dict[str, Any]
    confidence: float
    completed_at: float
    processing_time: float


class BaseAgent(ABC):
    """Base class for all AI agents in Market Pulse AGI"""
    
    def __init__(self, agent_id: str, capabilities: List[AgentCapability]):
        self.agent_id = agent_id
        self.capabilities = capabilities
        self.status = AgentStatus.IDLE
        self.current_task: Optional[AgentTask] = None
        self.task_history: List[AgentTask] = []
        self.result_history: List[TaskResult] = []
        self.reputation_score = 100.0
        
        # Initialize AEGIS node for secure communication
        self.aegis_node = AEGISNode(agent_id)
        self.crypto_engine = initialize_crypto({
            'security_level': 'HIGH',
            'node_id': agent_id
        })
        
        # Register task handler
        self.aegis_node.register_task_handler(self.handle_task)
        
        logger.info(f"Agent {agent_id} initialized with capabilities: {[c.value for c in capabilities]}")
    
    async def start(self):
        """Start the agent and connect to the network"""
        try:
            await self.aegis_node.start()
            self.status = AgentStatus.IDLE
            logger.info(f"Agent {self.agent_id} started successfully")
        except Exception as e:
            logger.error(f"Failed to start agent {self.agent_id}: {e}")
            self.status = AgentStatus.ERROR
            raise
    
    async def stop(self):
        """Stop the agent and disconnect from the network"""
        try:
            await self.aegis_node.stop()
            self.status = AgentStatus.OFFLINE
            logger.info(f"Agent {self.agent_id} stopped successfully")
        except Exception as e:
            logger.error(f"Error stopping agent {self.agent_id}: {e}")
    
    async def handle_task(self, message: SecureMessage):
        """Handle incoming task messages"""
        try:
            # Decrypt and parse the message
            task_data = json.loads(message.payload.decode())
            
            # Create task object
            task = AgentTask(
                task_id=task_data['task_id'],
                query_id=task_data['query_id'],
                task_type=task_data['task_type'],
                parameters=task_data['parameters'],
                assigned_at=task_data['assigned_at'],
                deadline=task_data['deadline']
            )
            
            # Process the task
            result = await self.process_task(task)
            
            # Send result back
            await self.send_result(task, result)
            
        except Exception as e:
            logger.error(f"Error handling task for agent {self.agent_id}: {e}")
            self.status = AgentStatus.ERROR
            # Send error response
            await self.send_error(message.sender_id, str(e))
    
    @abstractmethod
    async def process_task(self, task: AgentTask) -> TaskResult:
        """
        Process a task assigned to this agent.
        Must be implemented by subclasses.
        
        Args:
            task: The task to process
            
        Returns:
            TaskResult with processing results
        """
        pass
    
    async def send_result(self, task: AgentTask, result: TaskResult):
        """Send task result back to orchestrator"""
        try:
            result_data = asdict(result)
            await self.aegis_node.send_response(task.query_id, result_data)
            logger.info(f"Agent {self.agent_id} sent result for task {task.task_id}")
        except Exception as e:
            logger.error(f"Error sending result from agent {self.agent_id}: {e}")
    
    async def send_error(self, recipient_id: str, error_message: str):
        """Send error message to recipient"""
        try:
            error_data = {
                'agent_id': self.agent_id,
                'error': error_message,
                'timestamp': asyncio.get_event_loop().time()
            }
            await self.aegis_node.send_error(recipient_id, error_data)
            logger.info(f"Agent {self.agent_id} sent error to {recipient_id}")
        except Exception as e:
            logger.error(f"Error sending error from agent {self.agent_id}: {e}")
    
    def update_reputation(self, score_change: float):
        """Update agent reputation score"""
        self.reputation_score = max(0, min(1000, self.reputation_score + score_change))
        logger.info(f"Agent {self.agent_id} reputation updated to {self.reputation_score}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        return {
            'agent_id': self.agent_id,
            'status': self.status.value,
            'capabilities': [c.value for c in self.capabilities],
            'reputation_score': self.reputation_score,
            'current_task': asdict(self.current_task) if self.current_task else None,
            'tasks_processed': len(self.task_history),
            'results_generated': len(self.result_history)
        }


# Example implementation of a specialized agent
class ExampleFinancialAgent(BaseAgent):
    """Example implementation of a financial analysis agent"""
    
    def __init__(self, agent_id: str):
        super().__init__(agent_id, [AgentCapability.FINANCIAL_ANALYSIS])
    
    async def process_task(self, task: AgentTask) -> TaskResult:
        """Process financial analysis task"""
        self.status = AgentStatus.PROCESSING
        self.current_task = task
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Simulate financial analysis processing
            # In a real implementation, this would involve:
            # 1. Fetching financial data from APIs
            # 2. Running ML models for analysis
            # 3. Generating insights and predictions
            
            result_data = {
                'type': 'financial_analysis',
                'query_id': task.query_id,
                'analysis': {
                    'market_sentiment': 'positive',
                    'key_indicators': {
                        'gdp_growth': 2.1,
                        'inflation_rate': 3.2,
                        'unemployment_rate': 3.8
                    },
                    'stock_performance': {
                        'tech_sector': 'outperforming',
                        'consumer_goods': 'stable',
                        'energy': 'underperforming'
                    }
                },
                'sources': ['yahoo_finance', 'fred', 'sec_edgar'],
                'timestamp': start_time
            }
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            result = TaskResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                result_data=result_data,
                confidence=0.92,
                completed_at=asyncio.get_event_loop().time(),
                processing_time=processing_time
            )
            
            # Add to history
            self.task_history.append(task)
            self.result_history.append(result)
            
            self.status = AgentStatus.IDLE
            self.current_task = None
            
            return result
            
        except Exception as e:
            self.status = AgentStatus.ERROR
            self.current_task = None
            logger.error(f"Error processing task in financial agent {self.agent_id}: {e}")
            raise


# Example usage
if __name__ == "__main__":
    # This would typically be run as part of the agent network
    async def main():
        agent = ExampleFinancialAgent("financial_agent_001")
        await agent.start()
        
        # Agent would now listen for tasks
        # In a real scenario, this would run indefinitely
        try:
            await asyncio.sleep(3600)  # Run for 1 hour
        finally:
            await agent.stop()
    
    # Run the example
    asyncio.run(main())