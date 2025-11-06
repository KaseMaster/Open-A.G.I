#!/usr/bin/env python3
"""
Token Economy Simulation for Quantum Currency System
Implements multi-token inflation/deflation equilibrium simulation with cyclic flow
"""

import sys
import os
import json
import time
import hashlib
import secrets
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import our existing modules

@dataclass
class TokenState:
    """Represents the state of a token in the economy"""
    token_type: str  # "CHR", "FLX", "PSY", "ATR", "RES"
    supply: float
    price: float
    velocity: float  # Transactions per unit time
    inflation_rate: float
    deflation_rate: float
    market_cap: float
    timestamp: float

@dataclass
class EconomicAgent:
    """Represents an economic agent in the simulation"""
    agent_id: str
    chr_balance: float
    flx_balance: float
    psy_balance: float
    atr_balance: float
    res_balance: float
    activity_level: float  # 0.0 to 1.0
    reputation_score: float  # 0.0 to 1.0

@dataclass
class MarketEvent:
    """Represents a market event that affects the economy"""
    event_type: str  # "mint", "burn", "trade", "external_shock", "conversion"
    timestamp: float
    token_type: str
    amount: float
    agent_id: Optional[str] = None
    description: Optional[str] = None

class TokenEconomySimulation:
    """
    Implements multi-token inflation/deflation equilibrium simulation with cyclic flow
    """
    
    def __init__(self):
        # Initialize token states with proper initial supplies and prices
        self.token_states: Dict[str, TokenState] = {
            "CHR": TokenState(
                token_type="CHR",
                supply=1000000.0,
                price=1.0,
                velocity=0.5,  # Lower velocity for non-transferable token
                inflation_rate=0.0,
                deflation_rate=0.0,
                market_cap=1000000.0,
                timestamp=time.time()
            ),
            "FLX": TokenState(
                token_type="FLX",
                supply=500000.0,
                price=2.0,
                velocity=2.0,
                inflation_rate=0.0,
                deflation_rate=0.0,
                market_cap=1000000.0,
                timestamp=time.time()
            ),
            "PSY": TokenState(
                token_type="PSY",
                supply=250000.0,
                price=4.0,
                velocity=1.0,  # Medium velocity for semi-transferable token
                inflation_rate=0.0,
                deflation_rate=0.0,
                market_cap=1000000.0,
                timestamp=time.time()
            ),
            "ATR": TokenState(
                token_type="ATR",
                supply=125000.0,
                price=8.0,
                velocity=0.8,  # Lower velocity for stakable token
                inflation_rate=0.0,
                deflation_rate=0.0,
                market_cap=1000000.0,
                timestamp=time.time()
            ),
            "RES": TokenState(
                token_type="RES",
                supply=62500.0,
                price=16.0,
                velocity=1.5,  # Higher velocity for multiplicative token
                inflation_rate=0.0,
                deflation_rate=0.0,
                market_cap=1000000.0,
                timestamp=time.time()
            )
        }
        
        # Initialize economic agents
        self.agents: List[EconomicAgent] = []
        self._initialize_agents()
        
        # Initialize market events
        self.market_events: List[MarketEvent] = []
        
        # Simulation parameters
        self.time_step = 1.0  # Time step in hours
        self.current_time = 0.0
        self.simulation_history: List[Dict] = []
        
        # Token conversion rates (cyclic flow based on token properties)
        self.conversion_rates = {
            ("CHR", "ATR"): 0.1,   # 10% of CHR converts to ATR (stability)
            ("FLX", "PSY"): 0.05,  # 5% of FLX converts to PSY (synchronization)
            ("PSY", "ATR"): 0.03,  # 3% of PSY converts to ATR (stability)
            ("ATR", "RES"): 0.02,  # 2% of ATR converts to RES (expansion)
            ("RES", "CHR"): 0.15,  # 15% of RES converts back to CHR (ethical alignment)
            ("FLX", "RES"): 0.01,  # 1% of FLX converts to RES (expansion)
            ("CHR", "RES"): 0.05   # 5% of CHR converts to RES (expansion)
        }
        
        # Token-specific parameters
        self.token_params = {
            "CHR": {
                "inflation_sensitivity": 0.01,   # Less sensitive to supply changes
                "deflation_sensitivity": 0.02,   # More sensitive to demand changes
                "max_supply_change": 0.05        # Max 5% supply change per step
            },
            "FLX": {
                "inflation_sensitivity": 0.03,
                "deflation_sensitivity": 0.04,
                "max_supply_change": 0.10
            },
            "PSY": {
                "inflation_sensitivity": 0.02,
                "deflation_sensitivity": 0.03,
                "max_supply_change": 0.08
            },
            "ATR": {
                "inflation_sensitivity": 0.015,
                "deflation_sensitivity": 0.025,
                "max_supply_change": 0.07
            },
            "RES": {
                "inflation_sensitivity": 0.04,   # More sensitive to supply changes
                "deflation_sensitivity": 0.05,   # More sensitive to demand changes
                "max_supply_change": 0.15        # Max 15% supply change per step
            }
        }
        
        # Economic parameters
        self.inflation_sensitivity = 0.02  # General sensitivity to supply changes
        self.deflation_sensitivity = 0.03  # General sensitivity to demand changes
        self.market_sentiment = 0.0  # -1.0 (bearish) to 1.0 (bullish)
        
        # Network health metrics
        self.network_health = {
            "coherence_score": 0.8,
            "validator_performance": 0.85,
            "transaction_volume": 1000.0,
            "security_level": 0.95
        }
    
    def _initialize_agents(self):
        """Initialize economic agents for the simulation"""
        agent_types = [
            ("validator", 0.9),    # High reputation validators
            ("developer", 0.7),    # Active developers
            ("trader", 0.5),       # Active traders
            ("user", 0.3),         # Regular users
            ("speculator", 0.2)    # Low reputation speculators
        ]
        
        for i, (agent_type, reputation) in enumerate(agent_types):
            # Create multiple agents of each type
            for j in range(5):
                agent = EconomicAgent(
                    agent_id=f"{agent_type}-{i}-{j}",
                    chr_balance=np.random.exponential(1000),
                    flx_balance=np.random.exponential(500),
                    psy_balance=np.random.exponential(250),
                    atr_balance=np.random.exponential(125),
                    res_balance=np.random.exponential(62),
                    activity_level=np.random.beta(2, 2),  # Beta distribution for activity
                    reputation_score=reputation + np.random.normal(0, 0.1)
                )
                # Ensure reputation is within bounds
                agent.reputation_score = max(0.0, min(1.0, agent.reputation_score))
                self.agents.append(agent)
    
    def update_token_state(self, token_type: str, delta_supply: float = 0.0, 
                          delta_demand: float = 0.0, external_factor: float = 0.0):
        """
        Update the state of a token based on supply/demand changes
        
        Args:
            token_type: Type of token to update
            delta_supply: Change in supply (positive = inflation, negative = deflation)
            delta_demand: Change in demand (positive = increased demand, negative = decreased)
            external_factor: External market factors affecting price
        """
        if token_type not in self.token_states:
            return
        
        token = self.token_states[token_type]
        
        # Get token-specific parameters
        params = self.token_params.get(token_type, {
            "inflation_sensitivity": self.inflation_sensitivity,
            "deflation_sensitivity": self.deflation_sensitivity,
            "max_supply_change": 0.1
        })
        
        # Limit supply change to prevent extreme fluctuations
        delta_supply = max(-token.supply * params["max_supply_change"], 
                          min(token.supply * params["max_supply_change"], delta_supply))
        
        # Update supply
        token.supply += delta_supply
        
        # Ensure supply doesn't go negative
        token.supply = max(0.0, token.supply)
        
        # Calculate inflation/deflation rates
        if delta_supply != 0:
            if delta_supply > 0:
                token.inflation_rate = delta_supply / token.supply if token.supply > 0 else 0.0
            else:
                token.deflation_rate = abs(delta_supply) / token.supply if token.supply > 0 else 0.0
        
        # Update price based on supply/demand dynamics
        # Simplified model: price is inversely related to supply and directly related to demand
        supply_factor = 1.0 - (delta_supply / token.supply * params["inflation_sensitivity"]) if token.supply > 0 else 1.0
        demand_factor = 1.0 + (delta_demand * params["deflation_sensitivity"])
        external_factor = 1.0 + external_factor
        
        # Update price
        token.price *= supply_factor * demand_factor * external_factor
        
        # Ensure price doesn't go negative
        token.price = max(0.01, token.price)
        
        # Update market cap
        token.market_cap = token.supply * token.price
        
        # Update timestamp
        token.timestamp = time.time()
    
    def simulate_token_conversions(self):
        """Simulate conversions between tokens in the cyclic economy"""
        # For each conversion pair, simulate conversions based on rates
        for (from_token, to_token), rate in self.conversion_rates.items():
            if from_token in self.token_states and to_token in self.token_states:
                from_state = self.token_states[from_token]
                to_state = self.token_states[to_token]
                
                # Calculate amount to convert based on rate and supply
                convert_amount = from_state.supply * rate * np.random.uniform(0.8, 1.2)
                
                # Ensure we don't convert more than available
                convert_amount = min(convert_amount, from_state.supply * 0.1)  # Max 10% per step
                
                # Update token states
                self.update_token_state(from_token, -convert_amount)
                
                # Conversion rate depends on network health and token properties
                network_factor = self.network_health["coherence_score"]
                conversion_efficiency = 0.8 + 0.2 * network_factor  # 80-100% efficiency
                
                # Convert to target token with efficiency factor
                converted_amount = convert_amount * (from_state.price / to_state.price) * conversion_efficiency
                self.update_token_state(to_token, converted_amount)
                
                # Record conversion event
                if convert_amount > 0:
                    event = MarketEvent(
                        event_type="conversion",
                        timestamp=self.current_time,
                        token_type=f"{from_token}->{to_token}",
                        amount=convert_amount,
                        description=f"Converted {convert_amount:.2f} {from_token} to {to_token} at efficiency {conversion_efficiency:.2%}"
                    )
                    self.market_events.append(event)
    
    def simulate_agent_activities(self):
        """Simulate activities of economic agents"""
        for agent in self.agents:
            # Agent activity affects token demand/supply
            activity_impact = agent.activity_level * agent.reputation_score
            
            # Validators mint FLX based on CHR reputation and coherence
            if "validator" in agent.agent_id and agent.chr_balance > 100:
                # Mint amount depends on CHR balance and network coherence
                coherence_factor = self.network_health["coherence_score"]
                mint_amount = agent.chr_balance * 0.01 * activity_impact * coherence_factor
                self.update_token_state("FLX", mint_amount)
                
                # Validators also earn ATR for stability
                atr_reward = mint_amount * 0.1 * coherence_factor
                self.update_token_state("ATR", atr_reward * 0.5)
                
                # Record minting event
                event = MarketEvent(
                    event_type="mint",
                    timestamp=self.current_time,
                    token_type="FLX",
                    amount=mint_amount,
                    agent_id=agent.agent_id,
                    description=f"Validator {agent.agent_id} minted {mint_amount:.2f} FLX with {atr_reward:.2f} ATR reward"
                )
                self.market_events.append(event)
            
            # Developers convert FLX to PSY for synchronization
            if "developer" in agent.agent_id and agent.flx_balance > 50:
                convert_amount = agent.flx_balance * 0.02 * activity_impact
                if convert_amount > 0:
                    # Update agent balances
                    agent.flx_balance -= convert_amount
                    # Conversion depends on network synchronization needs
                    sync_factor = self.network_health["validator_performance"]
                    psy_amount = convert_amount * (0.3 + 0.4 * sync_factor)
                    agent.psy_balance += psy_amount
                    
                    # Update token states
                    self.update_token_state("FLX", -convert_amount)
                    self.update_token_state("PSY", psy_amount * 0.9)  # Some efficiency loss
                    
                    # Record conversion event
                    event = MarketEvent(
                        event_type="conversion",
                        timestamp=self.current_time,
                        token_type="FLX->PSY",
                        amount=convert_amount,
                        agent_id=agent.agent_id,
                        description=f"Developer {agent.agent_id} converted {convert_amount:.2f} FLX to {psy_amount:.2f} PSY"
                    )
                    self.market_events.append(event)
            
            # Traders exchange tokens
            if "trader" in agent.agent_id:
                # Random trades between tokens
                tokens = ["CHR", "FLX", "PSY", "ATR", "RES"]
                from_token = np.random.choice(tokens)
                to_token = np.random.choice([t for t in tokens if t != from_token])
                
                # Get agent balance for the from_token
                balance_attr = f"{from_token.lower()}_balance"
                if hasattr(agent, balance_attr):
                    balance = getattr(agent, balance_attr)
                    if balance > 10:
                        trade_amount = balance * 0.05 * activity_impact
                        # Update agent balances
                        setattr(agent, balance_attr, balance - trade_amount)
                        # Update token states
                        self.update_token_state(from_token, -trade_amount)
                        self.update_token_state(to_token, trade_amount * 0.95)  # 5% fee
                        
                        # Record trade event
                        event = MarketEvent(
                            event_type="trade",
                            timestamp=self.current_time,
                            token_type=f"{from_token}->{to_token}",
                            amount=trade_amount,
                            agent_id=agent.agent_id,
                            description=f"Trader {agent.agent_id} traded {trade_amount:.2f} {from_token} for {to_token}"
                        )
                        self.market_events.append(event)
            
            # Users convert CHR to RES for network expansion
            if "user" in agent.agent_id and agent.chr_balance > 200:
                convert_amount = agent.chr_balance * 0.01 * activity_impact
                if convert_amount > 0:
                    # Update agent balances
                    agent.chr_balance -= convert_amount
                    # Conversion depends on network expansion needs
                    expansion_factor = self.network_health["transaction_volume"] / 1000.0
                    res_amount = convert_amount * 0.1 * expansion_factor
                    agent.res_balance += res_amount
                    
                    # Update token states
                    self.update_token_state("CHR", -convert_amount)
                    self.update_token_state("RES", res_amount * 0.95)  # Some efficiency loss
                    
                    # Record conversion event
                    event = MarketEvent(
                        event_type="conversion",
                        timestamp=self.current_time,
                        token_type="CHR->RES",
                        amount=convert_amount,
                        agent_id=agent.agent_id,
                        description=f"User {agent.agent_id} converted {convert_amount:.2f} CHR to {res_amount:.2f} RES"
                    )
                    self.market_events.append(event)
    
    def apply_external_shocks(self):
        """Apply random external market shocks"""
        # Occasionally apply market shocks
        if np.random.random() < 0.1:  # 10% chance per step
            shock_magnitude = np.random.normal(0, 0.05)  # 5% standard deviation
            affected_token = np.random.choice(list(self.token_states.keys()))
            
            self.update_token_state(affected_token, external_factor=shock_magnitude)
            
            # Record shock event
            event = MarketEvent(
                event_type="external_shock",
                timestamp=self.current_time,
                token_type=affected_token,
                amount=shock_magnitude,
                description=f"External market shock affected {affected_token} by {shock_magnitude:.2%}"
            )
            self.market_events.append(event)
            
            # Update market sentiment
            self.market_sentiment += shock_magnitude * 0.1
            self.market_sentiment = max(-1.0, min(1.0, self.market_sentiment))
    
    def simulate_resonance_multiplication(self):
        """Simulate the multiplicative properties of RES tokens"""
        res_state = self.token_states["RES"]
        if res_state.supply > 0:
            # RES multiplication depends on network health and transaction volume
            health_factor = self.network_health["coherence_score"]
            volume_factor = min(2.0, self.network_health["transaction_volume"] / 1000.0)
            
            # Multiplication rate (1-5% per step based on network conditions)
            multiplication_rate = 0.01 + 0.04 * health_factor * volume_factor
            
            # Calculate multiplication amount
            multiplication_amount = res_state.supply * multiplication_rate
            
            # Apply multiplication
            self.update_token_state("RES", multiplication_amount)
            
            # Distribute some to network for expansion incentives
            network_share = multiplication_amount * 0.2  # 20% to network
            self.update_token_state("RES", network_share * 0.1)  # Some efficiency loss
            
            # Record multiplication event
            if multiplication_amount > 0:
                event = MarketEvent(
                    event_type="multiplication",
                    timestamp=self.current_time,
                    token_type="RES",
                    amount=multiplication_amount,
                    description=f"RES multiplied by {multiplication_rate:.2%}, added {multiplication_amount:.2f} tokens"
                )
                self.market_events.append(event)
    
    def update_network_health(self):
        """Update network health metrics based on token economy state"""
        # Calculate network health based on token economy metrics
        total_market_cap = sum(token.market_cap for token in self.token_states.values())
        avg_price_stability = np.mean([abs(token.price - 1.0) for token in self.token_states.values()])
        
        # Update network health metrics
        self.network_health["coherence_score"] = max(0.1, min(1.0, 0.8 + np.random.normal(0, 0.05)))
        self.network_health["validator_performance"] = max(0.1, min(1.0, 0.85 + np.random.normal(0, 0.03)))
        self.network_health["transaction_volume"] = max(100.0, self.network_health["transaction_volume"] * (1.0 + np.random.normal(0, 0.1)))
        self.network_health["security_level"] = max(0.1, min(1.0, 0.95 + np.random.normal(0, 0.01)))
    
    def calculate_equilibrium_metrics(self) -> Dict:
        """
        Calculate equilibrium metrics for the token economy
        
        Returns:
            Dictionary with equilibrium metrics
        """
        metrics = {
            "timestamp": self.current_time,
            "total_market_cap": sum(token.market_cap for token in self.token_states.values()),
            "market_sentiment": self.market_sentiment,
            "network_health": self.network_health,
            "total_supply": {token_type: token.supply for token_type, token in self.token_states.items()},
            "average_price": {token_type: token.price for token_type, token in self.token_states.items()},
            "inflation_rates": {token_type: token.inflation_rate for token_type, token in self.token_states.items()},
            "deflation_rates": {token_type: token.deflation_rate for token_type, token in self.token_states.items()},
            "token_ratios": {},
            "velocity_metrics": {token_type: token.velocity for token_type, token in self.token_states.items()}
        }
        
        # Calculate token ratios
        total_supply = sum(token.supply for token in self.token_states.values())
        if total_supply > 0:
            for token_type, token in self.token_states.items():
                metrics["token_ratios"][token_type] = token.supply / total_supply
        
        return metrics
    
    def run_simulation_step(self):
        """Run one step of the token economy simulation"""
        # Update network health
        self.update_network_health()
        
        # Simulate token conversions
        self.simulate_token_conversions()
        
        # Simulate agent activities
        self.simulate_agent_activities()
        
        # Apply external shocks
        self.apply_external_shocks()
        
        # Simulate RES multiplication
        self.simulate_resonance_multiplication()
        
        # Calculate and store equilibrium metrics
        metrics = self.calculate_equilibrium_metrics()
        self.simulation_history.append(metrics)
        
        # Update time
        self.current_time += self.time_step
    
    def run_simulation(self, steps: int = 100):
        """
        Run the token economy simulation for a specified number of steps
        
        Args:
            steps: Number of simulation steps to run
        """
        print(f"Running token economy simulation for {steps} steps...")
        
        for step in range(steps):
            self.run_simulation_step()
            
            # Print progress every 10 steps
            if step % 10 == 0:
                metrics = self.calculate_equilibrium_metrics()
                print(f"Step {step}: Total Market Cap = ${metrics['total_market_cap']:,.2f}")
        
        print("Simulation completed!")
    
    def get_token_cycle_efficiency(self) -> float:
        """
        Calculate the efficiency of the token cycle
        
        Returns:
            Efficiency score between 0.0 and 1.0
        """
        # Efficiency is measured by how well the token supplies maintain their target ratios
        target_ratios = {
            "CHR": 0.32,  # 32% of total
            "FLX": 0.32,  # 32% of total
            "PSY": 0.16,  # 16% of total
            "ATR": 0.10,  # 10% of total
            "RES": 0.10   # 10% of total
        }
        
        current_ratios = {}
        total_supply = sum(token.supply for token in self.token_states.values())
        
        if total_supply > 0:
            for token_type, token in self.token_states.items():
                current_ratios[token_type] = token.supply / total_supply
        
        # Calculate deviation from target ratios
        total_deviation = 0.0
        for token_type in target_ratios:
            target = target_ratios[token_type]
            current = current_ratios.get(token_type, 0.0)
            deviation = abs(target - current)
            total_deviation += deviation
        
        # Efficiency is 1.0 minus normalized deviation
        max_deviation = 2.0  # Maximum possible deviation
        efficiency = 1.0 - (total_deviation / max_deviation)
        
        return max(0.0, min(1.0, efficiency))
    
    def visualize_simulation_results(self):
        """Visualize the simulation results"""
        if not self.simulation_history:
            print("No simulation data to visualize")
            return
        
        # Extract data for plotting
        timestamps = [entry["timestamp"] for entry in self.simulation_history]
        market_caps = [entry["total_market_cap"] for entry in self.simulation_history]
        sentiments = [entry["market_sentiment"] for entry in self.simulation_history]
        
        # Create plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Total market cap over time
        ax1.plot(timestamps, market_caps, 'b-', linewidth=2)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Total Market Cap ($)')
        ax1.set_title('Total Market Cap Over Time')
        ax1.grid(True, alpha=0.3)
        
        # Market sentiment over time
        ax2.plot(timestamps, sentiments, 'g-', linewidth=2)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Market Sentiment')
        ax2.set_title('Market Sentiment Over Time')
        ax2.grid(True, alpha=0.3)
        
        # Token supplies
        token_types = list(self.token_states.keys())
        for token_type in token_types:
            supplies = [entry["total_supply"][token_type] for entry in self.simulation_history]
            ax3.plot(timestamps, supplies, label=token_type, linewidth=2)
        
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Token Supply')
        ax3.set_title('Token Supplies Over Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Token prices
        for token_type in token_types:
            prices = [entry["average_price"][token_type] for entry in self.simulation_history]
            ax4.plot(timestamps, prices, label=token_type, linewidth=2)
        
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Token Price ($)')
        ax4.set_title('Token Prices Over Time')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def generate_simulation_report(self) -> Dict:
        """
        Generate a comprehensive simulation report
        
        Returns:
            Dictionary with simulation report
        """
        final_metrics = self.calculate_equilibrium_metrics()
        
        report = {
            "report_timestamp": time.time(),
            "simulation_duration": self.current_time,
            "total_steps": len(self.simulation_history),
            "final_metrics": final_metrics,
            "token_cycle_efficiency": self.get_token_cycle_efficiency(),
            "market_events_count": len(self.market_events),
            "total_agents": len(self.agents),
            "network_health": self.network_health,
            "equilibrium_analysis": {}
        }
        
        # Analyze equilibrium stability
        if len(self.simulation_history) > 10:
            recent_history = self.simulation_history[-10:]
            market_caps = [entry["total_market_cap"] for entry in recent_history]
            
            # Calculate volatility
            if len(market_caps) > 1:
                volatility = np.std(market_caps) / np.mean(market_caps)
                report["equilibrium_analysis"]["market_cap_volatility"] = volatility
                report["equilibrium_analysis"]["market_stability"] = "High" if volatility < 0.05 else "Medium" if volatility < 0.15 else "Low"
        
        # Token-specific analysis
        report["token_analysis"] = {}
        for token_type, token in self.token_states.items():
            report["token_analysis"][token_type] = {
                "final_supply": token.supply,
                "final_price": token.price,
                "final_market_cap": token.market_cap,
                "inflation_rate": token.inflation_rate,
                "deflation_rate": token.deflation_rate,
                "price_stability": "Stable" if abs(token.price - 1.0) < 0.5 else "Unstable"
            }
        
        return report

def demo_token_economy_simulation():
    """Demonstrate token economy simulation capabilities"""
    print("📊 Token Economy Simulation Demo")
    print("=" * 35)
    
    # Create simulation instance
    simulation = TokenEconomySimulation()
    
    # Show initial token states
    print("\n💰 Initial Token States:")
    for token_type, token in simulation.token_states.items():
        print(f"   {token_type}:")
        print(f"      Supply: {token.supply:,.2f}")
        print(f"      Price: ${token.price:.2f}")
        print(f"      Market Cap: ${token.market_cap:,.2f}")
    
    # Show initial agent information
    print(f"\n👥 Initial Agents: {len(simulation.agents)}")
    agent_types = {}
    for agent in simulation.agents:
        agent_type = agent.agent_id.split('-')[0]
        agent_types[agent_type] = agent_types.get(agent_type, 0) + 1
    
    for agent_type, count in agent_types.items():
        print(f"   {agent_type.capitalize()}s: {count}")
    
    # Run simulation
    print("\n📈 Running Simulation:")
    simulation.run_simulation(steps=50)
    
    # Show final token states
    print("\n💰 Final Token States:")
    for token_type, token in simulation.token_states.items():
        print(f"   {token_type}:")
        print(f"      Supply: {token.supply:,.2f}")
        print(f"      Price: ${token.price:.2f}")
        print(f"      Market Cap: ${token.market_cap:,.2f}")
        print(f"      Inflation Rate: {token.inflation_rate:.4f}")
        print(f"      Deflation Rate: {token.deflation_rate:.4f}")
    
    # Show simulation metrics
    print("\n📊 Simulation Metrics:")
    metrics = simulation.calculate_equilibrium_metrics()
    print(f"   Total Market Cap: ${metrics['total_market_cap']:,.2f}")
    print(f"   Market Sentiment: {metrics['market_sentiment']:.3f}")
    print(f"   Token Cycle Efficiency: {simulation.get_token_cycle_efficiency():.3f}")
    print(f"   Network Health - Coherence: {metrics['network_health']['coherence_score']:.3f}")
    print(f"   Network Health - Validator Performance: {metrics['network_health']['validator_performance']:.3f}")
    
    # Show token ratios
    print("\n⚖️  Token Supply Ratios:")
    for token_type, ratio in metrics['token_ratios'].items():
        print(f"   {token_type}: {ratio:.3f} ({ratio*100:.1f}%)")
    
    # Show some market events
    print(f"\n📰 Market Events: {len(simulation.market_events)}")
    if simulation.market_events:
        # Show last 5 events
        for event in simulation.market_events[-5:]:
            print(f"   {event.timestamp:.1f}: {event.event_type} - {event.description}")
    
    # Generate and show report
    print("\n📋 Simulation Report:")
    report = simulation.generate_simulation_report()
    print(f"   Duration: {report['simulation_duration']:.1f} time units")
    print(f"   Steps: {report['total_steps']}")
    print(f"   Token Cycle Efficiency: {report['token_cycle_efficiency']:.3f}")
    print(f"   Market Events: {report['market_events_count']}")
    
    if "equilibrium_analysis" in report and "market_stability" in report["equilibrium_analysis"]:
        print(f"   Market Stability: {report['equilibrium_analysis']['market_stability']}")
    
    # Show token-specific analysis
    print("\n🔬 Token Analysis:")
    for token_type, analysis in report["token_analysis"].items():
        print(f"   {token_type}:")
        print(f"      Price Stability: {analysis['price_stability']}")
        print(f"      Final Market Cap: ${analysis['final_market_cap']:,.2f}")
    
    print("\n✅ Token economy simulation demo completed!")

if __name__ == "__main__":
    demo_token_economy_simulation()