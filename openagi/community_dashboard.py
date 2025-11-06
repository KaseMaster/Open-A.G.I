#!/usr/bin/env python3
"""
Community Dashboard for Quantum Currency
Implements real-time analytics and community participation features
"""

import sys
import os
import json
import time
import threading
import logging
import hashlib
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import our existing modules
from openagi.validator_console import ValidatorManagementConsole
from openagi.onchain_governance import OnChainGovernanceSystem
from openagi.token_economy_simulation import TokenEconomySimulation

@dataclass
class CommunityMetric:
    """Represents a community metric"""
    metric_name: str
    value: float
    timestamp: float
    description: str
    trend: str  # "up", "down", "stable"

@dataclass
class CommunityEvent:
    """Represents a community event"""
    event_id: str
    timestamp: float
    event_type: str  # "proposal", "vote", "transaction", "alert", "milestone"
    title: str
    description: str
    priority: str  # "low", "medium", "high"
    related_entities: List[str]  # Validator IDs, proposal IDs, etc.

@dataclass
class CommunityMember:
    """Represents a community member"""
    member_id: str
    username: str
    reputation_score: float
    activity_level: float  # 0.0 to 1.0
    roles: List[str]  # "validator", "developer", "contributor", "user"
    join_date: float
    last_active: float
    contributions: int

class CommunityDashboard:
    """
    Implements real-time analytics and community dashboard for open participation
    """
    
    def __init__(self, dashboard_name: str = "community-dashboard"):
        self.dashboard_name = dashboard_name
        self.dashboard_id = f"dashboard-{int(time.time())}-{hashlib.md5(dashboard_name.encode()).hexdigest()[:8]}"
        self.metrics: List[CommunityMetric] = []
        self.events: List[CommunityEvent] = []
        self.members: Dict[str, CommunityMember] = {}
        self.validator_console = ValidatorManagementConsole()
        self.governance_system = OnChainGovernanceSystem()
        self.economy_simulation = TokenEconomySimulation()
        self.dashboard_config = {
            "update_interval": 10.0,  # seconds
            "event_retention_days": 30,
            "max_events": 1000,
            "metrics_retention_days": 7
        }
        self._setup_logging()
        self._start_dashboard_thread()
    
    def _setup_logging(self):
        """Set up logging for the dashboard"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"community_dashboard_{self.dashboard_id}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(f"CommunityDashboard-{self.dashboard_id}")
    
    def _start_dashboard_thread(self):
        """Start background thread for dashboard updates"""
        self.dashboard_thread = threading.Thread(target=self._dashboard_loop, daemon=True)
        self.dashboard_thread.start()
        self.logger.info("Started dashboard update thread")
    
    def _dashboard_loop(self):
        """Background dashboard update loop"""
        while True:
            try:
                self._update_metrics()
                self._update_events()
                self._update_members()
                time.sleep(self.dashboard_config["update_interval"])
            except Exception as e:
                self.logger.error(f"Error in dashboard loop: {e}")
    
    def _update_metrics(self):
        """Update community metrics"""
        # Get data from various systems
        system_overview = self.validator_console.get_system_overview()
        governance_stats = self.governance_system.get_governance_stats()
        economy_metrics = self.economy_simulation.calculate_equilibrium_metrics()
        
        # Create metrics
        timestamp = time.time()
        
        # Network metrics
        self._add_metric("network_nodes", float(system_overview["total_nodes"]), timestamp,
                        "Total validator nodes in the network", "stable")
        self._add_metric("network_uptime", 
                        float(np.mean([m.uptime for m in self.validator_console.nodes.values()]) if self.validator_console.nodes else 0),
                        timestamp, "Average network uptime", "stable")
        self._add_metric("network_coherence", 
                        float(np.mean([m.harmonic_coherence for m in self.validator_console.nodes.values()]) if self.validator_console.nodes else 0),
                        timestamp, "Average harmonic coherence score", "stable")
        
        # Governance metrics
        self._add_metric("proposals_total", float(governance_stats["total_proposals"]), timestamp,
                        "Total governance proposals", "stable")
        self._add_metric("proposals_passed", float(governance_stats["passed_proposals"]), timestamp,
                        "Passed governance proposals", "stable")
        self._add_metric("voter_participation", 
                        float(governance_stats["total_votes"] / max(governance_stats["total_proposals"], 1)),
                        timestamp, "Average voter participation per proposal", "stable")
        
        # Economy metrics
        self._add_metric("total_market_cap", float(economy_metrics["total_market_cap"]), timestamp,
                        "Total market capitalization", "stable")
        self._add_metric("token_cycle_efficiency", float(self.economy_simulation.get_token_cycle_efficiency()), timestamp,
                        "Token cycle efficiency score", "stable")
        
        # Token-specific metrics
        for token_type, ratio in economy_metrics["token_ratios"].items():
            self._add_metric(f"token_ratio_{token_type.lower()}", float(ratio), timestamp,
                            f"Supply ratio for {token_type} token", "stable")
    
    def _add_metric(self, name: str, value: float, timestamp: float, description: str, trend: str):
        """Add a metric to the dashboard"""
        metric = CommunityMetric(
            metric_name=name,
            value=value,
            timestamp=timestamp,
            description=description,
            trend=trend
        )
        self.metrics.append(metric)
    
    def _update_events(self):
        """Update community events"""
        # Get recent events from various systems
        
        # Get recent alerts from validator console
        recent_alerts = self.validator_console.get_active_alerts()
        for alert in recent_alerts[-5:]:  # Last 5 alerts
            self._add_event(
                event_type="alert",
                title=f"Node Alert: {alert.node_id}",
                description=alert.message,
                priority=alert.severity,
                related_entities=[alert.node_id]
            )
        
        # Get recent proposals from governance system
        active_proposals = self.governance_system.get_active_proposals()
        for proposal_info in active_proposals[-3:]:  # Last 3 proposals
            if proposal_info:
                self._add_event(
                    event_type="proposal",
                    title=f"New Proposal: {proposal_info['title']}",
                    description=f"Proposal {proposal_info['proposal_id']} is now {proposal_info['status']}",
                    priority="medium",
                    related_entities=[proposal_info['proposal_id'], proposal_info['proposer']]
                )
        
        # Get recent transactions from economy simulation
        if self.economy_simulation.simulation_history:
            recent_history = self.economy_simulation.simulation_history[-1]
            self._add_event(
                event_type="milestone",
                title="Economy Update",
                description=f"Market cap updated to ${recent_history['total_market_cap']:,.2f}",
                priority="low",
                related_entities=[]
            )
    
    def _add_event(self, event_type: str, title: str, description: str, 
                   priority: str, related_entities: List[str]):
        """Add an event to the dashboard"""
        # Check if similar event already exists recently
        for event in self.events[-10:]:  # Check last 10 events
            if (event.event_type == event_type and 
                event.title == title and 
                time.time() - event.timestamp < 300):  # Within 5 minutes
                return  # Don't create duplicate events
        
        event_id = f"event-{int(time.time())}-{hashlib.md5(title.encode()).hexdigest()[:8]}"
        
        event = CommunityEvent(
            event_id=event_id,
            timestamp=time.time(),
            event_type=event_type,
            title=title,
            description=description,
            priority=priority,
            related_entities=related_entities
        )
        
        self.events.append(event)
        
        # Limit events to max_events
        if len(self.events) > self.dashboard_config["max_events"]:
            self.events = self.events[-self.dashboard_config["max_events"]:]
    
    def _update_members(self):
        """Update community members"""
        # Get members from various systems
        
        # Get validators as members
        for validator_id, validator in self.validator_console.staking_system.validators.items():
            if validator_id not in self.members:
                member = CommunityMember(
                    member_id=validator_id,
                    username=f"validator-{validator_id.split('-')[-1]}",
                    reputation_score=validator.chr_score,
                    activity_level=validator.uptime,
                    roles=["validator"],
                    join_date=time.time() - np.random.uniform(0, 365*24*3600),  # Random join date within a year
                    last_active=time.time(),
                    contributions=np.random.randint(10, 100)
                )
                self.members[validator_id] = member
            else:
                # Update existing member
                member = self.members[validator_id]
                member.reputation_score = validator.chr_score
                member.activity_level = validator.uptime
                member.last_active = time.time()
    
    def get_recent_metrics(self, limit: int = 20) -> List[CommunityMetric]:
        """
        Get recent metrics
        
        Args:
            limit: Maximum number of metrics to return
            
        Returns:
            List of recent metrics
        """
        sorted_metrics = sorted(self.metrics, key=lambda x: x.timestamp, reverse=True)
        return sorted_metrics[:limit]
    
    def get_metrics_by_name(self, metric_name: str) -> List[CommunityMetric]:
        """
        Get metrics by name
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            List of metrics with that name
        """
        return [m for m in self.metrics if m.metric_name == metric_name]
    
    def get_recent_events(self, limit: int = 20) -> List[CommunityEvent]:
        """
        Get recent events
        
        Args:
            limit: Maximum number of events to return
            
        Returns:
            List of recent events
        """
        sorted_events = sorted(self.events, key=lambda x: x.timestamp, reverse=True)
        return sorted_events[:limit]
    
    def get_events_by_type(self, event_type: str) -> List[CommunityEvent]:
        """
        Get events by type
        
        Args:
            event_type: Type of event
            
        Returns:
            List of events with that type
        """
        return [e for e in self.events if e.event_type == event_type]
    
    def get_community_members(self, limit: int = 50) -> List[CommunityMember]:
        """
        Get community members
        
        Args:
            limit: Maximum number of members to return
            
        Returns:
            List of community members
        """
        members_list = list(self.members.values())
        members_list.sort(key=lambda x: x.reputation_score, reverse=True)
        return members_list[:limit]
    
    def get_member(self, member_id: str) -> Optional[CommunityMember]:
        """
        Get a specific community member
        
        Args:
            member_id: ID of the member
            
        Returns:
            CommunityMember if found, None otherwise
        """
        return self.members.get(member_id)
    
    def get_dashboard_summary(self) -> Dict:
        """
        Get dashboard summary statistics
        
        Returns:
            Dictionary with dashboard summary
        """
        # Get recent metrics
        recent_metrics = self.get_recent_metrics(10)
        
        # Get recent events
        recent_events = self.get_recent_events(10)
        
        # Get community stats
        total_members = len(self.members)
        active_members = len([m for m in self.members.values() if time.time() - m.last_active < 24*3600])
        
        # Get event stats
        event_types = {}
        for event in self.events:
            event_types[event.event_type] = event_types.get(event.event_type, 0) + 1
        
        return {
            "dashboard_id": self.dashboard_id,
            "timestamp": time.time(),
            "total_metrics": len(self.metrics),
            "total_events": len(self.events),
            "total_members": total_members,
            "active_members": active_members,
            "recent_metrics": [
                {
                    "name": m.metric_name,
                    "value": m.value,
                    "description": m.description,
                    "trend": m.trend
                }
                for m in recent_metrics[:5]
            ],
            "recent_events": [
                {
                    "type": e.event_type,
                    "title": e.title,
                    "description": e.description,
                    "priority": e.priority,
                    "time_ago": int(time.time() - e.timestamp)
                }
                for e in recent_events[:5]
            ],
            "event_distribution": event_types,
            "validator_stats": self.validator_console.get_system_overview(),
            "governance_stats": self.governance_system.get_governance_stats(),
            "economy_stats": self.economy_simulation.calculate_equilibrium_metrics()
        }
    
    def search_community_data(self, query: str) -> Dict:
        """
        Search community data
        
        Args:
            query: Search query
            
        Returns:
            Dictionary with search results
        """
        results = {
            "metrics": [],
            "events": [],
            "members": []
        }
        
        # Search metrics
        query_lower = query.lower()
        for metric in self.metrics:
            if (query_lower in metric.metric_name.lower() or 
                query_lower in metric.description.lower()):
                results["metrics"].append({
                    "name": metric.metric_name,
                    "value": metric.value,
                    "description": metric.description,
                    "timestamp": metric.timestamp
                })
        
        # Search events
        for event in self.events:
            if (query_lower in event.title.lower() or 
                query_lower in event.description.lower() or
                query_lower in event.event_type.lower()):
                results["events"].append({
                    "type": event.event_type,
                    "title": event.title,
                    "description": event.description,
                    "timestamp": event.timestamp
                })
        
        # Search members
        for member in self.members.values():
            if (query_lower in member.username.lower() or
                query_lower in member.member_id.lower() or
                any(query_lower in role.lower() for role in member.roles)):
                results["members"].append({
                    "id": member.member_id,
                    "username": member.username,
                    "reputation": member.reputation_score,
                    "roles": member.roles
                })
        
        return results
    
    def generate_community_report(self) -> Dict:
        """
        Generate a comprehensive community report
        
        Returns:
            Dictionary with community report
        """
        summary = self.get_dashboard_summary()
        
        # Calculate community health score
        health_factors = []
        
        # Network health (20% weight)
        network_uptime = summary["validator_stats"].get("active_validators", 0) / max(summary["validator_stats"].get("total_validators", 1), 1)
        health_factors.append(network_uptime * 0.2)
        
        # Governance health (20% weight)
        governance_stats = summary["governance_stats"]
        if governance_stats["total_proposals"] > 0:
            governance_health = governance_stats["pass_rate"] * 0.2
        else:
            governance_health = 0.1  # Default if no proposals
        health_factors.append(governance_health)
        
        # Economy health (20% weight)
        economy_efficiency = summary["economy_stats"].get("token_cycle_efficiency", 0.5)
        health_factors.append(economy_efficiency * 0.2)
        
        # Community activity (20% weight)
        if summary["total_members"] > 0:
            activity_rate = summary["active_members"] / summary["total_members"]
        else:
            activity_rate = 0.0
        health_factors.append(activity_rate * 0.2)
        
        # Event activity (20% weight)
        recent_events = len([e for e in self.events if time.time() - e.timestamp < 7*24*3600])
        event_activity = min(recent_events / 50, 1.0)  # Normalize to 50 events per week
        health_factors.append(event_activity * 0.2)
        
        community_health = sum(health_factors) * 100  # Convert to percentage
        
        return {
            "report_timestamp": time.time(),
            "community_health": community_health,
            "health_status": "excellent" if community_health > 80 else "good" if community_health > 60 else "fair" if community_health > 40 else "poor",
            "summary": summary,
            "top_contributors": [
                {
                    "username": member.username,
                    "reputation": member.reputation_score,
                    "contributions": member.contributions
                }
                for member in sorted(self.members.values(), key=lambda x: x.contributions, reverse=True)[:10]
            ],
            "trending_metrics": [
                {
                    "name": metric.metric_name,
                    "value": metric.value,
                    "trend": metric.trend,
                    "description": metric.description
                }
                for metric in self.get_recent_metrics(10)
            ]
        }

def demo_community_dashboard():
    """Demonstrate community dashboard capabilities"""
    print("👥 Community Dashboard Demo")
    print("=" * 30)
    
    # Create dashboard instance
    dashboard = CommunityDashboard("Quantum Community Dashboard")
    
    # Let the dashboard initialize and collect some data
    print("\n📡 Initializing Dashboard...")
    time.sleep(5)  # Wait for some data collection
    
    # Show dashboard summary
    print("\n📊 Dashboard Summary:")
    summary = dashboard.get_dashboard_summary()
    print(f"   Dashboard ID: {summary['dashboard_id']}")
    print(f"   Total Metrics: {summary['total_metrics']}")
    print(f"   Total Events: {summary['total_events']}")
    print(f"   Community Members: {summary['total_members']}")
    print(f"   Active Members: {summary['active_members']}")
    
    # Show validator stats
    print("\n🏛️  Network Statistics:")
    validator_stats = summary["validator_stats"]
    print(f"   Total Nodes: {validator_stats.get('total_validators', 0)}")
    print(f"   Active Nodes: {validator_stats.get('active_validators', 0)}")
    print(f"   Total Staked: {validator_stats.get('total_staked', 0):,.2f}")
    print(f"   Total Delegated: {validator_stats.get('total_delegated', 0):,.2f}")
    
    # Show governance stats
    print("\n🗳️  Governance Statistics:")
    governance_stats = summary["governance_stats"]
    print(f"   Total Proposals: {governance_stats['total_proposals']}")
    print(f"   Passed Proposals: {governance_stats['passed_proposals']}")
    print(f"   Pass Rate: {governance_stats['pass_rate']:.1%}")
    print(f"   Total Votes: {governance_stats['total_votes']}")
    
    # Show economy stats
    print("\n💰 Economy Statistics:")
    economy_stats = summary["economy_stats"]
    print(f"   Total Market Cap: ${economy_stats['total_market_cap']:,.2f}")
    print(f"   Token Cycle Efficiency: {economy_stats.get('token_cycle_efficiency', 0):.3f}")
    
    # Show recent metrics
    print("\n📈 Recent Metrics:")
    recent_metrics = dashboard.get_recent_metrics(5)
    for metric in recent_metrics:
        trend_symbol = "↗️" if metric.trend == "up" else "↘️" if metric.trend == "down" else "➡️"
        print(f"   {trend_symbol} {metric.metric_name}: {metric.value:.3f} - {metric.description}")
    
    # Show recent events
    print("\n🔔 Recent Events:")
    recent_events = dashboard.get_recent_events(5)
    for event in recent_events:
        time_ago = int(time.time() - event.timestamp)
        time_unit = "seconds" if time_ago < 60 else "minutes" if time_ago < 3600 else "hours"
        time_value = time_ago if time_ago < 60 else time_ago // 60 if time_ago < 3600 else time_ago // 3600
        priority_symbol = "🔴" if event.priority == "high" else "🟡" if event.priority == "medium" else "🟢"
        print(f"   {priority_symbol} [{event.event_type.upper()}] {event.title}")
        print(f"      {event.description} ({time_value} {time_unit} ago)")
    
    # Show community members
    print("\n👥 Top Community Members:")
    top_members = dashboard.get_community_members(5)
    for member in top_members:
        role_badges = " ".join([f"[{role[:3].upper()}]" for role in member.roles])
        print(f"   {member.username} {role_badges}")
        print(f"      Reputation: {member.reputation_score:.3f}")
        print(f"      Contributions: {member.contributions}")
        print(f"      Activity: {member.activity_level:.1%}")
    
    # Show event distribution
    print("\n📊 Event Distribution:")
    event_dist = summary["event_distribution"]
    for event_type, count in event_dist.items():
        print(f"   {event_type.capitalize()}: {count}")
    
    # Search functionality
    print("\n🔍 Search Demo:")
    search_results = dashboard.search_community_data("validator")
    print(f"   Found {len(search_results['metrics'])} metrics")
    print(f"   Found {len(search_results['events'])} events")
    print(f"   Found {len(search_results['members'])} members")
    
    # Generate community report
    print("\n📋 Community Report:")
    report = dashboard.generate_community_report()
    print(f"   Community Health: {report['community_health']:.1f}/100")
    print(f"   Health Status: {report['health_status']}")
    
    # Show top contributors
    print("\n🏆 Top Contributors:")
    for contributor in report["top_contributors"][:3]:
        print(f"   {contributor['username']}: {contributor['contributions']} contributions (rep: {contributor['reputation']:.3f})")
    
    # Show trending metrics
    print("\n📈 Trending Metrics:")
    for metric in report["trending_metrics"][:3]:
        trend_symbol = "↗️" if metric['trend'] == "up" else "↘️" if metric['trend'] == "down" else "➡️"
        print(f"   {trend_symbol} {metric['name']}: {metric['value']:.3f} - {metric['description']}")
    
    print("\n✅ Community dashboard demo completed!")

if __name__ == "__main__":
    demo_community_dashboard()