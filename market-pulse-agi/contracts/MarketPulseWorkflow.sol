// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title MarketPulseWorkflow
 * @dev Smart contract for orchestrating market intelligence workflows in the Market Pulse AGI dApp
 */
contract MarketPulseWorkflow {
    // Structs for workflow management
    struct Query {
        uint256 id;
        address requester;
        string question;
        uint256 timestamp;
        uint256 deadline;
        uint256 reward;
        WorkflowStatus status;
    }
    
    struct AgentTask {
        uint256 queryId;
        address agent;
        string taskType;
        string parameters;
        bool completed;
        string resultHash;
        uint256 completedAt;
    }
    
    struct AgentInfo {
        address agentAddress;
        string[] capabilities;
        uint256 reputationScore;
        uint256 lastSeen;
        bool active;
    }
    
    enum WorkflowStatus {
        Submitted,
        Assigned,
        Processing,
        Completed,
        Failed
    }
    
    // State variables
    mapping(uint256 => Query) public queries;
    mapping(uint256 => AgentTask[]) public agentTasks;
    mapping(address => AgentInfo) public registeredAgents;
    mapping(string => address[]) public capabilityIndex;
    
    uint256 public queryCounter;
    uint256 public baseReward = 0.01 ether;
    
    // Events
    event QuerySubmitted(uint256 indexed queryId, address requester, string question);
    event TaskAssigned(uint256 indexed queryId, address agent, string taskType);
    event TaskCompleted(uint256 indexed queryId, address agent, string resultHash);
    event ReportGenerated(uint256 indexed queryId, string finalReportHash);
    event AgentRegistered(address agent, string[] capabilities);
    event AgentRewardPaid(address agent, uint256 amount);
    
    // Modifiers
    modifier onlyRegisteredAgent() {
        require(registeredAgents[msg.sender].active, "Agent not registered");
        _;
    }
    
    modifier onlyQueryRequester(uint256 queryId) {
        require(queries[queryId].requester == msg.sender, "Not query requester");
        _;
    }
    
    // Constructor
    constructor() {
        queryCounter = 1;
    }
    
    /**
     * @dev Submit a new market intelligence query
     * @param question The natural language question to analyze
     * @param deadline The deadline for completion (in seconds from now)
     */
    function submitQuery(string memory question, uint256 deadline) public payable {
        require(bytes(question).length > 0, "Question cannot be empty");
        require(deadline > 0, "Deadline must be in the future");
        require(msg.value >= baseReward, "Insufficient reward");
        
        uint256 queryId = queryCounter++;
        
        queries[queryId] = Query({
            id: queryId,
            requester: msg.sender,
            question: question,
            timestamp: block.timestamp,
            deadline: block.timestamp + deadline,
            reward: msg.value,
            status: WorkflowStatus.Submitted
        });
        
        emit QuerySubmitted(queryId, msg.sender, question);
    }
    
    /**
     * @dev Register an AI agent with its capabilities
     * @param capabilities Array of capabilities the agent supports
     */
    function registerAgent(string[] memory capabilities) public {
        require(capabilities.length > 0, "Agent must have at least one capability");
        
        // Update agent info
        registeredAgents[msg.sender] = AgentInfo({
            agentAddress: msg.sender,
            capabilities: capabilities,
            reputationScore: 100, // Starting reputation score
            lastSeen: block.timestamp,
            active: true
        });
        
        // Update capability index
        for (uint i = 0; i < capabilities.length; i++) {
            capabilityIndex[capabilities[i]].push(msg.sender);
        }
        
        emit AgentRegistered(msg.sender, capabilities);
    }
    
    /**
     * @dev Assign a task to an agent
     * @param queryId The ID of the query
     * @param agent The address of the agent
     * @param taskType The type of task
     * @param parameters Task parameters as JSON string
     */
    function assignTask(uint256 queryId, address agent, string memory taskType, string memory parameters) public onlyQueryRequester(queryId) {
        require(registeredAgents[agent].active, "Agent not registered");
        require(queries[queryId].status == WorkflowStatus.Submitted, "Query not in submitted state");
        
        AgentTask memory task = AgentTask({
            queryId: queryId,
            agent: agent,
            taskType: taskType,
            parameters: parameters,
            completed: false,
            resultHash: "",
            completedAt: 0
        });
        
        agentTasks[queryId].push(task);
        queries[queryId].status = WorkflowStatus.Assigned;
        
        emit TaskAssigned(queryId, agent, taskType);
    }
    
    /**
     * @dev Complete a task and submit results
     * @param queryId The ID of the query
     * @param resultHash IPFS hash of the result data
     */
    function completeTask(uint256 queryId, string memory resultHash) public onlyRegisteredAgent {
        require(bytes(resultHash).length > 0, "Result hash cannot be empty");
        require(queries[queryId].status == WorkflowStatus.Assigned || queries[queryId].status == WorkflowStatus.Processing, "Query not in processing state");
        
        // Find the task assigned to this agent
        AgentTask[] storage tasks = agentTasks[queryId];
        bool taskFound = false;
        
        for (uint i = 0; i < tasks.length; i++) {
            if (tasks[i].agent == msg.sender && !tasks[i].completed) {
                tasks[i].completed = true;
                tasks[i].resultHash = resultHash;
                tasks[i].completedAt = block.timestamp;
                taskFound = true;
                break;
            }
        }
        
        require(taskFound, "Task not found or already completed");
        
        // Check if all tasks are completed
        bool allCompleted = true;
        for (uint i = 0; i < tasks.length; i++) {
            if (!tasks[i].completed) {
                allCompleted = false;
                break;
            }
        }
        
        if (allCompleted) {
            queries[queryId].status = WorkflowStatus.Completed;
        } else {
            queries[queryId].status = WorkflowStatus.Processing;
        }
        
        // Pay the agent
        uint256 agentReward = queries[queryId].reward / agentTasks[queryId].length;
        payable(msg.sender).transfer(agentReward);
        
        emit TaskCompleted(queryId, msg.sender, resultHash);
        emit AgentRewardPaid(msg.sender, agentReward);
    }
    
    /**
     * @dev Generate final report and complete workflow
     * @param queryId The ID of the query
     * @param finalReportHash IPFS hash of the final report
     */
    function generateReport(uint256 queryId, string memory finalReportHash) public onlyQueryRequester(queryId) {
        require(bytes(finalReportHash).length > 0, "Report hash cannot be empty");
        require(queries[queryId].status == WorkflowStatus.Completed, "Query not completed yet");
        
        emit ReportGenerated(queryId, finalReportHash);
    }
    
    /**
     * @dev Get query information
     * @param queryId The ID of the query
     * @return Query information
     */
    function getQuery(uint256 queryId) public view returns (Query memory) {
        return queries[queryId];
    }
    
    /**
     * @dev Get tasks for a query
     * @param queryId The ID of the query
     * @return Array of tasks
     */
    function getTasks(uint256 queryId) public view returns (AgentTask[] memory) {
        return agentTasks[queryId];
    }
    
    /**
     * @dev Get agent information
     * @param agent The address of the agent
     * @return Agent information
     */
    function getAgent(address agent) public view returns (AgentInfo memory) {
        return registeredAgents[agent];
    }
    
    /**
     * @dev Find agents by capability
     * @param capability The capability to search for
     * @return Array of agent addresses
     */
    function findAgentsByCapability(string memory capability) public view returns (address[] memory) {
        return capabilityIndex[capability];
    }
    
    /**
     * @dev Update agent reputation score
     * @param agent The address of the agent
     * @param score The new reputation score
     */
    function updateAgentReputation(address agent, uint256 score) public {
        require(registeredAgents[agent].active, "Agent not registered");
        require(score <= 1000, "Reputation score cannot exceed 1000");
        
        registeredAgents[agent].reputationScore = score;
        registeredAgents[agent].lastSeen = block.timestamp;
    }
}