// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {IERC20} from "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import {Ownable} from "@openzeppelin/contracts/access/Ownable.sol";
import {ReentrancyGuard} from "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import {Pausable} from "@openzeppelin/contracts/utils/Pausable.sol";

/**
 * @title AEGISFaucet
 * @dev Faucet para distribución controlada de tokens AEGIS
 * Características:
 * - Límite de tiempo entre solicitudes por dirección
 * - Cantidad fija por solicitud
 * - Pausable en caso de emergencia
 * - Protección contra reentrancy
 * - Control de acceso para administración
 */
contract AEGISFaucet is Ownable, ReentrancyGuard, Pausable {
    
    // Token AEGIS que distribuye el faucet
    IERC20 public immutable aegisToken;
    
    // Configuración del faucet
    uint256 public faucetAmount = 100 * 10**18; // 100 AEGIS tokens por solicitud
    uint256 public cooldownTime = 24 hours; // 24 horas entre solicitudes
    uint256 public maxDailyLimit = 1000 * 10**18; // Límite diario total del faucet
    
    // Tracking de solicitudes
    mapping(address => uint256) public lastRequestTime;
    mapping(address => uint256) public totalReceived;
    
    // Tracking diario
    uint256 public dailyDistributed;
    uint256 public lastResetDay;
    
    // Estadísticas
    uint256 public totalDistributed;
    uint256 public totalRequests;
    address[] public recipients;
    
    // Eventos
    event TokensRequested(address indexed recipient, uint256 amount, uint256 timestamp);
    event FaucetConfigured(uint256 newAmount, uint256 newCooldown, uint256 newDailyLimit);
    event FaucetRefilled(uint256 amount);
    event EmergencyWithdrawal(address indexed token, uint256 amount);
    event DailyLimitReset(uint256 newDay, uint256 previousDistributed);
    
    /**
     * @dev Constructor del faucet
     * @param _aegisToken Dirección del contrato AEGISToken
     * @param _initialOwner Dirección del owner inicial
     */
    constructor(address _aegisToken, address _initialOwner) Ownable(_initialOwner) {
        require(_aegisToken != address(0), "AEGISFaucet: invalid token address");
        aegisToken = IERC20(_aegisToken);
        lastResetDay = block.timestamp / 1 days;
    }
    
    /**
     * @dev Función principal para solicitar tokens del faucet
     */
    function requestTokens() external nonReentrant whenNotPaused {
        address recipient = msg.sender;
        
        // Verificaciones de seguridad
        require(recipient != address(0), "AEGISFaucet: invalid recipient");
        require(!_isContract(recipient), "AEGISFaucet: contracts not allowed");
        
        // Verificar cooldown
        require(
            block.timestamp >= lastRequestTime[recipient] + cooldownTime,
            "AEGISFaucet: cooldown period not elapsed"
        );
        
        // Reset diario si es necesario
        _resetDailyLimitIfNeeded();
        
        // Verificar límite diario
        require(
            dailyDistributed + faucetAmount <= maxDailyLimit,
            "AEGISFaucet: daily limit exceeded"
        );
        
        // Verificar balance del faucet
        uint256 faucetBalance = aegisToken.balanceOf(address(this));
        require(faucetBalance >= faucetAmount, "AEGISFaucet: insufficient balance");
        
        // Actualizar tracking
        lastRequestTime[recipient] = block.timestamp;
        
        // Agregar a lista de recipients si es primera vez
        if (totalReceived[recipient] == 0) {
            recipients.push(recipient);
        }
        
        totalReceived[recipient] += faucetAmount;
        dailyDistributed += faucetAmount;
        totalDistributed += faucetAmount;
        totalRequests++;
        
        // Transferir tokens
        require(
            aegisToken.transfer(recipient, faucetAmount),
            "AEGISFaucet: token transfer failed"
        );
        
        emit TokensRequested(recipient, faucetAmount, block.timestamp);
    }
    
    /**
     * @dev Verifica si una dirección puede solicitar tokens
     * @param user Dirección a verificar
     * @return canRequest Si puede solicitar
     * @return timeRemaining Tiempo restante para próxima solicitud
     * @return reason Razón si no puede solicitar
     */
    function canRequestTokens(address user) external view returns (
        bool canRequest,
        uint256 timeRemaining,
        string memory reason
    ) {
        if (_isContract(user)) {
            return (false, 0, "Contracts not allowed");
        }
        
        if (paused()) {
            return (false, 0, "Faucet is paused");
        }
        
        uint256 nextRequestTime = lastRequestTime[user] + cooldownTime;
        if (block.timestamp < nextRequestTime) {
            return (false, nextRequestTime - block.timestamp, "Cooldown period active");
        }
        
        // Simular reset diario
        uint256 currentDay = block.timestamp / 1 days;
        uint256 simulatedDailyDistributed = (currentDay > lastResetDay) ? 0 : dailyDistributed;
        
        if (simulatedDailyDistributed + faucetAmount > maxDailyLimit) {
            return (false, 0, "Daily limit would be exceeded");
        }
        
        if (aegisToken.balanceOf(address(this)) < faucetAmount) {
            return (false, 0, "Insufficient faucet balance");
        }
        
        return (true, 0, "");
    }
    
    /**
     * @dev Configurar parámetros del faucet (solo owner)
     * @param _faucetAmount Nueva cantidad por solicitud
     * @param _cooldownTime Nuevo tiempo de cooldown
     * @param _maxDailyLimit Nuevo límite diario
     */
    function configureFaucet(
        uint256 _faucetAmount,
        uint256 _cooldownTime,
        uint256 _maxDailyLimit
    ) external onlyOwner {
        require(_faucetAmount > 0, "AEGISFaucet: amount must be greater than 0");
        require(_cooldownTime >= 1 hours, "AEGISFaucet: cooldown too short");
        require(_maxDailyLimit >= _faucetAmount, "AEGISFaucet: daily limit too low");
        
        faucetAmount = _faucetAmount;
        cooldownTime = _cooldownTime;
        maxDailyLimit = _maxDailyLimit;
        
        emit FaucetConfigured(_faucetAmount, _cooldownTime, _maxDailyLimit);
    }
    
    /**
     * @dev Pausar el faucet (solo owner)
     */
    function pause() external onlyOwner {
        _pause();
    }
    
    /**
     * @dev Despausar el faucet (solo owner)
     */
    function unpause() external onlyOwner {
        _unpause();
    }
    
    /**
     * @dev Retirada de emergencia de tokens (solo owner)
     * @param token Dirección del token a retirar
     * @param amount Cantidad a retirar
     */
    function emergencyWithdraw(address token, uint256 amount) external onlyOwner {
        require(token != address(0), "AEGISFaucet: invalid token address");
        
        if (token == address(aegisToken)) {
            // Para AEGIS tokens, verificar que no se retire todo
            uint256 balance = aegisToken.balanceOf(address(this));
            require(amount <= balance, "AEGISFaucet: insufficient balance");
        }
        
        IERC20(token).transfer(owner(), amount);
        emit EmergencyWithdrawal(token, amount);
    }
    
    /**
     * @dev Obtener estadísticas del faucet
     */
    function getFaucetStats() external view returns (
        uint256 balance,
        uint256 _totalDistributed,
        uint256 _totalRequests,
        uint256 _dailyDistributed,
        uint256 _maxDailyLimit,
        uint256 recipientCount,
        bool isPaused
    ) {
        return (
            aegisToken.balanceOf(address(this)),
            totalDistributed,
            totalRequests,
            dailyDistributed,
            maxDailyLimit,
            recipients.length,
            paused()
        );
    }
    
    /**
     * @dev Obtener información de un usuario específico
     * @param user Dirección del usuario
     */
    function getUserInfo(address user) external view returns (
        uint256 _totalReceived,
        uint256 _lastRequestTime,
        uint256 nextRequestTime,
        bool canRequest
    ) {
        nextRequestTime = lastRequestTime[user] + cooldownTime;
        canRequest = block.timestamp >= nextRequestTime && !paused();
        
        return (
            totalReceived[user],
            lastRequestTime[user],
            nextRequestTime,
            canRequest
        );
    }
    
    /**
     * @dev Reset del límite diario si es necesario
     */
    function _resetDailyLimitIfNeeded() internal {
        uint256 currentDay = block.timestamp / 1 days;
        if (currentDay > lastResetDay) {
            emit DailyLimitReset(currentDay, dailyDistributed);
            dailyDistributed = 0;
            lastResetDay = currentDay;
        }
    }
    
    /**
     * @dev Verificar si una dirección es un contrato
     * @param addr Dirección a verificar
     * @return True si es un contrato
     */
    function _isContract(address addr) internal view returns (bool) {
        uint256 size;
        assembly {
            size := extcodesize(addr)
        }
        return size > 0;
    }
    
    /**
     * @dev Función para recibir ETH (rechaza)
     */
    receive() external payable {
        revert("AEGISFaucet: does not accept ETH");
    }
    
    /**
     * @dev Función fallback (rechaza)
     */
    fallback() external payable {
        revert("AEGISFaucet: function not found");
    }
}