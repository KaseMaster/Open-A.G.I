// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {ERC20} from "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import {ERC20Burnable} from "@openzeppelin/contracts/token/ERC20/extensions/ERC20Burnable.sol";
import {ERC20Pausable} from "@openzeppelin/contracts/token/ERC20/extensions/ERC20Pausable.sol";
import {Ownable} from "@openzeppelin/contracts/access/Ownable.sol";
import {ERC20Permit} from "@openzeppelin/contracts/token/ERC20/extensions/ERC20Permit.sol";
import {IERC20} from "@openzeppelin/contracts/token/ERC20/IERC20.sol";

/**
 * @title AEGISToken
 * @dev Token ERC-20 para el ecosistema AEGIS con funcionalidades avanzadas:
 * - Minteable: Solo el owner puede crear nuevos tokens
 * - Burnable: Los holders pueden quemar sus tokens
 * - Pausable: El owner puede pausar transferencias en emergencias
 * - Permit: Permite aprobaciones sin gas usando firmas
 * - Control de acceso: Funciones administrativas protegidas
 */
contract AEGISToken is ERC20, ERC20Burnable, ERC20Pausable, Ownable, ERC20Permit {
    
    // Eventos personalizados
    event TokensMinted(address indexed to, uint256 amount);
    event TokensBurned(address indexed from, uint256 amount);
    event ContractPaused(address indexed by);
    event ContractUnpaused(address indexed by);
    
    // Constantes del token
    uint256 public constant MAX_SUPPLY = 1_000_000_000 * 10**18; // 1 billón de tokens máximo
    uint256 public constant INITIAL_SUPPLY = 100_000_000 * 10**18; // 100 millones iniciales
    
    /**
     * @dev Constructor que inicializa el token AEGIS
     * @param initialOwner Dirección que será el owner inicial del contrato
     */
    constructor(address initialOwner) 
        ERC20("AEGIS Token", "AEGIS") 
        Ownable(initialOwner)
        ERC20Permit("AEGIS Token")
    {
        // Mint del suministro inicial al owner
        _mint(initialOwner, INITIAL_SUPPLY);
        emit TokensMinted(initialOwner, INITIAL_SUPPLY);
    }
    
    /**
     * @dev Función para mintear nuevos tokens (solo owner)
     * @param to Dirección que recibirá los tokens
     * @param amount Cantidad de tokens a mintear
     */
    function mint(address to, uint256 amount) public onlyOwner {
        require(to != address(0), "AEGISToken: mint to zero address");
        require(amount > 0, "AEGISToken: mint amount must be greater than 0");
        require(totalSupply() + amount <= MAX_SUPPLY, "AEGISToken: would exceed max supply");
        
        _mint(to, amount);
        emit TokensMinted(to, amount);
    }
    
    /**
     * @dev Función para quemar tokens del caller
     * @param amount Cantidad de tokens a quemar
     */
    function burn(uint256 amount) public override {
        super.burn(amount);
        emit TokensBurned(msg.sender, amount);
    }
    
    /**
     * @dev Función para quemar tokens de otra dirección (requiere allowance)
     * @param account Dirección de la cual quemar tokens
     * @param amount Cantidad de tokens a quemar
     */
    function burnFrom(address account, uint256 amount) public override {
        super.burnFrom(account, amount);
        emit TokensBurned(account, amount);
    }
    
    /**
     * @dev Pausa todas las transferencias de tokens (solo owner)
     */
    function pause() public onlyOwner {
        _pause();
        emit ContractPaused(msg.sender);
    }
    
    /**
     * @dev Despausa las transferencias de tokens (solo owner)
     */
    function unpause() public onlyOwner {
        _unpause();
        emit ContractUnpaused(msg.sender);
    }
    
    /**
     * @dev Función de emergencia para retirar ETH accidentalmente enviado al contrato
     */
    function emergencyWithdrawETH() external onlyOwner {
        uint256 balance = address(this).balance;
        require(balance > 0, "AEGISToken: no ETH to withdraw");
        
        (bool success, ) = payable(owner()).call{value: balance}("");
        require(success, "AEGISToken: ETH withdrawal failed");
    }
    
    /**
     * @dev Función de emergencia para retirar tokens ERC-20 accidentalmente enviados
     * @param token Dirección del contrato del token a retirar
     * @param amount Cantidad a retirar
     */
    function emergencyWithdrawToken(address token, uint256 amount) external onlyOwner {
        require(token != address(this), "AEGISToken: cannot withdraw AEGIS tokens");
        require(token != address(0), "AEGISToken: invalid token address");
        
        IERC20(token).transfer(owner(), amount);
    }
    
    /**
     * @dev Override requerido por Solidity para resolver conflictos de herencia múltiple
     */
    function _update(address from, address to, uint256 value)
        internal
        override(ERC20, ERC20Pausable)
    {
        super._update(from, to, value);
    }
    
    /**
     * @dev Función de utilidad para obtener información básica del token
     * @return tokenName Nombre del token
     * @return tokenSymbol Símbolo del token
     * @return tokenDecimals Decimales del token
     * @return currentSupply Suministro total actual
     * @return maxSupply Suministro máximo permitido
     */
    function getTokenInfo() external view returns (
        string memory tokenName,
        string memory tokenSymbol,
        uint8 tokenDecimals,
        uint256 currentSupply,
        uint256 maxSupply
    ) {
        return (
            name(),
            symbol(),
            decimals(),
            totalSupply(),
            MAX_SUPPLY
        );
    }
}