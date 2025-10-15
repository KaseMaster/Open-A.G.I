// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract AEGISToken is ERC20, Ownable {
    uint256 public faucetAmount = 100 * 10 ** 18;
    uint256 public cooldownTime = 1 hours;
    mapping(address => uint256) public lastRequest;

    constructor(address initialOwner) ERC20("AEGIS", "AEGIS") Ownable(initialOwner) {
        _mint(initialOwner, 1000000 * 10 ** 18);
    }

    function mint(address to, uint256 amount) public onlyOwner {
        _mint(to, amount);
    }

    function requestTokens() public {
        require(block.timestamp >= lastRequest[msg.sender] + cooldownTime, "Cooldown active");
        lastRequest[msg.sender] = block.timestamp;
        _mint(msg.sender, faucetAmount);
    }

    function donate(address to, uint256 amount) public {
        _transfer(msg.sender, to, amount);
    }
}