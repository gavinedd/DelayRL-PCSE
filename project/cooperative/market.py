import numpy as np
from typing import List, Optional

class AgriculturalMarket:
    """Simulates a market where farms' collective actions influence prices"""
    def __init__(self, base_price: float = 100.0, supply_sensitivity: float = 0.1,
                 supply_decay: float = 0.05, volatility: float = 0.1,
                 min_price: float = 10.0):
        if base_price <= 0:
            raise ValueError("base_price must be positive")
        if supply_sensitivity <= 0:
            raise ValueError("supply_sensitivity must be positive")
        if supply_decay < 0 or supply_decay > 1:
            raise ValueError("supply_decay must be between 0 and 1")
            
        self.base_price = base_price
        self.supply_sensitivity = supply_sensitivity
        self.supply_decay = supply_decay
        self.volatility = volatility
        self.min_price = min_price
        
        self.total_supply = 0.0
        self.price_history: List[float] = []
        self.supply_history: List[float] = []
        
    def update_price(self, new_supply: float) -> float:
        """Update market price based on supply and market conditions"""
        # Apply supply decay
        self.total_supply = self.total_supply * (1 - self.supply_decay) + new_supply
        
        # Calculate base market price
        market_pressure = self.supply_sensitivity * self.total_supply
        base_market_price = self.base_price * (1 - market_pressure)
        
        # Add market volatility
        if self.price_history:
            volatility_factor = 1 + self.volatility * (2 * np.random.random() - 1)
            current_price = base_market_price * volatility_factor
        else:
            current_price = base_market_price
            
        # Ensure price doesn't go below minimum
        current_price = max(current_price, self.min_price)
        
        # Update histories
        self.price_history.append(current_price)
        self.supply_history.append(self.total_supply)
        
        return current_price
        
    def get_price_trend(self, window: int = 5) -> Optional[float]:
        """Calculate recent price trend"""
        if len(self.price_history) < window:
            return None
            
        recent_prices = self.price_history[-window:]
        if len(recent_prices) < 2:
            return 0.0
            
        return (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        
    def get_market_stability(self, window: int = 5) -> Optional[float]:
        """Calculate market stability based on price variance"""
        if len(self.price_history) < window:
            return None
            
        recent_prices = self.price_history[-window:]
        return 1.0 - np.std(recent_prices) / self.base_price
        
    def reset(self) -> None:
        """Reset market state"""
        self.total_supply = 0.0
        self.price_history = []
        self.supply_history = [] 