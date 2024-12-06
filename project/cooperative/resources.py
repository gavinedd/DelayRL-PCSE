from typing import Set, List
import numpy as np

class SharedResources:
    """Manages shared resources between cooperating farms"""
    def __init__(self, base_cost: float = 100.0, sharing_discount: float = 0.2,
                 max_sharing_farms: int = 10, min_cost: float = 10.0):
        if sharing_discount <= 0 or sharing_discount >= 1:
            raise ValueError("sharing_discount must be between 0 and 1")
        if base_cost <= 0:
            raise ValueError("base_cost must be positive")
            
        self.base_cost = base_cost
        self.sharing_discount = sharing_discount
        self.max_sharing_farms = max_sharing_farms
        self.min_cost = min_cost
        self.sharing_farms: Set[int] = set()
        self.sharing_history: List[int] = []
        
    def get_cost(self, farm_id: int) -> float:
        """Calculate operating costs with sharing benefits"""
        if farm_id not in self.sharing_farms:
            return self.base_cost
            
        # Calculate discounted cost with limits
        num_sharing = min(len(self.sharing_farms), self.max_sharing_farms)
        discount = self.sharing_discount * num_sharing
        
        # Ensure cost doesn't go below minimum
        cost = max(self.base_cost * (1 - discount), self.min_cost)
        return cost
        
    def update_sharing_status(self, farm_id: int, is_sharing: bool) -> None:
        """Update which farms are sharing resources"""
        if is_sharing:
            # Only add if below max limit
            if len(self.sharing_farms) < self.max_sharing_farms:
                self.sharing_farms.add(farm_id)
        else:
            self.sharing_farms.discard(farm_id)
            
        self.sharing_history.append(len(self.sharing_farms))
        
    def get_sharing_ratio(self) -> float:
        """Get the current ratio of sharing farms to maximum allowed"""
        return len(self.sharing_farms) / self.max_sharing_farms
        
    def reset(self) -> None:
        """Reset sharing status"""
        self.sharing_farms.clear()
        self.sharing_history = [] 