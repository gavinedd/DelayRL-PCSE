# Cooperative Farming Environment

This environment simulates a multi-agent farming system where individual farms can benefit from cooperation through shared resources, collective knowledge, and market dynamics.

## Core Components

### 1. Market System (`market.py`)
The market simulates a shared economy where all farms' actions influence prices:
- **Base Price**: Starting point for market prices
- **Supply Sensitivity**: How much total supply affects prices
- **Supply Decay**: Prevents market saturation over time
- **Volatility**: Random price fluctuations
- **Price Formula**: `base_price * (1 - supply_sensitivity * total_supply) * volatility_factor`

### 2. Shared Resources (`resources.py`)
Manages resource sharing between participating farms:
- **Base Cost**: Standard operating cost without sharing
- **Sharing Discount**: Cost reduction per sharing farm
- **Maximum Sharing**: Limit on number of sharing farms
- **Cost Formula**: `base_cost * (1 - sharing_discount * num_sharing_farms)`

### 3. Knowledge Pool
A shared learning system that enables farms to benefit from collective experience:

#### Core Functionality
- Maintains a fixed-size memory (default 1000 experiences) of successful farming actions
- Each experience contains:
  - Normalized state vectors
  - Actions taken
  - Rewards received
- Uses running statistics to normalize state data
- Implements adaptive similarity thresholds for experience matching

#### Knowledge Sharing Process
1. **Experience Storage**:
   - New experiences are normalized before storage
   - States are tracked using running means and standard deviations
   - Oldest experiences are automatically removed when capacity is reached

2. **Similarity Calculation**:
   - State similarity: Exponential decay of squared differences
   - Action similarity: Exponential decay of action differences
   - Combined similarity score: 70% state + 30% action weights

3. **Bonus Calculation**:
   - Identifies experiences above similarity threshold (default 0.2)
   - Calculates weighted average of similar experience rewards
   - Applies learning rate (default 0.1) to final bonus
   - Returns 0.0 if no similar experiences found

## Environment Structure

### Observation Space
Dict space with four components:
1. **Crop State**: Current crop conditions
2. **Weather**: Environmental conditions
3. **Previous Action**: Last action taken
4. **Cooperative Features**:
   - Market price (normalized)
   - Total supply ratio
   - Sharing farms ratio

### Action Space
- Continuous actions (Box space)
- Represents farming decisions (e.g., fertilizer application)
- Bounded between 0 and 1
- Scaled by action_multiplier

### Reward Calculation

The reward system combines multiple factors:

1. **Base Yield**:
   - Calculated from crop state and environmental conditions
   - Influenced by action effectiveness
   - Scaled based on current growth stage

2. **Market Component**:
   - Price factor based on current market conditions
   - Supply/demand dynamics affect final reward
   - Formula: `base_yield * current_market_price`

3. **Cooperation Bonus**:
   - Resource sharing benefits: `sharing_multiplier * base_cost_savings`
   - Knowledge pool bonus: `knowledge_multiplier * similarity_score`
   - Combined bonus capped at maximum_cooperation_bonus

Final Reward Formula:

```python
reward = (base_yield * market_price) * (1 + cooperation_bonus)
```

## Project Structure

```
DelayRL-PCSE/
├── pcse/           # PCSE crop simulation models (v5.5.6)
├── PCSE-Gym/       # Reinforcement learning environment
└── project/        # Custom project code
    └── cooperative/  # Cooperative farming environment
```

## Documentation

- [Cooperative Farming Environment](project/cooperative/README.md) - Details on the multi-agent cooperative system
- [PCSE Documentation](https://pcse.readthedocs.io/en/stable/)
- [PCSE-Gym Documentation](https://cropgym.ai/)