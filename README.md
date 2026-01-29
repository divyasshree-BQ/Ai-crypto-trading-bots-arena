# AI Crypto Trading Bots Arena

A paper trading competition where 5 AI bots with different strategies race against each other using real market data. **No real money is used.**

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# Run the arena
python3 bot_arena.py
```

Then view the race visualization:
```bash
python3 -m http.server 8000
# Open http://localhost:8000/bot_race.html
```

## The 5 Competing Bots

| Bot | Strategy | AI? | Confidence | Max Position | Special Rules |
|-----|----------|-----|------------|--------------|---------------|
| **Full-AI** | 100% AI autonomy | Yes | 30%+ | $10 | No restrictions |
| **AI-Limits** | AI + risk limits | Yes | 30%+ | $8 | Stop loss within 5% enforced |
| **AI-Rules** | AI suggests, rules approve | Yes | 30%+ | $10 | Blocks >30% price swings |
| **Rules-Only** | Pure momentum rules | No | - | $10 | Buy on 0.5-10% 24h gains |
| **Conservative** | AI with strict filters | Yes | 60%+ | $5 | Requires 1.5:1 reward/risk |

## Bot Strategies Explained

**Full-AI** - Maximum AI freedom
- AI makes all decisions with minimal constraints
- Aggressive trading, takes more opportunities
- Higher risk, potentially higher reward

**AI-Limits** - AI with guardrails
- AI trades freely but system enforces risk limits
- Stop losses must be within 5% of entry
- Risk multiplier of 0.8x reduces position exposure

**AI-Rules** - Hybrid approach
- AI suggests trades, rule-based system filters them
- Rejects trades during extreme volatility (>30% swings)
- Balances AI insight with safety rules

**Rules-Only** - No AI, pure logic
- Simple momentum strategy: buy when price up 0.5-10% in 24h
- Fixed targets: +3% profit, -2% stop loss
- Predictable, mechanical trading

**Conservative** - Quality over quantity
- Requires 60%+ AI confidence (vs 30% for others)
- Must have 1.5:1 reward-to-risk ratio
- Maximum 1 position at a time, $5 max size
- Fewest trades, but highest conviction

## Race Visualization

The `bot_race.html` dashboard shows:
- **Race Track**: Animated bots racing based on portfolio value
- **Leaderboard**: Rankings with PnL, win rate, trades, max drawdown
- **Activity Log**: Real-time actions (BUY/SELL/HOLD)
- **Tooltips**: Hover over bot names to see their rules

The visualization auto-updates every 2 seconds by reading `race_data.json`.

## Example Output

```
BOT ARENA - Multi-Bot Competition System
==================================================
Paper trading mode - no real money
==================================================

ARENA CYCLE 1 - 14:30:15
======================================================================
   [Full-AI] (full_ai)
      OK: BUY WETH (70%)
      OK: BUY cbBTC (65%)

   [AI-Limits] (ai_limits)
      OK: BUY WETH (80%)

   [Rules-Only] (rules_only)
      HOLD: No actions

──────────────────────────────────────────────────────────────────────
LEADERBOARD
──────────────────────────────────────────────────────────────────────
Bot              Value        PnL     Win%   Trades      DD%
──────────────────────────────────────────────────────────────────────
1st Full-AI     $  100.03     +0.03   100.0%       4     0.0%
2nd AI-Rules    $  100.00     +0.00     0.0%       0     0.0%
3rd Rules-Only  $  100.00     +0.00     0.0%       0     0.0%
4th Conservative $  100.00     +0.00     0.0%       0     0.0%
5th AI-Limits   $   99.99     -0.01    50.0%       2     0.0%
```

## Requirements

- Python 3.8+
- OpenAI API key (for AI-powered bots)
- Bitquery API key (for market data)

## Installation

1. Clone the repository
```bash
git clone https://github.com/divyasshree-BQ/Ai-crypto-trading-bots-arena
cd Ai-crypto-trading-bots-arena
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Get API Keys
- **Bitquery**: Create an API key at https://account.bitquery.io/user/api_v2/access_tokens
- **OpenAI**: Get your API key from https://platform.openai.com/api-keys

4. Configure environment variables
```bash
cp .env.example .env
```

Edit `.env` with your values:
```
OPENAI_API_KEY=sk-proj-...
BITQUERY_API_KEY=ory_at_...
```

## Project Structure

```
trading-bot/
├── bot_arena.py         # Multi-bot competition system
├── liquidity_data.py    # Bitquery API - market data & liquidity events
├── bot_race.html        # Race visualization dashboard
├── race_data.json       # Auto-generated race data
├── requirements.txt     # Python dependencies
└── .env                 # Environment variables
```

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                        Market Data Layer                        │
├─────────────────────────────────────────────────────────────────┤
│  Bitquery API                                                   │
│  ├─ DEX Trades (volume, price, buy/sell ratio)                 │
│  ├─ Liquidity Events (smart money flows)                       │
│  └─ Price changes (24h momentum)                               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     5 Competing Bots                            │
├─────────────────────────────────────────────────────────────────┤
│  Each bot receives the same market data and decides:           │
│  ├─ Full-AI: GPT-4o decides everything                         │
│  ├─ AI-Limits: GPT-4o + risk limits enforced                   │
│  ├─ AI-Rules: GPT-4o suggests, rules filter                    │
│  ├─ Rules-Only: Pure momentum rules, no AI                     │
│  └─ Conservative: GPT-4o with strict filters                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     Paper Trading Engine                        │
├─────────────────────────────────────────────────────────────────┤
│  ├─ Simulates buy/sell execution                               │
│  ├─ Tracks portfolio value per bot                             │
│  ├─ Calculates PnL, win rate, drawdown                         │
│  └─ Exports to race_data.json for visualization                │
└─────────────────────────────────────────────────────────────────┘
```

## Configuration

The arena runs with these defaults (configurable in `bot_arena.py`):
- **Starting capital**: $100 per bot
- **Cycles**: 20
- **Interval**: 30 seconds between cycles

## Performance Metrics

Each bot is tracked on:
- **Portfolio Value**: Current total value
- **PnL**: Profit/Loss from starting capital
- **Win Rate**: Percentage of profitable trades
- **Total Trades**: Number of executed trades
- **Max Drawdown**: Largest peak-to-trough decline

## License

MIT License

## Contributing

Contributions welcome. Please test thoroughly before submitting pull requests.
