#!/usr/bin/env python3
"""
Bot Arena - Multi-Bot Competition System
Spawns multiple trading bots with varying AI/human control levels
Paper trading mode - no real money, same market data
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from dotenv import load_dotenv

load_dotenv()

# Import market data fetcher
from liquidity_data import get_enhanced_market_data


class ControlLevel(Enum):
    """Bot control levels - how much AI vs rules/constraints"""
    FULL_AI = "full_ai"           # 100% AI - current bot
    AI_WITH_LIMITS = "ai_limits"  # AI trades, strict risk limits
    AI_SUGGESTIONS = "ai_suggest" # AI suggests, rules approve
    RULES_ONLY = "rules_only"     # Pure rule-based, no AI
    CONSERVATIVE = "conservative" # AI but very conservative


@dataclass
class PaperPosition:
    """Virtual position for paper trading"""
    market: str
    action: str
    entry_price: float
    target_price: float
    stop_loss: float
    amount_usd: float
    timestamp: str
    confidence: int = 0
    reasoning: str = ""


@dataclass
class BotPerformance:
    """Track bot performance metrics"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    peak_value: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    returns: List[float] = field(default_factory=list)


class PaperTradingBot:
    """
    Paper trading bot with configurable control level.
    Simulates trades without real execution.
    """

    def __init__(
        self,
        name: str,
        control_level: ControlLevel,
        starting_capital: float = 100.0,
        max_position_size: float = 10.0,
        max_positions: int = 3,
        min_confidence: int = 30,
        risk_multiplier: float = 1.0,  # Adjusts risk tolerance
    ):
        self.name = name
        self.control_level = control_level
        self.capital = starting_capital
        self.starting_capital = starting_capital
        self.max_position_size = max_position_size
        self.max_positions = max_positions
        self.min_confidence = min_confidence
        self.risk_multiplier = risk_multiplier

        # Trading state
        self.open_positions: List[PaperPosition] = []
        self.closed_positions: List[Dict] = []
        self.performance = BotPerformance()
        self.performance.peak_value = starting_capital

        # AI client (shared)
        self.openai_client = None
        self._init_ai()

    def _init_ai(self):
        """Initialize AI client if needed"""
        if self.control_level in [ControlLevel.FULL_AI, ControlLevel.AI_WITH_LIMITS,
                                   ControlLevel.AI_SUGGESTIONS, ControlLevel.CONSERVATIVE]:
            try:
                import openai
                api_key = os.getenv('OPENAI_API_KEY')
                if api_key:
                    self.openai_client = openai.OpenAI(api_key=api_key)
            except ImportError:
                pass

    def get_portfolio_value(self, market_data: Dict) -> float:
        """Calculate current portfolio value"""
        value = self.capital
        for pos in self.open_positions:
            current_price = self._get_price(pos.market, market_data)
            if current_price and pos.entry_price > 0:
                pnl_pct = (current_price - pos.entry_price) / pos.entry_price
                value += pos.amount_usd * (1 + pnl_pct)
        return value

    def _get_price(self, symbol: str, market_data: Dict) -> Optional[float]:
        """Get current price for a symbol"""
        trade_data = market_data.get('trade_data', {})
        markets = trade_data.get('top_markets', [])
        for m in markets:
            if m.get('symbol', '').upper() == symbol.upper():
                return m.get('recent_price', 0)
        # Fallback prices
        if symbol.upper() in ['USDC', 'USDT', 'DAI']:
            return 1.0
        if symbol.upper() == 'WETH':
            return 3000.0
        return None

    def generate_actions(self, market_data: Dict) -> List[Dict]:
        """Generate trading actions based on control level"""

        if self.control_level == ControlLevel.FULL_AI:
            return self._generate_full_ai_actions(market_data)

        elif self.control_level == ControlLevel.AI_WITH_LIMITS:
            return self._generate_ai_with_limits(market_data)

        elif self.control_level == ControlLevel.AI_SUGGESTIONS:
            return self._generate_ai_suggestions(market_data)

        elif self.control_level == ControlLevel.RULES_ONLY:
            return self._generate_rules_only(market_data)

        elif self.control_level == ControlLevel.CONSERVATIVE:
            return self._generate_conservative(market_data)

        return []

    def _generate_full_ai_actions(self, market_data: Dict) -> List[Dict]:
        """100% AI control - no restrictions"""
        if not self.openai_client:
            return []

        prompt = self._build_ai_prompt(market_data, restrictions="None - full autonomy")
        return self._call_ai(prompt)

    def _generate_ai_with_limits(self, market_data: Dict) -> List[Dict]:
        """AI trades but with strict risk limits enforced"""
        if not self.openai_client:
            return []

        restrictions = f"""
You trade freely, system enforces these limits after:
- Max position: ${self.max_position_size * self.risk_multiplier:.2f}
- Stop loss within 5% of entry
- Be aggressive - find opportunities!
"""
        prompt = self._build_ai_prompt(market_data, restrictions=restrictions)
        actions = self._call_ai(prompt)

        # Enforce limits
        validated = []
        for action in actions:
            if self._validate_risk_limits(action):
                validated.append(action)
        return validated

    def _generate_ai_suggestions(self, market_data: Dict) -> List[Dict]:
        """AI suggests, rule-based system approves"""
        if not self.openai_client:
            return []

        restrictions = "Suggest trades freely. System filters based on rules. Be aggressive!"
        prompt = self._build_ai_prompt(market_data, restrictions=restrictions)
        suggestions = self._call_ai(prompt)

        # Rule-based approval
        approved = []
        for suggestion in suggestions:
            if self._rule_based_approval(suggestion, market_data):
                approved.append(suggestion)
        return approved

    def _generate_rules_only(self, market_data: Dict) -> List[Dict]:
        """Pure rule-based trading - no AI"""
        actions = []
        trade_data = market_data.get('trade_data', {})
        markets = trade_data.get('top_markets', [])

        for market in markets[:5]:  # Top 5 markets
            symbol = market.get('symbol', '')
            price = market.get('recent_price', 0)
            volume_24h = market.get('volume_24h', 0)
            price_change = market.get('price_change_24h', 0)

            if not price or price <= 0:
                continue

            # Simple momentum rule: buy if up 0.5-10% (loosened from 2-5%)
            if 0.5 < price_change < 10 and len(self.open_positions) < self.max_positions:
                # Check if not already holding
                if not any(p.market.upper() == symbol.upper() for p in self.open_positions):
                    actions.append({
                        'action': 'BUY',
                        'market': symbol,
                        'confidence': 60,
                        'entry_price': price,
                        'target_price': price * 1.03,  # 3% target
                        'stop_loss': price * 0.98,     # 2% stop
                        'reasoning': f'Momentum rule: +{price_change:.1f}% in 24h'
                    })

            # Check open positions for exit
            for pos in self.open_positions:
                if pos.market.upper() == symbol.upper():
                    current_price = price
                    # Hit target or stop?
                    if current_price >= pos.target_price:
                        actions.append({
                            'action': 'CLOSE',
                            'market': symbol,
                            'confidence': 90,
                            'reasoning': 'Target reached'
                        })
                    elif current_price <= pos.stop_loss:
                        actions.append({
                            'action': 'CLOSE',
                            'market': symbol,
                            'confidence': 90,
                            'reasoning': 'Stop loss hit'
                        })

        return actions

    def _generate_conservative(self, market_data: Dict) -> List[Dict]:
        """AI but very conservative - high confidence, tight stops"""
        if not self.openai_client:
            return []

        restrictions = f"""
CONSERVATIVE MODE - still trade, but pick best setups:
- Prefer 60%+ confidence
- Tight stop loss (2-3% from entry)
- Good reward/risk ratio
- Max 1 position
- Find the BEST opportunity, not many
"""
        prompt = self._build_ai_prompt(market_data, restrictions=restrictions)
        actions = self._call_ai(prompt)

        # Extra conservative filter - only take best trades
        conservative = []
        for action in actions:
            conf = action.get('confidence', 0)
            if conf >= 60:  # Lowered from 80
                if action.get('action') in ['BUY', 'SELL']:
                    entry = action.get('entry_price', 0)
                    stop = action.get('stop_loss', 0)
                    target = action.get('target_price', 0)

                    if entry > 0 and stop > 0 and target > 0:
                        risk = abs(entry - stop)
                        reward = abs(target - entry)
                        # Require 1.5:1 reward/risk instead of 2:1
                        if risk > 0 and reward / risk >= 1.5:
                            conservative.append(action)
                else:
                    conservative.append(action)

        return conservative

    def _build_ai_prompt(self, market_data: Dict, restrictions: str) -> str:
        """Build AI prompt with market data"""
        trade_data = market_data.get('trade_data', {})

        # Compact market summary
        markets_summary = []
        for m in trade_data.get('top_markets', [])[:10]:
            markets_summary.append({
                's': m.get('symbol', ''),
                'p': round(m.get('recent_price', 0), 8),
                'v': round(m.get('volume_24h', 0), 2),
                'c': round(m.get('price_change_24h', 0), 2)
            })

        # Open positions summary
        positions_summary = []
        for pos in self.open_positions:
            positions_summary.append({
                'm': pos.market,
                'e': pos.entry_price,
                't': pos.target_price,
                's': pos.stop_loss
            })

        portfolio_value = self.get_portfolio_value(market_data)

        return f"""Trading Bot: {self.name}
Control: {self.control_level.value}

MARKET DATA (s=symbol, p=price, v=volume24h, c=change24h%):
{json.dumps(markets_summary, separators=(',', ':'))}

OPEN POSITIONS:
{json.dumps(positions_summary, separators=(',', ':'))}

PORTFOLIO:
- Capital: ${self.capital:.2f}
- Portfolio Value: ${portfolio_value:.2f}
- Open: {len(self.open_positions)}/{self.max_positions}
- Max Position: ${self.max_position_size:.2f}

RESTRICTIONS:
{restrictions}

Return JSON array of actions. Each needs: action, market, confidence, reasoning.
For BUY/SELL: also entry_price, target_price, stop_loss.
For CLOSE: just action, market, confidence, reasoning.

OUTPUT (JSON only):
"""

    def _call_ai(self, prompt: str) -> List[Dict]:
        """Call AI and parse response"""
        if not self.openai_client:
            return []

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",  # Cheaper for paper trading
                messages=[
                    {"role": "system", "content": "You are a trading AI. Return valid JSON arrays only."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            text = response.choices[0].message.content

            # Parse JSON
            start = text.find('[')
            end = text.rfind(']') + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except Exception as e:
            print(f"      [{self.name}] AI error: {e}")

        return []

    def _validate_risk_limits(self, action: Dict) -> bool:
        """Validate action against risk limits"""
        if action.get('action') not in ['BUY', 'SELL']:
            return True

        entry = action.get('entry_price', 0)
        stop = action.get('stop_loss', 0)

        if entry <= 0:
            return False

        # Stop loss within 5%
        if stop > 0:
            stop_pct = abs(entry - stop) / entry * 100
            if stop_pct > 5:
                return False

        # Position size limit
        amount = action.get('amount_usd', self.max_position_size)
        if amount > self.capital * 0.3:
            action['amount_usd'] = self.capital * 0.3

        return True

    def _rule_based_approval(self, action: Dict, market_data: Dict) -> bool:
        """Rule-based approval for AI suggestions"""
        if action.get('action') not in ['BUY', 'SELL']:
            return True  # Always allow closes/holds

        market = action.get('market', '')
        trade_data = market_data.get('trade_data', {})

        # Find market data
        market_info = None
        for m in trade_data.get('top_markets', []):
            if m.get('symbol', '').upper() == market.upper():
                market_info = m
                break

        # Allow known tokens even if not in market data
        known_tokens = ['WETH', 'ETH', 'USDC', 'USDT', 'DAI']
        if not market_info and market.upper() not in known_tokens:
            return False  # Unknown market

        # Rules for approval (loosened)
        if market_info:
            volume = market_info.get('volume_24h', 0)
            price_change = market_info.get('price_change_24h', 0)

            # Don't buy into huge pumps or dumps (>30% instead of >20%)
            if abs(price_change) > 30:
                return False

        # Confidence check (handle string or int)
        confidence = action.get('confidence', 0)
        if isinstance(confidence, str):
            try:
                confidence = int(float(confidence.replace('%', '')))
            except:
                confidence = 0
        if confidence < self.min_confidence:
            return False

        return True

    def execute_action(self, action: Dict, market_data: Dict) -> bool:
        """Execute a paper trade action"""
        action_type = action.get('action', '').upper()
        market = action.get('market', '')

        if action_type == 'BUY':
            return self._paper_buy(action, market_data)
        elif action_type == 'SELL':
            return self._paper_buy(action, market_data)  # Treat as buy for now
        elif action_type == 'CLOSE':
            return self._paper_close(action, market_data)

        return False

    def _paper_buy(self, action: Dict, market_data: Dict) -> bool:
        """Execute paper buy"""
        if len(self.open_positions) >= self.max_positions:
            return False

        amount = min(
            action.get('amount_usd', self.max_position_size),
            self.max_position_size,
            self.capital * 0.5  # Never more than 50% in one trade
        )

        if amount > self.capital:
            return False

        entry_price = action.get('entry_price', 0)
        if entry_price <= 0:
            entry_price = self._get_price(action['market'], market_data) or 0

        if entry_price <= 0:
            return False

        position = PaperPosition(
            market=action['market'],
            action='BUY',
            entry_price=entry_price,
            target_price=action.get('target_price', entry_price * 1.05),
            stop_loss=action.get('stop_loss', entry_price * 0.95),
            amount_usd=amount,
            timestamp=datetime.now().isoformat(),
            confidence=action.get('confidence', 0),
            reasoning=action.get('reasoning', '')
        )

        self.open_positions.append(position)
        self.capital -= amount
        self.performance.total_trades += 1

        return True

    def _paper_close(self, action: Dict, market_data: Dict) -> bool:
        """Execute paper close"""
        market = action.get('market', '').upper()

        for pos in self.open_positions[:]:
            if pos.market.upper() == market:
                current_price = self._get_price(pos.market, market_data)
                if not current_price:
                    current_price = pos.entry_price

                # Calculate PnL
                pnl_pct = (current_price - pos.entry_price) / pos.entry_price
                pnl_usd = pos.amount_usd * pnl_pct

                # Update capital
                self.capital += pos.amount_usd + pnl_usd

                # Track performance
                self.performance.total_pnl += pnl_usd
                self.performance.returns.append(pnl_pct * 100)

                if pnl_usd >= 0:
                    self.performance.winning_trades += 1
                else:
                    self.performance.losing_trades += 1

                # Record closed position
                self.closed_positions.append({
                    'market': pos.market,
                    'entry_price': pos.entry_price,
                    'exit_price': current_price,
                    'pnl_usd': pnl_usd,
                    'pnl_pct': pnl_pct * 100,
                    'closed_at': datetime.now().isoformat()
                })

                self.open_positions.remove(pos)
                return True

        return False

    def check_stops_and_targets(self, market_data: Dict) -> List[Dict]:
        """Check if any positions hit stop/target"""
        actions = []

        for pos in self.open_positions:
            current_price = self._get_price(pos.market, market_data)
            if not current_price:
                continue

            if current_price >= pos.target_price:
                actions.append({
                    'action': 'CLOSE',
                    'market': pos.market,
                    'confidence': 100,
                    'reasoning': f'Target hit: ${current_price:.4f} >= ${pos.target_price:.4f}'
                })
            elif current_price <= pos.stop_loss:
                actions.append({
                    'action': 'CLOSE',
                    'market': pos.market,
                    'confidence': 100,
                    'reasoning': f'Stop hit: ${current_price:.4f} <= ${pos.stop_loss:.4f}'
                })

        return actions

    def update_metrics(self, market_data: Dict):
        """Update performance metrics"""
        portfolio_value = self.get_portfolio_value(market_data)

        # Track peak and drawdown
        if portfolio_value > self.performance.peak_value:
            self.performance.peak_value = portfolio_value

        drawdown = (self.performance.peak_value - portfolio_value) / self.performance.peak_value * 100
        if drawdown > self.performance.max_drawdown:
            self.performance.max_drawdown = drawdown

        # Win rate
        total = self.performance.winning_trades + self.performance.losing_trades
        if total > 0:
            self.performance.win_rate = self.performance.winning_trades / total * 100

        # Sharpe ratio (simplified)
        if len(self.performance.returns) > 1:
            import statistics
            avg_return = statistics.mean(self.performance.returns)
            std_return = statistics.stdev(self.performance.returns) if len(self.performance.returns) > 1 else 1
            self.performance.sharpe_ratio = avg_return / std_return if std_return > 0 else 0


class BotArena:
    """
    Orchestrates multiple bots competing on same market data
    """

    def __init__(self):
        self.bots: List[PaperTradingBot] = []
        self.cycle = 0
        self.start_time = None
        self.race_history: List[Dict] = []  # Track all cycles for visualization

    def add_bot(self, bot: PaperTradingBot):
        """Add a bot to the arena"""
        self.bots.append(bot)

    def create_default_bots(self, starting_capital: float = 100.0):
        """Create the default set of competing bots"""

        # Bot A: Full AI (current behavior) - aggressive, low bar
        self.add_bot(PaperTradingBot(
            name="Full-AI",
            control_level=ControlLevel.FULL_AI,
            starting_capital=starting_capital,
            max_position_size=10,
            min_confidence=30
        ))

        # Bot B: AI with limits - still trades but enforces risk rules
        self.add_bot(PaperTradingBot(
            name="AI-Limits",
            control_level=ControlLevel.AI_WITH_LIMITS,
            starting_capital=starting_capital,
            max_position_size=8,
            min_confidence=30,  # Same as Full-AI
            risk_multiplier=0.8
        ))

        # Bot C: AI suggestions + rule approval - looser rules
        self.add_bot(PaperTradingBot(
            name="AI-Rules",
            control_level=ControlLevel.AI_SUGGESTIONS,
            starting_capital=starting_capital,
            max_position_size=10,
            min_confidence=30  # Same as Full-AI
        ))

        # Bot D: Pure rules (no AI) - looser momentum threshold
        self.add_bot(PaperTradingBot(
            name="Rules-Only",
            control_level=ControlLevel.RULES_ONLY,
            starting_capital=starting_capital,
            max_position_size=10,
            min_confidence=30
        ))

        # Bot E: Conservative AI - high bar but still achievable
        self.add_bot(PaperTradingBot(
            name="Conservative",
            control_level=ControlLevel.CONSERVATIVE,
            starting_capital=starting_capital,
            max_position_size=5,
            max_positions=1,
            min_confidence=60  # Lowered from 80
        ))

    def run_cycle(self, market_data: Dict):
        """Run one cycle for all bots"""
        self.cycle += 1
        cycle_actions: Dict[str, List[Dict]] = {}  # Track actions per bot

        print(f"\n{'='*70}")
        print(f"ARENA CYCLE {self.cycle} - {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*70}")

        for bot in self.bots:
            print(f"\n   [{bot.name}] ({bot.control_level.value})")
            cycle_actions[bot.name] = []

            # First check stops/targets
            stop_actions = bot.check_stops_and_targets(market_data)
            for action in stop_actions:
                print(f"      AUTO: {action['action']} {action['market']} - {action['reasoning']}")
                bot.execute_action(action, market_data)
                cycle_actions[bot.name].append({
                    "type": "AUTO",
                    "action": action['action'],
                    "market": action['market'],
                    "reason": action.get('reasoning', '')
                })

            # Generate new actions
            actions = bot.generate_actions(market_data)

            if actions:
                for action in actions[:2]:  # Max 2 actions per cycle
                    action_type = action.get('action', '')
                    market = action.get('market', '')
                    confidence = action.get('confidence', 0)

                    success = bot.execute_action(action, market_data)
                    status = "OK" if success else "SKIP"
                    print(f"      {status}: {action_type} {market} ({confidence}%)")
                    cycle_actions[bot.name].append({
                        "type": status,
                        "action": action_type,
                        "market": market,
                        "confidence": confidence
                    })
            else:
                print(f"      HOLD: No actions")
                cycle_actions[bot.name].append({"type": "HOLD", "action": "HOLD", "market": "", "confidence": 0})

            # Update metrics
            bot.update_metrics(market_data)

        # Record snapshot for race visualization
        snapshot = self.get_race_snapshot(market_data, cycle_actions)
        self.race_history.append(snapshot)
        self.export_race_data()  # Write JSON after each cycle

        # Print leaderboard
        self._print_leaderboard(market_data)

    def get_race_snapshot(self, market_data: Dict, cycle_actions: Dict[str, List[Dict]] = None) -> Dict:
        """Get current race state as structured data"""
        snapshot = {
            "cycle": self.cycle,
            "timestamp": datetime.now().isoformat(),
            "bots": {}
        }

        for bot in self.bots:
            value = bot.get_portfolio_value(market_data)
            pnl = value - bot.starting_capital

            snapshot["bots"][bot.name] = {
                "control_level": bot.control_level.value,
                "value": round(value, 2),
                "pnl": round(pnl, 2),
                "win_rate": round(bot.performance.win_rate, 1),
                "total_trades": bot.performance.total_trades,
                "winning_trades": bot.performance.winning_trades,
                "losing_trades": bot.performance.losing_trades,
                "max_drawdown": round(bot.performance.max_drawdown, 1),
                "open_positions": len(bot.open_positions),
                "actions": cycle_actions.get(bot.name, []) if cycle_actions else []
            }

        # Add rankings
        sorted_bots = sorted(snapshot["bots"].items(), key=lambda x: x[1]["value"], reverse=True)
        for rank, (name, _) in enumerate(sorted_bots, 1):
            snapshot["bots"][name]["rank"] = rank

        return snapshot

    def export_race_data(self, filepath: str = "race_data.json"):
        """Export full race history to JSON file"""
        data = {
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "total_cycles": self.cycle,
            "bots": [{"name": b.name, "control_level": b.control_level.value} for b in self.bots],
            "history": self.race_history
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def _print_leaderboard(self, market_data: Dict):
        """Print current standings"""
        print(f"\n{'─'*70}")
        print("LEADERBOARD")
        print(f"{'─'*70}")
        print(f"{'Bot':<15} {'Value':>10} {'PnL':>10} {'Win%':>8} {'Trades':>8} {'DD%':>8}")
        print(f"{'─'*70}")

        # Sort by portfolio value
        sorted_bots = sorted(
            self.bots,
            key=lambda b: b.get_portfolio_value(market_data),
            reverse=True
        )

        for i, bot in enumerate(sorted_bots):
            value = bot.get_portfolio_value(market_data)
            pnl = value - bot.starting_capital
            pnl_pct = (pnl / bot.starting_capital) * 100

            rank = ["1st", "2nd", "3rd", "4th", "5th"][i] if i < 5 else f"{i+1}th"

            print(f"{rank} {bot.name:<11} ${value:>8.2f} {pnl:>+9.2f} {bot.performance.win_rate:>7.1f}% {bot.performance.total_trades:>7} {bot.performance.max_drawdown:>7.1f}%")

        print(f"{'─'*70}")

    def run(self, cycles: int = 10, interval: int = 60):
        """Run the arena competition"""
        print("\n" + "="*70)
        print("BOT ARENA - Multi-Bot Competition")
        print("="*70)
        print(f"Bots: {len(self.bots)}")
        for bot in self.bots:
            print(f"  - {bot.name}: {bot.control_level.value}")
        print(f"Cycles: {cycles}")
        print(f"Interval: {interval}s")
        print("="*70)

        self.start_time = datetime.now()

        for _ in range(cycles):
            try:
                # Fetch market data (shared by all bots)
                market_data = get_enhanced_market_data()

                if not market_data:
                    print("No market data, skipping cycle...")
                    time.sleep(interval)
                    continue

                self.run_cycle(market_data)

                time.sleep(interval)

            except KeyboardInterrupt:
                print("\nArena stopped by user")
                break
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(10)

        # Final results
        self._print_final_results(market_data)

    def _print_final_results(self, market_data: Dict):
        """Print final competition results"""
        print("\n" + "="*70)
        print("FINAL RESULTS")
        print("="*70)

        sorted_bots = sorted(
            self.bots,
            key=lambda b: b.get_portfolio_value(market_data),
            reverse=True
        )

        winner = sorted_bots[0]
        print(f"\nWINNER: {winner.name} ({winner.control_level.value})")
        print(f"Final Value: ${winner.get_portfolio_value(market_data):.2f}")
        print(f"Total PnL: ${winner.performance.total_pnl:.2f}")
        print(f"Win Rate: {winner.performance.win_rate:.1f}%")

        print("\n" + "-"*70)
        print("All Bots Summary:")
        print("-"*70)

        for bot in sorted_bots:
            value = bot.get_portfolio_value(market_data)
            roi = ((value - bot.starting_capital) / bot.starting_capital) * 100
            print(f"\n{bot.name} ({bot.control_level.value}):")
            print(f"  Final: ${value:.2f} (ROI: {roi:+.1f}%)")
            print(f"  Trades: {bot.performance.total_trades} (W:{bot.performance.winning_trades} L:{bot.performance.losing_trades})")
            print(f"  Win Rate: {bot.performance.win_rate:.1f}%")
            print(f"  Max Drawdown: {bot.performance.max_drawdown:.1f}%")
            print(f"  Sharpe: {bot.performance.sharpe_ratio:.2f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Bot Arena - Multi-Bot Paper Trading Competition')
    parser.add_argument('--cycles', type=int, default=0, help='Number of cycles to run (0 = unlimited)')
    parser.add_argument('--interval', type=int, default=30, help='Seconds between cycles (default: 30)')
    parser.add_argument('--capital', type=float, default=100.0, help='Starting capital per bot (default: $100)')
    args = parser.parse_args()

    print("BOT ARENA - Multi-Bot Competition System")
    print("="*50)
    print("Paper trading mode - no real money")
    print("="*50)

    arena = BotArena()
    arena.create_default_bots(starting_capital=args.capital)

    if args.cycles == 0:
        print("\nRunning indefinitely (Ctrl+C to stop)...")
        arena.run(cycles=999999, interval=args.interval)
    else:
        arena.run(cycles=args.cycles, interval=args.interval)
