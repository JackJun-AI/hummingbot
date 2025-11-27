import asyncio
import json
import time
from decimal import Decimal
from typing import Dict, List, Optional

import pandas as pd
import pandas_ta as ta
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from pydantic import Field, field_validator
from pydantic_core.core_schema import ValidationInfo

from hummingbot.core.data_type.common import TradeType, PriceType
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy_v2.controllers.directional_trading_controller_base import (
    DirectionalTradingControllerBase,
    DirectionalTradingControllerConfigBase,
)
from hummingbot.strategy_v2.executors.position_executor.data_types import PositionExecutorConfig
from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction, ExecutorAction, StopExecutorAction


class AIAgentV1Config(DirectionalTradingControllerConfigBase):
    """AI Agent Trading Controller Configuration"""
    controller_name: str = "ai_agent_v1"
    
    # K线配置（回测必需，与 Bollinger V1 保持一致）
    candles_config: List[CandlesConfig] = []
    candles_connector: str = Field(
        default=None,
        json_schema_extra={
            "prompt": "Enter the connector for the candles data, leave empty to use the same exchange as the connector: ",
            "prompt_on_new": True
        }
    )
    candles_trading_pair: str = Field(
        default=None,
        json_schema_extra={
            "prompt": "Enter the trading pair for the candles data, leave empty to use the same trading pair as the connector: ",
            "prompt_on_new": True
        }
    )
    interval: str = Field(
        default="5m",
        json_schema_extra={
            "prompt": "Enter the candle interval (e.g., 1m, 5m, 1h, 1d): ",
            "prompt_on_new": True
        }
    )
    
    # 多币种配置
    trading_pairs: List[str] = Field(
        default=["BTC-USDT", "ETH-USDT", "SOL-USDT"],
        json_schema_extra={
            "prompt": "Enter comma-separated trading pairs (e.g., BTC-USDT,ETH-USDT): ",
            "prompt_on_new": True
        }
    )
    
    # AI 配置
    openrouter_api_key: str = Field(
        default="",
        json_schema_extra={
            "prompt": "Enter OpenRouter API key: ",
            "prompt_on_new": True,
            "is_secure": True
        }
    )
    
    llm_model: str = Field(
        default="anthropic/claude-3.5-sonnet",
        json_schema_extra={
            "prompt": "Enter LLM model (e.g., anthropic/claude-3.5-sonnet, deepseek/deepseek-chat): ",
            "prompt_on_new": True
        }
    )
    
    llm_temperature: Decimal = Field(
        default=Decimal("0.1"),
        json_schema_extra={
            "prompt": "Enter LLM temperature (0.0-1.0, lower = more conservative): ",
            "prompt_on_new": True
        }
    )
    
    llm_max_tokens: int = Field(
        default=4000,
        json_schema_extra={
            "prompt": "Enter LLM max tokens: ",
            "prompt_on_new": True
        }
    )
    
    # 决策间隔（秒）
    decision_interval: int = Field(
        default=180,  # 3分钟
        json_schema_extra={
            "prompt": "Enter decision interval in seconds (e.g., 180 for 3 minutes): ",
            "prompt_on_new": True,
            "is_updatable": True
        }
    )
    
    # K线配置
    candles_interval: str = Field(
        default="3m",
        json_schema_extra={
            "prompt": "Enter candles interval (e.g., 3m, 5m, 15m): ",
            "prompt_on_new": True
        }
    )
    
    candles_max_records: int = Field(
        default=100,
        json_schema_extra={
            "prompt": "Enter max candles records: ",
            "prompt_on_new": True
        }
    )
    
    # 风险控制
    max_concurrent_positions: int = Field(
        default=3,
        json_schema_extra={
            "prompt": "Enter max concurrent positions: ",
            "prompt_on_new": True,
            "is_updatable": True
        }
    )
    
    single_position_size_pct: Decimal = Field(
        default=Decimal("0.3"),
        json_schema_extra={
            "prompt": "Enter single position size as percentage of total_amount_quote (e.g., 0.3 for 30%): ",
            "prompt_on_new": True,
            "is_updatable": True
        }
    )
    
    # Validators（自动设置 candles_connector 和 candles_trading_pair）
    @field_validator("candles_connector", mode="before")
    @classmethod
    def set_candles_connector(cls, v, validation_info: ValidationInfo):
        if v is None or v == "":
            return validation_info.data.get("connector_name")
        return v

    @field_validator("candles_trading_pair", mode="before")
    @classmethod
    def set_candles_trading_pair(cls, v, validation_info: ValidationInfo):
        if v is None or v == "":
            return validation_info.data.get("trading_pair")
        return v


class AIAgentV1Controller(DirectionalTradingControllerBase):
    """
    AI Agent Trading Controller - V1 MVP
    
    Features:
    - Multi-pair monitoring
    - Funding rate tracking
    - LLM-based decision making (via OpenRouter)
    - Position and trade history tracking
    """
    
    def __init__(self, config: AIAgentV1Config, *args, **kwargs):
        self.config = config
        
        # 为每个交易对配置K线数据
        # 与 Bollinger V1 保持一致的初始化逻辑
        if len(self.config.candles_config) == 0:
            self.config.candles_config = [
                CandlesConfig(
                    connector=config.candles_connector,
                    trading_pair=pair,
                    interval=config.interval,  # 使用 interval 字段（与 candles_interval 保持同步）
                    max_records=config.candles_max_records
                ) for pair in config.trading_pairs
            ]
        
        super().__init__(config, *args, **kwargs)
        
        # 决策时间追踪
        self._last_decision_time = 0
        self._decision_in_progress = False
        
        # 初始化 LangChain LLM
        self._init_langchain_llm()
        
        self.logger().info(f"AI Agent V1 initialized - monitoring {len(config.trading_pairs)} pairs")
    
    def _init_langchain_llm(self):
        """初始化 LangChain LLM"""
        try:
            # 使用 LangChain 的 ChatOpenAI（兼容 OpenRouter）
            self.llm = ChatOpenAI(
                model=self.config.llm_model,
                openai_api_key=self.config.openrouter_api_key,
                openai_api_base="https://openrouter.ai/api/v1",
                temperature=float(self.config.llm_temperature),
                max_tokens=self.config.llm_max_tokens,
                timeout=30,
                max_retries=2,
            )
            
            # JSON 输出解析器
            self.json_parser = JsonOutputParser()
            
            self.logger().info(f"LangChain LLM initialized: {self.config.llm_model}")
            
        except Exception as e:
            self.logger().error(f"Failed to initialize LangChain LLM: {e}")
            self.llm = None
            self.json_parser = None
    
    async def update_processed_data(self):
        """
        更新处理后的数据
        
        ⚠️  注意：回测引擎只在开始前调用一次此方法，然后在循环中直接调用 determine_executor_actions()
        因此决策间隔逻辑已移到 determine_executor_actions() 中处理
        
        实盘模式：每个 tick 调用（但实际决策在 determine_executor_actions）
        回测模式：只调用一次（用于初始化）
        """
        # 简单标记数据已准备好
        self.processed_data["initialized"] = True
        
        # 实盘模式下，可以在这里预加载一些数据
        # 但不执行 AI 决策（决策在 determine_executor_actions 中）
        pass
    
    async def _build_trading_context(self) -> Dict:
        """
        构建 AI 决策所需的完整上下文
        """
        context = {
            "timestamp": self.market_data_provider.time(),
            "account": self._get_account_summary(),
            "positions": self._get_positions_summary(),
            "market_data": {},
            "funding_rates": {},
            "recent_trades": self._get_recent_trades(limit=10),
        }
        
        # 收集每个币种的市场数据
        self.logger().debug(f"Collecting market data for {len(self.config.trading_pairs)} pairs...")
        
        for pair in self.config.trading_pairs:
            try:
                self.logger().debug(f"Getting market info for {pair}...")
                market_info = await self._get_market_info(pair)
                
                # 🔧 即使有错误也要记录（便于调试）
                context["market_data"][pair] = market_info
                
                if "error" in market_info:
                    self.logger().warning(f"⚠️  {pair}: {market_info['error']}")
                else:
                    self.logger().debug(f"✅ {pair}: Price ${market_info.get('current_price', 0):.2f}")
                
                # 获取资金费率（仅 Perpetual）
                if "_perpetual" in self.config.connector_name:
                    funding_rate = await self._get_funding_rate(pair)
                    context["funding_rates"][pair] = funding_rate
                    
            except Exception as e:
                self.logger().error(f"❌ Failed to get market info for {pair}: {e}", exc_info=True)
                # 🔧 记录错误，而不是跳过
                context["market_data"][pair] = {"error": str(e)}
        
        self.logger().info(f"Market data collected: {len(context['market_data'])} pairs")
        
        return context
    
    def _get_account_summary(self) -> Dict:
        """获取账户摘要"""
        return {
            "total_amount_quote": float(self.config.total_amount_quote),
            "max_concurrent_positions": self.config.max_concurrent_positions,
            "single_position_size_pct": float(self.config.single_position_size_pct),
        }
    
    def _get_positions_summary(self) -> List[Dict]:
        """获取当前持仓摘要"""
        positions = []
        
        for executor in self.executors_info:
            if executor.is_active and executor.is_trading:
                try:
                    pos = {
                        "symbol": executor.config.trading_pair,
                        "side": executor.config.side.name,
                        "entry_price": float(executor.config.entry_price),
                        "amount": float(executor.config.amount),
                        "net_pnl_pct": float(executor.net_pnl_pct),
                        "net_pnl_quote": float(executor.net_pnl_quote),
                        "timestamp": executor.timestamp,
                        "executor_id": executor.id,
                    }
                    positions.append(pos)
                except Exception as e:
                    self.logger().warning(f"Error processing executor {executor.id}: {e}")
        
        return positions
    
    def _get_recent_trades(self, limit: int = 10) -> List[Dict]:
        """
        获取最近的交易记录
        
        从 executors_info 中提取已完成的交易
        """
        closed_executors = []
        
        for e in self.executors_info:
            if not (hasattr(e, 'status') and str(e.status) == 'RunnableStatus.TERMINATED'):
                continue
            
            # 获取入场价格
            entry_price = 0.0
            if hasattr(e, 'entry_price') and e.entry_price:
                entry_price = float(e.entry_price)
            elif hasattr(e, 'config') and hasattr(e.config, 'entry_price'):
                entry_price = float(e.config.entry_price)
            
            # 计算退出价格（基于 PnL）
            exit_price = 0.0
            if entry_price > 0 and hasattr(e, 'net_pnl_pct') and e.net_pnl_pct:
                pnl_pct = float(e.net_pnl_pct)
                side_multiplier = 1 if e.config.side.name == "BUY" else -1
                # exit_price = entry_price * (1 + pnl_pct * side_multiplier)
                # 简化：直接根据 PnL 百分比计算
                exit_price = entry_price * (1 + pnl_pct * side_multiplier)
            
            trade = {
                "symbol": e.config.trading_pair,
                "side": e.config.side.name,
                "entry_price": entry_price,
                "exit_price": exit_price,  # 🔧 添加退出价格
                "pnl_pct": float(e.net_pnl_pct) if hasattr(e, 'net_pnl_pct') else 0,
                "pnl_quote": float(e.net_pnl_quote) if hasattr(e, 'net_pnl_quote') else 0,
                "close_type": e.close_type.name if hasattr(e, 'close_type') and e.close_type else "UNKNOWN",
                "timestamp": e.timestamp if hasattr(e, 'timestamp') else 0,
                "close_timestamp": e.close_timestamp if hasattr(e, 'close_timestamp') else 0,
            }
            closed_executors.append(trade)
        
        return closed_executors[-limit:] if closed_executors else []
    
    async def _get_market_info(self, trading_pair: str) -> Dict:
        """获取单个币种的市场信息"""
        try:
            self.logger().debug(f"Fetching candles for {trading_pair}...")
            self.logger().debug(f"  Connector: {self.config.connector_name}")
            self.logger().debug(f"  Interval: {self.config.interval}")
            self.logger().debug(f"  Max records: {self.config.candles_max_records}")
            
            # 🔧 调试：检查 market_data_provider 中是否有这个交易对的数据
            available_keys = list(self.market_data_provider.candles_feeds.keys()) if hasattr(self.market_data_provider, 'candles_feeds') else []
            self.logger().debug(f"  Available candles feeds: {available_keys}")
            
            # 获取K线数据
            candles = self.market_data_provider.get_candles_df(
                connector_name=self.config.connector_name,
                trading_pair=trading_pair,
                interval=self.config.interval,
                max_records=self.config.candles_max_records
            )
            
            # 🔑 关键修复：过滤未来数据（防止 look-ahead bias）
            # 兼容实盘和回测两种模式
            if hasattr(self.market_data_provider, 'time'):
                current_time = self.market_data_provider.time()
                
                if not candles.empty and "timestamp" in candles.columns:
                    before_filter = len(candles)
                    candles = candles[candles["timestamp"] <= current_time]
                    after_filter = len(candles)
                    
                    # 只在回测时记录过滤信息（避免实盘日志过多）
                    if before_filter != after_filter:
                        self.logger().warning(
                            f"🔒 Time filter: {before_filter} → {after_filter} candles "
                            f"(current_time: {pd.to_datetime(current_time, unit='s')})"
                        )
                    
                    # 只保留最近 max_records 条（避免计算太多历史数据）
                    if len(candles) > self.config.candles_max_records:
                        candles = candles.tail(self.config.candles_max_records)
                        self.logger().warning(f"   Keeping last {self.config.candles_max_records} candles")
            
            self.logger().warning(f"Received {len(candles) if not candles.empty else 0} candles for {trading_pair}")
            
            if candles.empty or len(candles) < 20:
                self.logger().warning(
                    f"❌ Insufficient candles for {trading_pair}: {len(candles) if not candles.empty else 0} rows\n"
                    f"   Current time: {pd.to_datetime(current_time, unit='s') if 'current_time' in locals() else 'N/A'}\n"
                    f"   Available feeds: {available_keys}\n"
                    f"   Looking for: {self.config.connector_name}_{trading_pair}_{self.config.interval}"
                )
                return {"error": "insufficient_data", "symbol": trading_pair, "candles_count": len(candles) if not candles.empty else 0}
            
            # 计算技术指标（只使用当前时刻及之前的数据）
            close = candles["close"]
            high = candles["high"]
            low = candles["low"]
            
            self.logger().debug(f"Calculating indicators for {trading_pair}...")
            
            rsi = ta.rsi(close, length=14)
            macd = ta.macd(close, fast=12, slow=26, signal=9)
            ema_20 = ta.ema(close, length=20)
            
            current_price = float(close.iloc[-1])
            
            market_info = {
                "symbol": trading_pair,
                "current_price": current_price,
                "rsi": float(rsi.iloc[-1]) if not rsi.isna().iloc[-1] else None,
                "macd": float(macd[f"MACD_12_26_9"].iloc[-1]) if not macd.empty else None,
                "macd_signal": float(macd[f"MACDs_12_26_9"].iloc[-1]) if not macd.empty else None,
                "ema_20": float(ema_20.iloc[-1]) if not ema_20.isna().iloc[-1] else None,
                "price_change_24h_pct": self._calculate_price_change(candles),
                "volume_24h": float(candles["volume"].sum()),
            }
            
            # 🔧 修复：先格式化值，再构建日志字符串
            rsi_str = f"{market_info['rsi']:.1f}" if market_info['rsi'] is not None else 'N/A'
            macd_str = f"{market_info['macd']:.2f}" if market_info['macd'] is not None else 'N/A'
            ema_str = f"${market_info['ema_20']:.2f}" if market_info['ema_20'] is not None else 'N/A'
            
            self.logger().warning(
                f"✅ {trading_pair}: Price=${current_price:.2f}, "
                f"RSI={rsi_str}, MACD={macd_str}, EMA(20)={ema_str}"
            )
            self.logger().warning(market_info)
            return market_info
            
        except Exception as e:
            self.logger().error(f"❌ Error getting market info for {trading_pair}: {e}", exc_info=True)
            return {"error": str(e), "symbol": trading_pair}
    
    def _calculate_price_change(self, candles: pd.DataFrame) -> float:
        """计算24小时价格变化百分比"""
        if len(candles) < 2:
            return 0.0
        first_price = float(candles["close"].iloc[0])
        last_price = float(candles["close"].iloc[-1])
        return ((last_price - first_price) / first_price) * 100
    
    async def _get_funding_rate(self, trading_pair: str) -> Dict:
        """获取资金费率（仅 Perpetual 合约）"""
        try:
            # 方法 1: 使用 market_data_provider (推荐，适用于实盘和回测)
            if hasattr(self, 'market_data_provider'):
                try:
                    funding_info = self.market_data_provider.get_funding_info(
                        self.config.connector_name, 
                        trading_pair
                    )
                    return {
                        "rate": float(funding_info.rate),
                        "next_funding_time": funding_info.next_funding_utc_timestamp,
                    }
                except Exception as e:
                    self.logger().debug(f"market_data_provider.get_funding_info failed for {trading_pair}: {e}")
            
            # 方法 2: 直接从 connector 获取 (仅实盘可用)
            if hasattr(self, 'connectors'):
                connector = self.connectors.get(self.config.connector_name)
                if connector and hasattr(connector, 'get_funding_info'):
                    funding_info = connector.get_funding_info(trading_pair)
                    return {
                        "rate": float(funding_info.rate),
                        "next_funding_time": funding_info.next_funding_utc_timestamp,
                    }
            
            # 方法 3: 回测环境，返回默认值
            self.logger().debug(f"Funding info not available for {trading_pair} (backtest mode)")
            return {"rate": 0.0, "next_funding_time": 0}
                
        except Exception as e:
            self.logger().debug(f"Failed to get funding rate for {trading_pair}: {e}")
            return {"rate": 0.0, "next_funding_time": 0}
    
    async def _get_ai_decisions(self, context: Dict) -> List[Dict]:
        """
        调用 LLM 获取交易决策（使用 LangChain）
        """
        try:
            # Step 1: 构建 Prompt
            self.logger().debug("Building prompts...")
            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(context)
            

            self.logger().warning(f"System prompt: {system_prompt}")
            self.logger().warning(f"User prompt: {user_prompt}")
        
            
            # Step 2: 使用 LangChain 调用 LLM
            self.logger().info("Calling LLM API...")
            response = await self._call_langchain_llm(system_prompt, user_prompt)
            self.logger().info("LLM response received")
            
            # 打印完整响应用于调试
            self.logger().warning(f"LLM full response:\n{response}")
            
            # Step 3: 解析决策
            self.logger().debug("Parsing LLM response...")
            decisions = self._parse_ai_response(response)
            self.logger().info(f"Parsed {len(decisions)} raw decisions from LLM")
            
            # Step 4: 验证决策
            self.logger().debug("Validating decisions...")
            validated_decisions = self._validate_decisions(decisions, context)
            self.logger().info(f"Validated {len(validated_decisions)}/{len(decisions)} decisions")
            
            return validated_decisions
            
        except Exception as e:
            self.logger().error(f"Error in AI decision process: {e}", exc_info=True)
            return []
    
    def _build_system_prompt(self) -> str:
        """构建系统 Prompt"""
        return f"""You are an autonomous cryptocurrency trading agent with systematic, disciplined approach.

# ROLE & MISSION
Your mission: Maximize risk-adjusted returns through disciplined trading decisions based on technical analysis and risk management principles.

---

# TRADING ENVIRONMENT

## Your Trading Setup
- **Exchange**: {self.config.connector_name}
- **Available Pairs**: {', '.join(self.config.trading_pairs)}
- **Max Concurrent Positions**: {self.config.max_concurrent_positions}
- **Position Size**: {float(self.config.single_position_size_pct) * 100}% of capital per trade"""
        
        # 🔧 修复：stop_loss, take_profit, time_limit 可能为 None
        if self.config.triple_barrier_config.stop_loss is not None:
            system_prompt += f"\n- **Base Stop Loss**: {float(self.config.triple_barrier_config.stop_loss) * 100}%"
        
        if self.config.triple_barrier_config.take_profit is not None:
            system_prompt += f"\n- **Base Take Profit**: {float(self.config.triple_barrier_config.take_profit) * 100}%"
        
        if self.config.triple_barrier_config.time_limit is not None:
            system_prompt += f"\n- **Max Hold Time**: {self.config.triple_barrier_config.time_limit / 3600:.1f} hours"
        
        system_prompt += """

## Market Type
- **Perpetual Contracts**: No expiration, funding rate mechanism
- **Funding Rate Impact**:
  - Positive funding = Longs pay shorts (bullish sentiment)
  - Negative funding = Shorts pay longs (bearish sentiment)
  - Extreme funding rates (>0.01%) = Potential reversal signal

---

# AVAILABLE ACTIONS

You have exactly FOUR possible actions:

1. **open_long**: Open a LONG position (bet on price appreciation)
   - Use when: Bullish technical setup, positive momentum, clear uptrend

2. **open_short**: Open a SHORT position (bet on price depreciation)
   - Use when: Bearish technical setup, negative momentum, clear downtrend

3. **close_position**: Exit an existing position
   - Use when: Profit target reached, stop loss triggered, or thesis invalidated

4. **hold**: Maintain current positions OR wait for better opportunity
   - Use when: No clear edge exists, or existing positions are performing as expected

**IMPORTANT**: You are NOT required to trade if market conditions are unclear. Quality over quantity.

---

# TECHNICAL INDICATORS PROVIDED

**RSI (Relative Strength Index)**: Overbought/Oversold conditions
- RSI > 70 = Overbought (potential reversal down or trend continuation)
- RSI < 30 = Oversold (potential reversal up)
- RSI 40-60 = Neutral zone

**MACD (Moving Average Convergence Divergence)**: Momentum & Trend
- MACD > Signal = Bullish momentum
- MACD < Signal = Bearish momentum
- MACD crossover = Potential trend change

**EMA(20) (Exponential Moving Average)**: Trend direction
- Price > EMA = Uptrend
- Price < EMA = Downtrend

**24h Price Change**: Short-term momentum indicator

**Funding Rate** (Perpetual only): Market sentiment
- Positive funding = Bullish sentiment
- Negative funding = Bearish sentiment

---

# DECISION-MAKING FRAMEWORK

## Step 1: Analyze Current Positions
- Are existing positions performing as expected?
- Should any positions be closed (profit target, stop loss, invalidation)?

## Step 2: Identify Market Conditions
- What is the trend? (Use EMA, MACD, price action)
- Is momentum strong or weakening? (Use MACD, RSI)
- Is market overbought/oversold? (Use RSI)
- What is sentiment? (Use funding rate if available)

## Step 3: Scan for Opportunities
- Do any pairs show clear technical setups?
- Is risk/reward favorable (minimum 1.5:1)?
- Do you have available capital?

## Step 4: Risk Management Check
- Does this trade fit your risk parameters?
- Is stop loss placement logical?
- Is position size appropriate for confidence level?

## Step 5: Make Decision
- If clear edge exists → Trade
- If uncertain → Hold/Wait
- **Never force trades**

---

# RISK MANAGEMENT RULES (MANDATORY)

For EVERY trade, you must specify:

1. **stop_loss_pct** (float): Percentage stop loss
   - Typical range: 0.015 - 0.035 (1.5% - 3.5%)
   - Place beyond recent support/resistance

2. **take_profit_pct** (float): Percentage take profit
   - Typical range: 0.03 - 0.08 (3% - 8%)
   - Minimum 1.5:1 reward-to-risk ratio

3. **confidence** (int, 0-100): Your conviction level
   - 0-30: Low confidence (avoid trading)
   - 30-60: Moderate confidence (standard sizing)
   - 60-80: High confidence (acceptable)
   - 80-100: Very high confidence (rare, use cautiously)

4. **reasoning** (string): Your complete analysis
   - What technical signals support this trade?
   - What is the market context?
   - What could invalidate this thesis?
   - Why is risk/reward favorable?

---

# OUTPUT FORMAT

You must respond with a JSON array. **ALWAYS start with your reasoning, then decide.**

Format:
```json
[
  {{
    "reasoning": "BTC shows strong bullish divergence: RSI recovering from oversold (<30), MACD bullish crossover, price above EMA(20) at $42,500. Funding rate is neutral. Risk/reward is 1:2 with stop at $41,800 (-2.5%) and target at $44,200 (+4%). High confidence setup.",
    "action": "open_long",
    "symbol": "BTC-USDT",
    "stop_loss_pct": 0.025,
    "take_profit_pct": 0.05,
    "confidence": 75
  }}
]
```

**If no clear opportunity exists, return:**
```json
[
  {{
    "reasoning": "Market conditions are unclear. BTC range-bound with RSI neutral at 50, MACD flat. No clear directional bias. Waiting for better setup.",
    "action": "hold",
    "symbol": null,
    "stop_loss_pct": null,
    "take_profit_pct": null,
    "confidence": 0
  }}
]
```

---

# TRADING PHILOSOPHY

**Core Principles:**
1. **Capital Preservation First**: Protecting capital > chasing gains
2. **Discipline Over Emotion**: Follow your plan, don't move stops
3. **Quality Over Quantity**: Fewer high-conviction trades beat many random trades
4. **Respect the Trend**: Don't fight strong directional moves
5. **Learn from History**: Review recent trades to avoid repeated mistakes

**Common Pitfalls to Avoid:**
- ⚠️ Overtrading: Excessive trading erodes capital through fees
- ⚠️ Revenge Trading: Don't increase size after losses
- ⚠️ Analysis Paralysis: Don't wait for perfect setups
- ⚠️ Ignoring Risk: Always set stops, never "hope" for recovery

---

# FINAL INSTRUCTIONS

1. **Think before acting**: Analyze thoroughly before deciding
2. **Be honest about confidence**: Don't overstate conviction
3. **Provide detailed reasoning**: Explain your technical analysis
4. **Respect risk management**: Always set proper stops and targets
5. **Don't force trades**: It's OK to wait for better opportunities

Remember: Consistent, disciplined trading beats aggressive speculation. Focus on high-probability setups with favorable risk/reward.

Now analyze the market data and make your decision.
"""
    
    def _build_user_prompt(self, context: Dict) -> str:
        """构建用户 Prompt（包含实时数据）"""
        import json
        
        prompt_parts = []
        
        # 1. 账户信息
        prompt_parts.append(f"# ACCOUNT STATUS")
        prompt_parts.append(f"Total Capital: ${context['account']['total_amount_quote']:.2f}")
        prompt_parts.append(f"Active Positions: {len(context['positions'])}/{self.config.max_concurrent_positions}")
        prompt_parts.append(f"Available Slots: {self.config.max_concurrent_positions - len(context['positions'])}")
        
        # 2. 当前持仓
        if context["positions"]:
            prompt_parts.append(f"\n# CURRENT POSITIONS")
            for pos in context["positions"]:
                prompt_parts.append(
                    f"\n**{pos['symbol']} {pos['side']}**"
                    f"\n- Entry Price: ${pos['entry_price']:.2f}"
                    f"\n- Current PnL: {pos['net_pnl_pct']*100:.2f}% (${pos['net_pnl_quote']:.2f})"
                    f"\n- Position ID: {pos['executor_id']}"
                )
                prompt_parts.append("\n**Action Required?** Evaluate if this position should be closed based on:")
                prompt_parts.append("- Has profit target been reached?")
                prompt_parts.append("- Is stop loss triggered?")
                prompt_parts.append("- Is the thesis still valid?")
        else:
            prompt_parts.append(f"\n# CURRENT POSITIONS")
            prompt_parts.append(f"No active positions. You may open new positions if good opportunities exist.")
        
        # 3. 市场数据
        prompt_parts.append(f"\n# MARKET DATA")
        for symbol, data in context["market_data"].items():
            if "error" in data:
                continue
            
            funding_info = context["funding_rates"].get(symbol, {})
            funding_rate = funding_info.get("rate", 0.0)
            
            prompt_parts.append(
                f"\n## {symbol}"
                f"\n**Price & Trend:**"
                f"\n- Current Price: ${data['current_price']:.2f}"
                f"\n- 24h Change: {data['price_change_24h_pct']:.2f}%"
                f"\n- EMA(20): ${data['ema_20']:.2f if data['ema_20'] else 'N/A'}"
            )
            
            if data['ema_20']:
                if data['current_price'] > data['ema_20']:
                    prompt_parts.append(f"- Trend: UPTREND (Price > EMA)")
                else:
                    prompt_parts.append(f"- Trend: DOWNTREND (Price < EMA)")
            
            prompt_parts.append(
                f"\n**Technical Indicators:**"
                f"\n- RSI: {data['rsi']:.1f if data['rsi'] else 'N/A'}"
            )
            
            if data['rsi']:
                if data['rsi'] > 70:
                    prompt_parts.append(f"  → Overbought (potential reversal or strong trend)")
                elif data['rsi'] < 30:
                    prompt_parts.append(f"  → Oversold (potential reversal)")
                else:
                    prompt_parts.append(f"  → Neutral")
            
            prompt_parts.append(
                f"- MACD: {data['macd']:.2f if data['macd'] else 'N/A'}"
                f"\n- MACD Signal: {data['macd_signal']:.2f if data['macd_signal'] else 'N/A'}"
            )
            
            if data['macd'] and data['macd_signal']:
                if data['macd'] > data['macd_signal']:
                    prompt_parts.append(f"  → Bullish momentum (MACD > Signal)")
                else:
                    prompt_parts.append(f"  → Bearish momentum (MACD < Signal)")
            
            if funding_rate:
                prompt_parts.append(
                    f"\n**Funding Rate:** {funding_rate*100:.4f}% (8h)"
                )
                if funding_rate > 0.0001:
                    prompt_parts.append(f"  → Bullish sentiment (longs paying shorts)")
                elif funding_rate < -0.0001:
                    prompt_parts.append(f"  → Bearish sentiment (shorts paying longs)")
                else:
                    prompt_parts.append(f"  → Neutral sentiment")
        
        # 4. 历史交易记录
        if context["recent_trades"]:
            prompt_parts.append(f"\n# RECENT TRADES (Last {len(context['recent_trades'])})")
            prompt_parts.append("Learn from these trades to improve your strategy:\n")
            
            for trade in context["recent_trades"]:
                # 计算持仓时长
                duration_hours = 0
                if trade['close_timestamp'] and trade['timestamp']:
                    duration_hours = (trade['close_timestamp'] - trade['timestamp']) / 3600
                
                # 格式化交易信息
                pnl_emoji = "✅ PROFIT" if trade['pnl_quote'] > 0 else "❌ LOSS"
                prompt_parts.append(
                    f"- **{trade['symbol']} {trade['side']}**: "
                    f"Entry ${trade['entry_price']:.2f} → Exit ${trade['exit_price']:.2f}, "
                    f"PnL: {trade['pnl_pct']*100:.2f}% (${trade['pnl_quote']:.2f}) {pnl_emoji}, "
                    f"Close Reason: {trade['close_type']}, "
                    f"Duration: {duration_hours:.1f}h"
                )
        
        # 5. 决策指令
        prompt_parts.append(f"\n# YOUR DECISION")
        
        prompt_parts.append(
            f"\nBased on the above data, make your trading decision following these steps:\n"
            f"\n**Step 1: Analyze Current Positions**"
            f"\n- Should any existing positions be closed?"
            f"\n- Are they performing as expected?"
            f"\n\n**Step 2: Evaluate Market Conditions**"
            f"\n- What is the overall trend for each pair?"
            f"\n- Are there any clear technical setups?"
            f"\n- What is the risk/reward ratio?"
            f"\n\n**Step 3: Make Decision**"
            f"\n- If clear opportunity exists → Open position"
            f"\n- If existing position should exit → Close position"
            f"\n- If no clear edge → Hold/Wait"
            f"\n\n**Remember:**"
            f"\n- Quality over quantity (don't force trades)"
            f"\n- Always start with reasoning before deciding"
            f"\n- Set appropriate stop loss and take profit"
            f"\n- Be honest about your confidence level"
        )
        
        prompt_parts.append(f"\n\nProvide your decision in JSON format.")
        
        return "\n".join(prompt_parts)
    
    async def _call_langchain_llm(self, system_prompt: str, user_prompt: str) -> str:
        """使用 LangChain 调用 LLM"""
        if self.llm is None:
            raise RuntimeError("LangChain LLM not initialized")
        
        try:
            start_time = time.time()
            
            # 构建消息
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            self.logger().debug(f"Sending request to LLM ({self.config.llm_model})...")
            
            # 使用 ainvoke 异步调用
            response = await self.llm.ainvoke(messages)
            
            elapsed = time.time() - start_time
            content = response.content
            
            self.logger().info(f"LLM response received in {elapsed:.2f}s, length: {len(content)} chars")
            self.logger().debug(f"LLM Response preview: {content[:300]}...")
            
            return content
            
        except Exception as e:
            self.logger().error(f"LangChain LLM call failed: {e}", exc_info=True)
            raise
    
    def _parse_ai_response(self, response: str) -> List[Dict]:
        """解析 AI 返回的 JSON 决策"""
        try:
            self.logger().debug("Parsing LLM response...")
            
            # 尝试提取 JSON（可能被包裹在 ```json ``` 中）
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                json_str = response[start:end].strip()
                self.logger().debug("Found JSON in ```json``` block")
            elif "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                json_str = response[start:end].strip()
                self.logger().debug("Found JSON in ``` block")
            else:
                json_str = response.strip()
                self.logger().debug("Using raw response as JSON")
            
            self.logger().debug(f"JSON string to parse: {json_str[:200]}...")
            
            decisions = json.loads(json_str)
            
            if not isinstance(decisions, list):
                self.logger().warning("AI response is not a list, wrapping it")
                decisions = [decisions] if decisions else []
            
            self.logger().info(f"Successfully parsed {len(decisions)} decisions from LLM response")
            
            # 打印每个决策的基本信息
            for i, dec in enumerate(decisions, 1):
                self.logger().debug(f"Decision {i}: {dec}")
            
            return decisions
            
        except json.JSONDecodeError as e:
            self.logger().error(f"Failed to parse AI response as JSON: {e}")
            self.logger().error(f"Raw response (first 500 chars): {response[:500]}")
            return []
    
    def _validate_decisions(self, decisions: List[Dict], context: Dict) -> List[Dict]:
        """验证决策的合法性"""
        self.logger().info(f"Validating {len(decisions)} decisions...")
        
        validated = []
        current_positions = len(context["positions"])
        
        self.logger().debug(f"Current positions: {current_positions}, Max allowed: {self.config.max_concurrent_positions}")
        
        for i, decision in enumerate(decisions, 1):
            # 🔧 修复：先检查 reasoning（必须字段）
            reasoning = decision.get("reasoning", "")
            if not reasoning:
                self.logger().warning(f"❌ Decision {i}: missing reasoning field - {decision}")
                continue
            
            action = decision.get("action")
            symbol = decision.get("symbol")
            
            self.logger().debug(f"Validating decision {i}: {action} {symbol}")
            self.logger().debug(f"   Reasoning: {reasoning[:100]}...")  # 打印前100字符
            
            # hold 动作不需要 symbol
            if action == "hold":
                validated.append(decision)
                self.logger().info(f"✅ Decision {i}: HOLD - {reasoning[:50]}...")
                continue
            
            # 其他动作需要 symbol
            if not symbol:
                self.logger().warning(f"❌ Decision {i}: missing symbol for action {action}")
                continue
            
            # 检查交易对是否在配置中
            if symbol not in self.config.trading_pairs:
                self.logger().warning(f"❌ Decision {i}: symbol {symbol} not in configured pairs {self.config.trading_pairs}")
                continue
            
            # 检查仓位数量限制
            if action in ["open_long", "open_short"]:
                if current_positions >= self.config.max_concurrent_positions:
                    self.logger().warning(f"❌ Decision {i}: max positions ({self.config.max_concurrent_positions}) reached, skipping {action} for {symbol}")
                    continue
                
                # 检查止损止盈
                stop_loss_pct = decision.get("stop_loss_pct", 0.02)
                take_profit_pct = decision.get("take_profit_pct", 0.04)
                confidence = decision.get("confidence", 50)
                
                self.logger().debug(f"   SL: {stop_loss_pct*100:.1f}%, TP: {take_profit_pct*100:.1f}%, Confidence: {confidence}%")
                
                # 验证风险回报比
                if take_profit_pct < stop_loss_pct * 1.5:
                    self.logger().warning(f"⚠️  Decision {i}: R/R ratio too low for {symbol}, adjusting TP from {take_profit_pct*100:.1f}% to {stop_loss_pct*200:.1f}%")
                    decision["take_profit_pct"] = stop_loss_pct * 2
                
                current_positions += 1
                self.logger().debug(f"   ✅ Decision {i} validated, would be position #{current_positions}")
            
            validated.append(decision)
        
        self.logger().info(f"✅ Validation complete: {len(validated)}/{len(decisions)} decisions passed")
        
        if len(validated) < len(decisions):
            self.logger().warning(f"⚠️  {len(decisions) - len(validated)} decisions were rejected")
        
        return validated
    
    def _get_current_price(self, trading_pair: str) -> Optional[Decimal]:
        """
        获取当前价格 (支持回测和实盘)
        
        回测时从 K 线数据获取最新价格，因为 market_data_provider.prices 只支持单交易对
        """
        try:
            # 方法 1: 尝试从 market_data_provider 获取（实盘）
            price = self.market_data_provider.get_price_by_type(
                self.config.connector_name,
                trading_pair,
                price_type=PriceType.MidPrice
            )
            
            # 如果价格是默认值 1.0，说明回测时没有设置，从 K 线获取
            if price == Decimal("1") or price is None:
                # 方法 2: 从 K 线数据获取最新 close 价格（回测）
                candles_df = self.market_data_provider.get_candles_df(
                    connector_name=self.config.connector_name,
                    trading_pair=trading_pair,
                    interval=self.config.interval,
                    max_records=10  # 获取更多数据以确保有有效值
                )
                
                if not candles_df.empty:
                    # 获取最新 K 线的收盘价
                    latest_close = candles_df.iloc[-1]["close"]
                    price = Decimal(str(latest_close))
                    self.logger().debug(f"Got price from candles for {trading_pair}: {price}")
                else:
                    self.logger().error(f"❌ No candles data available for {trading_pair} - cannot get price!")
                    return None
            
            # 最后检查：确保价格有效
            if price is None or price <= 0:
                self.logger().error(f"❌ Invalid price for {trading_pair}: {price}")
                return None
            
            return price
            
        except Exception as e:
            self.logger().error(f"❌ Failed to get price for {trading_pair}: {e}", exc_info=True)
            return None
    
    def determine_executor_actions(self) -> List[ExecutorAction]:
        """
        根据 AI 决策生成 Executor Actions
        
        ⚠️  重要：回测引擎会在每个 tick 调用此方法，需要在这里控制决策频率
        """
        current_time = self.market_data_provider.time()
        
        # 🔧 修复：初始化决策时间
        if self._last_decision_time == 0:
            self._last_decision_time = current_time - self.config.decision_interval
            self.logger().info(
                f"⏱️  Decision timer initialized at timestamp {current_time} "
                f"(interval: {self.config.decision_interval}s)"
            )
        
        # 🔧 修复：检查决策间隔（防止回测时每个tick都决策）
        time_since_last = current_time - self._last_decision_time
        
        if time_since_last < self.config.decision_interval:
            # 未到决策时间，直接返回空列表
            return []
        
        # 🔑 到达决策时间，开始 AI 决策流程
        self.logger().info("=" * 80)
        self.logger().info(
            f"🤖 AI Decision Cycle Triggered "
            f"(time since last: {time_since_last:.0f}s, interval: {self.config.decision_interval}s)"
        )
        self.logger().info("=" * 80)
        
        # 🔧 修复：同步执行 AI 决策（避免事件循环冲突）
        try:
            ai_decisions = self._execute_ai_decision_cycle_sync()
            
        except Exception as e:
            self.logger().error(f"❌ AI decision cycle failed: {e}", exc_info=True)
            ai_decisions = []
        
        # 更新上次决策时间
        self._last_decision_time = current_time
        
        # 生成 Executor Actions
        if not ai_decisions:
            self.logger().warning("⚠️  No AI decisions generated")
            return []
        
        actions = []
        
        for i, decision in enumerate(ai_decisions, 1):
            action_type = decision.get("action")
            symbol = decision.get("symbol")
            
            self.logger().debug(f"Processing decision {i}/{len(ai_decisions)}: {action_type} {symbol}")
            
            try:
                if action_type == "open_long":
                    action = self._create_open_action(decision, TradeType.BUY)
                    if action:
                        actions.append(action)
                        self.logger().info(f"   ✅ Created LONG action for {symbol}")
                    else:
                        self.logger().warning(f"   ⚠️  Failed to create LONG action for {symbol}")
                        
                elif action_type == "open_short":
                    action = self._create_open_action(decision, TradeType.SELL)
                    if action:
                        actions.append(action)
                        self.logger().info(f"   ✅ Created SHORT action for {symbol}")
                    else:
                        self.logger().warning(f"   ⚠️  Failed to create SHORT action for {symbol}")
                        
                elif action_type == "close_position":
                    action = self._create_close_action(decision)
                    if action:
                        actions.append(action)
                        self.logger().info(f"   ✅ Created CLOSE action for {symbol}")
                    else:
                        self.logger().warning(f"   ⚠️  Failed to create CLOSE action for {symbol}")
                else:
                    self.logger().warning(f"   ⚠️  Unknown action type: {action_type}")
                        
            except Exception as e:
                self.logger().error(f"   ❌ Error creating action for {symbol}: {e}", exc_info=True)
        
        self.logger().info(f"📋 Generated {len(actions)} executor actions from {len(ai_decisions)} decisions")
        self.logger().info("=" * 80)
        
        # 过滤掉 None（安全检查）
        return [action for action in actions if action is not None]
    
    def _execute_ai_decision_cycle_sync(self) -> List[Dict]:
        """
        同步执行 AI 决策流程（避免事件循环冲突）
        
        🔧 修复：使用同步包装来避免 "event loop already running" 错误
        """
        import asyncio
        
        try:
            # 方法 1：检测是否有运行中的事件循环
            try:
                loop = asyncio.get_running_loop()
                # 如果有运行中的循环，使用 asyncio.create_task
                # 但在同步函数中无法直接 await，所以使用一个特殊方法
                self.logger().debug("Detected running event loop, using sync wrapper")
                
                # 创建新的事件循环在线程中运行（避免冲突）
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run, 
                        self._execute_ai_decision_cycle()
                    )
                    return future.result(timeout=30)  # 30秒超时
                    
            except RuntimeError:
                # 没有运行中的循环，直接运行
                self.logger().debug("No running event loop, using asyncio.run()")
                return asyncio.run(self._execute_ai_decision_cycle())
                
        except Exception as e:
            self.logger().error(f"Failed to execute AI decision cycle: {e}", exc_info=True)
            return []
    
    async def _execute_ai_decision_cycle(self) -> List[Dict]:
        """
        执行完整的 AI 决策流程（异步版本）
        """
        try:
            # Step 1: 构建上下文
            self.logger().info("📊 Building trading context...")
            context = await self._build_trading_context()
            self.logger().info(
                f"   ✅ Context: {len(context['market_data'])} pairs, "
                f"{len(context['positions'])} positions"
            )
            
            # Step 2: 调用 LLM
            self.logger().info("🧠 Calling LLM for decisions...")
            decisions = await self._get_ai_decisions(context)
            self.logger().info(f"   ✅ LLM returned {len(decisions)} decisions")
            
            # Step 3: 验证决策
            self.logger().debug("Validating decisions...")
            validated = self._validate_decisions(decisions, context)
            self.logger().info(f"   ✅ Validated {len(validated)}/{len(decisions)} decisions")
            
            return validated
            
        except Exception as e:
            self.logger().error(f"❌ Decision cycle failed: {e}", exc_info=True)
            return []
    
    def _create_open_action(self, decision: Dict, trade_type: TradeType) -> Optional[CreateExecutorAction]:
        """创建开仓 Action"""
        symbol = decision["symbol"]
        
        self.logger().info(f"🔍 Attempting to create {trade_type.name} action for {symbol}...")
        
        # 获取当前价格（用于计算仓位大小）
        # ⚠️  Workaround: 回测引擎不支持多交易对，需要从 K 线数据获取价格
        price = self._get_current_price(symbol)
        
        if price is None or price <= 0:
            self.logger().error(f"❌ Cannot get valid price for {symbol}, got {price} - SKIPPING this trade!")
            return None
        
        self.logger().info(f"   ✅ Got price for {symbol}: ${price:.2f}")
        
        # 计算仓位大小
        position_size_quote = self.config.total_amount_quote * self.config.single_position_size_pct
        amount = position_size_quote / price
        
        self.logger().debug(f"   Position size: ${position_size_quote:.2f} = {amount:.6f} {symbol.split('-')[0]}")
        
        # 止损止盈
        stop_loss_pct = Decimal(str(decision.get("stop_loss_pct", 0.02)))
        take_profit_pct = Decimal(str(decision.get("take_profit_pct", 0.04)))
        
        # 创建 Triple Barrier Config
        triple_barrier = self.config.triple_barrier_config.copy()
        triple_barrier.stop_loss = stop_loss_pct
        triple_barrier.take_profit = take_profit_pct
        
        # 确保 price 不是 None（双重检查）
        if price is None:
            self.logger().error(f"❌ CRITICAL: price became None after validation! Symbol: {symbol}")
            return None
        
        # ⚠️  重要：在回测中必须提供 entry_price，使用当前市价
        executor_config = PositionExecutorConfig(
            timestamp=self.market_data_provider.time(),
            connector_name=self.config.connector_name,
            trading_pair=symbol,
            side=trade_type,
            entry_price=price,  # 使用当前市价作为入场价（回测必需）
            amount=amount,
            triple_barrier_config=triple_barrier,
            leverage=self.config.leverage,
        )
        
        self.logger().info(
            f"📈 Creating {trade_type.name} position for {symbol} @ ${price:.2f}, "
            f"Amount: {amount:.4f}, SL: {stop_loss_pct*100:.1f}%, TP: {take_profit_pct*100:.1f}%"
        )
        
        return CreateExecutorAction(
            controller_id=self.config.id,
            executor_config=executor_config
        )
    
    def _create_close_action(self, decision: Dict) -> Optional[StopExecutorAction]:
        """创建平仓 Action"""
        executor_id = decision.get("executor_id")
        symbol = decision.get("symbol")
        
        # 查找对应的 Executor
        target_executor = None
        
        if executor_id:
            # 通过 ID 查找
            for executor in self.executors_info:
                if executor.id == executor_id and executor.is_active:
                    target_executor = executor
                    break
        else:
            # 通过 Symbol 查找
            for executor in self.executors_info:
                if executor.config.trading_pair == symbol and executor.is_active:
                    target_executor = executor
                    break
        
        if not target_executor:
            self.logger().warning(f"Cannot find active executor for {symbol}")
            return None
        
        self.logger().info(f"📉 Closing position for {symbol}, Executor ID: {target_executor.id}")
        
        return StopExecutorAction(
            controller_id=self.config.id,
            executor_id=target_executor.id
        )
    
    def to_format_status(self) -> List[str]:
        """格式化状态显示"""
        lines = []
        
        lines.append(f"🤖 AI Agent V1 Status")
        lines.append(f"=" * 50)
        
        # 监控币种
        lines.append(f"Monitoring Pairs: {', '.join(self.config.trading_pairs)}")
        
        # 持仓情况
        active_positions = [e for e in self.executors_info if e.is_active and e.is_trading]
        lines.append(f"Active Positions: {len(active_positions)}/{self.config.max_concurrent_positions}")
        
        for executor in active_positions:
            lines.append(
                f"  - {executor.config.trading_pair} {executor.config.side.name}: "
                f"PnL {executor.net_pnl_pct*100:.2f}% (${executor.net_pnl_quote:.2f})"
            )
        
        # 最近决策时间
        if self._last_decision_time > 0:
            time_since_last = self.market_data_provider.time() - self._last_decision_time
            lines.append(f"Last Decision: {int(time_since_last)}s ago")
            next_decision_in = max(0, self.config.decision_interval - time_since_last)
            lines.append(f"Next Decision: in {int(next_decision_in)}s")
        
        # 历史统计（从 executors_info 获取）
        closed_executors = [e for e in self.executors_info if hasattr(e, 'status') and str(e.status) == 'RunnableStatus.TERMINATED']
        if closed_executors:
            total_trades = len(closed_executors)
            winning_trades = sum(1 for e in closed_executors if hasattr(e, 'net_pnl_quote') and float(e.net_pnl_quote) > 0)
            win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
            lines.append(f"Trade History: {total_trades} trades, Win Rate: {win_rate:.1f}%")
        
        return lines

