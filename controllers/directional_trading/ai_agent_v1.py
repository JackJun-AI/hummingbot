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
from pydantic import Field

from hummingbot.core.data_type.common import TradeType
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
        if len(self.config.candles_config) == 0:
            self.config.candles_config = [
                CandlesConfig(
                    connector=config.connector_name,
                    trading_pair=pair,
                    interval=config.candles_interval,
                    max_records=config.candles_max_records
                ) for pair in config.trading_pairs
            ]
        
        super().__init__(config, *args, **kwargs)
        
        # 决策时间追踪
        self._last_decision_time = 0
        self._decision_in_progress = False
        
        # 历史交易记录（简化版，实际应该用数据库）
        self._trade_history: List[Dict] = []
        
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
                temperature=0.7,
                max_tokens=2000,
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
        每个 tick 都会调用，但只在达到决策间隔时才真正执行
        """
        current_time = self.market_data_provider.time()
        
        # 检查是否到达决策时间
        time_since_last = current_time - self._last_decision_time
        if time_since_last < self.config.decision_interval:
            return
        
        # 防止并发决策
        if self._decision_in_progress:
            self.logger().debug("Decision already in progress, skipping")
            return
        
        self._decision_in_progress = True
        
        try:
            self.logger().info("=" * 80)
            self.logger().info(f"🤖 Starting AI decision cycle (interval: {time_since_last}s)")
            self.logger().info("=" * 80)
            
            # Step 1: 收集交易上下文
            self.logger().info("📊 Step 1: Building trading context...")
            context = await self._build_trading_context()
            self.logger().info(f"   ✅ Context built - {len(context['market_data'])} pairs, "
                             f"{len(context['positions'])} positions")
            
            # Step 2: 调用 LLM 获取决策
            self.logger().info("🧠 Step 2: Calling LLM for decisions...")
            decisions = await self._get_ai_decisions(context)
            self.logger().info(f"   ✅ LLM returned {len(decisions)} decisions")
            
            # 打印决策详情
            if decisions:
                for i, dec in enumerate(decisions, 1):
                    self.logger().info(
                        f"   Decision {i}: {dec.get('action')} {dec.get('symbol')} "
                        f"(confidence: {dec.get('confidence', 0)}%)"
                    )
            else:
                self.logger().warning("   ⚠️  No decisions generated by LLM")
            
            # Step 3: 存储决策到 processed_data（供 determine_executor_actions 使用）
            self.processed_data["ai_decisions"] = decisions
            self.processed_data["context"] = context
            
            self._last_decision_time = current_time
            
            self.logger().info("=" * 80)
            self.logger().info(f"✅ Decision cycle completed - {len(decisions)} actions will be processed")
            self.logger().info("=" * 80)
            
        except Exception as e:
            self.logger().error(f"❌ Error in AI decision cycle: {e}", exc_info=True)
        finally:
            self._decision_in_progress = False
    
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
        for pair in self.config.trading_pairs:
            try:
                market_info = await self._get_market_info(pair)
                context["market_data"][pair] = market_info
                
                # 获取资金费率（仅 Perpetual）
                if "_perpetual" in self.config.connector_name:
                    funding_rate = await self._get_funding_rate(pair)
                    context["funding_rates"][pair] = funding_rate
                    
            except Exception as e:
                self.logger().warning(f"Failed to get market info for {pair}: {e}")
        
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
        """获取最近的交易记录"""
        # 简化版：从内存获取
        # 实际应该从数据库获取
        return self._trade_history[-limit:] if self._trade_history else []
    
    async def _get_market_info(self, trading_pair: str) -> Dict:
        """获取单个币种的市场信息"""
        try:
            # 获取K线数据
            candles = self.market_data_provider.get_candles_df(
                connector_name=self.config.connector_name,
                trading_pair=trading_pair,
                interval=self.config.candles_interval,
                max_records=self.config.candles_max_records
            )
            
            if candles.empty or len(candles) < 20:
                self.logger().warning(f"Insufficient candles for {trading_pair}")
                return {"error": "insufficient_data"}
            
            # 计算技术指标
            close = candles["close"]
            high = candles["high"]
            low = candles["low"]
            
            rsi = ta.rsi(close, length=14)
            macd = ta.macd(close, fast=12, slow=26, signal=9)
            ema_20 = ta.ema(close, length=20)
            
            current_price = float(close.iloc[-1])
            
            return {
                "symbol": trading_pair,
                "current_price": current_price,
                "rsi": float(rsi.iloc[-1]) if not rsi.isna().iloc[-1] else None,
                "macd": float(macd[f"MACD_12_26_9"].iloc[-1]) if not macd.empty else None,
                "macd_signal": float(macd[f"MACDs_12_26_9"].iloc[-1]) if not macd.empty else None,
                "ema_20": float(ema_20.iloc[-1]) if not ema_20.isna().iloc[-1] else None,
                "price_change_24h_pct": self._calculate_price_change(candles),
                "volume_24h": float(candles["volume"].sum()),
            }
            
        except Exception as e:
            self.logger().error(f"Error getting market info for {trading_pair}: {e}")
            return {"error": str(e)}
    
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
            
            self.logger().debug(f"System prompt length: {len(system_prompt)} chars")
            self.logger().debug(f"User prompt length: {len(user_prompt)} chars")
            
            # Step 2: 使用 LangChain 调用 LLM
            self.logger().info("Calling LLM API...")
            response = await self._call_langchain_llm(system_prompt, user_prompt)
            self.logger().info("LLM response received")
            
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
        return f"""You are an expert cryptocurrency trading AI agent.

**Trading Rules:**
- You can trade these pairs: {', '.join(self.config.trading_pairs)}
- Maximum concurrent positions: {self.config.max_concurrent_positions}
- Single position size: {float(self.config.single_position_size_pct) * 100}% of total capital
- Always use stop loss and take profit
- Risk/Reward ratio must be at least 1:2

**Available Actions:**
1. "open_long" - Open a long position
2. "open_short" - Open a short position
3. "close_position" - Close an existing position
4. "hold" - Take no action

**Output Format:**
You must respond with a JSON array in this exact format:
```json
[
  {{
    "action": "open_long",
    "symbol": "BTC-USDT",
    "reasoning": "Strong uptrend with RSI oversold",
    "stop_loss_pct": 0.02,
    "take_profit_pct": 0.04,
    "confidence": 75
  }},
  {{
    "action": "close_position",
    "symbol": "ETH-USDT",
    "executor_id": "abc123",
    "reasoning": "Take profit target reached"
  }}
]
```

**Important:**
- Only output valid JSON, no other text
- If no action is needed, return an empty array []
- Consider funding rates for perpetual contracts
- Learn from historical trades to avoid repeated mistakes
"""
    
    def _build_user_prompt(self, context: Dict) -> str:
        """构建用户 Prompt（包含实时数据）"""
        import json
        
        prompt_parts = []
        
        # 1. 账户信息
        prompt_parts.append(f"## Account Status")
        prompt_parts.append(f"Total Capital: ${context['account']['total_amount_quote']:.2f}")
        prompt_parts.append(f"Active Positions: {len(context['positions'])}/{self.config.max_concurrent_positions}")
        
        # 2. 当前持仓
        if context["positions"]:
            prompt_parts.append(f"\n## Current Positions")
            for pos in context["positions"]:
                prompt_parts.append(
                    f"- {pos['symbol']} {pos['side']}: Entry ${pos['entry_price']:.2f}, "
                    f"PnL: {pos['net_pnl_pct']*100:.2f}% (${pos['net_pnl_quote']:.2f}), "
                    f"ID: {pos['executor_id']}"
                )
        else:
            prompt_parts.append(f"\n## Current Positions: None")
        
        # 3. 市场数据
        prompt_parts.append(f"\n## Market Data")
        for symbol, data in context["market_data"].items():
            if "error" in data:
                continue
            
            funding_info = context["funding_rates"].get(symbol, {})
            funding_rate = funding_info.get("rate", 0.0)
            
            prompt_parts.append(
                f"\n### {symbol}\n"
                f"- Price: ${data['current_price']:.2f}\n"
                f"- 24h Change: {data['price_change_24h_pct']:.2f}%\n"
                f"- RSI: {data['rsi']:.1f if data['rsi'] else 'N/A'}\n"
                f"- MACD: {data['macd']:.2f if data['macd'] else 'N/A'}\n"
                f"- EMA(20): ${data['ema_20']:.2f if data['ema_20'] else 'N/A'}\n"
                f"- Funding Rate: {funding_rate*100:.4f}% (8h)" if funding_rate else ""
            )
        
        # 4. 历史交易记录
        if context["recent_trades"]:
            prompt_parts.append(f"\n## Recent Trades (Last {len(context['recent_trades'])})")
            for trade in context["recent_trades"]:
                prompt_parts.append(
                    f"- {trade['symbol']} {trade['side']}: "
                    f"Entry ${trade['entry_price']:.2f} → Exit ${trade['exit_price']:.2f}, "
                    f"PnL: {trade['pnl_pct']*100:.2f}%"
                )
        
        prompt_parts.append(f"\n## Your Decision:")
        prompt_parts.append(f"Based on the above information, what trades should we make?")
        
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
            action = decision.get("action")
            symbol = decision.get("symbol")
            
            self.logger().debug(f"Validating decision {i}: {action} {symbol}")
            
            # 基本字段检查
            if not action or not symbol:
                self.logger().warning(f"❌ Decision {i}: missing action or symbol - {decision}")
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
                
                self.logger().debug(f"   SL: {stop_loss_pct*100:.1f}%, TP: {take_profit_pct*100:.1f}%")
                
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
    
    def determine_executor_actions(self) -> List[ExecutorAction]:
        """
        根据 AI 决策生成 Executor Actions
        """
        # 获取 AI 决策
        ai_decisions = self.processed_data.get("ai_decisions", [])
        
        self.logger().info(f"🎯 Processing {len(ai_decisions)} AI decisions into executor actions...")
        
        if not ai_decisions:
            self.logger().warning("No AI decisions available to process")
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
        return actions
    
    def _create_open_action(self, decision: Dict, trade_type: TradeType) -> Optional[CreateExecutorAction]:
        """创建开仓 Action"""
        symbol = decision["symbol"]
        
        # 获取当前价格
        price = self.market_data_provider.get_price_by_type(
            self.config.connector_name,
            symbol,
            price_type=self.get_price_type(trade_type)
        )
        
        if price is None:
            self.logger().warning(f"Cannot get price for {symbol}")
            return None
        
        # 计算仓位大小
        position_size_quote = self.config.total_amount_quote * self.config.single_position_size_pct
        amount = position_size_quote / price
        
        # 止损止盈
        stop_loss_pct = Decimal(str(decision.get("stop_loss_pct", 0.02)))
        take_profit_pct = Decimal(str(decision.get("take_profit_pct", 0.04)))
        
        # 创建 Triple Barrier Config
        triple_barrier = self.config.triple_barrier_config.copy()
        triple_barrier.stop_loss = stop_loss_pct
        triple_barrier.take_profit = take_profit_pct
        
        executor_config = PositionExecutorConfig(
            timestamp=self.market_data_provider.time(),
            connector_name=self.config.connector_name,
            trading_pair=symbol,
            side=trade_type,
            entry_price=price,
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
        
        # 记录交易历史
        self._record_trade(target_executor)
        
        return StopExecutorAction(
            controller_id=self.config.id,
            executor_id=target_executor.id
        )
    
    def _record_trade(self, executor):
        """记录交易到历史"""
        try:
            trade_record = {
                "symbol": executor.config.trading_pair,
                "side": executor.config.side.name,
                "entry_price": float(executor.config.entry_price),
                "exit_price": float(self.market_data_provider.get_price_by_type(
                    self.config.connector_name,
                    executor.config.trading_pair,
                    price_type=self.get_price_type(executor.config.side)
                )),
                "pnl_pct": float(executor.net_pnl_pct),
                "pnl_quote": float(executor.net_pnl_quote),
                "timestamp": executor.timestamp,
                "close_timestamp": time.time(),
            }
            self._trade_history.append(trade_record)
            
            # 限制历史记录数量
            if len(self._trade_history) > 50:
                self._trade_history = self._trade_history[-50:]
                
        except Exception as e:
            self.logger().error(f"Failed to record trade: {e}")
    
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
        
        # 历史统计
        if self._trade_history:
            total_trades = len(self._trade_history)
            winning_trades = sum(1 for t in self._trade_history if t["pnl_quote"] > 0)
            win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
            lines.append(f"Trade History: {total_trades} trades, Win Rate: {win_rate:.1f}%")
        
        return lines

