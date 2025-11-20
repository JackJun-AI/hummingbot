import os
from decimal import Decimal
from typing import Dict, List, Optional

import pandas_ta as ta
from pydantic import Field, field_validator

from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.clock import Clock
from hummingbot.core.data_type.common import OrderType, PositionMode, PriceType, TradeType
from hummingbot.data_feed.candles_feed.candles_factory import CandlesConfig
from hummingbot.strategy.strategy_v2_base import StrategyV2Base, StrategyV2ConfigBase
from hummingbot.strategy_v2.executors.position_executor.data_types import PositionExecutorConfig, TripleBarrierConfig
from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction, StopExecutorAction


class BBRSIStrategyConfig(StrategyV2ConfigBase):
    """
    配置类：定义策略的所有参数
    """
    script_file_name: str = os.path.basename(__file__)
    markets: Dict[str, List[str]] = {}
    candles_config: List[CandlesConfig] = []
    controllers_config: List[str] = []
    
    # 交易对和交易所配置
    exchange: str = Field(default="binance_perpetual")
    trading_pair: str = Field(default="ETH-USDT")
    candles_exchange: str = Field(default="binance_perpetual") 
    candles_pair: str = Field(default="ETH-USDT")
    candles_interval: str = Field(default="5m")
    candles_length: int = Field(default=100, gt=0)
    
    # 资金管理
    order_amount_quote: Decimal = Field(default=50, gt=0)
    leverage: int = Field(default=10, gt=0)
    position_mode: PositionMode = Field(default="ONEWAY")
    
    # Bollinger Bands 参数
    bb_length: int = Field(default=20, gt=0)
    bb_std: float = Field(default=2.0, gt=0)
    bb_threshold: float = Field(default=0.2, ge=0, le=0.5)  # BB%阈值
    
    # RSI 参数
    rsi_length: int = Field(default=14, gt=0)
    rsi_low: float = Field(default=30, gt=0, lt=50)
    rsi_high: float = Field(default=70, gt=50, lt=100)
    
    # 风险管理
    stop_loss: Decimal = Field(default=Decimal("0.02"), gt=0)
    take_profit: Decimal = Field(default=Decimal("0.03"), gt=0)
    time_limit: int = Field(default=60 * 60, gt=0)  # 1小时

    @property
    def triple_barrier_config(self) -> TripleBarrierConfig:
        return TripleBarrierConfig(
            stop_loss=self.stop_loss,
            take_profit=self.take_profit,
            time_limit=self.time_limit,
            open_order_type=OrderType.MARKET,
            take_profit_order_type=OrderType.LIMIT,
            stop_loss_order_type=OrderType.MARKET,
            time_limit_order_type=OrderType.MARKET
        )

    @field_validator('position_mode', mode="before")
    @classmethod
    def validate_position_mode(cls, v: str) -> PositionMode:
        if v.upper() in PositionMode.__members__:
            return PositionMode[v.upper()]
        raise ValueError(f"Invalid position mode: {v}")


class BBRSIStrategy(StrategyV2Base):
    """
    Bollinger Bands + RSI 组合策略
    
    策略逻辑：
    - 做多条件：价格接近布林带下轨 AND RSI < 30 (超卖)
    - 做空条件：价格接近布林带上轨 AND RSI > 70 (超买)
    - 使用Triple Barrier进行风险管理
    """

    account_config_set = False

    @classmethod
    def init_markets(cls, config: BBRSIStrategyConfig):
        cls.markets = {config.exchange: {config.trading_pair}}

    def __init__(self, connectors: Dict[str, ConnectorBase], config: BBRSIStrategyConfig):
        if len(config.candles_config) == 0:
            config.candles_config.append(CandlesConfig(
                connector=config.candles_exchange,
                trading_pair=config.candles_pair,
                interval=config.candles_interval,
                max_records=config.candles_length + 50
            ))
        super().__init__(connectors, config)
        self.config = config
        self.current_rsi = None
        self.current_bb_percent = None
        self.current_signal = None

    def start(self, clock: Clock, timestamp: float) -> None:
        self._last_timestamp = timestamp
        self.apply_initial_setting()

    def create_actions_proposal(self) -> List[CreateExecutorAction]:
        create_actions = []
        signal = self.get_signal(self.config.candles_exchange, self.config.candles_pair)
        active_longs, active_shorts = self.get_active_executors_by_side(
            self.config.exchange, self.config.trading_pair
        )
        
        if signal is not None:
            mid_price = self.market_data_provider.get_price_by_type(
                self.config.exchange, self.config.trading_pair, PriceType.MidPrice
            )
            
            # 做多信号：布林带下轨附近 + RSI超卖
            if signal == 1 and len(active_longs) == 0:
                create_actions.append(CreateExecutorAction(
                    executor_config=PositionExecutorConfig(
                        timestamp=self.current_timestamp,
                        connector_name=self.config.exchange,
                        trading_pair=self.config.trading_pair,
                        side=TradeType.BUY,
                        entry_price=mid_price,
                        amount=self.config.order_amount_quote / mid_price,
                        triple_barrier_config=self.config.triple_barrier_config,
                        leverage=self.config.leverage
                    )))
            
            # 做空信号：布林带上轨附近 + RSI超买
            elif signal == -1 and len(active_shorts) == 0:
                create_actions.append(CreateExecutorAction(
                    executor_config=PositionExecutorConfig(
                        timestamp=self.current_timestamp,
                        connector_name=self.config.exchange,
                        trading_pair=self.config.trading_pair,
                        side=TradeType.SELL,
                        entry_price=mid_price,
                        amount=self.config.order_amount_quote / mid_price,
                        triple_barrier_config=self.config.triple_barrier_config,
                        leverage=self.config.leverage
                    )))
        
        return create_actions

    def stop_actions_proposal(self) -> List[StopExecutorAction]:
        stop_actions = []
        signal = self.get_signal(self.config.candles_exchange, self.config.candles_pair)
        active_longs, active_shorts = self.get_active_executors_by_side(
            self.config.exchange, self.config.trading_pair
        )
        
        if signal is not None:
            # 反向信号时平仓
            if signal == -1 and len(active_longs) > 0:
                stop_actions.extend([StopExecutorAction(executor_id=e.id) for e in active_longs])
            elif signal == 1 and len(active_shorts) > 0:
                stop_actions.extend([StopExecutorAction(executor_id=e.id) for e in active_shorts])
        
        return stop_actions

    def get_active_executors_by_side(self, connector_name: str, trading_pair: str):
        active_executors_by_connector_pair = self.filter_executors(
            executors=self.get_all_executors(),
            filter_func=lambda e: (e.connector_name == connector_name and 
                                 e.trading_pair == trading_pair and e.is_active)
        )
        active_longs = [e for e in active_executors_by_connector_pair if e.side == TradeType.BUY]
        active_shorts = [e for e in active_executors_by_connector_pair if e.side == TradeType.SELL]
        return active_longs, active_shorts

    def get_signal(self, connector_name: str, trading_pair: str) -> Optional[int]:
        """
        生成交易信号
        Returns:
            1: 做多信号
            -1: 做空信号
            0: 无信号
        """
        candles = self.market_data_provider.get_candles_df(
            connector_name, trading_pair, 
            self.config.candles_interval, 
            self.config.candles_length + 50
        )
        
        if candles.empty or len(candles) < self.config.candles_length:
            return 0
        
        # 计算技术指标
        candles.ta.bbands(length=self.config.bb_length, std=self.config.bb_std, append=True)
        candles.ta.rsi(length=self.config.rsi_length, append=True)
        
        # 获取最新值
        latest = candles.iloc[-1]
        close_price = latest['close']
        bb_lower = latest[f'BBL_{self.config.bb_length}_{self.config.bb_std}']
        bb_upper = latest[f'BBU_{self.config.bb_length}_{self.config.bb_std}']
        bb_middle = latest[f'BBM_{self.config.bb_length}_{self.config.bb_std}']
        rsi = latest[f'RSI_{self.config.rsi_length}']
        
        # 计算Bollinger Bands百分比位置
        bb_percent = (close_price - bb_lower) / (bb_upper - bb_lower)
        
        # 保存当前状态用于显示
        self.current_rsi = rsi
        self.current_bb_percent = bb_percent
        
        # 生成交易信号
        signal = 0
        
        # 做多条件：价格接近下轨 + RSI超卖
        if (bb_percent <= self.config.bb_threshold and 
            rsi <= self.config.rsi_low):
            signal = 1
        
        # 做空条件：价格接近上轨 + RSI超买  
        elif (bb_percent >= (1 - self.config.bb_threshold) and 
              rsi >= self.config.rsi_high):
            signal = -1
        
        self.current_signal = signal
        return signal

    def apply_initial_setting(self):
        if not self.account_config_set:
            for connector_name, connector in self.connectors.items():
                if self.is_perpetual(connector_name):
                    connector.set_position_mode(self.config.position_mode)
                    for trading_pair in self.market_data_provider.get_trading_pairs(connector_name):
                        connector.set_leverage(trading_pair, self.config.leverage)
            self.account_config_set = True

    def format_status(self) -> str:
        if not self.ready_to_trade:
            return "Market connectors are not ready."
        
        lines = []
        
        # 余额信息
        balance_df = self.get_balance_df()
        lines.extend(["", "  Balances:"] + ["    " + line for line in balance_df.to_string(index=False).split("\n")])
        
        # 技术指标状态
        if self.current_rsi is not None and self.current_bb_percent is not None:
            lines.extend([
                "",
                f"  📊 Technical Indicators:",
                f"    RSI: {self.current_rsi:.2f} (Buy ≤ {self.config.rsi_low}, Sell ≥ {self.config.rsi_high})",
                f"    BB%: {self.current_bb_percent:.2f} (Buy ≤ {self.config.bb_threshold}, Sell ≥ {1-self.config.bb_threshold:.2f})",
                f"    Signal: {'🟢 BUY' if self.current_signal == 1 else '🔴 SELL' if self.current_signal == -1 else '⚪ HOLD'}"
            ])
        
        # 活跃订单
        try:
            orders_df = self.active_orders_df()
            lines.extend(["", "  Active Orders:"] + ["    " + line for line in orders_df.to_string(index=False).split("\n")])
        except ValueError:
            lines.extend(["", "  No active orders."])
        
        return "\n".join(lines)
