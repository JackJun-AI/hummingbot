"""
Bollinger Bands + RSI Controller - Pine Script 风格的简洁实现
类似 TradingView 的简单直观写法
"""

from typing import List
import pandas_ta as ta
from pydantic import Field, field_validator
from pydantic_core.core_schema import ValidationInfo

from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy_v2.controllers.directional_trading_controller_base import (
    DirectionalTradingControllerBase,
    DirectionalTradingControllerConfigBase,
)


class BBRSISimpleConfig(DirectionalTradingControllerConfigBase):
    """配置类 - 类似 Pine Script 的输入参数"""
    controller_name: str = "bb_rsi_simple"
    candles_config: List[CandlesConfig] = []
    
    # === Pine Script 风格的输入参数 ===
    candles_connector: str = Field(default=None)
    candles_trading_pair: str = Field(default=None) 
    interval: str = Field(default="5m")
    
    # Bollinger Bands 设置
    bb_length: int = Field(default=20, description="Bollinger Bands Length")
    bb_std: float = Field(default=2.0, description="Bollinger Bands StdDev")
    bb_threshold: float = Field(default=0.2, description="BB Entry Zone (0.2 = 20%)")
    
    # RSI 设置  
    rsi_length: int = Field(default=14, description="RSI Length")
    rsi_oversold: float = Field(default=30, description="RSI Oversold Level")
    rsi_overbought: float = Field(default=70, description="RSI Overbought Level")

    @field_validator("candles_connector", mode="before")
    @classmethod
    def set_candles_connector(cls, v, validation_info: ValidationInfo):
        return v or validation_info.data.get("connector_name")

    @field_validator("candles_trading_pair", mode="before") 
    @classmethod
    def set_candles_trading_pair(cls, v, validation_info: ValidationInfo):
        return v or validation_info.data.get("trading_pair")


class BBRSISimpleController(DirectionalTradingControllerBase):
    """
    🎯 Bollinger Bands + RSI Controller
    
    Pine Script 风格的交易逻辑:
    - 做多: 价格在下轨区域 AND RSI超卖
    - 做空: 价格在上轨区域 AND RSI超买
    """

    def __init__(self, config: BBRSISimpleConfig, *args, **kwargs):
        self.config = config
        self.max_records = max(config.bb_length, config.rsi_length) + 20
        
        # 自动配置 K线数据
        if len(self.config.candles_config) == 0:
            self.config.candles_config = [CandlesConfig(
                connector=config.candles_connector,
                trading_pair=config.candles_trading_pair,
                interval=config.interval,
                max_records=self.max_records
            )]
        super().__init__(config, *args, **kwargs)

    async def update_processed_data(self):
        """
        核心策略逻辑 - 类似 Pine Script 的简洁写法
        """
        # === 获取数据 ===
        df = self.market_data_provider.get_candles_df(
            connector_name=self.config.candles_connector,
            trading_pair=self.config.candles_trading_pair,
            interval=self.config.interval,
            max_records=self.max_records
        )
        
        # === 计算指标 (Pine Script 风格) ===
        # Bollinger Bands
        df.ta.bbands(length=self.config.bb_length, std=self.config.bb_std, append=True)
        # RSI
        df.ta.rsi(length=self.config.rsi_length, append=True)
        
        # === 获取指标值 ===
        bb_percent = df[f"BBP_{self.config.bb_length}_{self.config.bb_std}"]  # BB%位置 (0-1)
        rsi = df[f"RSI_{self.config.rsi_length}"]                            # RSI值
        
        # === 交易条件 (Pine Script 风格的条件判断) ===
        # 做多条件: 价格在下轨区域 AND RSI超卖
        long_condition = (
            (bb_percent <= self.config.bb_threshold) &           # 价格接近下轨
            (rsi <= self.config.rsi_oversold)                   # RSI超卖
        )
        
        # 做空条件: 价格在上轨区域 AND RSI超买
        short_condition = (
            (bb_percent >= (1 - self.config.bb_threshold)) &    # 价格接近上轨
            (rsi >= self.config.rsi_overbought)                # RSI超买
        )
        
        # === 生成信号 ===
        df["signal"] = 0                                       # 默认无信号
        df.loc[long_condition, "signal"] = 1                  # 做多信号
        df.loc[short_condition, "signal"] = -1                # 做空信号
        
        # === 输出结果 ===
        self.processed_data["signal"] = df["signal"].iloc[-1]
        self.processed_data["features"] = df
        
        # === 调试信息 (可选) ===
        latest = df.iloc[-1]
        self.processed_data["debug_info"] = {
            "bb_percent": latest[f"BBP_{self.config.bb_length}_{self.config.bb_std}"],
            "rsi": latest[f"RSI_{self.config.rsi_length}"],
            "signal": latest["signal"],
            "long_ok": latest[f"BBP_{self.config.bb_length}_{self.config.bb_std}"] <= self.config.bb_threshold,
            "rsi_oversold": latest[f"RSI_{self.config.rsi_length}"] <= self.config.rsi_oversold
        }
