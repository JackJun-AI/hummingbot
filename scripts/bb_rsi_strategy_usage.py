#!/usr/bin/env python3
"""
使用示例：Bollinger Bands + RSI 策略

这个文件展示了如何配置和运行你的自定义策略
"""

import asyncio
from decimal import Decimal
from bb_rsi_custom_strategy import BBRSIStrategy, BBRSIStrategyConfig


# 策略配置示例
def create_default_config():
    """创建默认配置"""
    config = BBRSIStrategyConfig(
        # === 基础配置 ===
        exchange="binance_perpetual",           # 交易所
        trading_pair="ETH-USDT",               # 交易对
        candles_exchange="binance_perpetual",   # K线数据来源
        candles_pair="ETH-USDT",               # K线交易对
        candles_interval="5m",                 # K线周期
        candles_length=100,                    # K线数量
        
        # === 资金管理 ===
        order_amount_quote=Decimal("50"),      # 每次交易金额(USDT)
        leverage=10,                           # 杠杆倍数
        
        # === Bollinger Bands 参数 ===
        bb_length=20,                          # BB周期
        bb_std=2.0,                           # BB标准差倍数
        bb_threshold=0.2,                     # BB%阈值 (0.2 = 20%区域)
        
        # === RSI 参数 ===
        rsi_length=14,                        # RSI周期
        rsi_low=30,                          # RSI超卖线
        rsi_high=70,                         # RSI超买线
        
        # === 风险管理 ===
        stop_loss=Decimal("0.02"),           # 止损 2%
        take_profit=Decimal("0.03"),         # 止盈 3%
        time_limit=3600,                     # 时间限制 1小时
    )
    return config


def create_conservative_config():
    """创建保守配置"""
    config = BBRSIStrategyConfig(
        exchange="binance_perpetual",
        trading_pair="BTC-USDT",
        order_amount_quote=Decimal("100"),
        leverage=5,                           # 低杠杆
        
        bb_length=25,                        # 更长周期
        bb_std=2.5,                         # 更大标准差
        bb_threshold=0.15,                  # 更严格的阈值
        
        rsi_length=21,                      # 更长周期
        rsi_low=25,                        # 更严格的超卖
        rsi_high=75,                       # 更严格的超买
        
        stop_loss=Decimal("0.015"),        # 更小止损
        take_profit=Decimal("0.025"),      # 更小止盈
        time_limit=7200,                   # 更长持仓时间
    )
    return config


def create_aggressive_config():
    """创建激进配置"""
    config = BBRSIStrategyConfig(
        exchange="binance_perpetual",
        trading_pair="SOL-USDT",
        order_amount_quote=Decimal("200"),
        leverage=20,                        # 高杠杆
        
        bb_length=15,                      # 短周期
        bb_std=1.8,                       # 小标准差
        bb_threshold=0.25,                # 宽松阈值
        
        rsi_length=10,                    # 短周期
        rsi_low=35,                      # 宽松超卖
        rsi_high=65,                     # 宽松超买
        
        stop_loss=Decimal("0.03"),       # 大止损
        take_profit=Decimal("0.05"),     # 大止盈
        time_limit=1800,                 # 短持仓时间
    )
    return config


# 使用方法示例
if __name__ == "__main__":
    # 选择配置
    config = create_default_config()
    
    print("🚀 Bollinger Bands + RSI 策略配置")
    print("=" * 50)
    print(f"交易对: {config.trading_pair}")
    print(f"交易金额: {config.order_amount_quote} USDT")
    print(f"杠杆: {config.leverage}x")
    print(f"BB参数: 周期={config.bb_length}, 标准差={config.bb_std}")
    print(f"RSI参数: 周期={config.rsi_length}, 区间=[{config.rsi_low}, {config.rsi_high}]")
    print(f"风险管理: 止损={config.stop_loss}, 止盈={config.take_profit}")
    print("=" * 50)
    
    # 在实际使用中，这里会初始化连接器并启动策略
    # strategy = BBRSIStrategy(connectors, config)
    # strategy.start()
