"""
使用 Controller 的超简化策略实现
类似 Pine Script 的简洁风格 - 仅需30行核心代码！
"""

import os
from decimal import Decimal
from typing import Dict

from hummingbot.connector.connector_base import ConnectorBase  
from hummingbot.strategy.strategy_v2_base import StrategyV2Base, StrategyV2ConfigBase


# === 策略配置 (类似 Pine Script 的设置区域) ===
class SimpleControllerStrategyConfig(StrategyV2ConfigBase):
    script_file_name: str = os.path.basename(__file__)
    
    # 控制器配置文件路径
    controllers_config: list = ["conf/controllers/bb_rsi_simple.yml"]
    
    # 交易设置
    markets: Dict[str, list] = {"binance_perpetual": ["ETH-USDT"]}


class SimpleControllerStrategy(StrategyV2Base):
    """
    🎯 基于 Controller 的极简策略
    
    核心思想: Strategy 只负责资金管理和执行
              Controller 负责信号生成
    """
    
    def __init__(self, connectors: Dict[str, ConnectorBase], config: SimpleControllerStrategyConfig):
        super().__init__(connectors, config)
        self.config = config

    def format_status(self) -> str:
        """显示策略状态"""
        if not self.ready_to_trade:
            return "Market connectors are not ready."
        
        lines = []
        
        # 显示余额
        balance_df = self.get_balance_df()
        lines.extend(["", "💰 Balances:"] + 
                    ["    " + line for line in balance_df.to_string(index=False).split("\n")])
        
        # 显示 Controller 状态
        if self.controllers:
            for controller_id, controller in self.controllers.items():
                if hasattr(controller, 'processed_data') and 'debug_info' in controller.processed_data:
                    debug = controller.processed_data['debug_info']
                    signal = controller.processed_data.get('signal', 0)
                    
                    signal_emoji = "🟢 BUY" if signal == 1 else "🔴 SELL" if signal == -1 else "⚪ HOLD"
                    
                    lines.extend([
                        "",
                        f"📊 {controller_id} Status:",
                        f"    BB%: {debug.get('bb_percent', 0):.3f}",
                        f"    RSI: {debug.get('rsi', 0):.1f}",
                        f"    Signal: {signal_emoji}"
                    ])
        
        # 显示活跃订单
        try:
            orders_df = self.active_orders_df()
            lines.extend(["", "📋 Active Orders:"] + 
                        ["    " + line for line in orders_df.to_string(index=False).split("\n")])
        except ValueError:
            lines.extend(["", "📋 No active orders."])
        
        return "\n".join(lines)


# === 使用示例 ===
"""
1. 创建 Controller 配置文件: conf/controllers/bb_rsi_simple.yml
2. 在 Hummingbot 中运行: start --script controller_strategy_example.py
3. 享受 Pine Script 级别的简洁代码！

配置文件内容 (bb_rsi_simple.yml):
---
controller_name: bb_rsi_simple
controller_type: directional_trading
connector_name: binance_perpetual
trading_pair: ETH-USDT
interval: 5m
bb_length: 20
bb_std: 2.0
bb_threshold: 0.2
rsi_length: 14
rsi_oversold: 30
rsi_overbought: 70
order_amount_quote: 50
"""
