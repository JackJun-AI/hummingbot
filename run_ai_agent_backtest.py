#!/usr/bin/env python3
"""
AI Agent V1 正式回测脚本

使用方法:
    python run_ai_agent_backtest.py --config conf/controllers/ai_agent_backtest.yml --start 2024-11-18 --end 2024-11-25

功能:
    - 使用真实的历史数据进行回测
    - 调用真实的 OpenRouter LLM API
    - 完整的回测报告和性能分析
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add hummingbot to path
sys.path.insert(0, str(Path(__file__).parent))

from hummingbot.strategy_v2.backtesting.backtesting_engine_base import BacktestingEngineBase


async def run_backtest(config_path: str, start_date: str, end_date: str, 
                       backtesting_resolution: str = "5m", trade_cost: float = 0.001):
    """
    运行正式回测
    
    参数:
        config_path: 配置文件路径 (相对于 conf/controllers/)
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
        backtesting_resolution: 回测分辨率 (1m, 5m, 15m, 1h)
        trade_cost: 交易成本 (默认 0.1%)
    """
    
    print("=" * 80)
    print("🚀 AI Agent V1 正式回测")
    print("=" * 80)
    
    # 解析时间
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        start_timestamp = int(start_dt.timestamp())
        end_timestamp = int(end_dt.timestamp())
        
        days = (end_dt - start_dt).days
        
        print(f"\n📅 回测时间范围:")
        print(f"   开始时间: {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   结束时间: {end_dt.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   回测天数: {days} 天")
        print(f"   回测分辨率: {backtesting_resolution}")
        print(f"   交易成本: {trade_cost * 100}%")
        
    except ValueError as e:
        print(f"❌ 日期格式错误: {e}")
        print(f"   请使用 YYYY-MM-DD 格式，例如: 2024-11-18")
        return None
    
    # 检查配置文件
    if not os.path.exists(config_path):
        print(f"\n❌ 配置文件不存在: {config_path}")
        print(f"   请确认路径是否正确")
        return None
    
    print(f"\n⚙️  加载配置文件:")
    print(f"   配置路径: {config_path}")
    
    try:
        # 创建回测引擎
        engine = BacktestingEngineBase()
        
        # 加载控制器配置
        print(f"\n🔧 初始化回测引擎...")
        config_filename = os.path.basename(config_path)
        controller_config = engine.get_controller_config_instance_from_yml(config_filename)
        
        print(f"   控制器: {controller_config.controller_name}")
        print(f"   交易所: {controller_config.connector_name}")
        print(f"   交易对: {controller_config.trading_pairs if hasattr(controller_config, 'trading_pairs') else [controller_config.trading_pair]}")
        print(f"   初始资金: ${controller_config.total_amount_quote}")
        
        # 检查 API Key
        if hasattr(controller_config, 'openrouter_api_key'):
            api_key = controller_config.openrouter_api_key
            if not api_key or api_key.startswith("sk-or-v1-"):
                print(f"\n⚠️  OpenRouter API Key 配置检查:")
                if not api_key:
                    print(f"   ❌ API Key 未配置")
                    print(f"   请在配置文件中设置 openrouter_api_key")
                    print(f"   获取 API Key: https://openrouter.ai")
                    return None
                else:
                    print(f"   ✅ API Key 已配置: {api_key[:20]}...")
        
        # 运行回测
        print(f"\n📊 开始回测...")
        print(f"   (这可能需要几分钟，取决于时间范围和数据量)")
        
        results = await engine.run_backtesting(
            controller_config=controller_config,
            start=start_timestamp,
            end=end_timestamp,
            backtesting_resolution=backtesting_resolution,
            trade_cost=trade_cost
        )
        
        print(f"\n✅ 回测完成!")
        
        return results
        
    except Exception as e:
        print(f"\n❌ 回测失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def analyze_results(results, start_date: str, end_date: str):
    """分析回测结果"""
    
    if not results:
        print("\n❌ 没有回测结果可分析")
        return
    
    print("\n" + "=" * 80)
    print("📊 回测结果分析")
    print("=" * 80)
    
    executors = results.get("executors", [])
    summary = results.get("results", {})
    
    # 基本统计
    print(f"\n📈 交易统计:")
    print(f"   总交易次数: {len(executors)}")
    
    if not executors:
        print(f"   ⚠️  没有执行任何交易")
        print(f"\n💡 可能的原因:")
        print(f"   1. 决策间隔过长，回测时间太短")
        print(f"   2. AI 决策过于保守，没有触发交易信号")
        print(f"   3. 市场条件不满足开仓条件")
        print(f"   4. LLM API 调用失败或返回空决策")
        return
    
    # 详细交易记录
    print(f"\n📋 交易详情:")
    
    total_pnl_quote = 0
    total_pnl_pct = 0
    winning_trades = 0
    losing_trades = 0
    
    # for i, executor in enumerate(executors, 1):
    #     try:
    #         symbol = executor.config.trading_pair
    #         side = executor.config.side.name
    #         # ⚠️  使用 executor.entry_price (实际成交价) 而不是 config.entry_price
    #         entry_price = float(executor.entry_price) if hasattr(executor, 'entry_price') and executor.entry_price else 0
            
    #         # PnL
    #         pnl_quote = float(executor.net_pnl_quote) if hasattr(executor, 'net_pnl_quote') else 0
    #         pnl_pct = float(executor.net_pnl_pct) if hasattr(executor, 'net_pnl_pct') else 0
            
    #         # 状态
    #         status = executor.status if hasattr(executor, 'status') else "UNKNOWN"
            
    #         # 累计统计
    #         total_pnl_quote += pnl_quote
    #         total_pnl_pct += pnl_pct
            
    #         if pnl_quote > 0:
    #             winning_trades += 1
    #             result_emoji = "✅"
    #             result_text = "盈利"
    #         elif pnl_quote < 0:
    #             losing_trades += 1
    #             result_emoji = "❌"
    #             result_text = "亏损"
    #         else:
    #             result_emoji = "➖"
    #             result_text = "平手"
            
    #         print(f"\n   交易 #{i}:")
    #         print(f"   币种: {symbol} | 方向: {side} | 状态: {status}")
    #         print(f"   入场价: ${entry_price:.4f}")
    #         print(f"   PnL: ${pnl_quote:.2f} ({pnl_pct*100:.2f}%) {result_emoji} {result_text}")
            
    #         # 如果有更多详细信息
    #         if hasattr(executor, 'close_timestamp'):
    #             duration = executor.close_timestamp - executor.timestamp
    #             print(f"   持仓时长: {duration/3600:.1f} 小时")
                
    #     except Exception as e:
    #         print(f"   ⚠️  交易 #{i} 解析失败: {e}")
    
    # 总体表现
    print(f"\n" + "=" * 60)
    print(f"💰 总体表现")
    print(f"=" * 60)
    
    win_rate = (winning_trades / len(executors)) * 100 if executors else 0
    avg_pnl_pct = (total_pnl_pct / len(executors)) if executors else 0
    
    print(f"\n   总 PnL: ${total_pnl_quote:.2f}")
    print(f"   平均 PnL: {avg_pnl_pct*100:.2f}%")
    print(f"   胜率: {win_rate:.1f}% ({winning_trades}/{len(executors)})")
    print(f"   盈利交易: {winning_trades} 笔")
    print(f"   亏损交易: {losing_trades} 笔")
    
    # 详细统计（如果有）
    if summary:
        print(f"\n📊 详细统计:")
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                if 'pnl' in key.lower() or 'profit' in key.lower():
                    print(f"   {key}: ${value:.2f}" if abs(value) > 1 else f"   {key}: {value:.4f}")
                elif 'rate' in key.lower() or 'ratio' in key.lower():
                    print(f"   {key}: {value:.2%}")
                else:
                    print(f"   {key}: {value:.4f}")
            else:
                print(f"   {key}: {value}")
    
    # 时间分析
    print(f"\n⏱️  时间分析:")
    print(f"   回测时间: {start_date} ~ {end_date}")
    
    if executors:
        first_trade = executors[0]
        last_trade = executors[-1]
        
        if hasattr(first_trade, 'timestamp') and hasattr(last_trade, 'close_timestamp'):
            active_time = last_trade.close_timestamp - first_trade.timestamp
            print(f"   活跃交易时长: {active_time/3600:.1f} 小时 ({active_time/86400:.1f} 天)")
    
    # 风险指标
    print(f"\n⚠️  风险指标:")
    
    if winning_trades > 0 and losing_trades > 0:
        avg_win = sum(float(e.net_pnl_quote) for e in executors if float(e.net_pnl_quote) > 0) / winning_trades
        avg_loss = abs(sum(float(e.net_pnl_quote) for e in executors if float(e.net_pnl_quote) < 0) / losing_trades)
        profit_factor = avg_win / avg_loss if avg_loss > 0 else 0
        
        print(f"   平均盈利: ${avg_win:.2f}")
        print(f"   平均亏损: ${avg_loss:.2f}")
        print(f"   盈亏比: {profit_factor:.2f}")
    
    # 最大回撤
    if executors:
        cumulative_pnl = 0
        max_pnl = 0
        max_drawdown = 0
        
        for executor in executors:
            pnl = float(executor.net_pnl_quote) if hasattr(executor, 'net_pnl_quote') else 0
            cumulative_pnl += pnl
            max_pnl = max(max_pnl, cumulative_pnl)
            drawdown = max_pnl - cumulative_pnl
            max_drawdown = max(max_drawdown, drawdown)
        
        print(f"   最大回撤: ${max_drawdown:.2f}")
        print(f"   最高净值: ${max_pnl:.2f}")


def print_summary(results, config_path: str):
    """打印总结"""
    
    print("\n" + "=" * 80)
    print("📝 回测总结")
    print("=" * 80)
    
    if not results:
        print(f"\n❌ 回测失败")
        return
    
    executors = results.get("executors", [])
    total_pnl = sum(float(e.net_pnl_quote) for e in executors if hasattr(e, 'net_pnl_quote'))
    
    print(f"\n🎯 关键指标:")
    print(f"   ✅ 配置文件: {config_path}")
    print(f"   ✅ 回测完成: 是")
    print(f"   📊 总交易数: {len(executors)}")
    print(f"   💰 总 PnL: ${total_pnl:.2f}")
    
    if total_pnl > 0:
        print(f"   📈 策略表现: 盈利 ✅")
    elif total_pnl < 0:
        print(f"   📉 策略表现: 亏损 ❌")
    else:
        print(f"   ➖ 策略表现: 平手")
    
    print(f"\n💡 建议:")
    
    if len(executors) == 0:
        print(f"   1. 检查决策间隔是否过长")
        print(f"   2. 增加回测时间范围")
        print(f"   3. 检查 LLM API 是否正常")
        print(f"   4. 查看日志了解 AI 决策过程")
    elif total_pnl < 0:
        print(f"   1. 调整风险控制参数 (止损/止盈)")
        print(f"   2. 优化 LLM Prompt 策略")
        print(f"   3. 增加技术指标权重")
        print(f"   4. 考虑不同的市场条件")
    else:
        print(f"   1. 在不同时间段验证策略稳定性")
        print(f"   2. 进行纸上交易测试")
        print(f"   3. 逐步增加资金规模")
        print(f"   4. 持续监控和优化")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Agent V1 正式回测")
    parser.add_argument("--config", type=str, required=True, 
                        help="配置文件路径 (例如: ai_agent_backtest.yml)")
    parser.add_argument("--start", type=str, required=True,
                        help="开始日期 (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, required=True,
                        help="结束日期 (YYYY-MM-DD)")
    parser.add_argument("--resolution", type=str, default="5m",
                        help="回测分辨率 (1m, 5m, 15m, 1h)")
    parser.add_argument("--trade-cost", type=float, default=0.001,
                        help="交易成本 (默认 0.001 = 0.1%%)")
    
    args = parser.parse_args()
    
    # 运行回测
    results = asyncio.run(run_backtest(
        config_path=args.config,
        start_date=args.start,
        end_date=args.end,
        backtesting_resolution=args.resolution,
        trade_cost=args.trade_cost
    ))
    # print(results)
    # 分析结果
    if results:
        analyze_results(results, args.start, args.end)
        print_summary(results, args.config)
    else:
        print(f"\n❌ 回测未能完成，请检查上面的错误信息")


if __name__ == "__main__":
    main()
