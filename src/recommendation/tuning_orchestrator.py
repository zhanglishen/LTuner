#!/usr/bin/env python3
"""
调优编排器
整合监控、推荐、审批和应用的完整流程
支持 DBA 触发式调优和自动化调优
"""
import sys
sys.path.insert(0, '/root/GPTuner/src')

from typing import Dict, Optional
from datetime import datetime
import json

from monitoring.adaptive_monitor import AdaptiveMonitor
from recommendation.parameter_recommender import ParameterRecommender
from recommendation.config_manager import ConfigManager

# 导入规则引擎
try:
    from rule_engine.safety_engine import SafetyEngine
    SAFETY_ENGINE_AVAILABLE = True
except ImportError:
    SAFETY_ENGINE_AVAILABLE = False
    print("警告：规则引擎未找到，将跳过安全检查")


class TuningOrchestrator:
    """调优编排器 - 完整调优流程管理"""
    
    def __init__(self, dbms, use_rag=True, enable_safety_check=True):
        """
        初始化编排器
        
        Args:
            dbms: PgDBMS 实例
            use_rag: 是否使用 RAG 增强
            enable_safety_check: 是否启用安全检查
        """
        self.dbms = dbms
        self.enable_safety_check = enable_safety_check and SAFETY_ENGINE_AVAILABLE
        
        # 初始化各模块
        self.monitor = AdaptiveMonitor(dbms, use_rag=use_rag)
        self.recommender = ParameterRecommender(dbms, use_rag=use_rag)
        self.config_manager = ConfigManager(dbms)
        
        # 初始化安全引擎
        self.safety_engine = None
        if self.enable_safety_check:
            try:
                self.safety_engine = SafetyEngine(dbms)
            except Exception as e:
                print(f"警告：安全引擎初始化失败: {e}")
                self.enable_safety_check = False
        
        print("✅ 调优编排器已初始化")
        print(f"   监控模块: AdaptiveMonitor")
        print(f"   推荐模块: ParameterRecommender")
        print(f"   配置管理: ConfigManager")
        print(f"   安全检查: {'已启用' if self.enable_safety_check else '未启用'}")
        print(f"   RAG 增强: {'已启用' if use_rag else '未启用'}\n")
    
    def analyze_and_recommend(self, window_minutes: int = 120) -> Dict:
        """
        步骤 1: 分析当前状态并生成推荐
        
        Args:
            window_minutes: 监控时间窗口（分钟）
            
        Returns:
            完整分析和推荐报告
        """
        print(f"\n{'#'*60}")
        print("# 步骤 1: 工作负载分析与参数推荐")
        print(f"{'#'*60}\n")
        
        # 1. 采集性能指标并分析
        print("=" * 60)
        print("1.1 采集性能指标")
        print("=" * 60)
        monitor_report = self.monitor.collect_and_analyze(window_minutes)
        
        # 2. 生成参数推荐
        print("\n" + "=" * 60)
        print("1.2 生成参数推荐")
        print("=" * 60)
        recommendations = self.recommender.generate_recommendations(
            scenario=monitor_report['summary']['scenario'],
            metrics=monitor_report['metrics'],
            workload_analysis=monitor_report['workload_analysis']
        )
        
        # 3. 整合报告
        full_report = {
            'timestamp': datetime.now().isoformat(),
            'step': 'analyze_and_recommend',
            'monitor_report': monitor_report,
            'recommendations': recommendations,
            'summary': {
                'scenario': monitor_report['summary']['scenario'],
                'workload_type': monitor_report['workload_analysis']['summary']['workload_type'],
                'total_recommendations': recommendations['total_recommendations'],
                'high_priority_count': recommendations['summary']['high_priority_count'],
                'has_pressure': monitor_report['workload_analysis']['summary']['has_pressure']
            }
        }
        
        print(f"\n{'#'*60}")
        print("# 分析完成")
        print(f"{'#'*60}\n")
        
        return full_report
    
    def safety_check(self, recommendations: Dict) -> Dict:
        """
        步骤 2.5: 安全检查（可选）
        
        Args:
            recommendations: 推荐配置
            
        Returns:
            安全检查报告
        """
        if not self.enable_safety_check or not self.safety_engine:
            print(f"\n{'#'*60}")
            print("# 跳过安全检查（未启用）")
            print(f"{'#'*60}\n")
            return {'skipped': True, 'safe_to_apply': True}
        
        print(f"\n{'#'*60}")
        print("# 步骤 2.5: 安全检查")
        print(f"{'#'*60}\n")
        
        safety_report = self.safety_engine.run_safety_check(
            recommendations['recommendations']
        )
        
        print(f"{'#'*60}\n")
        
        return safety_report
    
    def preview_changes(self, recommendations: Dict) -> Dict:
        """
        步骤 2: 预览配置变更（模拟应用）
        
        Args:
            recommendations: 推荐配置
            
        Returns:
            预览结果
        """
        print(f"\n{'#'*60}")
        print("# 步骤 2: 预览配置变更")
        print(f"{'#'*60}\n")
        
        # 模拟应用（dry_run=True）
        preview_result = self.config_manager.apply_config_changes(
            recommendations['recommendations'],
            dry_run=True
        )
        
        preview_report = {
            'timestamp': datetime.now().isoformat(),
            'step': 'preview_changes',
            'preview_result': preview_result,
            'changes_count': {
                'successful': len(preview_result['successful']),
                'failed': len(preview_result['failed']),
                'skipped': len(preview_result['skipped'])
            }
        }
        
        print(f"{'#'*60}\n")
        
        return preview_report
    
    def request_approval(self, full_report: Dict) -> Dict:
        """
        步骤 3: 请求 DBA 审批
        
        Args:
            full_report: 完整分析报告
            
        Returns:
            审批信息
        """
        print(f"\n{'#'*60}")
        print("# 步骤 3: 等待 DBA 审批")
        print(f"{'#'*60}\n")
        
        # 打印审批摘要
        summary = full_report['summary']
        recommendations = full_report['recommendations']
        
        print("📋 审批摘要:\n")
        print(f"  场景类型: {summary['scenario']}")
        print(f"  推荐参数数: {summary['total_recommendations']}")
        print(f"  高优先级: {summary['high_priority_count']} 个")
        print(f"  资源压力: {'是' if summary['has_pressure'] else '否'}\n")
        
        # 打印高优先级推荐
        high_priority = [
            (knob, rec) for knob, rec in recommendations['recommendations'].items()
            if rec['priority'] == 'high'
        ]
        
        if high_priority:
            print("🔴 高优先级变更:\n")
            for knob, rec in high_priority[:5]:
                current = recommendations['current_config'].get(knob, 'unknown')
                print(f"  • {knob}")
                print(f"    当前: {current}")
                print(f"    推荐: {rec['recommended_value']}")
                print(f"    原因: {rec['reason']}\n")
        
        # 模拟交互式审批
        print("=" * 60)
        print("请审批此次调优方案:")
        print("  1. 批准 (approve)")
        print("  2. 拒绝 (reject)")
        print("  3. 修改 (modify)")
        print("=" * 60)
        
        # 在实际应用中，这里应该等待 DBA 输入
        # 现在返回一个模拟的审批结果
        approval_result = {
            'timestamp': datetime.now().isoformat(),
            'step': 'request_approval',
            'status': 'pending',  # 'approved', 'rejected', 'modified'
            'message': '等待 DBA 审批决策'
        }
        
        print(f"\n状态: {approval_result['status']}\n")
        print(f"{'#'*60}\n")
        
        return approval_result
    
    def apply_with_backup(self, recommendations: Dict, 
                         approval: Optional[Dict] = None) -> Dict:
        """
        步骤 4: 备份并应用配置
        
        Args:
            recommendations: 推荐配置
            approval: 审批信息（可选）
            
        Returns:
            应用结果
        """
        print(f"\n{'#'*60}")
        print("# 步骤 4: 备份并应用配置")
        print(f"{'#'*60}\n")
        
        # 1. 备份当前配置
        knob_names = list(recommendations['recommendations'].keys())
        backup_info = self.config_manager.backup_current_config(
            knob_names,
            description=f"调优前备份 - {recommendations['scenario']}"
        )
        
        # 2. 应用配置变更
        apply_result = self.config_manager.apply_config_changes(
            recommendations['recommendations'],
            dry_run=False
        )
        
        # 3. 重启数据库
        if len(apply_result['successful']) > 0:
            print("\n需要重启数据库使配置生效")
            restart_success = self.config_manager.restart_database()
            apply_result['restart_success'] = restart_success
        else:
            apply_result['restart_success'] = None
        
        # 整合结果
        apply_report = {
            'timestamp': datetime.now().isoformat(),
            'step': 'apply_with_backup',
            'backup_id': backup_info['backup_id'],
            'backup_file': backup_info.get('file', ''),
            'apply_result': apply_result,
            'summary': {
                'backup_created': True,
                'changes_applied': len(apply_result['successful']),
                'changes_failed': len(apply_result['failed']),
                'restart_required': len(apply_result['successful']) > 0,
                'restart_success': apply_result.get('restart_success')
            }
        }
        
        print(f"\n{'#'*60}")
        print("# 应用完成")
        print(f"{'#'*60}\n")
        print(f"✅ 备份ID: {backup_info['backup_id']}")
        print(f"✅ 应用成功: {len(apply_result['successful'])} 个参数")
        if apply_result.get('restart_success'):
            print(f"✅ 数据库已重启\n")
        
        return apply_report
    
    def rollback(self, backup_id: str) -> bool:
        """
        回滚到指定备份
        
        Args:
            backup_id: 备份ID
            
        Returns:
            是否成功
        """
        print(f"\n{'#'*60}")
        print(f"# 回滚配置")
        print(f"{'#'*60}\n")
        
        success = self.config_manager.restore_from_backup(backup_id)
        
        if success:
            print(f"✅ 已回滚到备份: {backup_id}\n")
        else:
            print(f"❌ 回滚失败\n")
        
        return success
    
    def run_full_tuning_workflow(self, 
                                 window_minutes: int = 120,
                                 auto_apply: bool = False,
                                 enable_safety_check: bool = True) -> Dict:
        """
        执行完整调优工作流
        
        Args:
            window_minutes: 监控时间窗口
            auto_apply: 是否自动应用（跳过审批）
            enable_safety_check: 是否启用安全检查
            
        Returns:
            完整工作流报告
        """
        print(f"\n{'='*60}")
        print(f"LTuner 自适应调优系统")
        print(f"模式: {'自动' if auto_apply else 'DBA 审批'}")
        print(f"安全检查: {'已启用' if enable_safety_check and self.enable_safety_check else '未启用'}")
        print(f"{'='*60}\n")
        
        workflow_report = {
            'start_time': datetime.now().isoformat(),
            'mode': 'auto' if auto_apply else 'manual',
            'safety_check_enabled': enable_safety_check and self.enable_safety_check,
            'steps': {}
        }
        
        try:
            # 步骤 1: 分析和推荐
            step1_report = self.analyze_and_recommend(window_minutes)
            workflow_report['steps']['step1_analyze'] = step1_report
            
            # 打印推荐摘要
            self.recommender.print_recommendations(step1_report['recommendations'])
            
            # 步骤 2: 预览变更
            step2_report = self.preview_changes(step1_report['recommendations'])
            workflow_report['steps']['step2_preview'] = step2_report
            
            # 步骤 2.5: 安全检查（可选）
            if enable_safety_check and self.enable_safety_check:
                safety_report = self.safety_check(step1_report)
                workflow_report['steps']['step2_5_safety_check'] = safety_report
                
                # 如果安全检查未通过，停止工作流
                if not safety_report.get('skipped', False):
                    if not safety_report.get('summary', {}).get('safe_to_apply', True):
                        print("\n\u274c 安全检查未通过，终止工作流")
                        print("\u8bf7修复以下问题后重试：\n")
                        
                        for invalid in safety_report.get('validation', {}).get('invalid_params', []):
                            print(f"  - {invalid['knob']}: {invalid['errors'][0] if invalid.get('errors') else '未知错误'}")
                        
                        workflow_report['end_time'] = datetime.now().isoformat()
                        workflow_report['status'] = 'failed_safety_check'
                        workflow_report['reason'] = '安全检查未通过'
                        return workflow_report
            
            # 步骤 3: 审批
            if not auto_apply:
                step3_report = self.request_approval(step1_report)
                workflow_report['steps']['step3_approval'] = step3_report
                
                # 在实际应用中，这里应该等待 DBA 审批
                print("⚠️  在生产环境中，此处会等待 DBA 审批")
                print("   当前为演示模式，假设已批准\n")
            
            # 步骤 4: 应用配置（仅在 auto_apply=True 或获得审批时）
            if auto_apply:
                step4_report = self.apply_with_backup(step1_report['recommendations'])
                workflow_report['steps']['step4_apply'] = step4_report
            else:
                print("ℹ️  配置未应用，等待 DBA 手动批准后执行\n")
            
            workflow_report['end_time'] = datetime.now().isoformat()
            workflow_report['status'] = 'completed'
            
        except Exception as e:
            workflow_report['status'] = 'failed'
            workflow_report['error'] = str(e)
            print(f"\n❌ 工作流执行失败: {e}\n")
        
        return workflow_report


# 测试代码
if __name__ == '__main__':
    from configparser import ConfigParser
    from dbms.postgres import PgDBMS
    
    # 加载配置
    config = ConfigParser()
    config.read('/root/GPTuner/configs/postgres.ini')
    
    # 初始化数据库
    dbms = PgDBMS.from_file(config)
    
    # 创建调优编排器
    orchestrator = TuningOrchestrator(dbms, use_rag=False)
    
    # 执行完整工作流（不自动应用）
    workflow_report = orchestrator.run_full_tuning_workflow(
        window_minutes=60,
        auto_apply=False
    )
    
    # 保存工作流报告
    output_file = '/root/GPTuner/tuning_workflow_report.json'
    with open(output_file, 'w') as f:
        json.dump(workflow_report, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 工作流报告已保存至: {output_file}")
