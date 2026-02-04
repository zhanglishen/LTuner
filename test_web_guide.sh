#!/bin/bash
# GPTuner Web 界面测试指南

echo "=================================="
echo "GPTuner Web 界面测试指南"
echo "=================================="
echo ""

# 1. 检查数据库连接
echo "1️⃣ 检查数据库连接..."
PGPASSWORD=password psql -h localhost -U admin -d benchbase -c "SELECT version();" > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "   ✅ 数据库连接正常"
    echo "   - 主机: localhost"
    echo "   - 端口: 5432"
    echo "   - 用户: admin"
    echo "   - 密码: password"
    echo "   - 数据库: benchbase"
else
    echo "   ❌ 数据库连接失败"
    exit 1
fi
echo ""

# 2. 检查 Streamlit 进程
echo "2️⃣ 检查 Streamlit 进程..."
if ps aux | grep -v grep | grep streamlit > /dev/null; then
    echo "   ✅ Streamlit 正在运行"
    ps aux | grep -v grep | grep streamlit | awk '{print "   PID:", $2, "| 内存:", $6"KB"}'
else
    echo "   ❌ Streamlit 未运行"
    echo "   启动命令: cd /root/GPTuner && /usr/local/python3.9/bin/streamlit run src/web/app.py --server.port 8501 --server.address 0.0.0.0 &"
    exit 1
fi
echo ""

# 3. 检查端口
echo "3️⃣ 检查 Web 端口..."
if netstat -tln | grep :8501 > /dev/null 2>&1; then
    echo "   ✅ 端口 8501 已监听"
else
    echo "   ⚠️  端口 8501 未监听，请等待 Streamlit 完全启动"
fi
echo ""

# 4. 测试数据库操作
echo "4️⃣ 测试数据库基本操作..."
PGPASSWORD=password psql -h localhost -U admin -d benchbase -c "
SELECT 
    setting as max_connections,
    (SELECT setting FROM pg_settings WHERE name='shared_buffers') as shared_buffers,
    (SELECT setting FROM pg_settings WHERE name='work_mem') as work_mem
FROM pg_settings WHERE name='max_connections';
" 2>&1 | head -10
echo ""

# 5. Web 访问指南
echo "=================================="
echo "📊 Web 界面访问方式"
echo "=================================="
echo ""
echo "本地访问: http://localhost:8501"
echo "远程访问: http://服务器IP:8501"
echo ""
echo "=================================="
echo "🚀 使用步骤"
echo "=================================="
echo ""
echo "步骤 1: 配置数据库连接"
echo "   - 进入「⚙️ 配置管理」页面"
echo "   - 填写数据库连接信息（默认已配置）："
echo "     • 主机: localhost"
echo "     • 端口: 5432"
echo "     • 用户: admin"
echo "     • 密码: password"
echo "     • 数据库: benchbase"
echo "   - 点击「🔍 测试连接」确认连接成功"
echo "   - 点击「💾 保存配置」"
echo ""
echo "步骤 2: 启动监控"
echo "   - 进入「📊 实时监控」页面"
echo "   - 点击「▶️ 启动监控」"
echo "   - 查看实时性能指标（TPS、QPS、缓存命中率、连接数）"
echo "   - 查看工作负载类型识别（OLTP/OLAP/HYBRID）"
echo ""
echo "步骤 3: 触发调优"
echo "   - 进入「🎯 调优推荐」页面"
echo "   - 点击「🚀 开始分析」"
echo "   - 系统会："
echo "     ① 采集过去 2 小时的性能指标"
echo "     ② 识别工作负载类型"
echo "     ③ 生成参数调优推荐"
echo "     ④ 执行安全检查（验证参数合法性）"
echo ""
echo "步骤 4: 审批推荐"
echo "   - 查看推荐参数详情："
echo "     • 当前值 vs 推荐值"
echo "     • 调整原因（可解释性）"
echo "     • 优先级（高/中/低）"
echo "   - 查看安全检查报告："
echo "     • 风险等级（无/低/中/高）"
echo "     • 风险因素"
echo "     • 安全建议"
echo "   - 操作选项："
echo "     ✅ 批准并应用（自动备份当前配置）"
echo "     ❌ 拒绝"
echo "     💾 导出报告"
echo ""
echo "步骤 5: 查看历史"
echo "   - 进入「📜 历史记录」页面"
echo "   - 查看所有备份列表"
echo "   - 可以："
echo "     ↩️ 回滚到任意历史版本"
echo "     💾 导出备份"
echo "     🗑️ 删除备份"
echo ""
echo "=================================="
echo "⚠️  注意事项"
echo "=================================="
echo ""
echo "1. 首次使用建议："
echo "   - 在测试环境验证功能"
echo "   - 启用安全检查（默认已启用）"
echo "   - 使用手动触发模式（默认）"
echo ""
echo "2. 安全保障："
echo "   - 每次应用配置前都会自动备份"
echo "   - 高风险配置会被自动禁止应用"
echo "   - 支持一键回滚"
echo ""
echo "3. 性能影响："
echo "   - 监控操作对数据库影响极小"
echo "   - 建议在低峰时段应用配置"
echo "   - 应用后需要重启数据库才能生效"
echo ""
echo "=================================="
echo "✅ 环境检查完成！"
echo "=================================="
