sudo rm -f /var/lib/postgresql/14/main/postgresql.auto.conf 2>/dev/null || true
sleep 2
# 通过 systemctl 重启（自动处理 auto.conf 问题）
sudo systemctl restart postgresql@14-main 2>/dev/null || sudo pg_ctlcluster 14 main restart 2>/dev/null || true
sleep 5