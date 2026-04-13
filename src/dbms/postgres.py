from dbms.dbms_template import DBMSTemplate
import psycopg2
import os
import time
import json
import subprocess

# PG 崩溃时返回的"惩罚性"性能值（让 SMAC 不再探索此方向）
_CRASH_PENALTY_TPS = -1.0
_CRASH_PENALTY_LAT = 1e9

class PgDBMS(DBMSTemplate):
    """ Instantiate DBMSTemplate to support PostgreSQL DBMS """
    def __init__(self, db, user, password, restart_cmd, recover_script, knob_info_path):
        super().__init__(db, user, password, restart_cmd, recover_script, knob_info_path)
        self.name = "postgres"
        self.crash_count = 0          # 记录总崩溃次数
        self.consecutive_crash = 0    # 连续崩溃次数（超过阈值才停止）
    
    def _connect(self, db=None):
        """ Establish connection to database, return success flag """
        self.failed_times = 0
        if db is None:
            db = self.db
        print(f'Trying to connect to {db} with user {self.user}')
        for _ in range(6):           # 最多重试 6 次（30 秒）
            try:            
                self.connection = psycopg2.connect(
                    database=db, user=self.user,
                    password=self.password, host="localhost"
                )
                print(f"Success to connect to {db} with user {self.user}")
                self.failed_times = 0
                return True
            except Exception as e:
                self.failed_times += 1
                print(f'Exception while trying to connect: {e}')
                if self.failed_times >= 4:
                    self.failed_times = 0
                    self.recover_dbms()
                    time.sleep(5)    # 等恢复完成再重连
                    continue
                print("Reconnect again")
                time.sleep(5)
        print("[ERROR] Cannot connect after recovery, skip this config")
        return False
            
    def _disconnect(self):
        """ Disconnect from database. """
        if self.connection:
            try:
                print('Disconnecting ...')
                self.connection.close()
                print('Disconnecting done ...')
            except:
                pass
            finally:
                self.connection = None

    def recover_dbms(self):
        """Recover the dbms if the dbms has a crash - 带超时保护"""
        self.crash_count += 1
        self.consecutive_crash += 1
        print(f"[RECOVER] Crash #{self.crash_count}, running recovery script...")
        # 带超时保护的 recover，避免 su - postgres 挂起
        try:
            result = subprocess.run(
                f"sh {self.recover_script}",
                shell=True, timeout=60,
                capture_output=True, text=True
            )
            print(f"[RECOVER] Script exit code: {result.returncode}")
        except subprocess.TimeoutExpired:
            print("[RECOVER] Recovery script timed out! Trying systemctl...")
            os.system("sudo systemctl restart postgresql@14-main 2>/dev/null || true")
            time.sleep(5)
        print("DBMS recovered")

    def copy_db(self, target_db, source_db):
        # for tpcc, recover the data for the target db(benchbase)
        self.update_dbms(f'drop database if exists {target_db}')
        print('Dropped old database')
        self.update_dbms(f'create database {target_db} with template {source_db}')
        print('Initialized new database')

    def reset_config(self):
        """ Reset all parameters to default values. """
        self.update_dbms('alter system reset all;')
        
    def reconfigure(self):
        """ 
            Restart to make parameter settings take effect. Returns true if successful.
            崩溃时自动恢复并返回 False（不中断整体实验）。
        """
        self._disconnect()
        print(f"[RECONFIGURE] Restarting PG with cmd: {self.restart_cmd[:60]}...")
        
        # 用子进程+超时执行 restart，避免挂起
        try:
            result = subprocess.run(
                self.restart_cmd,
                shell=True, timeout=45,
                capture_output=True, text=True
            )
        except subprocess.TimeoutExpired:
            print("[RECONFIGURE] Restart timed out! Triggering recovery...")
            self.recover_dbms()
            time.sleep(5)
            success = self._connect()
            return success

        time.sleep(3)
        success = self._connect()
        if success:
            self.consecutive_crash = 0   # 连接成功，重置连续崩溃计数
            return True
        else:
            print("[RECONFIGURE] PG failed to start, recovering...")
            self.recover_dbms()
            time.sleep(5)
            # 恢复后再尝试连接一次
            success = self._connect()
            if success:
                self.consecutive_crash = 0
            return success  # 即使恢复失败也返回 False 而不是抛出异常
    
    def get_sql_result(self, sql):
        """ Execute sql query on dbms and return the result and its description """
        self.connection.autocommit = True
        cursor = self.connection.cursor()
        cursor.execute(sql)
        result = cursor.fetchall()
        description = cursor.description
        cursor.close()
        
        return result, description
    
    def extract_knob_info(self, dest_path):
        """ execute "pg_settings" sql on dbms for knob information and store the query result in json format """
        knob_info = {}
        knobs_sql = "SELECT name FROM pg_settings;"
        knobs, _ = self.get_sql_result(knobs_sql)
        for knob in knobs:
            knob = knob[0]  # Extract the knob name from the result tuple
            knob_details_sql = f"SELECT * FROM pg_settings WHERE name = '{knob}';"
            knob_detail, description = self.get_sql_result(knob_details_sql)
            if knob_detail:
                column_names = [desc[0] for desc in description]
                knob_detail = knob_detail[0]
                knob_attributes = {}
                for i, column_name in enumerate(column_names):
                    knob_attributes[column_name] = knob_detail[i]
                knob_info[knob] = knob_attributes
        with open(dest_path, "w") as json_file:
            json.dump(knob_info, json_file, indent=4, sort_keys=True)
        print(f"The knob info is written to {dest_path}")

    def update_dbms(self, sql):
        """ Execute sql query on dbms to update knob value and return success flag """
        try:
            self.connection.autocommit = True
            cursor = self.connection.cursor()
            cursor.execute(sql)
            cursor.close()
            return True
        except Exception as e:
            print(f"Failed to execute {sql} to update dbms for error: {e}")
            return False 

    def set_knob(self, knob, knob_value):
        query_one = f'alter system set {knob} to \'{knob_value}\';'
        success =  self.update_dbms(query_one)
        if success:
            self.config[knob] = knob_value
        return success 
    
    def get_knob_value(self, knob):
        """ Get the current value for a knob """
        result, _ = self.get_sql_result(f"show {knob}")
        return result
        
    def check_knob_exists(self, knob):
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM pg_settings WHERE name = %s", (knob,))
        row = cursor.fetchone()
        cursor.close()
        return row is not None

    def exec_quries(self, sql):
        """ Executes all SQL queries in given file and returns success flag. """
        try:
            self.connection.autocommit = True
            cursor = self.connection.cursor()
            sql_statements = sql.split(';')
            for statement in sql_statements:
                if statement.strip():
                    cursor.execute(statement)
            # cursor.execute(sql)
            cursor.close()
            return True
        except Exception as e:
            print(f'Exception execution {sql}: {e}')
        return False