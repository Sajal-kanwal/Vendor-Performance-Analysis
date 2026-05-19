"""
Industry-Grade CSV to MySQL Database Loader
Handles multiple CSV files with error handling, logging, and data validation
"""

import os
import sys
import logging
import pandas as pd
import mysql.connector
from mysql.connector import Error, pooling
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import json
from dataclasses import dataclass
import re


@dataclass
class DatabaseConfig:
    """Database connection configuration"""
    host: str
    user: str
    password: str
    database: str
    port: int = 3306
    pool_size: int = 5


class CSVToMySQLLoader:
    """Advanced CSV to MySQL loader with enterprise features"""

    def __init__(self, db_config: DatabaseConfig, log_dir: str = "logs"):
        self.db_config = db_config
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.connection_pool = None
        self._setup_logging()
        self._create_connection_pool()

    def _setup_logging(self):
        """Configure comprehensive logging"""
        log_file = self.log_dir / f"csv_loader_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("CSV to MySQL Loader initialized")

    def _create_connection_pool(self):
        """Create MySQL connection pool for better performance"""
        try:
            self.connection_pool = pooling.MySQLConnectionPool(
                pool_name="csv_loader_pool",
                pool_size=self.db_config.pool_size,
                host=self.db_config.host,
                user=self.db_config.user,
                password=self.db_config.password,
                database=self.db_config.database,
                port=self.db_config.port
            )
            self.logger.info("Database connection pool created successfully")
        except Error as e:
            self.logger.error(f"Error creating connection pool: {e}")
            raise

    def _get_connection(self):
        """Get connection from pool"""
        try:
            return self.connection_pool.get_connection()
        except Error as e:
            self.logger.error(f"Error getting connection from pool: {e}")
            raise

    def _sanitize_table_name(self, filename: str) -> str:
        """Convert filename to valid MySQL table name"""
        table_name = Path(filename).stem
        table_name = re.sub(r'[^a-zA-Z0-9_]', '_', table_name)
        table_name = re.sub(r'^[0-9]', '_', table_name)
        return table_name.lower()

    def _infer_mysql_type(self, dtype, max_length: int = None) -> str:
        """Infer MySQL data type from pandas dtype"""
        dtype_str = str(dtype)

        if 'int' in dtype_str:
            if 'int8' in dtype_str or 'int16' in dtype_str:
                return 'SMALLINT'
            elif 'int32' in dtype_str:
                return 'INT'
            else:
                return 'BIGINT'
        elif 'float' in dtype_str or 'double' in dtype_str:
            return 'DOUBLE'
        elif 'datetime' in dtype_str:
            return 'DATETIME'
        elif 'date' in dtype_str:
            return 'DATE'
        elif 'bool' in dtype_str:
            return 'BOOLEAN'
        else:
            if max_length and max_length <= 255:
                return f'VARCHAR({max_length})'
            elif max_length and max_length <= 65535:
                return 'TEXT'
            else:
                return 'LONGTEXT'

    def _create_table_from_dataframe(self, df: pd.DataFrame, table_name: str,
                                     connection, if_exists: str = 'replace') -> bool:
        """Create MySQL table based on DataFrame structure"""
        try:
            cursor = connection.cursor()

            if if_exists == 'replace':
                cursor.execute(f"DROP TABLE IF EXISTS `{table_name}`")
                self.logger.info(f"Dropped existing table: {table_name}")
            elif if_exists == 'append':
                cursor.execute(f"SHOW TABLES LIKE '{table_name}'")
                if cursor.fetchone():
                    self.logger.info(f"Table {table_name} exists, appending data")
                    return True

            columns = []
            for col in df.columns:
                col_name = re.sub(r'[^a-zA-Z0-9_]', '_', str(col))

                if df[col].dtype == 'object':
                    max_len = df[col].astype(str).str.len().max()
                    max_len = min(max_len * 2, 65535) if max_len else 255
                    mysql_type = self._infer_mysql_type(df[col].dtype, max_len)
                else:
                    mysql_type = self._infer_mysql_type(df[col].dtype)

                columns.append(f"`{col_name}` {mysql_type}")

            columns.append("PRIMARY KEY (`id`)")
            columns.insert(0, "`id` INT AUTO_INCREMENT")

            create_table_sql = f"""
            CREATE TABLE `{table_name}` (
                {', '.join(columns)}
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """

            cursor.execute(create_table_sql)
            connection.commit()
            self.logger.info(f"Table {table_name} created successfully")
            return True

        except Error as e:
            self.logger.error(f"Error creating table {table_name}: {e}")
            connection.rollback()
            return False
        finally:
            cursor.close()

    def _insert_data_batch(self, df: pd.DataFrame, table_name: str,
                           connection, batch_size: int = 1000) -> Dict:
        """Insert data in batches for better performance"""
        cursor = connection.cursor()
        total_rows = len(df)
        inserted_rows = 0
        failed_rows = 0

        try:
            df_clean = df.copy()
            for col in df_clean.columns:
                df_clean[col] = df_clean[col].replace({pd.NA: None, pd.NaT: None})

            columns = [re.sub(r'[^a-zA-Z0-9_]', '_', str(col)) for col in df_clean.columns]
            placeholders = ', '.join(['%s'] * len(columns))
            column_names = ', '.join([f'`{col}`' for col in columns])

            insert_sql = f"INSERT INTO `{table_name}` ({column_names}) VALUES ({placeholders})"

            for i in range(0, total_rows, batch_size):
                batch = df_clean.iloc[i:i + batch_size]
                values = [tuple(row) for row in batch.values]

                try:
                    cursor.executemany(insert_sql, values)
                    connection.commit()
                    inserted_rows += len(values)
                    self.logger.info(f"Inserted batch {i // batch_size + 1}: {len(values)} rows")
                except Error as e:
                    self.logger.error(f"Error inserting batch {i // batch_size + 1}: {e}")
                    failed_rows += len(values)
                    connection.rollback()

            return {
                'total_rows': total_rows,
                'inserted_rows': inserted_rows,
                'failed_rows': failed_rows,
                'success_rate': (inserted_rows / total_rows * 100) if total_rows > 0 else 0
            }

        except Error as e:
            self.logger.error(f"Error during batch insert: {e}")
            connection.rollback()
            return {
                'total_rows': total_rows,
                'inserted_rows': inserted_rows,
                'failed_rows': total_rows - inserted_rows,
                'success_rate': 0
            }
        finally:
            cursor.close()

    def load_csv_file(self, csv_file: str, table_name: Optional[str] = None,
                      if_exists: str = 'replace', batch_size: int = 1000,
                      encoding: str = 'utf-8') -> Dict:
        """Load single CSV file to MySQL"""
        self.logger.info(f"Processing file: {csv_file}")

        try:
            df = pd.read_csv(csv_file, encoding=encoding)
            self.logger.info(f"CSV loaded: {len(df)} rows, {len(df.columns)} columns")

            if df.empty:
                self.logger.warning(f"File {csv_file} is empty, skipping")
                return {'status': 'skipped', 'reason': 'empty file'}

            table_name = table_name or self._sanitize_table_name(csv_file)
            connection = self._get_connection()

            try:
                if not self._create_table_from_dataframe(df, table_name, connection, if_exists):
                    return {'status': 'failed', 'reason': 'table creation failed'}

                insert_stats = self._insert_data_batch(df, table_name, connection, batch_size)

                return {
                    'status': 'success',
                    'file': csv_file,
                    'table': table_name,
                    **insert_stats
                }

            finally:
                connection.close()

        except Exception as e:
            self.logger.error(f"Error processing {csv_file}: {e}")
            return {
                'status': 'failed',
                'file': csv_file,
                'reason': str(e)
            }

    def load_directory(self, directory: str, pattern: str = "*.csv",
                       if_exists: str = 'replace', batch_size: int = 1000) -> List[Dict]:
        """Load all CSV files from directory"""
        self.logger.info(f"Scanning directory: {directory}")

        csv_files = list(Path(directory).glob(pattern))
        self.logger.info(f"Found {len(csv_files)} CSV files")

        results = []
        for csv_file in csv_files:
            result = self.load_csv_file(str(csv_file), if_exists=if_exists,
                                        batch_size=batch_size)
            results.append(result)

        return results

    def generate_report(self, results: List[Dict]) -> str:
        """Generate summary report"""
        report_file = self.log_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_files': len(results),
            'successful': sum(1 for r in results if r.get('status') == 'success'),
            'failed': sum(1 for r in results if r.get('status') == 'failed'),
            'total_rows_processed': sum(r.get('total_rows', 0) for r in results),
            'total_rows_inserted': sum(r.get('inserted_rows', 0) for r in results),
            'details': results
        }

        with open(report_file, 'w') as f:
            json.dump(summary, f, indent=2)

        self.logger.info(f"Report generated: {report_file}")
        return str(report_file)


def main():
    """
    Main execution function.

    Reads database credentials from environment variables (via config.py)
    and loads all CSV files from the configured data directory.
    """
    from config import get_config

    cfg = get_config()

    # Build DatabaseConfig from centralized settings
    db_config = DatabaseConfig(
        host=cfg.database.host,
        user=cfg.database.user,
        password=cfg.database.password,
        database=cfg.database.database,
        port=cfg.database.port,
        pool_size=cfg.database.pool_size,
    )

    # Initialize loader
    loader = CSVToMySQLLoader(db_config)

    # Load all CSV files from the configured data directory
    results = loader.load_directory(
        directory=str(cfg.paths.data_dir),
        pattern='*.csv',
        if_exists='replace',  # 'replace', 'append', or 'fail'
        batch_size=cfg.pipeline.batch_size,
    )

    # Generate report
    report_file = loader.generate_report(results)

    # Print summary
    print("\n" + "=" * 50)
    print("LOAD SUMMARY")
    print("=" * 50)
    for result in results:
        if result.get('status') == 'success':
            print(f"✓ {result['file']} -> {result['table']}")
            print(f"  Rows: {result['inserted_rows']}/{result['total_rows']} "
                  f"({result['success_rate']:.1f}%)")
        else:
            print(f"✗ {result.get('file', 'Unknown')} - {result.get('reason', 'Unknown error')}")
    print(f"\nDetailed report: {report_file}")


if __name__ == "__main__":
    main()
