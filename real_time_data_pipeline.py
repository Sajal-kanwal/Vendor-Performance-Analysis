"""
Enterprise Real-Time Data Pipeline - Server to MySQL Database
Supports multiple data sources: API, FTP, SFTP, S3, CSV, JSON, Excel
Automated scheduling, monitoring, and alert system
"""

import os
import sys
import logging
import pandas as pd
import mysql.connector
from mysql.connector import Error, pooling
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Callable
import json
import time
import schedule
import requests
from dataclasses import dataclass, field
import hashlib
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from abc import ABC, abstractmethod
import threading
from queue import Queue
import ftplib
import paramiko
import pyarrow.parquet as pq
import re


@dataclass
class DatabaseConfig:
    """Database connection configuration"""
    host: str
    user: str
    password: str
    database: str
    port: int = 3306
    pool_size: int = 10


@dataclass
class EmailConfig:
    """Email alert configuration"""
    enabled: bool = False
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    sender_email: str = ""
    sender_password: str = ""
    recipient_emails: List[str] = field(default_factory=list)


@dataclass
class DataSourceConfig:
    """Data source configuration"""
    source_type: str  # 'api', 'ftp', 'sftp', 's3', 'local', 'database'
    name: str
    enabled: bool = True
    schedule_interval: int = 300  # seconds
    table_name: Optional[str] = None
    sync_mode: str = 'incremental'  # 'full', 'incremental', 'upsert'

    # API specific
    api_url: Optional[str] = None
    api_headers: Dict = field(default_factory=dict)
    api_params: Dict = field(default_factory=dict)
    api_auth: Optional[tuple] = None

    # FTP/SFTP specific
    ftp_host: Optional[str] = None
    ftp_user: Optional[str] = None
    ftp_password: Optional[str] = None
    ftp_path: Optional[str] = None
    ftp_port: int = 21

    # S3 specific
    s3_bucket: Optional[str] = None
    s3_key: Optional[str] = None
    s3_access_key: Optional[str] = None
    s3_secret_key: Optional[str] = None

    # Local file specific
    file_path: Optional[str] = None
    file_pattern: Optional[str] = "*.csv"

    # Database specific
    source_db_config: Optional[Dict] = None
    source_query: Optional[str] = None

    # Common
    data_format: str = 'csv'  # 'csv', 'json', 'excel', 'parquet'
    primary_key: Optional[str] = None
    timestamp_column: Optional[str] = None
    last_sync_time: Optional[datetime] = None


class DataExtractor(ABC):
    """Abstract base class for data extractors"""

    @abstractmethod
    def extract(self, config: DataSourceConfig) -> pd.DataFrame:
        """Extract data from source"""
        pass


class APIExtractor(DataExtractor):
    """Extract data from REST APIs"""

    def __init__(self, logger):
        self.logger = logger

    def extract(self, config: DataSourceConfig) -> pd.DataFrame:
        """Fetch data from API endpoint"""
        try:
            self.logger.info(f"Fetching data from API: {config.api_url}")

            response = requests.get(
                config.api_url,
                headers=config.api_headers,
                params=config.api_params,
                auth=config.api_auth,
                timeout=60
            )
            response.raise_for_status()

            if config.data_format == 'json':
                data = response.json()
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                elif isinstance(data, dict):
                    if 'data' in data:
                        df = pd.DataFrame(data['data'])
                    elif 'results' in data:
                        df = pd.DataFrame(data['results'])
                    else:
                        df = pd.DataFrame([data])
                else:
                    raise ValueError("Unexpected JSON structure")
            else:
                df = pd.read_csv(pd.io.common.BytesIO(response.content))

            self.logger.info(f"Extracted {len(df)} rows from API")
            return df

        except Exception as e:
            self.logger.error(f"API extraction failed: {e}")
            raise


class FTPExtractor(DataExtractor):
    """Extract data from FTP servers"""

    def __init__(self, logger):
        self.logger = logger

    def extract(self, config: DataSourceConfig) -> pd.DataFrame:
        """Download and read file from FTP"""
        try:
            self.logger.info(f"Connecting to FTP: {config.ftp_host}")

            ftp = ftplib.FTP()
            ftp.connect(config.ftp_host, config.ftp_port)
            ftp.login(config.ftp_user, config.ftp_password)

            files = []
            ftp.cwd(config.ftp_path or '/')
            ftp.retrlines('LIST', files.append)

            # Get the most recent file matching pattern
            latest_file = None
            latest_time = None

            for file_info in files:
                parts = file_info.split()
                filename = parts[-1]
                if config.file_pattern.replace('*', '') in filename:
                    if latest_file is None:
                        latest_file = filename

            if not latest_file:
                raise FileNotFoundError("No matching files found on FTP")

            # Download file to memory
            from io import BytesIO
            buffer = BytesIO()
            ftp.retrbinary(f'RETR {latest_file}', buffer.write)
            buffer.seek(0)

            if config.data_format == 'csv':
                df = pd.read_csv(buffer)
            elif config.data_format == 'excel':
                df = pd.read_excel(buffer)
            else:
                raise ValueError(f"Unsupported format: {config.data_format}")

            ftp.quit()
            self.logger.info(f"Extracted {len(df)} rows from FTP file: {latest_file}")
            return df

        except Exception as e:
            self.logger.error(f"FTP extraction failed: {e}")
            raise


class SFTPExtractor(DataExtractor):
    """Extract data from SFTP servers"""

    def __init__(self, logger):
        self.logger = logger

    def extract(self, config: DataSourceConfig) -> pd.DataFrame:
        """Download and read file from SFTP"""
        try:
            self.logger.info(f"Connecting to SFTP: {config.ftp_host}")

            transport = paramiko.Transport((config.ftp_host, config.ftp_port or 22))
            transport.connect(username=config.ftp_user, password=config.ftp_password)
            sftp = paramiko.SFTPClient.from_transport(transport)

            from io import BytesIO
            buffer = BytesIO()

            # List files and get latest
            files = sftp.listdir(config.ftp_path or '.')
            matching_files = [f for f in files if config.file_pattern.replace('*', '') in f]

            if not matching_files:
                raise FileNotFoundError("No matching files found on SFTP")

            latest_file = matching_files[-1]
            remote_path = f"{config.ftp_path}/{latest_file}".replace('//', '/')

            sftp.getfo(remote_path, buffer)
            buffer.seek(0)

            if config.data_format == 'csv':
                df = pd.read_csv(buffer)
            elif config.data_format == 'excel':
                df = pd.read_excel(buffer)

            sftp.close()
            transport.close()

            self.logger.info(f"Extracted {len(df)} rows from SFTP file: {latest_file}")
            return df

        except Exception as e:
            self.logger.error(f"SFTP extraction failed: {e}")
            raise


class LocalFileExtractor(DataExtractor):
    """Extract data from local files"""

    def __init__(self, logger):
        self.logger = logger

    def extract(self, config: DataSourceConfig) -> pd.DataFrame:
        """Read data from local file system"""
        try:
            if config.file_path:
                files = [Path(config.file_path)]
            else:
                files = list(Path('.').glob(config.file_pattern))

            if not files:
                raise FileNotFoundError("No matching files found")

            latest_file = max(files, key=lambda f: f.stat().st_mtime)
            self.logger.info(f"Reading file: {latest_file}")

            if config.data_format == 'csv':
                df = pd.read_csv(latest_file)
            elif config.data_format == 'json':
                df = pd.read_json(latest_file)
            elif config.data_format == 'excel':
                df = pd.read_excel(latest_file)
            elif config.data_format == 'parquet':
                df = pd.read_parquet(latest_file)
            else:
                raise ValueError(f"Unsupported format: {config.data_format}")

            self.logger.info(f"Extracted {len(df)} rows from local file")
            return df

        except Exception as e:
            self.logger.error(f"Local file extraction failed: {e}")
            raise


class DatabaseExtractor(DataExtractor):
    """Extract data from another database"""

    def __init__(self, logger):
        self.logger = logger

    def extract(self, config: DataSourceConfig) -> pd.DataFrame:
        """Query data from source database"""
        try:
            self.logger.info("Connecting to source database")

            conn = mysql.connector.connect(**config.source_db_config)
            df = pd.read_sql(config.source_query, conn)
            conn.close()

            self.logger.info(f"Extracted {len(df)} rows from source database")
            return df

        except Exception as e:
            self.logger.error(f"Database extraction failed: {e}")
            raise


class RealTimeDataPipeline:
    """Enterprise real-time data pipeline orchestrator"""

    def __init__(self, db_config: DatabaseConfig, email_config: EmailConfig = None,
                 log_dir: str = "logs"):
        self.db_config = db_config
        self.email_config = email_config or EmailConfig()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        self.connection_pool = None
        self.data_sources: List[DataSourceConfig] = []
        self.extractors = {}
        self.job_queue = Queue()
        self.running = False
        self.stats = {
            'total_syncs': 0,
            'successful_syncs': 0,
            'failed_syncs': 0,
            'total_rows_processed': 0
        }

        self._setup_logging()
        self._create_connection_pool()
        self._initialize_extractors()
        self._create_metadata_tables()

    def _setup_logging(self):
        """Configure logging"""
        log_file = self.log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Real-Time Data Pipeline initialized")

    def _create_connection_pool(self):
        """Create MySQL connection pool"""
        try:
            self.connection_pool = pooling.MySQLConnectionPool(
                pool_name="pipeline_pool",
                pool_size=self.db_config.pool_size,
                host=self.db_config.host,
                user=self.db_config.user,
                password=self.db_config.password,
                database=self.db_config.database,
                port=self.db_config.port
            )
            self.logger.info("Database connection pool created")
        except Error as e:
            self.logger.error(f"Connection pool creation failed: {e}")
            raise

    def _initialize_extractors(self):
        """Initialize data extractors"""
        self.extractors = {
            'api': APIExtractor(self.logger),
            'ftp': FTPExtractor(self.logger),
            'sftp': SFTPExtractor(self.logger),
            'local': LocalFileExtractor(self.logger),
            'database': DatabaseExtractor(self.logger)
        }

    def _create_metadata_tables(self):
        """Create metadata tracking tables"""
        try:
            conn = self.connection_pool.get_connection()
            cursor = conn.cursor()

            # Sync history table
            cursor.execute("""
                           CREATE TABLE IF NOT EXISTS `pipeline_sync_history`
                           (
                               `id`
                               INT
                               AUTO_INCREMENT
                               PRIMARY
                               KEY,
                               `source_name`
                               VARCHAR
                           (
                               255
                           ),
                               `table_name` VARCHAR
                           (
                               255
                           ),
                               `sync_start` DATETIME,
                               `sync_end` DATETIME,
                               `rows_processed` INT,
                               `status` VARCHAR
                           (
                               50
                           ),
                               `error_message` TEXT,
                               INDEX idx_source_name
                           (
                               source_name
                           ),
                               INDEX idx_sync_start
                           (
                               sync_start
                           )
                               ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                           """)

            # Data quality metrics table
            cursor.execute("""
                           CREATE TABLE IF NOT EXISTS `pipeline_data_quality`
                           (
                               `id`
                               INT
                               AUTO_INCREMENT
                               PRIMARY
                               KEY,
                               `table_name`
                               VARCHAR
                           (
                               255
                           ),
                               `check_time` DATETIME,
                               `total_rows` INT,
                               `null_count` INT,
                               `duplicate_count` INT,
                               `data_hash` VARCHAR
                           (
                               64
                           ),
                               INDEX idx_table_name
                           (
                               table_name
                           )
                               ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                           """)

            conn.commit()
            cursor.close()
            conn.close()
            self.logger.info("Metadata tables created/verified")

        except Error as e:
            self.logger.error(f"Metadata table creation failed: {e}")

    def add_data_source(self, source_config: DataSourceConfig):
        """Register a new data source"""
        self.data_sources.append(source_config)
        self.logger.info(f"Data source registered: {source_config.name}")

    def _sanitize_name(self, name: str) -> str:
        """Sanitize table/column names"""
        name = re.sub(r'[^a-zA-Z0-9_]', '_', str(name))
        name = re.sub(r'^[0-9]', '_', name)
        return name.lower()

    def _infer_mysql_type(self, dtype, max_length: int = None) -> str:
        """Infer MySQL data type"""
        dtype_str = str(dtype)

        if 'int' in dtype_str:
            return 'BIGINT' if 'int64' in dtype_str else 'INT'
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
                return f'VARCHAR({min(max_length * 2, 255)})'
            else:
                return 'TEXT'

    def _create_or_update_table(self, df: pd.DataFrame, table_name: str,
                                connection, primary_key: str = None):
        """Create or update table schema"""
        cursor = connection.cursor()

        try:
            cursor.execute(f"SHOW TABLES LIKE '{table_name}'")
            table_exists = cursor.fetchone() is not None

            if not table_exists:
                columns = []

                for col in df.columns:
                    col_name = self._sanitize_name(col)

                    if df[col].dtype == 'object':
                        max_len = df[col].astype(str).str.len().max()
                        max_len = max_len if max_len else 255
                        mysql_type = self._infer_mysql_type(df[col].dtype, max_len)
                    else:
                        mysql_type = self._infer_mysql_type(df[col].dtype)

                    columns.append(f"`{col_name}` {mysql_type}")

                if primary_key:
                    pk_col = self._sanitize_name(primary_key)
                    columns.append(f"PRIMARY KEY (`{pk_col}`)")
                else:
                    columns.insert(0, "`id` INT AUTO_INCREMENT")
                    columns.append("PRIMARY KEY (`id`)")

                columns.append("`_pipeline_updated_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP")

                create_sql = f"""
                    CREATE TABLE `{table_name}` (
                        {', '.join(columns)}
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """

                cursor.execute(create_sql)
                connection.commit()
                self.logger.info(f"Table {table_name} created")

        except Error as e:
            self.logger.error(f"Table creation/update failed: {e}")
            connection.rollback()
            raise
        finally:
            cursor.close()

    def _sync_full(self, df: pd.DataFrame, table_name: str, connection):
        """Full table replacement"""
        cursor = connection.cursor()
        try:
            cursor.execute(f"TRUNCATE TABLE `{table_name}`")
            self._insert_batch(df, table_name, connection)
            self.logger.info(f"Full sync completed for {table_name}")
        finally:
            cursor.close()

    def _sync_incremental(self, df: pd.DataFrame, table_name: str,
                          connection, timestamp_col: str, last_sync: datetime):
        """Incremental sync based on timestamp"""
        if timestamp_col and last_sync:
            df_filtered = df[pd.to_datetime(df[timestamp_col]) > last_sync]
            self.logger.info(f"Incremental sync: {len(df_filtered)} new rows")
            if not df_filtered.empty:
                self._insert_batch(df_filtered, table_name, connection)
        else:
            self._insert_batch(df, table_name, connection)

    def _sync_upsert(self, df: pd.DataFrame, table_name: str,
                     connection, primary_key: str):
        """Upsert (insert or update) based on primary key"""
        if not primary_key:
            self.logger.warning("No primary key specified, falling back to insert")
            self._insert_batch(df, table_name, connection)
            return

        cursor = connection.cursor()
        try:
            pk_col = self._sanitize_name(primary_key)
            columns = [self._sanitize_name(col) for col in df.columns]

            placeholders = ', '.join(['%s'] * len(columns))
            column_names = ', '.join([f'`{col}`' for col in columns])

            updates = ', '.join([f'`{col}`=VALUES(`{col}`)' for col in columns if col != pk_col])

            upsert_sql = f"""
                INSERT INTO `{table_name}` ({column_names})
                VALUES ({placeholders})
                ON DUPLICATE KEY UPDATE {updates}
            """

            values = [tuple(row) for row in df.values]
            cursor.executemany(upsert_sql, values)
            connection.commit()

            self.logger.info(f"Upsert completed: {len(values)} rows")

        except Error as e:
            self.logger.error(f"Upsert failed: {e}")
            connection.rollback()
            raise
        finally:
            cursor.close()

    def _insert_batch(self, df: pd.DataFrame, table_name: str,
                      connection, batch_size: int = 1000):
        """Insert data in batches"""
        cursor = connection.cursor()

        try:
            df_clean = df.copy()
            for col in df_clean.columns:
                df_clean[col] = df_clean[col].replace({pd.NA: None, pd.NaT: None})

            columns = [self._sanitize_name(col) for col in df_clean.columns]
            placeholders = ', '.join(['%s'] * len(columns))
            column_names = ', '.join([f'`{col}`' for col in columns])

            insert_sql = f"INSERT INTO `{table_name}` ({column_names}) VALUES ({placeholders})"

            total_rows = len(df_clean)
            for i in range(0, total_rows, batch_size):
                batch = df_clean.iloc[i:i + batch_size]
                values = [tuple(row) for row in batch.values]
                cursor.executemany(insert_sql, values)
                connection.commit()

            self.logger.info(f"Inserted {total_rows} rows into {table_name}")

        except Error as e:
            self.logger.error(f"Batch insert failed: {e}")
            connection.rollback()
            raise
        finally:
            cursor.close()

    def _log_sync_history(self, source_name: str, table_name: str,
                          start_time: datetime, end_time: datetime,
                          rows_processed: int, status: str, error_msg: str = None):
        """Log sync operation to history"""
        try:
            conn = self.connection_pool.get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                           INSERT INTO pipeline_sync_history
                           (source_name, table_name, sync_start, sync_end, rows_processed, status, error_message)
                           VALUES (%s, %s, %s, %s, %s, %s, %s)
                           """, (source_name, table_name, start_time, end_time, rows_processed, status, error_msg))

            conn.commit()
            cursor.close()
            conn.close()

        except Error as e:
            self.logger.error(f"Failed to log sync history: {e}")

    def _send_alert(self, subject: str, message: str):
        """Send email alert"""
        if not self.email_config.enabled:
            return

        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_config.sender_email
            msg['To'] = ', '.join(self.email_config.recipient_emails)
            msg['Subject'] = subject

            msg.attach(MIMEText(message, 'html'))

            server = smtplib.SMTP(self.email_config.smtp_server, self.email_config.smtp_port)
            server.starttls()
            server.login(self.email_config.sender_email, self.email_config.sender_password)
            server.send_message(msg)
            server.quit()

            self.logger.info(f"Alert sent: {subject}")

        except Exception as e:
            self.logger.error(f"Failed to send alert: {e}")

    def sync_data_source(self, source_config: DataSourceConfig):
        """Sync single data source"""
        start_time = datetime.now()
        status = "failed"
        error_msg = None
        rows_processed = 0

        try:
            self.logger.info(f"Starting sync for: {source_config.name}")

            # Extract data
            extractor = self.extractors.get(source_config.source_type)
            if not extractor:
                raise ValueError(f"Unknown source type: {source_config.source_type}")

            df = extractor.extract(source_config)
            rows_processed = len(df)

            if df.empty:
                self.logger.warning(f"No data extracted from {source_config.name}")
                return

            # Prepare table
            table_name = source_config.table_name or self._sanitize_name(source_config.name)
            connection = self.connection_pool.get_connection()

            try:
                self._create_or_update_table(df, table_name, connection, source_config.primary_key)

                # Sync based on mode
                if source_config.sync_mode == 'full':
                    self._sync_full(df, table_name, connection)
                elif source_config.sync_mode == 'incremental':
                    self._sync_incremental(df, table_name, connection,
                                           source_config.timestamp_column,
                                           source_config.last_sync_time)
                elif source_config.sync_mode == 'upsert':
                    self._sync_upsert(df, table_name, connection, source_config.primary_key)

                source_config.last_sync_time = datetime.now()
                status = "success"

                self.stats['successful_syncs'] += 1
                self.stats['total_rows_processed'] += rows_processed

            finally:
                connection.close()

        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Sync failed for {source_config.name}: {e}")
            self.stats['failed_syncs'] += 1

            # Send alert on failure
            self._send_alert(
                f"Pipeline Alert: Sync Failed - {source_config.name}",
                f"<p>Source: {source_config.name}</p>"
                f"<p>Error: {error_msg}</p>"
                f"<p>Time: {datetime.now()}</p>"
            )

        finally:
            end_time = datetime.now()
            self.stats['total_syncs'] += 1

            # Log to history
            self._log_sync_history(
                source_config.name,
                source_config.table_name or self._sanitize_name(source_config.name),
                start_time,
                end_time,
                rows_processed,
                status,
                error_msg
            )

    def schedule_jobs(self):
        """Schedule all data sources"""
        for source in self.data_sources:
            if not source.enabled:
                continue

            schedule.every(source.schedule_interval).seconds.do(
                self.sync_data_source, source
            )

            self.logger.info(f"Scheduled {source.name} every {source.schedule_interval}s")

    def run(self):
        """Start the pipeline"""
        self.running = True
        self.logger.info("Pipeline started")

        # Initial sync for all sources
        for source in self.data_sources:
            if source.enabled:
                self.sync_data_source(source)

        # Schedule recurring jobs
        self.schedule_jobs()

        # Run scheduler
        while self.running:
            schedule.run_pending()
            time.sleep(1)

    def stop(self):
        """Stop the pipeline"""
        self.running = False
        self.logger.info("Pipeline stopped")

        # Print final stats
        self.logger.info(f"""
        Pipeline Statistics:
        - Total Syncs: {self.stats['total_syncs']}
        - Successful: {self.stats['successful_syncs']}
        - Failed: {self.stats['failed_syncs']}
        - Total Rows: {self.stats['total_rows_processed']}
        """)

    def get_status(self) -> Dict:
        """Get pipeline status"""
        return {
            'running': self.running,
            'total_sources': len(self.data_sources),
            'enabled_sources': sum(1 for s in self.data_sources if s.enabled),
            'stats': self.stats,
            'last_sync_times': {
                s.name: s.last_sync_time.isoformat() if s.last_sync_time else None
                for s in self.data_sources
            }
        }


def main():
    """Example usage with multiple data sources"""

    # Database configuration
    db_config = DatabaseConfig(
        host='localhost',
        user='root',
        password='your_password',
        database='enterprise_db',
        pool_size=10
    )

    # Email alerts configuration
    email_config = EmailConfig(
        enabled=True,
        smtp_server='smtp.gmail.com',
        smtp_port=587,
        sender_email='alerts@company.com',
        sender_password='your_app_password',
        recipient_emails=['admin@company.com', 'team@company.com']
    )

    # Initialize pipeline
    pipeline = RealTimeDataPipeline(db_config, email_config)

    # Data Source 1: REST API (every 5 minutes)
    api_source = DataSourceConfig(
        source_type='api',
        name='sales_api',
        api_url='https://api.company.com/sales',
        api_headers={'Authorization': 'Bearer YOUR_TOKEN'},
        data_format='json',
        table_name='sales_data',
        sync_mode='incremental',
        timestamp_column='created_at',
        schedule_interval=300
    )
    pipeline.add_data_source(api_source)

    # Data Source 2: FTP Server (every 15 minutes)
    ftp_source = DataSourceConfig(
        source_type='ftp',
        name='inventory_ftp',
        ftp_host='ftp.warehouse.com',
        ftp_user='user',
        ftp_password='pass',
        ftp_path='/exports',
        file_pattern='inventory_*.csv',
        data_format='csv',
        table_name='inventory',
        sync_mode='full',
        schedule_interval=900
    )
    pipeline.add_data_source(ftp_source)

    # Data Source 3: SFTP Server (every 10 minutes)
    sftp_source = DataSourceConfig(
        source_type='sftp',
        name='customer_sftp',
        ftp_host='sftp.partner.com',
        ftp_user='sftp_user',
        ftp_password='sftp_pass',
        ftp_path='/customer_data',
        ftp_port=22,
        file_pattern='customers_*.csv',
        data_format='csv',
        table_name='customers',
        sync_mode='upsert',
        primary_key='customer_id',
        schedule_interval=600
    )
    pipeline.add_data_source(sftp_source)

    # Data Source 4: Local Files (every 2 minutes)
    local_source = DataSourceConfig(
        source_type='local',
        name='local_logs',
        file_path='./data',
        file_pattern='logs_*.json',
        data_format='json',
        table_name='application_logs',
        sync_mode='incremental',
        timestamp_column='timestamp',
        schedule_interval=120
    )
    pipeline.add_data_source(local_source)

    # Data Source 5: Another Database (every 30 minutes)
    db_source = DataSourceConfig(
        source_type='database',
        name='legacy_db',
        source_db_config={
            'host': 'legacy.company.com',
            'user': 'readonly',
            'password': 'pass',
            'database': 'legacy_db'
        },
        source_query="""
                     SELECT *
                     FROM orders
                     WHERE updated_at > DATE_SUB(NOW(), INTERVAL 30 MINUTE)
                     """,
        table_name='legacy_orders',
        sync_mode='incremental',
        timestamp_column='updated_at',
        schedule_interval=1800
    )
    pipeline.add_data_source(db_source)

    # Start pipeline in separate thread for graceful shutdown
    import signal

    def signal_handler(sig, frame):
        print("\n\nStopping pipeline...")
        pipeline.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    print("=" * 70)
    print("ENTERPRISE DATA PIPELINE - STARTED")
    print("=" * 70)
    print(f"Active Sources: {len(pipeline.data_sources)}")
    print("Press Ctrl+C to stop\n")

    # Run pipeline
    try:
        pipeline.run()
    except KeyboardInterrupt:
        pipeline.stop()


if __name__ == "__main__":
    main()