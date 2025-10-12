#!/usr/bin/env python3
"""
Sistema de Backup Automático para AEGIS
Implementa respaldos incrementales, completos y en la nube
"""

import os
import json
import asyncio
import hashlib
import tarfile
import zipfile
import shutil
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging

try:
    from cryptography.fernet import Fernet
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

try:
    import boto3
    from botocore.exceptions import ClientError
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False

# Configurar logging
logger = logging.getLogger(__name__)

class BackupType(Enum):
    """Tipos de backup disponibles"""
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"
    SNAPSHOT = "snapshot"

class BackupStatus(Enum):
    """Estados de backup"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class CompressionType(Enum):
    """Tipos de compresión"""
    NONE = "none"
    ZIP = "zip"
    TAR_GZ = "tar.gz"
    TAR_BZ2 = "tar.bz2"

@dataclass
class BackupJob:
    """Definición de un trabajo de backup"""
    id: str
    name: str
    backup_type: BackupType
    source_paths: List[str]
    destination: str
    schedule: str  # Cron-like expression
    compression: CompressionType
    encryption: bool
    retention_days: int
    exclude_patterns: List[str]
    created_at: datetime
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    status: BackupStatus = BackupStatus.PENDING

@dataclass
class BackupResult:
    """Resultado de un backup"""
    job_id: str
    backup_type: BackupType
    start_time: datetime
    end_time: Optional[datetime]
    status: BackupStatus
    file_path: Optional[str]
    file_size: int
    files_count: int
    compression_ratio: float
    error_message: Optional[str] = None
    checksum: Optional[str] = None

class FileHasher:
    """Utilidad para calcular hashes de archivos"""
    
    @staticmethod
    def calculate_file_hash(file_path: str, algorithm: str = "sha256") -> str:
        """Calcula el hash de un archivo"""
        hash_obj = hashlib.new(algorithm)
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_obj.update(chunk)
            return hash_obj.hexdigest()
        except Exception as e:
            logger.error(f"Error calculando hash de {file_path}: {e}")
            return ""

    @staticmethod
    def get_directory_manifest(directory: str) -> Dict[str, str]:
        """Genera un manifiesto con hashes de todos los archivos"""
        manifest = {}
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, directory)
                manifest[relative_path] = FileHasher.calculate_file_hash(file_path)
        return manifest

class EncryptionManager:
    """Gestor de encriptación para backups"""
    
    def __init__(self, key_path: str):
        self.key_path = key_path
        self._key = None
        self._fernet = None
        
    def _load_or_generate_key(self) -> bytes:
        """Carga o genera una clave de encriptación"""
        if not CRYPTO_AVAILABLE:
            raise RuntimeError("Cryptography no está disponible")
            
        if os.path.exists(self.key_path):
            with open(self.key_path, 'rb') as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            os.makedirs(os.path.dirname(self.key_path), exist_ok=True)
            with open(self.key_path, 'wb') as f:
                f.write(key)
            return key
    
    def get_fernet(self) -> Fernet:
        """Obtiene instancia de Fernet para encriptación"""
        if not self._fernet:
            if not self._key:
                self._key = self._load_or_generate_key()
            self._fernet = Fernet(self._key)
        return self._fernet
    
    def encrypt_file(self, input_path: str, output_path: str) -> bool:
        """Encripta un archivo"""
        try:
            fernet = self.get_fernet()
            with open(input_path, 'rb') as infile:
                data = infile.read()
            
            encrypted_data = fernet.encrypt(data)
            
            with open(output_path, 'wb') as outfile:
                outfile.write(encrypted_data)
            
            return True
        except Exception as e:
            logger.error(f"Error encriptando {input_path}: {e}")
            return False
    
    def decrypt_file(self, input_path: str, output_path: str) -> bool:
        """Desencripta un archivo"""
        try:
            fernet = self.get_fernet()
            with open(input_path, 'rb') as infile:
                encrypted_data = infile.read()
            
            decrypted_data = fernet.decrypt(encrypted_data)
            
            with open(output_path, 'wb') as outfile:
                outfile.write(decrypted_data)
            
            return True
        except Exception as e:
            logger.error(f"Error desencriptando {input_path}: {e}")
            return False

class CompressionManager:
    """Gestor de compresión para backups"""
    
    @staticmethod
    def compress_directory(source_dir: str, output_path: str, 
                          compression: CompressionType, 
                          exclude_patterns: List[str] = None) -> tuple[bool, int, int]:
        """
        Comprime un directorio
        Returns: (success, original_size, compressed_size)
        """
        exclude_patterns = exclude_patterns or []
        
        def should_exclude(path: str) -> bool:
            for pattern in exclude_patterns:
                if pattern in path:
                    return True
            return False
        
        try:
            original_size = 0
            files_added = 0
            
            if compression == CompressionType.ZIP:
                with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for root, dirs, files in os.walk(source_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            if not should_exclude(file_path):
                                arcname = os.path.relpath(file_path, source_dir)
                                zipf.write(file_path, arcname)
                                original_size += os.path.getsize(file_path)
                                files_added += 1
            
            elif compression in [CompressionType.TAR_GZ, CompressionType.TAR_BZ2]:
                mode = 'w:gz' if compression == CompressionType.TAR_GZ else 'w:bz2'
                with tarfile.open(output_path, mode) as tar:
                    for root, dirs, files in os.walk(source_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            if not should_exclude(file_path):
                                arcname = os.path.relpath(file_path, source_dir)
                                tar.add(file_path, arcname)
                                original_size += os.path.getsize(file_path)
                                files_added += 1
            
            elif compression == CompressionType.NONE:
                # Copia simple sin compresión
                shutil.copytree(source_dir, output_path, 
                              ignore=lambda dir, files: [f for f in files 
                                                        if should_exclude(os.path.join(dir, f))])
                for root, dirs, files in os.walk(output_path):
                    for file in files:
                        original_size += os.path.getsize(os.path.join(root, file))
                        files_added += 1
            
            compressed_size = os.path.getsize(output_path) if compression != CompressionType.NONE else original_size
            return True, original_size, compressed_size
            
        except Exception as e:
            logger.error(f"Error comprimiendo {source_dir}: {e}")
            return False, 0, 0

class CloudStorageManager:
    """Gestor de almacenamiento en la nube"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.s3_client = None
        
    def _get_s3_client(self):
        """Obtiene cliente S3"""
        if not AWS_AVAILABLE:
            raise RuntimeError("boto3 no está disponible")
            
        if not self.s3_client:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=self.config.get('aws_access_key'),
                aws_secret_access_key=self.config.get('aws_secret_key'),
                region_name=self.config.get('aws_region', 'us-east-1')
            )
        return self.s3_client
    
    def upload_to_s3(self, local_path: str, bucket: str, key: str) -> bool:
        """Sube archivo a S3"""
        try:
            s3 = self._get_s3_client()
            s3.upload_file(local_path, bucket, key)
            logger.info(f"Archivo subido a S3: s3://{bucket}/{key}")
            return True
        except Exception as e:
            logger.error(f"Error subiendo a S3: {e}")
            return False
    
    def download_from_s3(self, bucket: str, key: str, local_path: str) -> bool:
        """Descarga archivo de S3"""
        try:
            s3 = self._get_s3_client()
            s3.download_file(bucket, key, local_path)
            logger.info(f"Archivo descargado de S3: s3://{bucket}/{key}")
            return True
        except Exception as e:
            logger.error(f"Error descargando de S3: {e}")
            return False

class BackupDatabase:
    """Base de datos para gestionar backups"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Inicializa la base de datos"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS backup_jobs (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    backup_type TEXT NOT NULL,
                    source_paths TEXT NOT NULL,
                    destination TEXT NOT NULL,
                    schedule_expr TEXT NOT NULL,
                    compression TEXT NOT NULL,
                    encryption BOOLEAN NOT NULL,
                    retention_days INTEGER NOT NULL,
                    exclude_patterns TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_run TEXT,
                    next_run TEXT,
                    status TEXT NOT NULL
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS backup_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    backup_type TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    status TEXT NOT NULL,
                    file_path TEXT,
                    file_size INTEGER NOT NULL,
                    files_count INTEGER NOT NULL,
                    compression_ratio REAL NOT NULL,
                    error_message TEXT,
                    checksum TEXT,
                    FOREIGN KEY (job_id) REFERENCES backup_jobs (id)
                )
            ''')
    
    def save_job(self, job: BackupJob):
        """Guarda un trabajo de backup"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO backup_jobs 
                (id, name, backup_type, source_paths, destination, schedule_expr,
                 compression, encryption, retention_days, exclude_patterns,
                 created_at, last_run, next_run, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                job.id, job.name, job.backup_type.value,
                json.dumps(job.source_paths), job.destination, job.schedule,
                job.compression.value, job.encryption, job.retention_days,
                json.dumps(job.exclude_patterns), job.created_at.isoformat(),
                job.last_run.isoformat() if job.last_run else None,
                job.next_run.isoformat() if job.next_run else None,
                job.status.value
            ))
    
    def get_job(self, job_id: str) -> Optional[BackupJob]:
        """Obtiene un trabajo por ID"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('SELECT * FROM backup_jobs WHERE id = ?', (job_id,))
            row = cursor.fetchone()
            
            if row:
                return BackupJob(
                    id=row[0], name=row[1], backup_type=BackupType(row[2]),
                    source_paths=json.loads(row[3]), destination=row[4],
                    schedule=row[5], compression=CompressionType(row[6]),
                    encryption=bool(row[7]), retention_days=row[8],
                    exclude_patterns=json.loads(row[9]),
                    created_at=datetime.fromisoformat(row[10]),
                    last_run=datetime.fromisoformat(row[11]) if row[11] else None,
                    next_run=datetime.fromisoformat(row[12]) if row[12] else None,
                    status=BackupStatus(row[13])
                )
        return None
    
    def get_all_jobs(self) -> List[BackupJob]:
        """Obtiene todos los trabajos"""
        jobs = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('SELECT * FROM backup_jobs')
            for row in cursor.fetchall():
                jobs.append(BackupJob(
                    id=row[0], name=row[1], backup_type=BackupType(row[2]),
                    source_paths=json.loads(row[3]), destination=row[4],
                    schedule=row[5], compression=CompressionType(row[6]),
                    encryption=bool(row[7]), retention_days=row[8],
                    exclude_patterns=json.loads(row[9]),
                    created_at=datetime.fromisoformat(row[10]),
                    last_run=datetime.fromisoformat(row[11]) if row[11] else None,
                    next_run=datetime.fromisoformat(row[12]) if row[12] else None,
                    status=BackupStatus(row[13])
                ))
        return jobs
    
    def save_result(self, result: BackupResult):
        """Guarda resultado de backup"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO backup_results 
                (job_id, backup_type, start_time, end_time, status, file_path,
                 file_size, files_count, compression_ratio, error_message, checksum)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result.job_id, result.backup_type.value, result.start_time.isoformat(),
                result.end_time.isoformat() if result.end_time else None,
                result.status.value, result.file_path, result.file_size,
                result.files_count, result.compression_ratio,
                result.error_message, result.checksum
            ))

class BackupScheduler:
    """Programador de tareas de backup"""
    
    def __init__(self):
        self.running = False
        self.tasks = {}
    
    def parse_schedule(self, schedule_expr: str) -> int:
        """
        Parsea expresión de programación simple
        Formatos soportados:
        - "daily" -> cada 24 horas
        - "hourly" -> cada hora
        - "weekly" -> cada 7 días
        - "30m" -> cada 30 minutos
        - "2h" -> cada 2 horas
        """
        schedule_expr = schedule_expr.lower().strip()
        
        if schedule_expr == "daily":
            return 24 * 60 * 60
        elif schedule_expr == "hourly":
            return 60 * 60
        elif schedule_expr == "weekly":
            return 7 * 24 * 60 * 60
        elif schedule_expr.endswith('m'):
            return int(schedule_expr[:-1]) * 60
        elif schedule_expr.endswith('h'):
            return int(schedule_expr[:-1]) * 60 * 60
        elif schedule_expr.endswith('d'):
            return int(schedule_expr[:-1]) * 24 * 60 * 60
        else:
            # Default: daily
            return 24 * 60 * 60
    
    def calculate_next_run(self, job: BackupJob) -> datetime:
        """Calcula próxima ejecución"""
        interval_seconds = self.parse_schedule(job.schedule)
        
        if job.last_run:
            return job.last_run + timedelta(seconds=interval_seconds)
        else:
            return datetime.now() + timedelta(seconds=interval_seconds)
    
    def should_run_now(self, job: BackupJob) -> bool:
        """Determina si un trabajo debe ejecutarse ahora"""
        if not job.next_run:
            return True
        return datetime.now() >= job.next_run

class AEGISBackupSystem:
    """Sistema principal de backup para AEGIS"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_path = config.get("base_path", "aegis_backups")
        self.db_path = os.path.join(self.base_path, "backup_system.db")
        
        # Inicializar componentes
        self.database = BackupDatabase(self.db_path)
        self.scheduler = BackupScheduler()
        self.encryption_manager = None
        self.cloud_manager = None
        
        # Estado del sistema
        self.running = False
        self.current_jobs = {}
        
        # Configurar encriptación si está habilitada
        if config.get("encryption", {}).get("enabled", False):
            key_path = os.path.join(self.base_path, "encryption.key")
            self.encryption_manager = EncryptionManager(key_path)
        
        # Configurar almacenamiento en la nube si está habilitado
        if config.get("cloud_storage", {}).get("enabled", False):
            self.cloud_manager = CloudStorageManager(config.get("cloud_storage", {}))
        
        # Crear directorios necesarios
        os.makedirs(self.base_path, exist_ok=True)
        
        logger.info("Sistema de backup AEGIS inicializado")
    
    def create_backup_job(self, name: str, source_paths: List[str], 
                         backup_type: BackupType = BackupType.INCREMENTAL,
                         schedule: str = "daily",
                         compression: CompressionType = CompressionType.TAR_GZ,
                         encryption: bool = False,
                         retention_days: int = 30,
                         exclude_patterns: List[str] = None) -> str:
        """Crea un nuevo trabajo de backup"""
        
        job_id = hashlib.md5(f"{name}_{datetime.now().isoformat()}".encode()).hexdigest()[:12]
        destination = os.path.join(self.base_path, "jobs", job_id)
        
        job = BackupJob(
            id=job_id,
            name=name,
            backup_type=backup_type,
            source_paths=source_paths,
            destination=destination,
            schedule=schedule,
            compression=compression,
            encryption=encryption,
            retention_days=retention_days,
            exclude_patterns=exclude_patterns or [],
            created_at=datetime.now()
        )
        
        # Calcular próxima ejecución
        job.next_run = self.scheduler.calculate_next_run(job)
        
        # Guardar en base de datos
        self.database.save_job(job)
        
        logger.info(f"Trabajo de backup creado: {name} (ID: {job_id})")
        return job_id
    
    async def execute_backup_job(self, job_id: str) -> BackupResult:
        """Ejecuta un trabajo de backup"""
        job = self.database.get_job(job_id)
        if not job:
            raise ValueError(f"Trabajo no encontrado: {job_id}")
        
        logger.info(f"Iniciando backup: {job.name}")
        
        # Crear resultado inicial
        result = BackupResult(
            job_id=job_id,
            backup_type=job.backup_type,
            start_time=datetime.now(),
            end_time=None,
            status=BackupStatus.RUNNING,
            file_path=None,
            file_size=0,
            files_count=0,
            compression_ratio=0.0
        )
        
        try:
            # Actualizar estado del trabajo
            job.status = BackupStatus.RUNNING
            job.last_run = result.start_time
            self.database.save_job(job)
            
            # Crear directorio de destino
            os.makedirs(job.destination, exist_ok=True)
            
            # Generar nombre del archivo de backup
            timestamp = result.start_time.strftime("%Y%m%d_%H%M%S")
            backup_filename = f"{job.name}_{job.backup_type.value}_{timestamp}"
            
            if job.compression != CompressionType.NONE:
                if job.compression == CompressionType.ZIP:
                    backup_filename += ".zip"
                elif job.compression == CompressionType.TAR_GZ:
                    backup_filename += ".tar.gz"
                elif job.compression == CompressionType.TAR_BZ2:
                    backup_filename += ".tar.bz2"
            
            backup_path = os.path.join(job.destination, backup_filename)
            
            # Ejecutar backup según el tipo
            if job.backup_type == BackupType.FULL:
                success, original_size, compressed_size = await self._execute_full_backup(
                    job, backup_path)
            elif job.backup_type == BackupType.INCREMENTAL:
                success, original_size, compressed_size = await self._execute_incremental_backup(
                    job, backup_path)
            else:
                # Por ahora, tratar otros tipos como full backup
                success, original_size, compressed_size = await self._execute_full_backup(
                    job, backup_path)
            
            if success:
                # Calcular checksum
                result.checksum = FileHasher.calculate_file_hash(backup_path)
                result.file_path = backup_path
                result.file_size = compressed_size
                result.compression_ratio = compressed_size / original_size if original_size > 0 else 0
                
                # Encriptar si está habilitado
                if job.encryption and self.encryption_manager:
                    encrypted_path = backup_path + ".enc"
                    if self.encryption_manager.encrypt_file(backup_path, encrypted_path):
                        os.remove(backup_path)  # Eliminar archivo sin encriptar
                        result.file_path = encrypted_path
                        result.file_size = os.path.getsize(encrypted_path)
                
                # Subir a la nube si está configurado
                if self.cloud_manager and self.config.get("cloud_storage", {}).get("auto_upload", False):
                    bucket = self.config["cloud_storage"]["bucket"]
                    key = f"aegis_backups/{job_id}/{os.path.basename(result.file_path)}"
                    self.cloud_manager.upload_to_s3(result.file_path, bucket, key)
                
                result.status = BackupStatus.COMPLETED
                job.status = BackupStatus.COMPLETED
                
                logger.info(f"Backup completado: {job.name} -> {result.file_path}")
            else:
                result.status = BackupStatus.FAILED
                result.error_message = "Error durante la compresión"
                job.status = BackupStatus.FAILED
        
        except Exception as e:
            result.status = BackupStatus.FAILED
            result.error_message = str(e)
            job.status = BackupStatus.FAILED
            logger.error(f"Error en backup {job.name}: {e}")
        
        finally:
            result.end_time = datetime.now()
            
            # Calcular próxima ejecución
            job.next_run = self.scheduler.calculate_next_run(job)
            
            # Guardar resultados
            self.database.save_job(job)
            self.database.save_result(result)
        
        return result
    
    async def _execute_full_backup(self, job: BackupJob, output_path: str) -> tuple[bool, int, int]:
        """Ejecuta backup completo"""
        # Crear directorio temporal para consolidar archivos
        temp_dir = os.path.join(job.destination, "temp_full")
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            # Copiar todos los archivos fuente al directorio temporal
            for source_path in job.source_paths:
                if os.path.isfile(source_path):
                    dest_file = os.path.join(temp_dir, os.path.basename(source_path))
                    shutil.copy2(source_path, dest_file)
                elif os.path.isdir(source_path):
                    dest_dir = os.path.join(temp_dir, os.path.basename(source_path))
                    shutil.copytree(source_path, dest_dir, 
                                  ignore=lambda dir, files: [f for f in files 
                                                           if any(pattern in os.path.join(dir, f) 
                                                                 for pattern in job.exclude_patterns)])
            
            # Comprimir directorio temporal
            success, original_size, compressed_size = CompressionManager.compress_directory(
                temp_dir, output_path, job.compression, job.exclude_patterns)
            
            return success, original_size, compressed_size
            
        finally:
            # Limpiar directorio temporal
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
    
    async def _execute_incremental_backup(self, job: BackupJob, output_path: str) -> tuple[bool, int, int]:
        """Ejecuta backup incremental"""
        # Para backup incremental, necesitamos comparar con el último backup
        # Por simplicidad, implementamos como backup completo por ahora
        # En una implementación completa, compararíamos manifiestos de archivos
        return await self._execute_full_backup(job, output_path)
    
    async def cleanup_old_backups(self):
        """Limpia backups antiguos según políticas de retención"""
        jobs = self.database.get_all_jobs()
        
        for job in jobs:
            cutoff_date = datetime.now() - timedelta(days=job.retention_days)
            
            # Buscar archivos de backup antiguos
            if os.path.exists(job.destination):
                for file in os.listdir(job.destination):
                    file_path = os.path.join(job.destination, file)
                    if os.path.isfile(file_path):
                        file_time = datetime.fromtimestamp(os.path.getctime(file_path))
                        if file_time < cutoff_date:
                            try:
                                os.remove(file_path)
                                logger.info(f"Backup antiguo eliminado: {file_path}")
                            except Exception as e:
                                logger.error(f"Error eliminando backup antiguo {file_path}: {e}")
    
    async def start_scheduler(self):
        """Inicia el programador de backups"""
        self.running = True
        logger.info("Programador de backups iniciado")
        
        while self.running:
            try:
                jobs = self.database.get_all_jobs()
                
                for job in jobs:
                    if job.status != BackupStatus.RUNNING and self.scheduler.should_run_now(job):
                        # Ejecutar backup en background
                        asyncio.create_task(self.execute_backup_job(job.id))
                
                # Limpiar backups antiguos cada hora
                if datetime.now().minute == 0:
                    await self.cleanup_old_backups()
                
                await asyncio.sleep(60)  # Verificar cada minuto
                
            except Exception as e:
                logger.error(f"Error en programador de backups: {e}")
                await asyncio.sleep(60)
    
    def stop_scheduler(self):
        """Detiene el programador de backups"""
        self.running = False
        logger.info("Programador de backups detenido")
    
    def get_backup_status(self) -> Dict[str, Any]:
        """Obtiene estado del sistema de backup"""
        jobs = self.database.get_all_jobs()
        
        status = {
            "total_jobs": len(jobs),
            "active_jobs": len([j for j in jobs if j.status != BackupStatus.FAILED]),
            "running_jobs": len([j for j in jobs if j.status == BackupStatus.RUNNING]),
            "failed_jobs": len([j for j in jobs if j.status == BackupStatus.FAILED]),
            "scheduler_running": self.running,
            "storage_path": self.base_path,
            "encryption_enabled": self.encryption_manager is not None,
            "cloud_storage_enabled": self.cloud_manager is not None
        }
        
        return status
    
    def list_backup_jobs(self) -> List[Dict[str, Any]]:
        """Lista todos los trabajos de backup"""
        jobs = self.database.get_all_jobs()
        return [asdict(job) for job in jobs]

# Instancia global del sistema de backup
_backup_system: Optional[AEGISBackupSystem] = None

async def start_backup_system(config: Dict[str, Any]):
    """Inicia el sistema de backup"""
    global _backup_system
    
    if _backup_system:
        logger.warning("Sistema de backup ya está iniciado")
        return
    
    _backup_system = AEGISBackupSystem(config)
    
    # Crear trabajos de backup por defecto para AEGIS
    default_jobs = [
        {
            "name": "AEGIS_Config_Backup",
            "source_paths": ["aegis_config", "config"],
            "backup_type": BackupType.INCREMENTAL,
            "schedule": "6h",
            "compression": CompressionType.TAR_GZ,
            "encryption": True,
            "retention_days": 30,
            "exclude_patterns": ["*.tmp", "*.log"]
        },
        {
            "name": "AEGIS_Logs_Backup",
            "source_paths": ["aegis_logs"],
            "backup_type": BackupType.FULL,
            "schedule": "daily",
            "compression": CompressionType.TAR_BZ2,
            "encryption": False,
            "retention_days": 7,
            "exclude_patterns": ["*.lock"]
        },
        {
            "name": "AEGIS_Data_Backup",
            "source_paths": ["aegis_metrics", "aegis_alerts"],
            "backup_type": BackupType.INCREMENTAL,
            "schedule": "12h",
            "compression": CompressionType.ZIP,
            "encryption": True,
            "retention_days": 60,
            "exclude_patterns": ["*.tmp", "*.cache"]
        }
    ]
    
    # Crear trabajos por defecto si no existen
    existing_jobs = {job.name for job in _backup_system.database.get_all_jobs()}
    
    for job_config in default_jobs:
        if job_config["name"] not in existing_jobs:
            _backup_system.create_backup_job(**job_config)
    
    # Iniciar programador
    asyncio.create_task(_backup_system.start_scheduler())
    
    logger.info("Sistema de backup AEGIS iniciado exitosamente")

async def stop_backup_system():
    """Detiene el sistema de backup"""
    global _backup_system
    
    if _backup_system:
        _backup_system.stop_scheduler()
        _backup_system = None
        logger.info("Sistema de backup AEGIS detenido")

def get_backup_system() -> Optional[AEGISBackupSystem]:
    """Obtiene la instancia del sistema de backup"""
    return _backup_system

# Decorador para backup automático de funciones críticas
def backup_on_change(backup_paths: List[str], job_name: str = None):
    """Decorador que crea backup cuando se modifica contenido crítico"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)
            
            # Crear backup automático si el sistema está disponible
            if _backup_system:
                name = job_name or f"Auto_Backup_{func.__name__}"
                try:
                    await _backup_system.execute_backup_job(
                        _backup_system.create_backup_job(
                            name=name,
                            source_paths=backup_paths,
                            backup_type=BackupType.INCREMENTAL,
                            schedule="manual",
                            compression=CompressionType.TAR_GZ,
                            encryption=True,
                            retention_days=7
                        )
                    )
                except Exception as e:
                    logger.error(f"Error en backup automático: {e}")
            
            return result
        return wrapper
    return decorator

if __name__ == "__main__":
    # Configuración de ejemplo
    config = {
        "base_path": "aegis_backups",
        "encryption": {"enabled": True},
        "cloud_storage": {
            "enabled": False,
            "provider": "aws_s3",
            "bucket": "aegis-backups",
            "auto_upload": True,
            "aws_access_key": "",
            "aws_secret_key": "",
            "aws_region": "us-east-1"
        }
    }
    
    async def test_backup():
        await start_backup_system(config)
        
        # Esperar un poco para ver el sistema en acción
        await asyncio.sleep(10)
        
        # Mostrar estado
        if _backup_system:
            status = _backup_system.get_backup_status()
            print(f"Estado del sistema: {status}")
            
            jobs = _backup_system.list_backup_jobs()
            print(f"Trabajos configurados: {len(jobs)}")
        
        await stop_backup_system()
    
    asyncio.run(test_backup())