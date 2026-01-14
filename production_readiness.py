#!/usr/bin/env python3
"""
AEGIS Framework - Production Readiness Check
Verificación completa de preparación para despliegue en producción
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

class ProductionReadinessChecker:
    """Verificador de preparación para producción"""

    def __init__(self):
        self.issues = []
        self.warnings = []
        self.passed_checks = []

    def check(self, description: str, condition: bool, error_msg: str = "", warning_msg: str = ""):
        """Verificar una condición y registrar el resultado"""
        if condition:
            self.passed_checks.append(f"✅ {description}")
            return True
        else:
            if error_msg:
                self.issues.append(f"❌ {description}: {error_msg}")
            elif warning_msg:
                self.warnings.append(f"⚠️ {description}: {warning_msg}")
            return False

    def run_all_checks(self) -> bool:
        """Ejecutar todas las verificaciones de preparación"""
        print("🔍 AEGIS Framework - Verificación de Preparación para Producción")
        print("=" * 70)

        # Verificaciones de sistema
        self._check_system_requirements()
        self._check_directory_structure()
        self._check_configuration_files()
        self._check_security_configuration()
        self._check_docker_setup()
        self._check_networking()
        self._check_monitoring_setup()

        # Mostrar resultados
        self._print_results()

        # Retornar si está listo para producción
        return len(self.issues) == 0

    def _check_system_requirements(self):
        """Verificar requisitos del sistema"""
        print("\n📋 Verificando requisitos del sistema...")

        # Python version
        python_version = sys.version_info
        self.check(
            "Python 3.11+ instalado",
            python_version >= (3, 11),
            f"Python {python_version.major}.{python_version.minor} encontrado, se requiere 3.11+"
        )

        # Docker disponible
        try:
            result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
            self.check("Docker instalado", result.returncode == 0)
        except FileNotFoundError:
            self.check("Docker instalado", False, "Docker no está instalado")

        # Docker Compose disponible
        try:
            result = subprocess.run(["docker-compose", "--version"], capture_output=True, text=True)
            docker_compose_ok = result.returncode == 0
        except FileNotFoundError:
            docker_compose_ok = False

        if not docker_compose_ok:
            try:
                result = subprocess.run(["docker", "compose", "version"], capture_output=True, text=True)
                docker_compose_ok = result.returncode == 0
            except FileNotFoundError:
                docker_compose_ok = False

        self.check("Docker Compose disponible", docker_compose_ok, "Docker Compose no está disponible")

        # OpenSSL disponible
        try:
            result = subprocess.run(["openssl", "version"], capture_output=True, text=True)
            self.check("OpenSSL instalado", result.returncode == 0)
        except FileNotFoundError:
            self.check("OpenSSL instalado", False, "OpenSSL no está instalado")

    def _check_directory_structure(self):
        """Verificar estructura de directorios"""
        print("\n📁 Verificando estructura de directorios...")

        required_dirs = [
            "config",
            "data",
            "logs",
            "backups",
            "secrets",
            "certs",
            "keys"
        ]

        for dir_name in required_dirs:
            exists = os.path.isdir(dir_name)
            self.check(f"Directorio {dir_name} existe", exists, f"Crear directorio {dir_name}")

    def _check_configuration_files(self):
        """Verificar archivos de configuración"""
        print("\n⚙️ Verificando archivos de configuración...")

        config_files = [
            ("config/production_config.json", "Configuración de producción"),
            ("docker-compose.prod.yml", "Docker Compose de producción"),
            ("pyproject.toml", "Configuración del proyecto Python")
        ]

        for file_path, description in config_files:
            exists = os.path.isfile(file_path)
            self.check(f"{description} existe", exists, f"Crear archivo {file_path}")

        # Verificar configuración de producción
        if os.path.isfile("config/production_config.json"):
            try:
                with open("config/production_config.json", "r") as f:
                    config = json.load(f)

                # Verificar campos críticos
                required_fields = [
                    "app.node_id",
                    "crypto.security_level",
                    "consensus.algorithm",
                    "state_persistence.replication_factor"
                ]

                for field_path in required_fields:
                    keys = field_path.split(".")
                    value = config
                    try:
                        for key in keys:
                            value = value[key]
                        self.check(f"Configuración {field_path} presente", True)
                    except (KeyError, TypeError):
                        self.check(f"Configuración {field_path} presente", False, f"Campo {field_path} faltante en configuración")

            except json.JSONDecodeError as e:
                self.check("Configuración JSON válida", False, f"Error de sintaxis JSON: {e}")

    def _check_security_configuration(self):
        """Verificar configuración de seguridad"""
        print("\n🔒 Verificando configuración de seguridad...")

        # Verificar secrets
        secret_files = [
            "secrets/postgres_password.txt",
            "secrets/grafana_password.txt"
        ]

        for secret_file in secret_files:
            exists = os.path.isfile(secret_file)
            if exists:
                # Verificar permisos
                stat_info = os.stat(secret_file)
                permissions = oct(stat_info.st_mode)[-3:]
                secure_perms = permissions in ["600", "400"]
                self.check(f"Archivo secreto {secret_file} tiene permisos seguros", secure_perms,
                          f"Permisos actuales: {permissions}, deberían ser 600 o 400")
            else:
                self.check(f"Archivo secreto {secret_file} existe", False,
                          f"Generar secreto con: openssl rand -base64 32 > {secret_file}")

        # Verificar certificados SSL
        ssl_files = ["certs/aegis.crt", "certs/aegis.key"]
        for ssl_file in ssl_files:
            exists = os.path.isfile(ssl_file)
            self.check(f"Certificado SSL {ssl_file} existe", exists,
                      f"Generar certificado SSL o configurar ruta correcta")

    def _check_docker_setup(self):
        """Verificar configuración de Docker"""
        print("\n🐳 Verificando configuración de Docker...")

        compose_file = "docker-compose.prod.yml"
        if os.path.isfile(compose_file):
            try:
                result = subprocess.run(["docker-compose", "-f", compose_file, "config", "--quiet"],
                                      capture_output=True, text=True)
                self.check("Docker Compose válido", result.returncode == 0,
                          f"Error en docker-compose.prod.yml: {result.stderr}")

                # Contar servicios
                result = subprocess.run(["docker-compose", "-f", compose_file, "config", "--services"],
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    services = result.stdout.strip().split("\n")
                    self.check("Número adecuado de servicios", len(services) >= 8,
                              f"Encontrados {len(services)} servicios, se esperan al menos 8")

            except FileNotFoundError:
                self.check("Docker Compose disponible para validación", False,
                          "docker-compose no está en PATH")
        else:
            self.check("Archivo docker-compose.prod.yml existe", False)

    def _check_networking(self):
        """Verificar configuración de red"""
        print("\n🌐 Verificando configuración de red...")

        # Verificar puertos disponibles (básico)
        import socket
        ports_to_check = [8443, 8444, 3000, 9090]

        for port in ports_to_check:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(("127.0.0.1", port))
            port_free = result != 0
            sock.close()

            if port_free:
                self.check(f"Puerto {port} disponible", True)
            else:
                self.check(f"Puerto {port} disponible", False,
                          f"Puerto {port} en uso - liberar o cambiar configuración",
                          "Puerto en uso, pero puede ser liberado durante despliegue")

    def _check_monitoring_setup(self):
        """Verificar configuración de monitoreo"""
        print("\n📊 Verificando configuración de monitoreo...")

        monitoring_files = [
            "config/prometheus.prod.yml",
            "config/nginx/nginx.prod.conf"
        ]

        for file_path in monitoring_files:
            exists = os.path.isfile(file_path)
            self.check(f"Archivo de monitoreo {file_path} existe", exists,
                      f"Crear archivo de configuración {file_path}")

        # Verificar que el script de despliegue existe y es ejecutable
        deploy_script = "deploy-production.sh"
        exists = os.path.isfile(deploy_script)
        if exists:
            executable = os.access(deploy_script, os.X_OK)
            self.check("Script de despliegue ejecutable", executable,
                      f"Hacer ejecutable: chmod +x {deploy_script}")
        else:
            self.check(f"Script de despliegue {deploy_script} existe", False)

    def _print_results(self):
        """Imprimir resultados de la verificación"""
        print("\n" + "=" * 70)
        print("📊 RESULTADOS DE LA VERIFICACIÓN")
        print("=" * 70)

        # Checks exitosos
        if self.passed_checks:
            print(f"\n✅ VERIFICACIONES EXITOSAS ({len(self.passed_checks)}):")
            for check in self.passed_checks:
                print(f"   {check}")

        # Advertencias
        if self.warnings:
            print(f"\n⚠️ ADVERTENCIAS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"   {warning}")

        # Problemas críticos
        if self.issues:
            print(f"\n❌ PROBLEMAS CRÍTICOS ({len(self.issues)}):")
            for issue in self.issues:
                print(f"   {issue}")

        # Resumen final
        print(f"\n📈 RESUMEN:")
        print(f"   ✅ Checks exitosos: {len(self.passed_checks)}")
        print(f"   ⚠️ Advertencias: {len(self.warnings)}")
        print(f"   ❌ Problemas críticos: {len(self.issues)}")

        if len(self.issues) == 0:
            print("\n🎉 ¡SISTEMA LISTO PARA PRODUCCIÓN!")
            if len(self.warnings) == 0:
                print("   No hay problemas ni advertencias.")
            else:
                print(f"   Revisar {len(self.warnings)} advertencias antes del despliegue.")
        else:
            print(f"\n🚫 ¡CORREGIR {len(self.issues)} PROBLEMAS CRÍTICOS ANTES DEL DESPLIEGUE!")
            print("   Ejecutar: python production_readiness.py --fix")

def main():
    """Función principal"""
    checker = ProductionReadinessChecker()
    ready = checker.run_all_checks()

    if not ready:
        print("\n💡 Para corregir problemas automáticamente:")
        print("   python production_readiness.py --fix")
        sys.exit(1)
    else:
        print("\n🚀 Listo para despliegue:")
        print("   ./deploy-production.sh")
        sys.exit(0)

if __name__ == "__main__":
    main()
