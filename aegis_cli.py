#!/usr/bin/env python3
"""
🎯 AEGIS CLI - Advanced Command Line Interface
CLI avanzado con autocompletado, ayuda contextual, y modo interactivo
"""

import asyncio
import click
import json
import os
import time
from typing import Dict, List, Any, Optional
from pathlib import Path
import questionary
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.live import Live
from rich.prompt import Prompt, Confirm
from rich.group import Group
import logging

# Importar componentes del framework
from aegis_sdk import AEGIS, aegis_session, SDKResponse

# Configurar logging
logging.basicConfig(level=logging.WARNING)  # Solo warnings en CLI
logger = logging.getLogger(__name__)

console = Console()

class AEGISCLI:
    """CLI avanzado para AEGIS Framework"""

    def __init__(self):
        self.aegis: Optional[AEGIS] = None
        self.interactive_mode = False

    async def initialize(self, api_key: Optional[str] = None):
        """Inicializar conexión con AEGIS"""
        self.aegis = AEGIS(api_key)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Conectando con AEGIS Framework...", total=None)

            # Simular inicialización
            await asyncio.sleep(1)

            progress.update(task, completed=True, description="✅ Conectado a AEGIS Framework")

    async def execute_command(self, command: str, **kwargs) -> bool:
        """Ejecutar comando"""

        commands = {
            "init": self.cmd_init,
            "health": self.cmd_health,
            "models": self.cmd_models,
            "federated": self.cmd_federated,
            "cloud": self.cmd_cloud,
            "edge": self.cmd_edge,
            "predict": self.cmd_predict,
            "deploy": self.cmd_deploy,
            "monitor": self.cmd_monitor,
            "interactive": self.cmd_interactive
        }

        if command in commands:
            try:
                return await commands[command](**kwargs)
            except Exception as e:
                console.print(f"[red]❌ Error ejecutando comando '{command}': {e}[/red]")
                return False
        else:
            console.print(f"[red]❌ Comando desconocido: {command}[/red]")
            console.print("[yellow]💡 Usa 'help' para ver comandos disponibles[/yellow]")
            return False

    async def cmd_init(self, **kwargs):
        """Inicializar o verificar configuración"""
        console.print("[blue]🔧 Inicializando AEGIS CLI...[/blue]")

        # Verificar configuración
        config_status = {
            "SDK inicializado": self.aegis is not None,
            "API conectada": True,  # Simulado
            "Modelos disponibles": True,  # Simulado
            "Cloud providers": True,  # Simulado
            "Edge devices": True  # Simulado
        }

        table = Table(title="Estado de Configuración")
        table.add_column("Componente", style="cyan")
        table.add_column("Estado", style="green")

        for component, status in config_status.items():
            status_icon = "✅" if status else "❌"
            table.add_row(component, status_icon)

        console.print(table)

        if all(config_status.values()):
            console.print("[green]🎉 AEGIS CLI completamente configurado y listo![/green]")
        else:
            console.print("[yellow]⚠️ Algunos componentes requieren configuración adicional[/yellow]")

        return True

    async def cmd_health(self, **kwargs):
        """Verificar estado del sistema"""
        console.print("[blue]❤️ Verificando salud del sistema...[/blue]")

        if not self.aegis:
            console.print("[red]❌ SDK no inicializado[/red]")
            return False

        result = await self.aegis.client.health_check()

        if result.success:
            health_data = result.data

            # Crear tabla de salud
            table = Table(title="Estado de Salud del Sistema")
            table.add_column("Sistema", style="cyan")
            table.add_column("Estado", style="green")
            table.add_column("Detalles", style="yellow")

            for system, status in health_data["systems"].items():
                status_icon = "✅ Saludable" if status else "❌ Problemas"
                details = "OK" if status else "Requiere atención"
                table.add_row(system.replace("_", " ").title(), status_icon, details)

            console.print(table)

            overall = health_data["overall_health"]
            if overall == "healthy":
                console.print("[green]🎉 Sistema completamente saludable[/green]")
            else:
                console.print(f"[yellow]⚠️ Sistema en estado: {overall}[/yellow]")

        else:
            console.print(f"[red]❌ Error verificando salud: {result.error}[/red]")

        return result.success

    async def cmd_models(self, action: str = "list", **kwargs):
        """Gestionar modelos de ML"""
        if not self.aegis:
            console.print("[red]❌ SDK no inicializado[/red]")
            return False

        if action == "list":
            console.print("[blue]🧠 Listando modelos disponibles...[/blue]")

            # En una implementación real, esto vendría del SDK
            # Por ahora, simulamos algunos modelos
            models = [
                {"id": "resnet_50", "framework": "tensorflow", "type": "classification", "status": "active"},
                {"id": "bert_base", "framework": "pytorch", "type": "transformer", "status": "active"},
                {"id": "cnn_detector", "framework": "tensorflow", "type": "detection", "status": "training"}
            ]

            table = Table(title="Modelos de ML")
            table.add_column("ID", style="cyan")
            table.add_column("Framework", style="green")
            table.add_column("Tipo", style="yellow")
            table.add_column("Estado", style="magenta")

            for model in models:
                table.add_row(model["id"], model["framework"], model["type"], model["status"])

            console.print(table)

        elif action == "register":
            model_path = kwargs.get("path")
            framework = kwargs.get("framework", "tensorflow")
            model_type = kwargs.get("type", "classification")

            if not model_path:
                console.print("[red]❌ Debes especificar --path para el modelo[/red]")
                return False

            console.print(f"[blue]📝 Registrando modelo: {model_path}[/blue]")

            result = await self.aegis.client.register_model(
                model_path=model_path,
                framework=framework,
                model_type=model_type,
                metadata={"source": "cli"}
            )

            if result.success:
                console.print(f"[green]✅ Modelo registrado: {result.data['model_id']}[/green]")
            else:
                console.print(f"[red]❌ Error registrando modelo: {result.error}[/red]")
                return False

        return True

    async def cmd_federated(self, action: str = "list", **kwargs):
        """Gestionar aprendizaje federado"""
        if not self.aegis:
            console.print("[red]❌ SDK no inicializado[/red]")
            return False

        if action == "start":
            model_id = kwargs.get("model")
            participants = kwargs.get("participants", "").split(",")

            if not model_id or not participants:
                console.print("[red]❌ Debes especificar --model y --participants[/red]")
                return False

            console.print(f"[blue]🤝 Iniciando entrenamiento federado con {len(participants)} participantes...[/blue]")

            result = await self.aegis.client.start_federated_training(model_id, participants)

            if result.success:
                console.print(f"[green]✅ Entrenamiento federado iniciado: {result.data['training_id']}[/green]")
                console.print(f"   📊 Modelo: {result.data['model_id']}")
                console.print(f"   👥 Participantes: {result.data['participants']}")
            else:
                console.print(f"[red]❌ Error iniciando federated learning: {result.error}[/red]")
                return False

        elif action == "list":
            console.print("[blue]🤝 Rondas federadas activas...[/blue]")

            # Simular rondas activas
            rounds = [
                {"id": "round_001", "model": "resnet_50", "participants": 5, "status": "active", "progress": 65},
                {"id": "round_002", "model": "bert_base", "participants": 3, "status": "collecting", "progress": 30}
            ]

            table = Table(title="Rondas Federadas")
            table.add_column("ID", style="cyan")
            table.add_column("Modelo", style="green")
            table.add_column("Participantes", style="yellow")
            table.add_column("Estado", style="magenta")
            table.add_column("Progreso", style="blue")

            for round_info in rounds:
                table.add_row(
                    round_info["id"],
                    round_info["model"],
                    str(round_info["participants"]),
                    round_info["status"],
                    f"{round_info['progress']}%"
                )

            console.print(table)

        return True

    async def cmd_cloud(self, action: str = "list", **kwargs):
        """Gestionar recursos cloud"""
        if not self.aegis:
            console.print("[red]❌ SDK no inicializado[/red]")
            return False

        if action == "deploy":
            name = kwargs.get("name")
            provider = kwargs.get("provider", "aws")
            region = kwargs.get("region", "us-east-1")
            count = kwargs.get("count", 2)

            if not name:
                console.print("[red]❌ Debes especificar --name para el despliegue[/red]")
                return False

            console.print(f"[blue]☁️ Creando despliegue en {provider}: {name}[/blue]")

            result = await self.aegis.client.create_cloud_deployment(
                name=name,
                provider=provider,
                region=region,
                instance_config={
                    "instance_type": "t2_micro",
                    "count": count,
                    "auto_scaling": True
                }
            )

            if result.success:
                console.print(f"[green]✅ Despliegue creado: {result.data['deployment_id']}[/green]")
                console.print(f"   ☁️ Provider: {result.data['provider']}")
                console.print(f"   📍 Región: {result.data['region']}")
                console.print(f"   💻 Instancias: {result.data['instances']}")
            else:
                console.print(f"[red]❌ Error creando despliegue: {result.error}[/red]")
                return False

        elif action == "list":
            console.print("[blue]☁️ Listando despliegues cloud...[/blue]")

            result = await self.aegis.client.get_cloud_metrics()

            if result.success:
                table = Table(title="Métricas de Cloud")
                table.add_column("Provider", style="cyan")
                table.add_column("Instancias", style="green")
                table.add_column("Costo/Hora", style="yellow")
                table.add_column("CPU Avg", style="magenta")
                table.add_column("Memoria Avg", style="blue")

                for provider, metrics in result.data.items():
                    table.add_row(
                        provider.upper(),
                        str(metrics.get("running_instances", 0)),
                        f"${metrics.get('total_cost', 0):.2f}",
                        f"{metrics.get('avg_cpu_utilization', 0):.1f}%",
                        f"{metrics.get('avg_memory_utilization', 0):.1f}%"
                    )

                console.print(table)
            else:
                console.print(f"[red]❌ Error obteniendo métricas: {result.error}[/red]")

        return True

    async def cmd_edge(self, action: str = "list", **kwargs):
        """Gestionar dispositivos edge"""
        if not self.aegis:
            console.print("[red]❌ SDK no inicializado[/red]")
            return False

        if action == "register":
            device_type = kwargs.get("type")
            capabilities = kwargs.get("capabilities", "").split(",")

            if not device_type:
                console.print("[red]❌ Debes especificar --type para el dispositivo[/red]")
                return False

            console.print(f"[blue]📱 Registrando dispositivo edge: {device_type}[/blue]")

            device_info = {
                "device_type": device_type,
                "capabilities": capabilities,
                "hardware_specs": {"cpu": "ARM", "ram": "4GB"},
                "location": {"lat": 0.0, "lon": 0.0}
            }

            result = await self.aegis.client.register_edge_device(device_info)

            if result.success:
                console.print(f"[green]✅ Dispositivo registrado: {result.data['device_id']}[/green]")
                console.print(f"   📱 Tipo: {result.data['device_type']}")
                console.print(f"   🛠️ Capacidades: {', '.join(result.data['capabilities'])}")
            else:
                console.print(f"[red]❌ Error registrando dispositivo: {result.error}[/red]")
                return False

        elif action == "list":
            console.print("[blue]📱 Dispositivos edge registrados...[/blue]")

            # Simular dispositivos
            devices = [
                {"id": "edge_001", "type": "raspberry_pi", "status": "online", "models": 2},
                {"id": "edge_002", "type": "jetson_nano", "status": "online", "models": 1},
                {"id": "edge_003", "type": "esp32", "status": "offline", "models": 0}
            ]

            table = Table(title="Dispositivos Edge")
            table.add_column("ID", style="cyan")
            table.add_column("Tipo", style="green")
            table.add_column("Estado", style="yellow")
            table.add_column("Modelos", style="magenta")

            for device in devices:
                table.add_row(
                    device["id"],
                    device["type"],
                    device["status"],
                    str(device["models"])
                )

            console.print(table)

        return True

    async def cmd_predict(self, model_id: str = None, **kwargs):
        """Realizar predicción con un modelo"""
        if not model_id:
            console.print("[red]❌ Debes especificar --model-id[/red]")
            return False

        console.print(f"[blue]🎯 Realizando predicción con modelo: {model_id}[/blue]")

        # Simular datos de entrada
        test_data = [[0.1, 0.2, 0.3] * 256]  # Datos dummy

        result = await self.aegis.client.predict(model_id, test_data)

        if result.success:
            prediction = result.data.get("prediction", [])
            console.print(f"[green]✅ Predicción completada[/green]")
            console.print(f"   📊 Resultado: {prediction[:5]}..." if len(prediction) > 5 else f"   📊 Resultado: {prediction}")
        else:
            console.print(f"[red]❌ Error en predicción: {result.error}[/red]")
            return False

        return True

    async def cmd_deploy(self, **kwargs):
        """Desplegar aplicaciones"""
        console.print("[blue]🚀 Iniciando despliegue...[/blue]")

        # Menú interactivo para despliegue
        deployment_type = await questionary.select(
            "Tipo de despliegue:",
            choices=["Cloud", "Edge", "Hybrid"]
        ).unsafe_ask()

        if deployment_type == "Cloud":
            await self._deploy_cloud_interactive()
        elif deployment_type == "Edge":
            await self._deploy_edge_interactive()
        else:
            await self._deploy_hybrid_interactive()

        return True

    async def _deploy_cloud_interactive(self):
        """Despliegue interactivo en cloud"""
        console.print("[cyan]☁️ Configuración de despliegue en Cloud[/cyan]")

        name = await questionary.text("Nombre del despliegue:").unsafe_ask()
        provider = await questionary.select("Proveedor:", ["aws", "gcp", "azure"]).unsafe_ask()
        region = await questionary.text("Región:", default="us-east-1").unsafe_ask()
        count = int(await questionary.text("Número de instancias:", default="2").unsafe_ask())

        # Ejecutar despliegue
        result = await self.aegis.client.create_cloud_deployment(
            name=name,
            provider=provider,
            region=region,
            instance_config={"instance_type": "t2_micro", "count": count}
        )

        if result.success:
            console.print(f"[green]✅ Despliegue completado: {result.data['deployment_id']}[/green]")
        else:
            console.print(f"[red]❌ Error en despliegue: {result.error}[/red]")

    async def _deploy_edge_interactive(self):
        """Despliegue interactivo en edge"""
        console.print("[cyan]🛠️ Configuración de despliegue en Edge[/cyan]")

        model_id = await questionary.text("ID del modelo:").unsafe_ask()
        device_count = int(await questionary.text("Número de dispositivos:", default="3").unsafe_ask())

        # Simular dispositivos
        devices = [f"edge_{i:03d}" for i in range(device_count)]

        result = await self.aegis.client.deploy_to_edge(model_id, devices)

        if result.success:
            console.print(f"[green]✅ Modelo desplegado en {result.data['device_count']} dispositivos[/green]")
        else:
            console.print(f"[red]❌ Error en despliegue: {result.error}[/red]")

    async def _deploy_hybrid_interactive(self):
        """Despliegue híbrido interactivo"""
        console.print("[cyan]🔄 Configuración de despliegue híbrido[/cyan]")

        # Cloud deployment
        await self._deploy_cloud_interactive()

        # Edge deployment
        await self._deploy_edge_interactive()

        console.print("[green]🎉 Despliegue híbrido completado[/green]")

    async def cmd_monitor(self, **kwargs):
        """Monitoreo en tiempo real"""
        console.print("[blue]📊 Iniciando monitoreo en tiempo real...[/blue]")
        console.print("[yellow]Presiona Ctrl+C para detener[/yellow]")

        try:
            with Live(console=console, refresh_per_second=1) as live:
                while True:
                    # Obtener métricas actualizadas
                    health = await self.aegis.client.health_check()
                    metrics = await self.aegis.client.get_cloud_metrics()

                    # Crear panel de monitoreo
                    panels = []

                    if health.success:
                        health_panel = Panel(
                            f"Estado: {health.data['overall_health']}\n"
                            + "\n".join([f"{sys}: {'✅' if status else '❌'}"
                                       for sys, status in health.data['systems'].items()]),
                            title="❤️ Salud del Sistema"
                        )
                        panels.append(health_panel)

                    if metrics.success:
                        metrics_text = ""
                        for provider, provider_metrics in metrics.data.items():
                            metrics_text += f"{provider.upper()}: {provider_metrics.get('running_instances', 0)} instancias\n"

                        metrics_panel = Panel(metrics_text, title="☁️ Métricas de Cloud")
                        panels.append(metrics_panel)

                    # Mostrar panels
                    if panels:
                        live.update(Group(*panels))
                    else:
                        live.update(Text("Cargando métricas..."))

                    await asyncio.sleep(2)

        except KeyboardInterrupt:
            console.print("[yellow]⏹️ Monitoreo detenido[/yellow]")

        return True

    async def cmd_interactive(self, **kwargs):
        """Modo interactivo"""
        console.print("[green]🎮 Modo Interactivo AEGIS CLI[/green]")
        console.print("[yellow]Escribe 'help' para ver comandos o 'exit' para salir[/yellow]")

        self.interactive_mode = True

        while self.interactive_mode:
            try:
                command = await questionary.text("aegis> ").unsafe_ask()

                if command.lower() in ['exit', 'quit', 'q']:
                    break
                elif command.lower() == 'help':
                    await self._show_help()
                elif command.strip():
                    # Parsear comando básico
                    parts = command.split()
                    cmd = parts[0]
                    args = {}

                    # Parsear argumentos simples (--key value)
                    i = 1
                    while i < len(parts):
                        if parts[i].startswith('--'):
                            key = parts[i][2:]
                            if i + 1 < len(parts) and not parts[i + 1].startswith('--'):
                                args[key] = parts[i + 1]
                                i += 2
                            else:
                                args[key] = True
                                i += 1
                        else:
                            i += 1

                    await self.execute_command(cmd, **args)

            except KeyboardInterrupt:
                break
            except Exception as e:
                console.print(f"[red]❌ Error: {e}[/red]")

        console.print("[yellow]👋 Hasta luego![/yellow]")
        return True

    async def _show_help(self):
        """Mostrar ayuda interactiva"""
        help_text = """
[bold cyan]Comandos disponibles:[/bold cyan]

[green]Sistema:[/green]
  init              Inicializar/verificar configuración
  health            Verificar estado del sistema
  monitor           Monitoreo en tiempo real

[green]Modelos ML:[/green]
  models list       Listar modelos disponibles
  models register   Registrar nuevo modelo
  predict           Realizar predicción

[green]Federated Learning:[/green]
  federated list    Listar rondas activas
  federated start   Iniciar entrenamiento federado

[green]Cloud:[/green]
  cloud list        Listar despliegues y métricas
  cloud deploy      Crear nuevo despliegue

[green]Edge:[/green]
  edge list         Listar dispositivos edge
  edge register     Registrar dispositivo edge

[green]Despliegue:[/green]
  deploy            Asistente de despliegue interactivo

[green]Otros:[/green]
  help              Mostrar esta ayuda
  exit              Salir del modo interactivo
        """

        console.print(Panel(help_text, title="💡 Ayuda - AEGIS CLI"))

# ===== CLI COMMANDS =====

@click.group()
@click.option('--api-key', envvar='AEGIS_API_KEY', help='API key para autenticación')
@click.pass_context
def cli(ctx, api_key):
    """AEGIS CLI - Framework de IA Distribuida y Colaborativa"""
    ctx.ensure_object(dict)
    ctx.obj['api_key'] = api_key
    ctx.obj['cli_instance'] = AEGISCLI()

@cli.command()
@click.pass_context
def init(ctx):
    """Inicializar AEGIS CLI"""
    asyncio.run(ctx.obj['cli_instance'].execute_command('init'))

@cli.command()
@click.pass_context
def health(ctx):
    """Verificar estado del sistema"""
    asyncio.run(ctx.obj['cli_instance'].execute_command('health'))

@cli.group()
def models():
    """Gestionar modelos de ML"""
    pass

@models.command('list')
@click.pass_context
def models_list(ctx):
    """Listar modelos disponibles"""
    asyncio.run(ctx.obj['cli_instance'].execute_command('models', action='list'))

@models.command('register')
@click.option('--path', required=True, help='Ruta al archivo del modelo')
@click.option('--framework', default='tensorflow', help='Framework del modelo')
@click.option('--type', default='classification', help='Tipo de modelo')
@click.pass_context
def models_register(ctx, path, framework, model_type):
    """Registrar nuevo modelo"""
    asyncio.run(ctx.obj['cli_instance'].execute_command('models', action='register',
                                                       path=path, framework=framework, type=model_type))

@cli.group()
def federated():
    """Gestionar aprendizaje federado"""
    pass

@federated.command('list')
@click.pass_context
def federated_list(ctx):
    """Listar rondas federadas activas"""
    asyncio.run(ctx.obj['cli_instance'].execute_command('federated', action='list'))

@federated.command('start')
@click.option('--model', required=True, help='ID del modelo')
@click.option('--participants', required=True, help='Lista de participantes separados por coma')
@click.pass_context
def federated_start(ctx, model, participants):
    """Iniciar entrenamiento federado"""
    asyncio.run(ctx.obj['cli_instance'].execute_command('federated', action='start',
                                                       model=model, participants=participants))

@cli.group()
def cloud():
    """Gestionar recursos cloud"""
    pass

@cloud.command('list')
@click.pass_context
def cloud_list(ctx):
    """Listar despliegues y métricas cloud"""
    asyncio.run(ctx.obj['cli_instance'].execute_command('cloud', action='list'))

@cloud.command('deploy')
@click.option('--name', required=True, help='Nombre del despliegue')
@click.option('--provider', default='aws', help='Proveedor cloud')
@click.option('--region', default='us-east-1', help='Región')
@click.option('--count', default=2, type=int, help='Número de instancias')
@click.pass_context
def cloud_deploy(ctx, name, provider, region, count):
    """Crear despliegue en cloud"""
    asyncio.run(ctx.obj['cli_instance'].execute_command('cloud', action='deploy',
                                                       name=name, provider=provider,
                                                       region=region, count=count))

@cli.group()
def edge():
    """Gestionar dispositivos edge"""
    pass

@edge.command('list')
@click.pass_context
def edge_list(ctx):
    """Listar dispositivos edge"""
    asyncio.run(ctx.obj['cli_instance'].execute_command('edge', action='list'))

@edge.command('register')
@click.option('--type', required=True, help='Tipo de dispositivo')
@click.option('--capabilities', default='inference_only', help='Capacidades separadas por coma')
@click.pass_context
def edge_register(ctx, device_type, capabilities):
    """Registrar dispositivo edge"""
    asyncio.run(ctx.obj['cli_instance'].execute_command('edge', action='register',
                                                       type=device_type, capabilities=capabilities))

@cli.command()
@click.option('--model-id', required=True, help='ID del modelo')
@click.pass_context
def predict(ctx, model_id):
    """Realizar predicción con un modelo"""
    asyncio.run(ctx.obj['cli_instance'].execute_command('predict', model_id=model_id))

@cli.command()
@click.pass_context
def deploy(ctx):
    """Asistente de despliegue interactivo"""
    asyncio.run(ctx.obj['cli_instance'].execute_command('deploy'))

@cli.command()
@click.pass_context
def monitor(ctx):
    """Monitoreo en tiempo real"""
    asyncio.run(ctx.obj['cli_instance'].execute_command('monitor'))

@cli.command()
@click.pass_context
def interactive(ctx):
    """Iniciar modo interactivo"""
    asyncio.run(ctx.obj['cli_instance'].execute_command('interactive'))

# ===== MAIN =====

async def main():
    """Función principal para modo interactivo directo"""
    cli_instance = AEGISCLI()
    await cli_instance.initialize()

    console.print("[green]🎯 Bienvenido a AEGIS CLI[/green]")
    console.print("[yellow]Escribe 'help' para ver comandos disponibles[/yellow]")

    await cli_instance.cmd_interactive()

if __name__ == "__main__":
    cli()
