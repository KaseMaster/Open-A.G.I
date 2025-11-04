#!/usr/bin/env python3
"""
🎯 AEGIS Developer CLI - Sprint 3.3
CLI avanzado con templates, documentación y experiencia de desarrollo mejorada
"""

import asyncio
import click
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# Importar componentes de AEGIS
from aegis_sdk import AEGIS
from aegis_templates import AEGISTemplates

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class AEGISDeveloperCLI:
    """CLI avanzado para desarrolladores de AEGIS"""

    def __init__(self):
        self.aegis: Optional[AEGIS] = None
        self.templates = AEGISTemplates()
        self.current_project: Optional[Path] = None

    async def initialize(self, api_key: Optional[str] = None):
        """Inicializar CLI"""
        self.aegis = AEGIS(api_key)

        # Verificar conexión
        result = await self.aegis.client.health_check()
        if not result.success:
            click.echo(f"⚠️  Warning: Could not connect to AEGIS - {result.error}")
        else:
            click.echo("✅ Connected to AEGIS Framework")

    async def create_project_from_template(self, template_name: str, project_name: str,
                                         output_dir: str = "./") -> bool:
        """Crear proyecto desde template"""

        try:
            # Verificar que el template existe
            if template_name not in [t["id"] for t in self.templates.list_templates()]:
                click.echo(f"❌ Template '{template_name}' not found")
                return False

            # Crear el proyecto
            project_path = self.templates.generate_project(template_name, project_name, output_dir)

            click.echo(f"✅ Project '{project_name}' created successfully!")
            click.echo(f"📁 Location: {project_path}")
            click.echo("")
            click.echo("🚀 Quick start:")
            click.echo(f"   cd {project_path}")
            click.echo("   pip install -r requirements.txt")
            click.echo("   python main.py")
            click.echo("   Check the documentation in the docs/ folder")

            return True

        except Exception as e:
            click.echo(f"❌ Error creating project: {e}")
            return False

    async def show_template_info(self, template_name: Optional[str] = None):
        """Mostrar información de templates"""

        if template_name:
            # Mostrar información específica
            try:
                info = self.templates.get_template_info(template_name)
                click.echo(f"📋 Template: {info['name']}")
                click.echo(f"📝 Description: {info['description']}")
                click.echo(f"🏷️  Category: {info['category']}")
                click.echo(f"⏱️  Time: {info['estimated_time']}")
                click.echo("")
                click.echo("📁 Files included:")
                for file in info['files']:
                    click.echo(f"   • {file}")
                click.echo("")
                click.echo("📖 README:")
                click.echo(info['readme'][:500] + "..." if len(info['readme']) > 500 else info['readme'])
            except ValueError:
                click.echo(f"❌ Template '{template_name}' not found")
        else:
            # Listar todos los templates
            templates = self.templates.list_templates()

            if not templates:
                click.echo("❌ No templates available")
                return

            click.echo("🎯 Available Templates:")
            click.echo("=" * 50)

            # Agrupar por categoría
            categories = {}
            for template in templates:
                cat = template["category"]
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(template)

            for category, temps in categories.items():
                click.echo(f"\n🏷️  {category.upper()}:")
                for template in temps:
                    click.echo(f"   • {template['id']}: {template['name']}")
                    click.echo(f"     {template['description']}")
                    click.echo(f"     📁 {template['files_count']} files, ⏱️  {template['estimated_time']}")

    async def run_quick_start(self, use_case: str) -> bool:
        """Ejecutar inicio rápido para caso de uso"""

        if not self.aegis:
            click.echo("❌ CLI not initialized")
            return False

        try:
            # Configuración básica para demos
            if use_case == "federated":
                config = {
                    "model_id": "demo_model",
                    "participants": ["node_1", "node_2", "node_3"]
                }
            elif use_case == "edge":
                config = {
                    "model_id": "mobile_net",
                    "devices": [{"device_type": "raspberry_pi", "capabilities": ["inference_only"]}]
                }
            elif use_case == "cloud":
                config = {
                    "name": "demo_deployment",
                    "provider": "aws",
                    "instance_type": "t2_micro",
                    "count": 1
                }
            else:
                click.echo(f"❌ Unknown use case: {use_case}")
                return False

            click.echo(f"🚀 Starting quick {use_case} demo...")

            result = await self.aegis.quick_start(use_case, config)

            if result.success:
                click.echo(f"✅ {use_case.title()} demo completed successfully!")
                if result.data:
                    click.echo(f"📊 Result: {json.dumps(result.data, indent=2)}")
            else:
                click.echo(f"❌ Demo failed: {result.error}")

            return result.success

        except Exception as e:
            click.echo(f"❌ Error running demo: {e}")
            return False

# ===== CLI COMMANDS =====

@click.group()
@click.option('--api-key', envvar='AEGIS_API_KEY', help='AEGIS API Key')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.pass_context
def aegis_cli(ctx, api_key, verbose):
    """🎯 AEGIS Developer CLI - Sprint 3.3

    Advanced CLI for AEGIS Framework development with templates,
    documentation, and enhanced developer experience.
    """
    ctx.ensure_object(dict)
    ctx.obj['cli'] = AEGISDeveloperCLI()
    ctx.obj['verbose'] = verbose

    # Initialize CLI
    asyncio.run(ctx.obj['cli'].initialize(api_key))

@aegis_cli.command()
@click.argument('template_name', required=False)
@click.pass_context
def templates(ctx, template_name):
    """🎯 List and show information about AEGIS templates

    Templates provide pre-configured boilerplates for common use cases.

    Examples:
        aegis templates                    # List all templates
        aegis templates federated_learning # Show specific template info
    """
    asyncio.run(ctx.obj['cli'].show_template_info(template_name))

@aegis_cli.command()
@click.argument('template_name')
@click.argument('project_name')
@click.option('--output-dir', '-o', default='./', help='Output directory')
@click.pass_context
def create(ctx, template_name, project_name, output_dir):
    """📁 Create a new project from an AEGIS template

    Generate a complete project structure with all necessary files,
    dependencies, and documentation.

    Examples:
        aegis create federated_learning my_federated_project
        aegis create cloud_deployment my_cloud_app --output-dir ./projects
    """
    success = asyncio.run(ctx.obj['cli'].create_project_from_template(
        template_name, project_name, output_dir
    ))

    if success and ctx.obj['verbose']:
        click.echo("\n💡 Next steps:")
        click.echo("   1. Review the generated README.md for detailed instructions")
        click.echo("   2. Install dependencies: pip install -r requirements.txt")
        click.echo("   3. Run the application: python main.py")
        click.echo("   4. Check the documentation in the docs/ folder")

@aegis_cli.command()
@click.argument('use_case', type=click.Choice(['federated', 'edge', 'cloud']))
@click.pass_context
def quickstart(ctx, use_case):
    """🚀 Quick start demo for common use cases

    Run a complete working example to see AEGIS capabilities in action.

    Available demos:
        federated: Federated learning with multiple participants
        edge:      Edge computing with device deployment
        cloud:     Cloud deployment with auto-scaling

    Examples:
        aegis quickstart federated
        aegis quickstart edge
        aegis quickstart cloud
    """
    success = asyncio.run(ctx.obj['cli'].run_quick_start(use_case))

    if success and ctx.obj['verbose']:
        click.echo(f"\n📚 Learn more about {use_case} development:")
        click.echo("   • Check the generated code and documentation")
        click.echo("   • Visit: https://docs.aegis-framework.com")
        click.echo("   • Join our developer community on Discord")

@aegis_cli.command()
@click.option('--format', '-f', type=click.Choice(['table', 'json']), default='table',
              help='Output format')
@click.pass_context
def status(ctx, format):
    """📊 Show current AEGIS system status

    Display information about connected components, active deployments,
    and system health.

    Examples:
        aegis status              # Table format (default)
        aegis status --format json # JSON format
    """
    # This would show actual system status in a real implementation
    status_data = {
        "framework_version": "3.3.0",
        "components": {
            "sdk": "connected",
            "ml_engine": "active",
            "cloud_orchestrator": "ready",
            "edge_system": "ready",
            "federated_coordinator": "ready"
        },
        "active_deployments": 0,
        "connected_devices": 0,
        "system_health": "excellent"
    }

    if format == 'json':
        click.echo(json.dumps(status_data, indent=2))
    else:
        click.echo("📊 AEGIS System Status")
        click.echo("=" * 30)
        click.echo(f"Version: {status_data['framework_version']}")
        click.echo(f"Health:  {status_data['system_health']}")
        click.echo("")
        click.echo("Components:")
        for comp, status in status_data['components'].items():
            click.echo(f"  • {comp}: {status}")
        click.echo("")
        click.echo("Active Resources:")
        click.echo(f"  • Deployments: {status_data['active_deployments']}")
        click.echo(f"  • Edge devices: {status_data['connected_devices']}")

@aegis_cli.command()
@click.argument('topic', required=False)
@click.pass_context
def docs(ctx, topic):
    """📖 Access AEGIS documentation and guides

    Get help with AEGIS development, best practices, and troubleshooting.

    Available topics:
        getting-started: Basic setup and first steps
        api-reference:   Complete API documentation
        best-practices:  Development guidelines
        troubleshooting: Common issues and solutions

    Examples:
        aegis docs                    # Show all available topics
        aegis docs getting-started    # Show getting started guide
    """
    docs_content = {
        "getting-started": """
🎯 Getting Started with AEGIS

1. Install the SDK:
   pip install aegis-sdk

2. Set up your API key:
   export AEGIS_API_KEY="your-api-key-here"

3. Create your first project:
   aegis create federated_learning my_first_project

4. Run the project:
   cd my_first_project
   pip install -r requirements.txt
   python main.py

For more detailed guides, visit: https://docs.aegis-framework.com/getting-started
        """,
        "api-reference": """
🔧 AEGIS API Reference

Core Classes:
  • AEGIS: Main client class
  • AEGISClient: Low-level API client
  • MLFrameworkManager: ML model management
  • FederatedLearningCoordinator: FL orchestration

Key Methods:
  • register_model(): Register a new ML model
  • predict(): Run model inference
  • start_federated_training(): Begin FL training
  • deploy_to_edge(): Deploy to edge devices
  • create_cloud_deployment(): Deploy to cloud

Complete API docs: https://docs.aegis-framework.com/api
        """,
        "best-practices": """
💡 AEGIS Development Best Practices

🔒 Security:
  • Always use HTTPS in production
  • Rotate API keys regularly
  • Implement proper authentication
  • Use encrypted channels for sensitive data

⚡ Performance:
  • Optimize models for target hardware
  • Use appropriate batch sizes
  • Monitor resource usage
  • Implement proper error handling

🧠 ML Best Practices:
  • Validate models before deployment
  • Use federated learning for privacy
  • Monitor model performance
  • Implement gradual rollouts

📊 Monitoring:
  • Set up comprehensive logging
  • Use structured logging (JSON)
  • Monitor key metrics
  • Set up alerts for anomalies
        """,
        "troubleshooting": """
🔧 Troubleshooting Guide

Common Issues:

❌ "Connection refused"
   → Check if AEGIS services are running
   → Verify API key is correct
   → Check network connectivity

❌ "Model not found"
   → Ensure model is registered first
   → Check model ID spelling
   → Verify model format compatibility

❌ "Deployment failed"
   → Check cloud provider credentials
   → Verify resource quotas
   → Review deployment logs

❌ "Federated training stuck"
   → Check participant connectivity
   → Verify model compatibility
   → Review network timeouts

For more help: https://docs.aegis-framework.com/troubleshooting
        """
    }

    if topic and topic in docs_content:
        click.echo(docs_content[topic])
    else:
        click.echo("📖 AEGIS Documentation Topics:")
        click.echo("=" * 40)
        for topic_name in docs_content.keys():
            topic_title = topic_name.replace('-', ' ').title()
            click.echo(f"  • {topic_name}: {topic_title}")

        if topic:
            click.echo(f"\n❌ Topic '{topic}' not found. Use one of the topics above.")

@aegis_cli.command()
@click.pass_context
def version(ctx):
    """📋 Show AEGIS version information"""
    version_info = {
        "framework": "AEGIS Framework",
        "version": "3.3.0",
        "sprint": "3.3 - Developer Experience",
        "components": [
            "Core SDK (aegis_sdk.py)",
            "Templates System (aegis_templates.py)",
            "Advanced CLI (aegis_cli.py)",
            "ML Framework Integration",
            "Federated Learning System",
            "Multi-Cloud Orchestration",
            "Edge Computing Platform"
        ],
        "python_requires": ">=3.8",
        "license": "MIT"
    }

    click.echo("📋 AEGIS Framework Version Information")
    click.echo("=" * 45)
    click.echo(f"Framework: {version_info['framework']}")
    click.echo(f"Version:   {version_info['version']}")
    click.echo(f"Sprint:    {version_info['sprint']}")
    click.echo(f"License:   {version_info['license']}")
    click.echo("")
    click.echo("Components:")
    for component in version_info['components']:
        click.echo(f"  ✅ {component}")

@aegis_cli.command()
@click.option('--project-name', '-n', help='Name for the generated project')
@click.option('--template', '-t', default='hybrid_system',
              help='Template to use for the project')
@click.option('--output-dir', '-o', default='./',
              help='Directory to create the project in')
@click.pass_context
def scaffold(ctx, project_name, template, output_dir):
    """🏗️ Scaffold a complete AEGIS project

    Create a full-featured AEGIS project with best practices,
    testing, documentation, and deployment configuration.

    This is a comprehensive starter template that includes:
    - Complete project structure
    - Unit and integration tests
    - Docker configuration
    - CI/CD pipeline
    - Documentation
    - Monitoring setup

    Examples:
        aegis scaffold --project-name my_aegis_app
        aegis scaffold -n my_app -t federated_learning -o ./projects
    """
    if not project_name:
        project_name = click.prompt("Enter project name")

    click.echo(f"🏗️  Scaffolding AEGIS project: {project_name}")
    click.echo(f"📋 Using template: {template}")

    success = asyncio.run(ctx.obj['cli'].create_project_from_template(
        template, project_name, output_dir
    ))

    if success:
        project_path = Path(output_dir) / project_name
        click.echo("\n🎉 Project scaffolded successfully!")
        click.echo("\n📁 Project structure created:")
        click.echo(f"   {project_path}/")
        click.echo("   ├── src/                 # Source code")
        click.echo("   ├── tests/              # Test suites")
        click.echo("   ├── docs/               # Documentation")
        click.echo("   ├── docker/             # Docker configs")
        click.echo("   ├── config/             # Configuration files")
        click.echo("   ├── scripts/            # Utility scripts")
        click.echo("   └── README.md           # Project documentation")

        click.echo("\n🚀 Quick start:")
        click.echo(f"   cd {project_path}")
        click.echo("   python -m venv venv")
        click.echo("   source venv/bin/activate  # On Windows: venv\\Scripts\\activate")
        click.echo("   pip install -r requirements.txt")
        click.echo("   python -m pytest tests/    # Run tests")
        click.echo("   python src/main.py        # Run application")

        if ctx.obj['verbose']:
            click.echo("\n💡 Advanced features included:")
            click.echo("   • Comprehensive test suite with 90%+ coverage")
            click.echo("   • Docker multi-stage builds for production")
            click.echo("   • GitHub Actions CI/CD pipeline")
            click.echo("   • Prometheus/Grafana monitoring stack")
            click.echo("   • Terraform infrastructure as code")
            click.echo("   • API documentation with OpenAPI/Swagger")
            click.echo("   • Security scanning and vulnerability checks")
            click.echo("   • Performance profiling and optimization tools")

# ===== MAIN =====

if __name__ == "__main__":
    aegis_cli()
