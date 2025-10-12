# Guía de Contribución - AEGIS

¡Gracias por tu interés en contribuir al proyecto AEGIS! Esta guía te ayudará a entender cómo puedes participar en el desarrollo de este sistema avanzado de gestión de información y seguridad.

## Tabla de Contenidos

- [Código de Conducta](#código-de-conducta)
- [Cómo Contribuir](#cómo-contribuir)
- [Configuración del Entorno de Desarrollo](#configuración-del-entorno-de-desarrollo)
- [Estándares de Código](#estándares-de-código)
- [Proceso de Pull Request](#proceso-de-pull-request)
- [Reportar Bugs](#reportar-bugs)
- [Solicitar Características](#solicitar-características)
- [Documentación](#documentación)
- [Testing](#testing)
- [Seguridad](#seguridad)
- [Comunidad](#comunidad)

## Código de Conducta

### Nuestro Compromiso

En el interés de fomentar un ambiente abierto y acogedor, nosotros como contribuidores y mantenedores nos comprometemos a hacer de la participación en nuestro proyecto y nuestra comunidad una experiencia libre de acoso para todos.

### Estándares

Ejemplos de comportamiento que contribuyen a crear un ambiente positivo incluyen:

- Usar lenguaje acogedor e inclusivo
- Ser respetuoso de diferentes puntos de vista y experiencias
- Aceptar críticas constructivas de manera elegante
- Enfocarse en lo que es mejor para la comunidad
- Mostrar empatía hacia otros miembros de la comunidad

### Responsabilidades

Los mantenedores del proyecto son responsables de clarificar los estándares de comportamiento aceptable y se espera que tomen acciones correctivas apropiadas y justas en respuesta a cualquier instancia de comportamiento inaceptable.

## Cómo Contribuir

### Tipos de Contribuciones

Valoramos todos los tipos de contribuciones:

1. **Código**: Nuevas características, corrección de bugs, optimizaciones
2. **Documentación**: Mejoras en docs, tutoriales, ejemplos
3. **Testing**: Nuevos tests, mejoras en cobertura
4. **Seguridad**: Auditorías, reportes de vulnerabilidades
5. **Diseño**: UI/UX del dashboard, diagramas de arquitectura
6. **Traducción**: Internacionalización del proyecto

### Proceso General

1. **Fork** el repositorio
2. **Clone** tu fork localmente
3. **Crea** una rama para tu contribución
4. **Desarrolla** tu contribución
5. **Testa** tus cambios
6. **Commit** siguiendo nuestras convenciones
7. **Push** a tu fork
8. **Crea** un Pull Request

## Configuración del Entorno de Desarrollo

### Requisitos Previos

```bash
# Python 3.8 o superior
python --version

# Git
git --version

# Docker (opcional pero recomendado)
docker --version
```

### Configuración Inicial

```bash
# 1. Fork y clone el repositorio
git clone https://github.com/tu-usuario/aegis.git
cd aegis

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/macOS
# o
venv\Scripts\activate     # Windows

# 3. Instalar dependencias de desarrollo
pip install -r requirements-dev.txt

# 4. Instalar pre-commit hooks
pre-commit install

# 5. Configurar variables de entorno
cp .env.example .env
# Editar .env con tus configuraciones

# 6. Ejecutar tests para verificar configuración
python -m pytest tests/
```

### Configuración con Docker

```bash
# 1. Construir imagen de desarrollo
docker-compose -f docker-compose.dev.yml build

# 2. Iniciar servicios de desarrollo
docker-compose -f docker-compose.dev.yml up -d

# 3. Ejecutar tests en contenedor
docker-compose -f docker-compose.dev.yml exec aegis python -m pytest
```

### Herramientas de Desarrollo

#### Editor/IDE Recomendado

- **VS Code** con extensiones:
  - Python
  - Pylance
  - Black Formatter
  - GitLens
  - Docker

#### Configuración de VS Code

```json
// .vscode/settings.json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length=88"],
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests/"],
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true
    }
}
```

## Estándares de Código

### Estilo de Código Python

Seguimos **PEP 8** con algunas adaptaciones:

```python
# Longitud máxima de línea: 88 caracteres
# Usar Black para formateo automático
# Usar type hints en todas las funciones

from typing import Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)

class ExampleClass:
    """Clase de ejemplo siguiendo nuestros estándares.
    
    Attributes:
        name: Nombre del ejemplo
        config: Configuración del ejemplo
    """
    
    def __init__(self, name: str, config: Optional[Dict] = None) -> None:
        self.name = name
        self.config = config or {}
        
    def process_data(self, data: List[Dict]) -> Dict[str, Union[int, str]]:
        """Procesa datos de entrada.
        
        Args:
            data: Lista de diccionarios con datos
            
        Returns:
            Diccionario con resultados procesados
            
        Raises:
            ValueError: Si los datos están malformados
        """
        if not data:
            raise ValueError("Los datos no pueden estar vacíos")
            
        result = {
            "processed_count": len(data),
            "status": "success"
        }
        
        logger.info(f"Procesados {len(data)} elementos")
        return result
```

### Convenciones de Naming

```python
# Variables y funciones: snake_case
user_name = "aegis_user"
def calculate_hash():
    pass

# Clases: PascalCase
class CryptoManager:
    pass

# Constantes: UPPER_SNAKE_CASE
MAX_RETRY_ATTEMPTS = 3
DEFAULT_TIMEOUT = 30

# Archivos: snake_case
# crypto_framework.py
# p2p_network.py
```

### Documentación de Código

```python
def complex_function(param1: str, param2: int, param3: Optional[bool] = None) -> Dict:
    """Función compleja que requiere documentación detallada.
    
    Esta función realiza operaciones complejas sobre los parámetros
    de entrada y retorna un diccionario con los resultados.
    
    Args:
        param1: Descripción del primer parámetro
        param2: Descripción del segundo parámetro
        param3: Parámetro opcional, por defecto None
        
    Returns:
        Dict: Diccionario con las claves:
            - 'result': Resultado de la operación
            - 'status': Estado de la operación
            - 'metadata': Información adicional
            
    Raises:
        ValueError: Si param1 está vacío
        TypeError: Si param2 no es un entero
        
    Example:
        >>> result = complex_function("test", 42, True)
        >>> print(result['status'])
        'success'
        
    Note:
        Esta función puede ser costosa computacionalmente
        para valores grandes de param2.
    """
    pass
```

### Manejo de Errores

```python
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class AegisError(Exception):
    """Excepción base para errores de AEGIS."""
    pass

class CryptoError(AegisError):
    """Errores relacionados con criptografía."""
    pass

def secure_operation(data: str) -> Optional[str]:
    """Operación segura con manejo de errores apropiado."""
    try:
        # Operación que puede fallar
        result = perform_crypto_operation(data)
        logger.info("Operación criptográfica exitosa")
        return result
        
    except CryptoError as e:
        logger.error(f"Error criptográfico: {e}")
        raise  # Re-raise para que el caller maneje
        
    except Exception as e:
        logger.error(f"Error inesperado: {e}")
        raise AegisError(f"Fallo en operación segura: {e}") from e
```

### Logging

```python
import logging
import structlog

# Configuración de logging estructurado
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

def example_with_logging():
    """Ejemplo de uso correcto de logging."""
    logger.info("Iniciando operación", operation="example")
    
    try:
        # Operación
        result = perform_operation()
        logger.info("Operación completada", 
                   operation="example", 
                   result_count=len(result))
        return result
        
    except Exception as e:
        logger.error("Error en operación", 
                    operation="example", 
                    error=str(e),
                    exc_info=True)
        raise
```

## Proceso de Pull Request

### Antes de Crear el PR

1. **Sincroniza** tu fork con el repositorio principal
2. **Ejecuta** todos los tests
3. **Verifica** el linting y formateo
4. **Actualiza** la documentación si es necesario
5. **Añade** tests para nuevas funcionalidades

```bash
# Sincronizar fork
git remote add upstream https://github.com/aegis-project/aegis.git
git fetch upstream
git checkout main
git merge upstream/main

# Ejecutar verificaciones
python -m pytest tests/
python -m black --check .
python -m pylint src/
python -m mypy src/
```

### Estructura del PR

#### Título
```
[TIPO] Descripción concisa del cambio

Ejemplos:
[FEAT] Agregar algoritmo de consenso híbrido
[FIX] Corregir memory leak en P2P network
[DOCS] Actualizar guía de instalación
[TEST] Agregar tests para crypto framework
[REFACTOR] Mejorar estructura del storage system
```

#### Descripción

```markdown
## Descripción
Breve descripción de los cambios realizados.

## Tipo de Cambio
- [ ] Bug fix (cambio que corrige un issue)
- [ ] Nueva característica (cambio que agrega funcionalidad)
- [ ] Breaking change (cambio que rompe compatibilidad)
- [ ] Documentación (cambios solo en documentación)

## ¿Cómo se ha probado?
Describe las pruebas realizadas para verificar los cambios.

## Checklist
- [ ] Mi código sigue las guías de estilo del proyecto
- [ ] He realizado una auto-revisión de mi código
- [ ] He comentado mi código, especialmente en áreas difíciles
- [ ] He realizado cambios correspondientes en la documentación
- [ ] Mis cambios no generan nuevas advertencias
- [ ] He agregado tests que prueban que mi fix es efectivo o que mi característica funciona
- [ ] Los tests unitarios nuevos y existentes pasan localmente
- [ ] Cualquier cambio dependiente ha sido fusionado y publicado

## Screenshots (si aplica)
Agregar screenshots para cambios en UI.

## Información Adicional
Cualquier información adicional relevante.
```

### Revisión de Código

#### Para Revisores

- **Funcionalidad**: ¿El código hace lo que se supone que debe hacer?
- **Legibilidad**: ¿Es fácil de entender?
- **Mantenibilidad**: ¿Será fácil de mantener?
- **Performance**: ¿Hay impactos en el rendimiento?
- **Seguridad**: ¿Introduce vulnerabilidades?
- **Tests**: ¿Está adecuadamente probado?

#### Para Contribuidores

- Responde a comentarios de manera constructiva
- Realiza cambios solicitados promptamente
- Mantén el PR actualizado con la rama principal
- Sé paciente durante el proceso de revisión

## Reportar Bugs

### Antes de Reportar

1. **Busca** en issues existentes
2. **Verifica** que uses la versión más reciente
3. **Reproduce** el bug de manera consistente
4. **Recopila** información del sistema

### Template de Bug Report

```markdown
**Descripción del Bug**
Una descripción clara y concisa del bug.

**Para Reproducir**
Pasos para reproducir el comportamiento:
1. Ve a '...'
2. Haz clic en '....'
3. Desplázate hacia abajo hasta '....'
4. Ve el error

**Comportamiento Esperado**
Una descripción clara y concisa de lo que esperabas que pasara.

**Screenshots**
Si aplica, agrega screenshots para ayudar a explicar tu problema.

**Información del Sistema:**
 - OS: [e.g. Ubuntu 20.04]
 - Python Version: [e.g. 3.9.7]
 - AEGIS Version: [e.g. 1.0.0]
 - Docker Version (si aplica): [e.g. 20.10.8]

**Logs**
```
Pega aquí los logs relevantes
```

**Contexto Adicional**
Agrega cualquier otro contexto sobre el problema aquí.
```

## Solicitar Características

### Template de Feature Request

```markdown
**¿Tu solicitud de característica está relacionada con un problema? Por favor describe.**
Una descripción clara y concisa de cuál es el problema. Ej. Siempre me frustra cuando [...]

**Describe la solución que te gustaría**
Una descripción clara y concisa de lo que quieres que pase.

**Describe alternativas que has considerado**
Una descripción clara y concisa de cualquier solución o característica alternativa que hayas considerado.

**Contexto adicional**
Agrega cualquier otro contexto o screenshots sobre la solicitud de característica aquí.

**Impacto**
- [ ] Mejora la seguridad
- [ ] Mejora el rendimiento
- [ ] Mejora la usabilidad
- [ ] Agrega nueva funcionalidad
- [ ] Otro: ___________

**Prioridad**
- [ ] Crítica
- [ ] Alta
- [ ] Media
- [ ] Baja
```

## Documentación

### Tipos de Documentación

1. **API Reference**: Documentación automática desde docstrings
2. **User Guides**: Tutoriales y guías paso a paso
3. **Developer Docs**: Documentación técnica interna
4. **Architecture Docs**: Diagramas y explicaciones de arquitectura

### Escribir Documentación

```markdown
# Título de la Sección

## Introducción
Breve introducción al tema.

## Requisitos Previos
- Conocimiento de Python
- Familiaridad con conceptos de criptografía
- Instalación de AEGIS

## Paso a Paso

### 1. Configuración Inicial
```python
# Código de ejemplo
from aegis import CryptoFramework

crypto = CryptoFramework()
```

### 2. Uso Básico
Explicación del uso básico con ejemplos.

## Ejemplos Avanzados
Ejemplos más complejos para usuarios avanzados.

## Troubleshooting
Problemas comunes y sus soluciones.

## Referencias
- [Enlace a documentación relacionada]
- [Enlace a especificaciones técnicas]
```

### Generación de Documentación

```bash
# Generar documentación API
python -m sphinx-build -b html docs/ docs/_build/

# Servir documentación localmente
python -m http.server 8000 --directory docs/_build/
```

## Testing

### Tipos de Tests

1. **Unit Tests**: Prueban componentes individuales
2. **Integration Tests**: Prueban interacciones entre componentes
3. **Performance Tests**: Prueban rendimiento y escalabilidad
4. **Security Tests**: Prueban vulnerabilidades de seguridad

### Escribir Tests

```python
import pytest
from unittest.mock import Mock, patch
from aegis.crypto_framework import CryptoFramework

class TestCryptoFramework:
    """Tests para el framework criptográfico."""
    
    @pytest.fixture
    def crypto_framework(self):
        """Fixture para instancia de CryptoFramework."""
        return CryptoFramework()
    
    def test_key_generation(self, crypto_framework):
        """Test generación de claves."""
        # Arrange
        key_size = 256
        
        # Act
        key = crypto_framework.generate_key(key_size)
        
        # Assert
        assert key is not None
        assert len(key) == key_size // 8
        
    def test_encryption_decryption(self, crypto_framework):
        """Test cifrado y descifrado."""
        # Arrange
        plaintext = "mensaje secreto"
        key = crypto_framework.generate_key(256)
        
        # Act
        ciphertext = crypto_framework.encrypt(plaintext, key)
        decrypted = crypto_framework.decrypt(ciphertext, key)
        
        # Assert
        assert ciphertext != plaintext
        assert decrypted == plaintext
        
    @patch('aegis.crypto_framework.os.urandom')
    def test_key_generation_with_mock(self, mock_urandom, crypto_framework):
        """Test generación de claves con mock."""
        # Arrange
        mock_urandom.return_value = b'x' * 32
        
        # Act
        key = crypto_framework.generate_key(256)
        
        # Assert
        mock_urandom.assert_called_once_with(32)
        assert key == b'x' * 32
        
    def test_invalid_key_size(self, crypto_framework):
        """Test manejo de tamaño de clave inválido."""
        # Act & Assert
        with pytest.raises(ValueError, match="Tamaño de clave inválido"):
            crypto_framework.generate_key(0)
```

### Ejecutar Tests

```bash
# Todos los tests
python -m pytest

# Tests específicos
python -m pytest tests/test_crypto.py

# Con cobertura
python -m pytest --cov=src --cov-report=html

# Tests de rendimiento
python -m pytest tests/performance/ -v

# Tests de seguridad
python -m pytest tests/security/ -v
```

### Configuración de pytest

```ini
# pytest.ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --strict-markers
    --strict-config
    --cov=src
    --cov-report=term-missing
    --cov-report=html
    --cov-fail-under=80
markers =
    unit: Unit tests
    integration: Integration tests
    performance: Performance tests
    security: Security tests
    slow: Slow running tests
```

## Seguridad

### Reportar Vulnerabilidades

**NO** reportes vulnerabilidades de seguridad a través de issues públicos.

#### Proceso de Reporte Seguro

1. **Email**: Envía detalles a security@aegis-project.org
2. **Encriptación**: Usa nuestra clave PGP pública si es posible
3. **Información**: Incluye pasos para reproducir
4. **Tiempo**: Permite tiempo razonable para respuesta

#### Template de Reporte de Seguridad

```
Asunto: [SECURITY] Descripción breve de la vulnerabilidad

Descripción:
Descripción detallada de la vulnerabilidad.

Impacto:
Descripción del impacto potencial.

Pasos para Reproducir:
1. Paso 1
2. Paso 2
3. Paso 3

Información del Sistema:
- Versión de AEGIS
- Sistema Operativo
- Configuración relevante

Mitigación Sugerida:
Si tienes sugerencias para mitigar la vulnerabilidad.
```

### Mejores Prácticas de Seguridad

```python
# ✅ CORRECTO: Validación de entrada
def process_user_input(user_input: str) -> str:
    if not isinstance(user_input, str):
        raise TypeError("Input debe ser string")
    
    if len(user_input) > 1000:
        raise ValueError("Input demasiado largo")
    
    # Sanitizar input
    sanitized = html.escape(user_input)
    return sanitized

# ✅ CORRECTO: Manejo seguro de secretos
import os
from cryptography.fernet import Fernet

def get_encryption_key() -> bytes:
    key = os.environ.get('AEGIS_ENCRYPTION_KEY')
    if not key:
        raise ValueError("AEGIS_ENCRYPTION_KEY no configurada")
    return key.encode()

# ❌ INCORRECTO: Hardcoded secrets
SECRET_KEY = "mi-clave-super-secreta"  # NUNCA hacer esto

# ❌ INCORRECTO: SQL injection vulnerable
def get_user(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"  # Vulnerable
    return execute_query(query)

# ✅ CORRECTO: Consulta parametrizada
def get_user(user_id: int):
    query = "SELECT * FROM users WHERE id = %s"
    return execute_query(query, (user_id,))
```

## Comunidad

### Canales de Comunicación

- **GitHub Discussions**: Para preguntas generales y discusiones
- **GitHub Issues**: Para bugs y feature requests
- **Discord**: Para chat en tiempo real (próximamente)
- **Email**: Para contacto directo con mantenedores

### Eventos y Reuniones

- **Weekly Dev Sync**: Martes 15:00 UTC
- **Monthly Community Call**: Primer viernes del mes 16:00 UTC
- **Quarterly Roadmap Review**: Cada trimestre

### Reconocimientos

Reconocemos las contribuciones de varias maneras:

- **Contributors**: Listado en README y documentación
- **Hall of Fame**: Para contribuciones significativas
- **Swag**: Stickers y merchandise para contribuidores activos
- **Conference Talks**: Oportunidades para presentar el proyecto

### Mentoría

Ofrecemos programas de mentoría para nuevos contribuidores:

- **First-time Contributors**: Guía personalizada para primeras contribuciones
- **Good First Issues**: Issues marcados como buenos para principiantes
- **Pair Programming**: Sesiones de programación en pareja
- **Code Review Mentoring**: Aprender mejores prácticas de revisión

## Recursos Adicionales

### Documentación Técnica
- [Architecture Guide](ARCHITECTURE_GUIDE.md)
- [API Reference](API_REFERENCE.md)
- [Security Guide](SECURITY_GUIDE.md)
- [Deployment Guide](DEPLOYMENT_GUIDE.md)

### Herramientas de Desarrollo
- [Black](https://black.readthedocs.io/): Formateo de código
- [Pylint](https://pylint.org/): Linting
- [MyPy](https://mypy.readthedocs.io/): Type checking
- [Pytest](https://pytest.org/): Testing framework
- [Pre-commit](https://pre-commit.com/): Git hooks

### Recursos de Aprendizaje
- [Python Best Practices](https://docs.python-guide.org/)
- [Cryptography Concepts](https://cryptography.io/)
- [P2P Network Design](https://en.wikipedia.org/wiki/Peer-to-peer)
- [Consensus Algorithms](https://en.wikipedia.org/wiki/Consensus_algorithm)

---

¡Gracias por contribuir a AEGIS! Tu participación hace que este proyecto sea mejor para toda la comunidad. 🚀

Para preguntas sobre esta guía, por favor abre un issue o contacta a los mantenedores directamente.