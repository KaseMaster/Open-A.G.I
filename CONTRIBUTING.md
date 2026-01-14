# Contribuir a Open A.G.I

## Flujo recomendado

1. Crea una rama desde `main`:
   - `feat/...` para features
   - `fix/...` para bugs
   - `chore/...` para mantenimiento/seguridad
2. Haz commits pequeños y descriptivos.
3. Abre un Pull Request hacia `main`.
4. Verifica localmente antes de pedir revisión:
   - Python: `python -m pytest -q` (o subset relevante)
   - Node: `npm audit --audit-level=high` en los paquetes tocados
   - Frontends: `npm run build` en los paquetes tocados

## Seguridad / Dependencias

- Evita subir `node_modules/` al repo.
- Prefiere remediaciones sin cambios breaking; si son inevitables, explícitalo en el PR.
- Si una vulnerabilidad es transitiva y “No fix available”, documenta el riesgo y el plan de mitigación.

