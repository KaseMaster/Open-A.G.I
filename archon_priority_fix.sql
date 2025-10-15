-- SCRIPT DE REPARACIÓN ARCHON MCP - COLUMNA PRIORITY
-- Autor: AEGIS - Analista Experto en Gestión de Información y Seguridad
-- Fecha: 2024-12-16
-- 
-- INSTRUCCIONES:
-- 1. Conectar a Supabase usando el SQL Editor
-- 2. Ejecutar este script completo
-- 3. Verificar que no hay errores
-- 4. Reiniciar el servidor Archon MCP

-- 1. Verificar si la tabla archon_tasks existe
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'archon_tasks') THEN
        -- Crear tabla archon_tasks si no existe
        CREATE TABLE archon_tasks (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            project_id UUID REFERENCES archon_projects(id) ON DELETE CASCADE,
            title TEXT NOT NULL,
            description TEXT,
            status TEXT DEFAULT 'todo' CHECK (status IN ('todo', 'doing', 'review', 'done')),
            assignee TEXT DEFAULT 'User',
            feature TEXT,
            priority INTEGER DEFAULT 50 CHECK (priority >= 0 AND priority <= 100),
            task_order INTEGER DEFAULT 0,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        
        RAISE NOTICE 'Tabla archon_tasks creada exitosamente';
    END IF;
END $$;

-- 2. Verificar y añadir columna priority si no existe
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT FROM information_schema.columns 
        WHERE table_name = 'archon_tasks' AND column_name = 'priority'
    ) THEN
        -- Añadir columna priority
        ALTER TABLE archon_tasks 
        ADD COLUMN priority INTEGER DEFAULT 50 CHECK (priority >= 0 AND priority <= 100);
        
        RAISE NOTICE 'Columna priority añadida exitosamente';
        
        -- Actualizar prioridades basadas en estado
        UPDATE archon_tasks SET priority = 
            CASE 
                WHEN status = 'doing' THEN 90
                WHEN status = 'todo' THEN 70
                WHEN status = 'review' THEN 60
                WHEN status = 'done' THEN 30
                ELSE 50
            END;
        
        RAISE NOTICE 'Prioridades actualizadas basadas en estado';
    ELSE
        RAISE NOTICE 'Columna priority ya existe';
    END IF;
END $$;

-- 3. Verificar que la tabla archon_projects existe
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'archon_projects') THEN
        -- Crear tabla archon_projects si no existe
        CREATE TABLE archon_projects (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            title TEXT NOT NULL,
            description TEXT,
            github_repo TEXT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        
        -- Insertar proyecto por defecto
        INSERT INTO archon_projects (title, description) 
        VALUES ('Proyecto por Defecto', 'Proyecto creado automáticamente por el sistema de reparación')
        ON CONFLICT DO NOTHING;
        
        RAISE NOTICE 'Tabla archon_projects creada con proyecto por defecto';
    END IF;
END $$;

-- 4. Crear índices para optimizar rendimiento
CREATE INDEX IF NOT EXISTS idx_archon_tasks_project_id ON archon_tasks(project_id);
CREATE INDEX IF NOT EXISTS idx_archon_tasks_status ON archon_tasks(status);
CREATE INDEX IF NOT EXISTS idx_archon_tasks_priority ON archon_tasks(priority DESC);
CREATE INDEX IF NOT EXISTS idx_archon_tasks_created_at ON archon_tasks(created_at DESC);

-- 5. Verificar integridad de datos
SELECT 
    'archon_tasks' as tabla,
    COUNT(*) as total_registros,
    COUNT(CASE WHEN priority IS NOT NULL THEN 1 END) as con_priority,
    MIN(priority) as min_priority,
    MAX(priority) as max_priority,
    AVG(priority) as avg_priority
FROM archon_tasks
UNION ALL
SELECT 
    'archon_projects' as tabla,
    COUNT(*) as total_registros,
    COUNT(*) as con_priority,
    NULL as min_priority,
    NULL as max_priority,
    NULL as avg_priority
FROM archon_projects;

-- 6. Mostrar estructura final de la tabla
SELECT 
    column_name,
    data_type,
    is_nullable,
    column_default
FROM information_schema.columns 
WHERE table_name = 'archon_tasks'
ORDER BY ordinal_position;