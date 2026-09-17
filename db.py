import psycopg2
from psycopg2.extras import RealDictCursor
import os


def get_database_connection():
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5432")),
        database=os.getenv("DB_NAME", "evalio"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "postgres"),
    )
    return conn


def init_db():
    conn = get_database_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS hackathons (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            description TEXT DEFAULT '',
            theme TEXT DEFAULT '',
            is_allowed BOOLEAN DEFAULT FALSE,
            criteria TEXT DEFAULT '',
            deadline TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cur.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'project_type') THEN
                CREATE TYPE project_type AS ENUM (
                    'VANILLA_JS',
                    'REACT',
                    'NEXT_JS',
                    'VUE',
                    'NUXT',
                    'ANGULAR',
                    'SVELTE',
                    'SVELTEKIT',
                    'ASTRO',
                    'REMIX',
                    'TAILWIND',
                    'NODE_EXPRESS',
                    'FASTAPI',
                    'DJANGO',
                    'SPRING_BOOT',
                    'GIN',
                    'RAILS',
                    'LARAVEL',
                    'ACTIX',
                    'SWIFT_UI',
                    'KOTLIN_JETPACK',
                    'REACT_NATIVE',
                    'EXPO',
                    'FLUTTER',
                    'DOTNET_MAUI',
                    'IONIC',
                    'NATIVESCRIPT',
                    'OTHER'
                );
            END IF;
        END
        $$;
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS projects (
            id SERIAL PRIMARY KEY,
            project_id VARCHAR(255) UNIQUE NOT NULL,
            hackathon_id INTEGER REFERENCES hackathons(id) ON DELETE SET NULL,
            short_description TEXT DEFAULT '',
            long_description TEXT DEFAULT '',
            github_link TEXT DEFAULT '',
            demo_link TEXT DEFAULT NULL,
            theme TEXT DEFAULT '',
            is_reviewed BOOLEAN DEFAULT FALSE,
            code_agent_analysis JSONB DEFAULT '[]'::jsonb,
            market_agent_analysis JSONB DEFAULT '[]'::jsonb,
            overall_score DECIMAL(3,2) DEFAULT NULL,
            score_explanation TEXT DEFAULT '',
            project_type project_type DEFAULT 'OTHER',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS evaluations (
            id SERIAL PRIMARY KEY,
            project_id VARCHAR(255) REFERENCES projects(project_id) ON DELETE CASCADE,
            criteria_name VARCHAR(255) NOT NULL,
            score DECIMAL(3,2) DEFAULT 0.00,
            remarks TEXT DEFAULT '',
            agent_type VARCHAR(50) DEFAULT 'code',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Add new columns if they don't exist
    try:
        cur.execute(
            "ALTER TABLE projects ADD COLUMN overall_score DECIMAL(3,2) DEFAULT NULL"
        )
    except:
        pass
    try:
        cur.execute("ALTER TABLE projects ADD COLUMN score_explanation TEXT DEFAULT ''")
    except:
        pass
    try:
        cur.execute("ALTER TABLE projects ADD COLUMN demo_link TEXT DEFAULT NULL")
    except:
        pass
    try:
        cur.execute(
            "ALTER TABLE projects ADD COLUMN project_type project_type DEFAULT 'OTHER'"
        )
    except:
        pass

    conn.commit()
    cur.close()
    conn.close()
