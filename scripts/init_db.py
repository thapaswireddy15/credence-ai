"""Create the MySQL database and tables, and export the DDL to sql/schema.sql.

Run:  python -m scripts.init_db            (create database + tables)
      python -m scripts.init_db --ddl-only (only regenerate sql/schema.sql)
"""

import argparse
import sys
from pathlib import Path

from sqlalchemy.dialects import mysql
from sqlalchemy.schema import CreateIndex, CreateTable

from app import models  # noqa: F401  (registers tables)
from app.config import get_settings
from app.database import Base, check_db_connection, engine, init_db

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "sql" / "schema.sql"


def export_ddl() -> None:
    dialect = mysql.dialect()
    db = get_settings().database_url.database
    lines = [
        "-- Credence AI MySQL schema (generated from app/models.py by scripts/init_db.py)",
        f"CREATE DATABASE IF NOT EXISTS `{db}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;",
        f"USE `{db}`;",
        "",
    ]
    for table in Base.metadata.sorted_tables:
        lines.append(str(CreateTable(table).compile(dialect=dialect)).strip() + ";\n")
        for index in sorted(table.indexes, key=lambda i: i.name):
            lines.append(str(CreateIndex(index).compile(dialect=dialect)).strip() + ";")
        lines.append("")
    SCHEMA_PATH.parent.mkdir(exist_ok=True)
    SCHEMA_PATH.write_text("\n".join(lines))
    print(f"Wrote {SCHEMA_PATH.relative_to(SCHEMA_PATH.parent.parent)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ddl-only", action="store_true")
    args = parser.parse_args()

    export_ddl()
    if args.ddl_only:
        return 0

    url = get_settings().database_url
    print(f"Connecting to MySQL at {url.host}:{url.port} as {url.username} ...")
    try:
        init_db()
    except Exception as exc:
        print(f"FAILED: {exc}\nCheck MYSQL_* / DATABASE_URL in your .env file.")
        return 1
    ok = check_db_connection()
    print(f"Database `{url.database}` ready with tables: {', '.join(Base.metadata.tables)}"
          if ok else "Tables created but connection check failed.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
