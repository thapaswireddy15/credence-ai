"""MySQL connection management with SQLAlchemy ORM + PyMySQL."""

from collections.abc import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL, Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings


class Base(DeclarativeBase):
    pass


def make_engine(url: URL) -> Engine:
    return create_engine(
        url,
        pool_pre_ping=True,  # transparently replace connections MySQL has closed
        pool_recycle=280,  # stay below typical MySQL / cloud idle timeouts
        future=True,
    )


def ensure_database_exists(url: URL) -> None:
    """Create the target MySQL database if it does not exist yet."""
    database = url.database
    if not database:
        raise ValueError("DATABASE_URL must include a database name")
    server_engine = create_engine(url.set(database=""), future=True)
    try:
        with server_engine.begin() as conn:
            conn.execute(
                text(f"CREATE DATABASE IF NOT EXISTS `{database}` "
                     "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
            )
    finally:
        server_engine.dispose()


engine = make_engine(get_settings().database_url)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def init_db(bind: Engine = engine, create_database: bool = True) -> None:
    """Create the database (optional) and all tables."""
    from app import models  # noqa: F401  (register models on Base.metadata)

    if create_database:
        ensure_database_exists(bind.url)
    Base.metadata.create_all(bind=bind)
    _apply_additive_migrations(bind)


def _apply_additive_migrations(bind: Engine) -> None:
    """Bring tables created by older versions up to date (add-only, never drops data)."""
    from sqlalchemy import inspect

    columns = {c["name"] for c in inspect(bind).get_columns("predictions")}
    if "imputed_features" not in columns:
        with bind.begin() as conn:
            conn.execute(text("ALTER TABLE predictions ADD COLUMN imputed_features VARCHAR(300) NULL "
                              "AFTER model_version"))


def check_db_connection(bind: Engine = engine) -> bool:
    try:
        with bind.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency: one session per request, always closed."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
