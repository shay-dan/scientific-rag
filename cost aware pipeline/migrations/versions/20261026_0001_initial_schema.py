"""Adopt the existing schema and create it for new installations.

Revision ID: 20261026_0001
Revises:
Create Date: 2026-10-26
"""

from __future__ import annotations

from collections.abc import Callable

from alembic import op
from pgvector.sqlalchemy import Vector
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "20261026_0001"
down_revision = None
branch_labels = None
depends_on = None


def _tables() -> set[str]:
    return set(sa.inspect(op.get_bind()).get_table_names())


def _columns(table_name: str) -> set[str]:
    return {item["name"] for item in sa.inspect(op.get_bind()).get_columns(table_name)}


def _indexes(table_name: str) -> set[str]:
    return {item["name"] for item in sa.inspect(op.get_bind()).get_indexes(table_name)}


def _ensure_table(name: str, create: Callable[[], None]) -> None:
    if name not in _tables():
        create()


def _ensure_index(name: str, table_name: str, columns: list[str], **kwargs) -> None:
    if name not in _indexes(table_name):
        op.create_index(name, table_name, columns, **kwargs)


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    _ensure_table("research_runs", lambda: op.create_table(
        "research_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("requested_by", sa.String(length=128), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("answer", sa.Text(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("model_metadata", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    ))
    if "requested_by" not in _columns("research_runs"):
        op.add_column("research_runs", sa.Column("requested_by", sa.String(length=128), nullable=True))
    _ensure_index("ix_research_runs_requested_by", "research_runs", ["requested_by"])

    _ensure_table("evaluation_datasets", lambda: op.create_table(
        "evaluation_datasets",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(length=256), nullable=False),
        sa.Column("created_by", sa.String(length=128), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name", name="evaluation_datasets_name_key"),
    ))

    _ensure_table("source_documents", lambda: op.create_table(
        "source_documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("research_run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("external_id", sa.String(length=512), nullable=True),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("source", sa.String(length=64), nullable=False),
        sa.Column("source_url", sa.Text(), nullable=True),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(["research_run_id"], ["research_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    ))

    _ensure_table("parent_documents", lambda: op.create_table(
        "parent_documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("source_document_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(["source_document_id"], ["source_documents.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("source_document_id", "ordinal", name="uq_parent_document_ordinal"),
    ))

    _ensure_table("research_evidence", lambda: op.create_table(
        "research_evidence",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("research_run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(["research_run_id"], ["research_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("research_run_id", "ordinal", name="uq_research_evidence_ordinal"),
    ))
    _ensure_index("ix_research_evidence_run_ordinal", "research_evidence", ["research_run_id", "ordinal"])

    _ensure_table("research_feedback", lambda: op.create_table(
        "research_feedback",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("research_run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("submitted_by", sa.String(length=128), nullable=False),
        sa.Column("rating", sa.String(length=32), nullable=False),
        sa.Column("categories", sa.JSON(), nullable=False),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["research_run_id"], ["research_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    ))
    _ensure_index("ix_research_feedback_research_run_id", "research_feedback", ["research_run_id"])
    _ensure_index("ix_research_feedback_submitted_by", "research_feedback", ["submitted_by"])

    _ensure_table("audit_events", lambda: op.create_table(
        "audit_events",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("actor", sa.String(length=128), nullable=False),
        sa.Column("action", sa.String(length=64), nullable=False),
        sa.Column("target_type", sa.String(length=64), nullable=True),
        sa.Column("target_id", sa.String(length=128), nullable=True),
        sa.Column("details", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    ))
    _ensure_index("ix_audit_events_actor", "audit_events", ["actor"])
    _ensure_index("ix_audit_events_action", "audit_events", ["action"])
    _ensure_index("ix_audit_events_created_at", "audit_events", ["created_at"])

    _ensure_table("evaluation_cases", lambda: op.create_table(
        "evaluation_cases",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("dataset_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("reference_answer", sa.Text(), nullable=False),
        sa.Column("reference_context", sa.Text(), nullable=True),
        sa.Column("reference_claims", sa.JSON(), server_default=sa.text("'[]'::json"), nullable=False),
        sa.Column("provenance", sa.String(length=32), nullable=False),
        sa.ForeignKeyConstraint(["dataset_id"], ["evaluation_datasets.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    ))
    if "reference_claims" not in _columns("evaluation_cases"):
        op.add_column(
            "evaluation_cases",
            sa.Column("reference_claims", sa.JSON(), server_default=sa.text("'[]'::json"), nullable=False),
        )

    _ensure_table("evaluation_runs", lambda: op.create_table(
        "evaluation_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("dataset_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("metrics", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["dataset_id"], ["evaluation_datasets.id"]),
        sa.PrimaryKeyConstraint("id"),
    ))

    _ensure_table("evaluation_results", lambda: op.create_table(
        "evaluation_results",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("evaluation_run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("evaluation_case_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("research_run_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("answer", sa.Text(), nullable=True),
        sa.Column("retrieved_contexts", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("judge_scores", sa.JSON(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["evaluation_case_id"], ["evaluation_cases.id"]),
        sa.ForeignKeyConstraint(["evaluation_run_id"], ["evaluation_runs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["research_run_id"], ["research_runs.id"]),
        sa.PrimaryKeyConstraint("id"),
    ))

    _ensure_table("document_chunks", lambda: op.create_table(
        "document_chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("source_document_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("parent_document_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.Column("search_vector", postgresql.TSVECTOR(), nullable=True),
        sa.Column("embedding", Vector(dim=768), nullable=True),
        sa.Column("token_count", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(["parent_document_id"], ["parent_documents.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["source_document_id"], ["source_documents.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    ))
    if "parent_document_id" not in _columns("document_chunks"):
        op.add_column("document_chunks", sa.Column("parent_document_id", postgresql.UUID(as_uuid=True), nullable=True))
        op.create_foreign_key(
            "document_chunks_parent_document_id_fkey",
            "document_chunks",
            "parent_documents",
            ["parent_document_id"],
            ["id"],
            ondelete="CASCADE",
        )
    _ensure_index("ix_document_chunks_parent_document_id", "document_chunks", ["parent_document_id"])
    _ensure_index(
        "ix_document_chunks_search_vector",
        "document_chunks",
        ["search_vector"],
        postgresql_using="gin",
    )
    _ensure_index(
        "ix_document_chunks_embedding_hnsw",
        "document_chunks",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )

    expected_columns = {
        "research_runs": {"id", "question", "requested_by", "status", "answer", "error", "model_metadata", "created_at", "completed_at"},
        "source_documents": {"id", "research_run_id", "external_id", "title", "source", "source_url", "metadata_json"},
        "parent_documents": {"id", "source_document_id", "ordinal", "content", "metadata_json"},
        "research_evidence": {"id", "research_run_id", "ordinal", "content", "metadata_json"},
        "research_feedback": {"id", "research_run_id", "submitted_by", "rating", "categories", "comment", "created_at"},
        "audit_events": {"id", "actor", "action", "target_type", "target_id", "details", "created_at"},
        "evaluation_datasets": {"id", "name", "created_by", "created_at"},
        "evaluation_cases": {"id", "dataset_id", "question", "reference_answer", "reference_context", "reference_claims", "provenance"},
        "evaluation_runs": {"id", "dataset_id", "status", "metrics", "created_at", "completed_at"},
        "evaluation_results": {"id", "evaluation_run_id", "evaluation_case_id", "research_run_id", "answer", "retrieved_contexts", "status", "judge_scores", "error"},
        "document_chunks": {"id", "source_document_id", "parent_document_id", "content", "metadata_json", "search_vector", "embedding", "token_count"},
    }
    for table_name, required in expected_columns.items():
        missing = required - _columns(table_name)
        if missing:
            raise RuntimeError(f"Cannot adopt incomplete table {table_name}; missing columns: {sorted(missing)}")


def downgrade() -> None:
    op.drop_table("document_chunks")
    op.drop_table("evaluation_results")
    op.drop_table("evaluation_runs")
    op.drop_table("evaluation_cases")
    op.drop_table("audit_events")
    op.drop_table("research_feedback")
    op.drop_table("research_evidence")
    op.drop_table("parent_documents")
    op.drop_table("source_documents")
    op.drop_table("evaluation_datasets")
    op.drop_table("research_runs")
    # The vector extension may be shared by other schemas/applications, so the
    # downgrade deliberately does not remove it.
