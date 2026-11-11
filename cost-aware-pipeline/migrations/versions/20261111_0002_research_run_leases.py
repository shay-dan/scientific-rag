"""Add fencing and lease columns for recoverable research jobs (roadmap R2).

``attempt`` is an optimistic version counter used to fence stale workers.
``lease_owner`` and ``lease_expires_at`` record which worker holds a running
run and until when; the worker renews the lease while it is alive.
Existing rows keep attempt=0 and no lease.

Revision ID: 20261111_0002
Revises: 20261026_0001
Create Date: 2026-11-11
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20261111_0002"
down_revision = "20261026_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("research_runs", sa.Column("attempt", sa.Integer(), nullable=False, server_default=sa.text("0")))
    op.add_column("research_runs", sa.Column("lease_owner", sa.String(length=128), nullable=True))
    op.add_column("research_runs", sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True))
    op.create_index(
        "ix_research_runs_active",
        "research_runs",
        ["status", "lease_expires_at"],
        postgresql_where=sa.text("status IN ('queued', 'running')"),
    )


def downgrade() -> None:
    op.drop_index("ix_research_runs_active", table_name="research_runs")
    op.drop_column("research_runs", "lease_expires_at")
    op.drop_column("research_runs", "lease_owner")
    op.drop_column("research_runs", "attempt")
