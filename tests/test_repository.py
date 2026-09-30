import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from agents.investigator import Investigator
from config import Settings
from services.cache.db import connect
from services.cache.repository import HistoryError, InvestigationRepository
from services.cache.search_cache import SearchCache, make_cache_key
from services.demo import demo_client, demo_opportunity, demo_resume

NOW = datetime(2026, 9, 30, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def report():
    settings = Settings(_env_file=None, llm_provider="none")
    return Investigator(settings, demo_client(NOW, settings)).run(demo_opportunity(NOW), resume=demo_resume(), now=NOW, is_demo=True)


@pytest.fixture
def repo(tmp_path):
    return InvestigationRepository(tmp_path / "history.db")


def count(repo, table):
    with connect(repo._db_path) as conn:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def test_save_then_reopen_returns_an_identical_report(repo, report):
    repo.save(report)
    assert repo.get(report.id) == report


def test_every_brief_table_is_populated(repo, report):
    repo.save(report)
    assert count(repo, "investigations") == count(repo, "opportunities") == count(repo, "reports") == 1
    assert count(repo, "search_results") == len(report.method.searches) == 9
    assert count(repo, "evidence") == len(report.evidence)
    with connect(repo._db_path) as conn:
        row = conn.execute("SELECT * FROM investigations").fetchone()
    assert (row["company"], row["role"], row["status"], row["is_demo"], row["input_url"]) == (
        "Lumenpath Labs", "Software Engineer Intern", "complete", 1, report.opportunity.source_url)


def test_saving_the_same_report_twice_replaces_it(repo, report):
    repo.save(report)
    repo.save(report)
    assert (count(repo, "investigations"), count(repo, "evidence"), count(repo, "search_results")) == (1, len(report.evidence), 9)


def test_history_lists_newest_first_and_respects_the_limit(repo, report):
    for i in range(3):
        repo.save(report.model_copy(update={"id": f"r{i}", "created_at": NOW + timedelta(minutes=i)}))
    summaries = repo.list()
    assert [s.id for s in summaries] == ["r2", "r1", "r0"]
    assert summaries[0].company == "Lumenpath Labs" and summaries[0].is_demo is True
    assert len(repo.list(limit=2)) == 2


def test_delete_removes_the_report_and_all_child_rows(repo, report):
    repo.save(report)
    assert repo.delete(report.id) is True
    assert repo.get(report.id) is None
    assert all(count(repo, t) == 0 for t in ("investigations", "opportunities", "search_results", "evidence", "reports"))
    assert repo.delete(report.id) is False


def test_delete_all(repo, report):
    for i in range(3):
        repo.save(report.model_copy(update={"id": f"r{i}"}))
    assert repo.delete_all() == 3 and repo.list() == []


def test_unknown_id_and_empty_history(repo):
    assert repo.get("nope") is None and repo.list() == []


def test_a_report_that_no_longer_validates_is_unreadable_not_a_crash(repo, report):
    repo.save(report)
    with connect(repo._db_path) as conn:
        conn.execute("UPDATE reports SET report_json = ? WHERE investigation_id = ?", ('{"id": "x", "old": "format"}', report.id))
    assert repo.get(report.id) is None
    assert [s.id for s in repo.list()] == [report.id]  # still listed, so the user can delete it

    with connect(repo._db_path) as conn:
        conn.execute("UPDATE reports SET report_json = 'not json' WHERE investigation_id = ?", (report.id,))
    assert repo.get(report.id) is None


def test_database_failures_surface_as_a_safe_history_error(tmp_path, report):
    blocked = tmp_path / "file.txt"
    blocked.write_text("x")
    repo = InvestigationRepository(blocked / "sub" / "history.db")  # parent is a file: cannot create the db
    with pytest.raises(HistoryError) as info:
        repo.save(report)
    assert "data folder" in info.value.user_message
    assert repo.get("x") is None and repo.list() == []  # reads degrade gracefully


def test_history_and_search_cache_can_share_one_database(tmp_path, report):
    db = tmp_path / "shared.db"
    SearchCache(db).put(make_cache_key("google", {"q": "x"}), "google", "x", {"organic_results": []})
    repo = InvestigationRepository(db)
    repo.save(report)
    assert SearchCache(db).get(make_cache_key("google", {"q": "x"}), 24) is not None
    assert repo.get(report.id) == report


def test_foreign_keys_are_enforced(repo, report):
    repo.save(report)
    with pytest.raises(sqlite3.IntegrityError):
        with connect(repo._db_path) as conn:
            conn.execute("INSERT INTO evidence (investigation_id, evidence_id, item_json) VALUES ('ghost', 'E1', '{}')")


def test_search_cache_degrades_silently_when_the_data_folder_is_unusable(tmp_path):
    blocked = tmp_path / "file.txt"
    blocked.write_text("x")
    cache = SearchCache(blocked / "sub" / "cache.db")
    key = make_cache_key("google", {"q": "x"})
    cache.put(key, "google", "x", {"organic_results": []})  # must not raise
    assert cache.get(key, 24) is None
