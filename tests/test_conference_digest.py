from datetime import date, timedelta
import json
from urllib.parse import unquote

import pytest

from zotero_arxiv_daily import conference_digest

from tests.canned_responses import make_sample_paper


def test_daily_candidates_rotate_and_exclude_duplicate_titles():
    catalogue = [make_sample_paper(title=f"Conference Paper {i}", abstract="") for i in range(4000)]
    day = date(2026, 11, 15)
    today = conference_digest.daily_conference_candidates(catalogue, [], day)
    tomorrow = conference_digest.daily_conference_candidates(catalogue, [], day + timedelta(days=1))
    assert len(today) >= 20 and len(tomorrow) >= 20
    assert {paper.title for paper in today}.isdisjoint(paper.title for paper in tomorrow)

    saved = today[0].title.upper() + "."
    preprint = today[1].title
    repeat = make_sample_paper(title=today[2].title + "!", abstract="Published abstract")
    updated = conference_digest.daily_conference_candidates(
        [repeat, *reversed(catalogue)], [saved, preprint], day,
    )
    assert updated == [repeat, *today[3:]]


def test_candidate_limit_preserves_smaller_venues(monkeypatch):
    day = date(2026, 11, 15)
    large = [make_sample_paper(title=f"Large Venue Paper {i}", source="NeurIPS") for i in range(5000)]
    small = [make_sample_paper(title=f"Small Venue Paper {i}", source="USENIX Security") for i in range(300)]
    assert len(conference_digest.daily_conference_candidates(small, [], day)) >= 2
    monkeypatch.setattr(conference_digest, "CANDIDATE_LIMIT", 8)

    candidates = conference_digest.daily_conference_candidates(large + small, [], day)

    assert len(candidates) == 8
    assert sum(paper.source == "USENIX Security" for paper in candidates) >= 2


def test_virtual_catalogue_requires_visible_accepted_main_track_papers(monkeypatch):
    main = {
        "name": "An accepted paper",
        "sourceurl": "https://openreview.net/group?id=NeurIPS.cc/2026/Conference",
        "visible": True,
        "decision": "Accept (poster)",
        "virtualsite_url": "/virtual/2026/poster/123",
        "authors": [{"fullname": "A. Researcher"}],
    }
    entries = [
        main,
        main | {"name": "Rejected", "decision": "Reject"},
        main | {"name": "Withdrawn", "visible": False},
        main | {"name": "Workshop", "sourceurl": main["sourceurl"] + "/Workshop"},
        main | {"name": "Oral duplicate", "virtualsite_url": "/virtual/2026/oral/123"},
    ]
    monkeypatch.setattr(conference_digest, "_get", lambda _: json.dumps({"results": entries}).encode())

    paper, = conference_digest._virtual_papers("NeurIPS")
    assert paper.title == main["name"]
    assert paper.authors == ["A. Researcher"]
    assert paper.url == "https://neurips.cc/virtual/2026/poster/123"


def test_one_unavailable_venue_does_not_stop_the_others(monkeypatch):
    def unavailable():
        raise TimeoutError("catalogue timed out")

    acl = make_sample_paper(source="ACL")
    monkeypatch.setattr(conference_digest, "_anthology_papers", lambda venue, _: [acl] if venue == "ACL" else [])
    monkeypatch.setattr(conference_digest, "_cvpr_papers", unavailable)
    monkeypatch.setattr(conference_digest, "_virtual_papers", lambda _: [])
    for source in ("icml", "aaai", "usenix", "sp", "ccs", "ndss", "www"):
        monkeypatch.setattr(conference_digest, f"_{source}_papers", lambda: [])

    assert conference_digest.fetch_conference_papers() == [acl]


def test_emnlp_uses_only_main_proceedings(monkeypatch):
    catalogue = b"""<collection>
      <volume id="main"><paper id="1"><title>Main Event</title><abstract>An abstract</abstract></paper></volume>
      <volume id="demos"><paper id="1"><title>A Demo</title></paper></volume>
    </collection>"""
    monkeypatch.setattr(conference_digest, "_get", lambda _: catalogue)

    paper, = conference_digest._anthology_papers("EMNLP", "main")
    assert paper.title == "Main Event"
    assert paper.abstract == "An abstract"
    assert paper.url == "https://aclanthology.org/2026.emnlp-main.1/"


@pytest.mark.parametrize(
    "fetcher,source,fixture,link_suffix",
    [
        ("usenix", "USENIX Security",
         '<article class="node node-paper"><h2><a href="/conference/usenixsecurity26/presentation/demo">One Paper</a></h2>'
         '<div class="field-name-field-paper-people-text"><p>A. Author, <em>University</em></p></div></article>',
         "/conference/usenixsecurity26/presentation/demo"),
        ("sp", "IEEE S&P",
         '<div id="cycle1"><div class="list-group-item"><b><a href="#x">One Paper</a></b>'
         '<div class="authorlist">A. Author<sup>1</sup><br>University</div></div></div>',
         "#:~:text=One Paper"),
        ("ccs", "CCS",
         '<table><tr><th>Title</th><th>Author</th></tr>'
         '<tr><td>One Paper</td><td>A. Author (University)<br>B. Author (Other)</td></tr></table>',
         "#:~:text=One Paper"),
        ("ndss", "NDSS",
         '<div><h2 class="pt-cv-title"><a href="https://www.ndss-symposium.org/ndss-paper/demo/">One Paper</a></h2>'
         '<div><div class="pt-cv-ctf-display_authors">A. Author (University)</div></div></div>',
         "/ndss-paper/demo/"),
        ("www", "WWW",
         '<div class="papers-section"><ul><li><span class="paper-id">(rfp0001)</span> One Paper — '
         '<span class="paper-authors">A. Author and B. Author</span></li></ul></div>',
         "#:~:text=One Paper"),
    ],
)
def test_new_venue_catalogues_produce_official_links(monkeypatch, fetcher, source, fixture, link_suffix):
    page = f'<html><head><meta charset="utf-8"></head><body>{fixture}</body></html>'
    monkeypatch.setattr(conference_digest, "_get", lambda _: page.encode())

    paper, = getattr(conference_digest, f"_{fetcher}_papers")()

    assert (paper.source, paper.title, paper.authors[0]) == (source, "One Paper", "A. Author")
    assert unquote(paper.url).endswith(link_suffix)
