"""数据源插件测试 — GitHub / RSS / PubMed。"""

from __future__ import annotations

import json
from unittest.mock import patch

# ═══════════════════════════════════════════════════════════════════════════
# 插件发现测试
# ═══════════════════════════════════════════════════════════════════════════


class TestPluginDiscovery:
    def test_github_plugin_discovered(self):
        from horizonrl.plugins.registry import PluginRegistry

        registry = PluginRegistry()
        result = registry.discover("plugins")
        assert "github_search" in result

    def test_rss_plugin_discovered(self):
        from horizonrl.plugins.registry import PluginRegistry

        registry = PluginRegistry()
        result = registry.discover("plugins")
        assert "rss_feed" in result

    def test_pubmed_plugin_discovered(self):
        from horizonrl.plugins.registry import PluginRegistry

        registry = PluginRegistry()
        result = registry.discover("plugins")
        assert "pubmed_search" in result


# ═══════════════════════════════════════════════════════════════════════════
# GitHub Plugin 测试
# ═══════════════════════════════════════════════════════════════════════════


class TestGithubPlugin:
    def test_config_defaults(self):
        from plugins.github_search import GithubPluginConfig

        cfg = GithubPluginConfig()
        assert cfg.search_type == "repositories"
        assert cfg.per_page == 10

    def test_build_params(self):
        from plugins.github_search import GithubPlugin

        plugin = GithubPlugin()
        params = plugin.build_params("搜索Python机器学习项目", "")
        assert "query" in params
        assert params["query"] == "Python机器学习项目"
        assert params["num_results"] == 10

    def test_extract_evidence(self):
        from plugins.github_search import GithubPlugin

        plugin = GithubPlugin()
        output = json.dumps([
            {"title": "user/repo", "url": "https://github.com/user/repo",
             "description": "A great repo", "provider": "github"},
        ])
        evs = plugin.extract_evidence(output)
        assert len(evs) == 1
        assert evs[0].source_type == "github"
        assert "user/repo" in evs[0].content

    def test_extract_evidence_fallback(self):
        from plugins.github_search import GithubPlugin

        plugin = GithubPlugin()
        evs = plugin.extract_evidence("not json")
        assert len(evs) == 1
        assert evs[0].source_type == "github"

    def test_fallback_results(self):
        from plugins.github_search import GithubPlugin

        results = GithubPlugin._fallback_results("test query")
        assert len(results) == 1
        assert results[0]["is_mock"] is True
        assert "github.com/search" in results[0]["url"]

    def test_execute_with_mock_response(self):
        import asyncio
        import json as _json

        from plugins.github_search import GithubPlugin

        plugin = GithubPlugin()

        mock_resp = _json.dumps({
            "items": [
                {"full_name": "torvalds/linux", "html_url": "https://github.com/torvalds/linux",
                 "description": "Linux kernel", "stargazers_count": 180000,
                 "language": "C", "topics": []},
            ]
        }).encode("utf-8")

        async def run():
            with patch.object(plugin, "_fetch_json", return_value=_json.loads(mock_resp)):
                result = await plugin.execute(query="linux kernel")
                return result

        result = asyncio.run(run())
        items = _json.loads(result)
        assert len(items) == 1
        assert items[0]["title"] == "torvalds/linux"
        assert items[0]["provider"] == "github"

    def test_execute_handles_api_error(self):
        import asyncio
        import json as _json

        from plugins.github_search import GithubPlugin

        plugin = GithubPlugin()

        async def run():
            with patch.object(plugin, "_fetch_json", return_value={}):
                result = await plugin.execute(query="nonexistent_xyzzy42")
                return result

        result = asyncio.run(run())
        items = _json.loads(result)
        assert len(items) == 1
        assert items[0]["is_mock"] is True

    def test_provider_info(self):
        from plugins.github_search import GithubPlugin

        info = GithubPlugin.get_provider_info()
        assert info["provider_id"] == "github"
        assert info["env_var"] == "GITHUB_TOKEN"


# ═══════════════════════════════════════════════════════════════════════════
# RSS Plugin 测试
# ═══════════════════════════════════════════════════════════════════════════


SAMPLE_RSS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Test Feed</title>
    <link>http://example.com</link>
    <item>
      <title>Article One</title>
      <link>http://example.com/1</link>
      <description>First test article about machine learning</description>
      <pubDate>Mon, 01 Jan 2024 00:00:00 GMT</pubDate>
    </item>
    <item>
      <title>Article Two</title>
      <link>http://example.com/2</link>
      <description>Second test article about deep learning</description>
      <pubDate>Tue, 02 Jan 2024 00:00:00 GMT</pubDate>
    </item>
  </channel>
</rss>"""

SAMPLE_ATOM_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Test Atom Feed</title>
  <entry>
    <title>Atom Entry One</title>
    <link href="http://example.com/a1"/>
    <summary>First atom entry</summary>
  </entry>
</feed>
"""


class TestRssPlugin:
    def test_config_defaults(self):
        from plugins.rss_feed import RssPluginConfig

        cfg = RssPluginConfig()
        assert cfg.max_entries == 20

    def test_build_params_extracts_url(self):
        from plugins.rss_feed import RssPlugin

        plugin = RssPlugin()
        params = plugin.build_params("https://example.com/feed.xml 搜索AI", "")
        assert params["url"] == "https://example.com/feed.xml"
        assert "AI" in params["query"]

    def test_build_params_no_url(self):
        from plugins.rss_feed import RssPlugin

        plugin = RssPlugin()
        params = plugin.build_params("搜索AI最新进展", "")
        assert params["url"] == ""
        assert "搜索AI最新进展" in params["query"]

    def test_parse_rss_feed(self):
        from plugins.rss_feed import RssPlugin

        plugin = RssPlugin()
        entries = plugin._parse_feed(SAMPLE_RSS_XML)
        assert len(entries) == 2
        assert entries[0]["title"] == "Article One"
        assert entries[0]["link"] == "http://example.com/1"

    def test_parse_atom_feed_stdlib(self):
        from plugins.rss_feed import _parse_feed_stdlib
        entries = _parse_feed_stdlib(SAMPLE_ATOM_XML)
        assert len(entries) == 1
        assert entries[0]["title"] == "Atom Entry One"

    def test_extract_evidence(self):
        from plugins.rss_feed import RssPlugin

        plugin = RssPlugin()
        output = json.dumps([
            {"title": "News", "link": "http://x.com", "summary": "summary text"},
        ])
        evs = plugin.extract_evidence(output)
        assert len(evs) == 1
        assert evs[0].source_type == "rss"
        assert "News" in evs[0].content

    def test_execute_with_mock_fetch(self):
        import asyncio
        import json as _json

        from plugins.rss_feed import RssPlugin

        plugin = RssPlugin()

        async def run():
            with patch.object(plugin, "_fetch_url", return_value=SAMPLE_RSS_XML):
                result = await plugin.execute(url="http://example.com/feed", query="deep")
                return result

        result = asyncio.run(run())
        items = _json.loads(result)
        assert len(items) == 1
        assert items[0]["title"] == "Article Two"

    def test_execute_no_url_returns_error(self):
        import asyncio
        import json as _json

        from plugins.rss_feed import RssPlugin

        plugin = RssPlugin()
        result = asyncio.run(
            plugin.execute(url="")
        )
        data = _json.loads(result)
        assert "error" in data


# ═══════════════════════════════════════════════════════════════════════════
# PubMed Plugin 测试
# ═══════════════════════════════════════════════════════════════════════════


SAMPLE_ESEARCH_JSON = {
    "esearchresult": {
        "idlist": ["12345", "67890"],
    },
}

SAMPLE_EFETCH_XML = """<?xml version="1.0" encoding="UTF-8"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>12345</PMID>
      <Article>
        <ArticleTitle>Deep learning for cancer detection</ArticleTitle>
        <Abstract>
          <AbstractText>This paper presents a novel method...</AbstractText>
        </Abstract>
        <AuthorList>
          <Author>
            <LastName>Smith</LastName><ForeName>John</ForeName>
          </Author>
        </AuthorList>
        <Journal>
          <Title>Nature Medicine</Title>
          <JournalIssue>
            <PubDate><Year>2024</Year></PubDate>
          </JournalIssue>
        </Journal>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>"""


class TestPubmedPlugin:
    def test_config_defaults(self):
        from plugins.pubmed_search import PubmedPluginConfig

        cfg = PubmedPluginConfig()
        assert cfg.max_results == 10

    def test_build_params(self):
        from plugins.pubmed_search import PubmedPlugin

        plugin = PubmedPlugin()
        params = plugin.build_params("搜索癌症免疫治疗最新进展", "")
        assert params["query"] == "癌症免疫治疗最新进展"
        assert params["max_results"] == 10

    def test_parse_efetch_xml(self):
        from plugins.pubmed_search import PubmedPlugin

        plugin = PubmedPlugin()
        articles = plugin._parse_efetch_xml(SAMPLE_EFETCH_XML)
        assert len(articles) == 1
        assert articles[0]["pmid"] == "12345"
        assert articles[0]["title"] == "Deep learning for cancer detection"
        assert articles[0]["journal"] == "Nature Medicine"
        assert articles[0]["year"] == "2024"
        assert "Smith" in articles[0]["authors"][0]

    def test_extract_evidence(self):
        from plugins.pubmed_search import PubmedPlugin

        plugin = PubmedPlugin()
        output = json.dumps([
            {"title": "Paper", "url": "https://pubmed.ncbi.nlm.nih.gov/12345/",
             "abstract": "An important finding", "provider": "pubmed"},
        ])
        evs = plugin.extract_evidence(output)
        assert len(evs) == 1
        assert evs[0].source_type == "paper"
        assert "Paper" in evs[0].content

    def test_extract_evidence_fallback(self):
        from plugins.pubmed_search import PubmedPlugin

        plugin = PubmedPlugin()
        output = json.dumps([
            {"provider": "pubmed_fallback", "is_mock": True, "title": "Fallback"},
        ])
        evs = plugin.extract_evidence(output)
        assert len(evs) == 1
        assert evs[0].is_mock is True

    def test_fallback_results(self):
        from plugins.pubmed_search import PubmedPlugin

        results = PubmedPlugin._fallback_results("cancer")
        assert len(results) == 1
        assert results[0]["is_mock"] is True
        assert "pubmed.ncbi.nlm.nih.gov" in results[0]["url"]

    def test_execute_with_mock(self):
        import asyncio
        import json as _json

        from plugins.pubmed_search import PubmedPlugin

        plugin = PubmedPlugin()

        async def run():
            with patch.object(plugin, "_esearch", return_value=["12345"]):
                with patch.object(plugin, "_efetch", return_value=[{
                    "pmid": "12345",
                    "title": "Test Article",
                    "abstract": "Test abstract text",
                    "authors": ["Author One"],
                    "journal": "Test Journal",
                    "year": "2024",
                }]):
                    result = await plugin.execute(query="test query")
                    return result

        result = asyncio.run(run())
        items = _json.loads(result)
        assert len(items) == 1
        assert items[0]["title"] == "Test Article"
        assert items[0]["provider"] == "pubmed"

    def test_execute_handles_api_error(self):
        import asyncio
        import json as _json

        from plugins.pubmed_search import PubmedPlugin

        plugin = PubmedPlugin()

        async def run():
            with patch.object(plugin, "_esearch", return_value=[]):
                result = await plugin.execute(query="test query")
                return result

        result = asyncio.run(run())
        items = _json.loads(result)
        assert len(items) == 1
        assert items[0]["is_mock"] is True

    def test_provider_info(self):
        from plugins.pubmed_search import PubmedPlugin

        info = PubmedPlugin.get_provider_info()
        assert info["provider_id"] == "ncbi"
        assert info["env_var"] == "NCBI_API_KEY"


# ═══════════════════════════════════════════════════════════════════════════
# 集成: AgentWorker 自动识别插件工具
# ═══════════════════════════════════════════════════════════════════════════


class TestAgentWorkerDataSourceIntegration:
    def test_worker_builds_params_for_github(self):
        from horizonrl.agent.worker import AgentWorker
        from horizonrl.schemas.task import TaskSpec
        from horizonrl.tools.manager import ToolManager
        from plugins.github_search import GithubPlugin

        mgr = ToolManager()
        mgr.register_plugin("github_search", GithubPlugin())
        worker = AgentWorker(worker_id="w1", tool_manager=mgr)
        task = TaskSpec(
            id="t1", name="test", description="搜索PyTorch项目",
            tool_names=["github_search"], depends_on=[], context="",
        )
        params = worker._build_params("github_search", task)
        assert params["query"] == "PyTorch项目"
        assert params["num_results"] == 10

    def test_worker_builds_params_for_pubmed(self):
        from horizonrl.agent.worker import AgentWorker
        from horizonrl.schemas.task import TaskSpec
        from horizonrl.tools.manager import ToolManager
        from plugins.pubmed_search import PubmedPlugin

        mgr = ToolManager()
        mgr.register_plugin("pubmed_search", PubmedPlugin())
        worker = AgentWorker(worker_id="w1", tool_manager=mgr)
        task = TaskSpec(
            id="t1", name="test", description="搜索癌症免疫治疗",
            tool_names=["pubmed_search"], depends_on=[], context="",
        )
        params = worker._build_params("pubmed_search", task)
        assert params["query"] == "癌症免疫治疗"
        assert params["max_results"] == 10

    def test_worker_builds_params_for_rss(self):
        from horizonrl.agent.worker import AgentWorker
        from horizonrl.schemas.task import TaskSpec
        from horizonrl.tools.manager import ToolManager
        from plugins.rss_feed import RssPlugin

        mgr = ToolManager()
        mgr.register_plugin("rss_feed", RssPlugin())
        worker = AgentWorker(worker_id="w1", tool_manager=mgr)
        task = TaskSpec(
            id="t1", name="test",
            description="https://example.com/feed.xml 搜索AI",
            tool_names=["rss_feed"], depends_on=[], context="",
        )
        params = worker._build_params("rss_feed", task)
        assert params["url"] == "https://example.com/feed.xml"
        assert "AI" in params["query"]
