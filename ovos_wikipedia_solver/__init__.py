# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import concurrent.futures
from functools import lru_cache
from typing import Optional, Tuple, Dict, Any

import requests
from ovos_bm25_solver import BM25MultipleChoiceSolver
from ovos_plugin_manager.keywords import load_keyword_extract_plugin
from ovos_plugin_manager.solvers import load_tldr_solver_plugin
from ovos_plugin_manager.templates.keywords import KeywordExtractor
from ovos_plugin_manager.templates.language import LanguageTranslator, LanguageDetector
from ovos_plugin_manager.templates.solvers import QuestionSolver, TldrSolver
from ovos_utils import flatten_list
from ovos_utils.log import LOG
from ovos_utils.parse import fuzzy_match, MatchStrategy
from ovos_utils.text_utils import rm_parentheses
from quebra_frases import sentence_tokenize

from ovos_wikipedia_solver.version import VERSION_BUILD, VERSION_MAJOR, VERSION_MINOR


class WikipediaSolver(QuestionSolver):
    """
    A solver for answering questions using Wikipedia search and summaries.
    """
    USER_AGENT = f"ovos-wikipedia-solver/{VERSION_MAJOR}.{VERSION_MINOR}.{VERSION_BUILD} (https://github.com/OpenVoiceOS/ovos-wikipedia-solver)"

    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 enable_tx=False,
                 translator: Optional[LanguageTranslator] = None,
                 detector: Optional[LanguageDetector] = None):
        super().__init__(config, enable_tx=enable_tx, priority=40, translator=translator, detector=detector)
        self.kword_extractors: Dict[str, KeywordExtractor] = {}
        self.summarizers: Dict[str, TldrSolver] = {}

    def get_summarizer(self, lang: str) -> Optional[TldrSolver]:
        if lang not in self.summarizers:
            summarizer_plugin: str = self.config.get("summarizer") or "ovos-summarizer-bm25"
            summarizer_class = load_tldr_solver_plugin(summarizer_plugin)
            if not summarizer_class:
                return None
            try:
                summarizer = summarizer_class(internal_lang=lang, enable_tx=self.enable_tx)
            except:
                summarizer = summarizer_class()  # some plugins dont accept all kwargs
            if self.enable_tx:  # share objects to avoid re-init
                summarizer._detector = self.detector
                summarizer._translator = self.translator
                summarizer.enable_tx = self.enable_tx
            self.summarizers[lang] = summarizer

        return self.summarizers[lang]

    def get_keyword_extractor(self, lang: str) -> Optional[KeywordExtractor]:
        if lang not in self.summarizers:
            kw_plugin: str = self.config.get("keyword_extractor") or "ovos-rake-keyword-extractor"
            kword_extractor_class = load_keyword_extract_plugin(kw_plugin)
            if not kword_extractor_class:
                return None
            kword_extractor = kword_extractor_class()
            if self.enable_tx:  # share objects to avoid re-init
                kword_extractor._detector = self.detector
                kword_extractor._translator = self.translator
                kword_extractor_class.enable_tx = self.enable_tx
            self.kword_extractors[lang] = kword_extractor
        return self.kword_extractors[lang]

    @classmethod
    @lru_cache(maxsize=128)
    def get_page_data(cls, pid: str, lang: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Fetch detailed data for a specific Wikipedia page.

        Args:
            pid (str): Page ID.
            lang (str): Language code.

        Returns:
            Tuple[Optional[str], Optional[str], Optional[str]]: Page title, summary, and image URL.
        """
        url = (f"https://{lang}.wikipedia.org/w/api.php?format=json&action=query&"
               f"prop=extracts|pageimages&exintro&explaintext&redirects=1&pageids={pid}")
        try:
            disambiguation_indicators = ["may refer to:", "refers to:"]
            response = requests.get(url, timeout=5, headers={"User-Agent": cls.USER_AGENT}).json()
            page = response["query"]["pages"][pid]
            summary = rm_parentheses(page.get("extract", ""))
            if any(i in summary for i in disambiguation_indicators):
                return None, None, None  # Disambiguation list page
            img = None
            if "thumbnail" in page:
                thumbnail = page["thumbnail"]["source"]
                parts = thumbnail.split("/")[:-1]
                img = "/".join(part for part in parts if part != "thumb")
            return page["title"], summary, img
        except Exception as e:
            LOG.error(f"Error fetching page data for PID {pid}: {e}")
            return None, None, None

    @staticmethod
    @lru_cache(maxsize=128)
    def score_page(query: str, title: str, summary: str, idx: int) -> float:
        """
        Score a Wikipedia page based on its relevance to a query.

        Args:
            query (str): User query.
            title (str): Page title.
            summary (str): Page summary.
            idx (int): Index in the original search result order.

        Returns:
            float: Relevance score.
        """
        page_mod = 1 - (idx * 0.05)  # Favor original order returned by Wikipedia
        title_score = max(fuzzy_match(query, title, MatchStrategy.DAMERAU_LEVENSHTEIN_SIMILARITY),
            fuzzy_match(query, rm_parentheses(title), MatchStrategy.DAMERAU_LEVENSHTEIN_SIMILARITY))
        summary_score = fuzzy_match(summary, title, MatchStrategy.TOKEN_SET_RATIO)
        return title_score * summary_score * page_mod

    def get_data(self, query: str, lang: Optional[str] = None, units: Optional[str] = None,
                 skip_disambiguation: bool = False):
        """Fetch Wikipedia search results and detailed data concurrently."""
        LOG.debug(f"WikiSolver query: {query}")
        lang = (lang or self.default_lang).split("-")[0]
        search_url = (f"https://{lang}.wikipedia.org/w/api.php?action=query&list=search&"
                      f"srsearch={query}&format=json")
        try:
            search_results = requests.get(search_url, timeout=5, headers={"User-Agent": self.USER_AGENT}).json().get(
                "query", {}).get("search", [])
        except Exception as e:
            LOG.error(f"Error fetching search results: {e}")
            search_results = []

        if not search_results:
            kwx = self.get_keyword_extractor(lang)
            fallback_query = max(kwx.extract(query, lang=lang))
            if fallback_query and fallback_query != query:
                LOG.debug(f"WikiSolver Fallback, new query: {fallback_query}")
                return self.get_data(fallback_query, lang=lang, units=units)
            return {}

        top_k = 3 if not skip_disambiguation else 1
        LOG.debug(f"Matched {len(search_results)} Wikipedia pages, using top {top_k}")
        search_results = search_results[:top_k]

        # Prepare for parallel fetch and maintain original order
        summaries = [None] * len(search_results)  # List to hold results in original order
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_to_idx = {executor.submit(self.get_page_data, str(r["pageid"]), lang): idx for idx, r in
                enumerate(search_results) if "(disambiguation)" not in r["title"]}

            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]  # Get original index from future
                title, ans, img = future.result()
                if title and ans:
                    summaries[idx] = (title, ans, img)

        summaries = [s for s in summaries if s is not None]
        if not summaries:
            return {}

        reranked = []
        shorts = []
        summarizer = self.get_summarizer(lang)
        for idx, (title, summary, img) in enumerate(summaries):
            try:
                short = summarizer.tldr(summary, lang=lang)
            except Exception as e:
                short = summary

            score = self.score_page(query, title, short, idx)
            reranked.append((idx, score))
            shorts.append(short)

        reranked = sorted(reranked, key=lambda x: x[1], reverse=True)
        selected = reranked[0][0]

        return {"title": summaries[selected][0], "short_answer": shorts[selected], "summary": summaries[selected][1],
            "img": summaries[selected][2], }

    def get_spoken_answer(self, query: str, lang: Optional[str] = None, units: Optional[str] = None,
                          skip_disambiguation: bool = False):
        data = self.get_data(query, lang=lang, units=units, skip_disambiguation=skip_disambiguation)
        return data.get("short_answer", "")

    def get_image(self, query: str, lang: Optional[str] = None, units: Optional[str] = None,
                  skip_disambiguation: bool = True):
        data = self.get_data(query, lang=lang, units=units, skip_disambiguation=skip_disambiguation)
        return data.get("img", "")

    def get_expanded_answer(self, query: str, lang: Optional[str] = None, units: Optional[str] = None,
                            skip_disambiguation: bool = False):
        """
        return a list of ordered steps to expand the answer, eg, "tell me more"
        {
            "title": "optional",
            "summary": "speak this",
            "img": "optional/path/or/url
        }
        """
        data = self.get_data(query, lang=lang, units=units, skip_disambiguation=skip_disambiguation)
        ans = flatten_list([sentence_tokenize(s) for s in data["summary"].split("\n")])
        steps = [{"title": data.get("title", query).title(), "summary": s, "img": data.get("img")} for s in ans]
        return steps


WIKIPEDIA_PERSONA = {"name": "Wikipedia",
                     "solvers": ["ovos-solver-plugin-wikipedia",
                                 "ovos-solver-failure-plugin"]}

if __name__ == "__main__":
    LOG.set_level("ERROR")

    s = WikipediaSolver()
    print(s.get_spoken_answer("quem é Elon Musk", "pt"))
    # ('who is Elon Musk', <CQSMatchLevel.GENERAL: 3>, 'The Musk family is a wealthy family of South African origin that is largely active in the United States and Canada.',
    # {'query': 'who is Elon Musk', 'image': None, 'title': 'Musk Family',
    # 'answer': 'The Musk family is a wealthy family of South African origin that is largely active in the United States and Canada.'})

    query = "who is Isaac Newton"
    print(s.get_keyword_extractor("en").extract(query, "en"))
    # assert s.extract_keyword(query, "en-us") == "Isaac Newton"

    print(s.get_spoken_answer("venus", "en"))
    print(s.get_spoken_answer("elon musk", "en"))
    print(s.get_spoken_answer("mercury", "en"))
