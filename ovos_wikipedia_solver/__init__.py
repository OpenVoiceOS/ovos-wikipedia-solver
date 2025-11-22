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
        """
        Initialize a WikipediaSolver instance, configure base QuestionSolver, and prepare plugin caches.

        Parameters:
            config (dict | None): Optional configuration for the solver.
            enable_tx (bool): If True, enables translation support and causes translator/detector to be propagated to plugins.
            translator (LanguageTranslator | None): Optional translator to be forwarded to language-aware plugins when translation is enabled.
            detector (LanguageDetector | None): Optional language detector to be forwarded to language-aware plugins when translation is enabled.

        Detailed behavior:
            - Calls the superclass initializer with the provided parameters and a fixed priority of 40.
            - Creates empty caches for per-language keyword extractors (`kword_extractors`) and summarizers (`summarizers`).
        """
        super().__init__(config, enable_tx=enable_tx, priority=40, translator=translator, detector=detector)
        self.kword_extractors: Dict[str, KeywordExtractor] = {}
        self.summarizers: Dict[str, TldrSolver] = {}

    def get_summarizer(self, lang: str) -> Optional[TldrSolver]:
        """
        Lazily load, configure, cache, and return a language-specific TLDR summarizer plugin.

        If a summarizer plugin is configured or the default plugin is available, instantiate it for the given language, attach shared translator/detector objects when translation is enabled, cache the instance, and return it.

        Returns:
            A TldrSolver instance configured for `lang`, or `None` if the plugin cannot be found or instantiated.
        """
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
        """
        Get a keyword extractor instance for the given language, creating and caching a plugin instance when needed.

        Returns:
            KeywordExtractor | None: A `KeywordExtractor` configured for `lang`, or `None` if the configured plugin cannot be loaded.
        """
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
        Fetch the title, plain-text intro summary, and an image URL for a Wikipedia page.

        Parameters:
            pid (str): Wikipedia pageid.
            lang (str): Language code (e.g., "en", "pt"); used as the wiki subdomain.

        Returns:
            Tuple[Optional[str], Optional[str], Optional[str]]: (title, summary, image_url).
                - title: Page title or `None` if not available.
                - summary: Plain-text intro with parenthetical content removed, or `None` when the page is a disambiguation list or on error.
                - image_url: Derived image URL (thumbnail path normalized by removing "thumb" segments) or `None` if no image is available.
                Returns (None, None, None) on disambiguation pages or on request/parse errors.
        """
        url = (
            f"https://{lang}.wikipedia.org/w/api.php?format=json&action=query&"
            f"prop=extracts|pageimages&exintro&explaintext&redirects=1&pageids={pid}"
        )
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
        Compute a relevance score for a Wikipedia page given a search query.

        Parameters:
            query (str): The user's search query.
            title (str): The page title.
            summary (str): The page summary text.
            idx (int): The page's index in the original search results; lower indices are weighted more favorably.

        Returns:
            float: Relevance score where higher values indicate greater relevance.
        """
        page_mod = 1 - (idx * 0.05)  # Favor original order returned by Wikipedia
        title_score = max(
            fuzzy_match(query, title, MatchStrategy.DAMERAU_LEVENSHTEIN_SIMILARITY),
            fuzzy_match(query, rm_parentheses(title), MatchStrategy.DAMERAU_LEVENSHTEIN_SIMILARITY)
        )
        summary_score = fuzzy_match(summary, title, MatchStrategy.TOKEN_SET_RATIO)
        return title_score * summary_score * page_mod

    def get_data(self, query: str, lang: Optional[str] = None, units: Optional[str] = None,
                 skip_disambiguation: bool = False):
        """
        Searches Wikipedia for a query, fetches page extracts in the requested language, generates a short answer, and returns the best-matching result.

        Performs a site search using the language's Wikipedia, falls back to a keyword-extracted query if no results are found, concurrently fetches page data (skipping disambiguation pages), creates a short summary using the configured summarizer plugin, scores and re-ranks candidates, and returns the top entry.

        Parameters:
            query (str): User query to search on Wikipedia.
            lang (Optional[str]): Language code or locale to use (e.g., "en" or "pt-BR"); defaults to the solver's default language if omitted.
            units (Optional[str]): Units preference (accepted but not used by this method).
            skip_disambiguation (bool): If True, limit search to a single top result to avoid multiple disambiguation candidates.

        Returns:
            dict: A dictionary with keys:
                - "title": selected page title (str)
                - "short_answer": concise summary of the page (str)
                - "summary": full page extract (str)
                - "img": image URL if available (str or None)
            Returns an empty dict if no suitable page data could be retrieved.
        """
        LOG.debug(f"WikiSolver query: {query}")
        lang = (lang or self.default_lang).split("-")[0]
        search_url = (
            f"https://{lang}.wikipedia.org/w/api.php?action=query&list=search&"
            f"srsearch={query}&format=json"
        )
        try:
            search_results = requests.get(search_url,
                                          timeout=5,
                                          headers={"User-Agent": self.USER_AGENT}
                                          ).json().get("query", {}).get("search", [])
        except Exception as e:
            LOG.error(f"Error fetching search results: {e}")
            search_results = []

        if not search_results:
            kwx = self.get_keyword_extractor(lang)
            keywords = kwx.extract(query, lang=lang)
            if keywords:
                fallback_query = max(keywords)
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
            future_to_idx = {
                executor.submit(self.get_page_data, str(r["pageid"]), lang): idx
                for idx, r in enumerate(search_results)
                if "(disambiguation)" not in r["title"]
            }

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

        return {
            "title": summaries[selected][0],
            "short_answer": shorts[selected],
            "summary": summaries[selected][1],
            "img": summaries[selected][2],
        }

    def get_spoken_answer(self, query: str, lang: Optional[str] = None, units: Optional[str] = None,
                          skip_disambiguation: bool = False):
        """
        Return a concise spoken answer for the given query using Wikipedia search and summarization.

        Parameters:
          query (str): The user's question or search query.
          lang (Optional[str]): Language tag (e.g., "en") to use for search and summarization; when None, the solver's default language is used.
          units (Optional[str]): Unit system hint (unused by Wikipedia lookup; provided for API compatibility).
          skip_disambiguation (bool): If True, prefer a single top result and avoid expanding disambiguation results.

        Returns:
          short_answer (str): A brief spoken answer extracted or generated from the top Wikipedia page, or an empty string if no answer is found.
        """
        data = self.get_data(query, lang=lang, units=units, skip_disambiguation=skip_disambiguation)
        return data.get("short_answer", "")

    def get_image(self, query: str, lang: Optional[str] = None, units: Optional[str] = None,
                  skip_disambiguation: bool = True):
        """
        Retrieve the image URL for the best-matching Wikipedia page for a query.

        Parameters:
          query (str): Search query.
          lang (Optional[str]): Language code to use for the search; if omitted the solver's default language is used.
          units (Optional[str]): Units preference passed through to underlying helpers (unused by this method).
          skip_disambiguation (bool): If True, prefer a single top result to avoid disambiguation pages.

        Returns:
          img (str): Image URL for the selected page, or an empty string if no image is available.
        """
        data = self.get_data(query, lang=lang, units=units, skip_disambiguation=skip_disambiguation)
        return data.get("img", "")

    def get_expanded_answer(self, query: str,
                            lang: Optional[str] = None,
                            units: Optional[str] = None,
                            skip_disambiguation: bool = False):
        """
        Produce an ordered list of step dictionaries that expand the page summary into sentence-level items.

        Each step corresponds to a sentence from the page summary. The step title defaults to the page title (or the original query capitalized) and the image is copied from the page data when available.

        Returns:
            List[dict]: A list of steps where each dict contains:
                - title (str): Page title or capitalized query.
                - summary (str): A single sentence to speak.
                - img (str | None): Image URL or path, or None if unavailable.
        """
        data = self.get_data(query, lang=lang, units=units, skip_disambiguation=skip_disambiguation)
        ans = flatten_list([sentence_tokenize(s) for s in data["summary"].split("\n")])
        steps = [{
            "title": data.get("title", query).title(),
            "summary": s,
            "img": data.get("img")
        } for s in ans]
        return steps


WIKIPEDIA_PERSONA = {
    "name": "Wikipedia",
    "solvers": [
        "ovos-solver-plugin-wikipedia",
        "ovos-solver-failure-plugin"
    ]
}

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