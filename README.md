# Wikipedia Plugin

```python
from ovos_wikipedia_solver import WikipediaSolver

s = WikipediaSolver()
print(s.get_spoken_answer("quem é Elon Musk", "pt"))

query = "who is Isaac Newton"
print(s.extract_keyword(query, "en-us"))
assert s.extract_keyword(query, "en-us") == "Isaac Newton"

print(s.get_spoken_answer("venus", "en"))
print(s.get_spoken_answer("elon musk", "en"))
print(s.get_spoken_answer("mercury", "en"))
```