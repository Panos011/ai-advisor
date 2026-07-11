import json
import unittest

from vendor_report import aggregate, parse_events, render_overview, render_tool, to_json


def _line(tools, goal="writing blog posts", action="recommend", q="i need a writing tool"):
    payload = {"goal": goal, "action": action, "mode": "best_fit", "personalized": False,
               "tools": tools, "q": q}
    return f"INFO:api:SHORTLIST {json.dumps(payload)}"


class ParseEventsTests(unittest.TestCase):
    def test_extracts_payloads_and_skips_junk(self):
        lines = [
            "INFO:api:ROUTING action=recommend hits=2 q='x'",   # no marker
            _line(["Writerly", "Draftly"]),
            "Jul 10 12:00:00 render[1]: " + _line(["Descript"]),  # prefixed
            "INFO:api:SHORTLIST {not json",                       # malformed
            _line([]),                                            # no tools
        ]
        events = parse_events(lines)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["tools"], ["Writerly", "Draftly"])
        self.assertEqual(events[1]["tools"], ["Descript"])

    def test_strips_blank_tool_names(self):
        events = parse_events([_line(["  Writerly  ", "", "Draftly"])])
        self.assertEqual(events[0]["tools"], ["Writerly", "Draftly"])


class AggregateTests(unittest.TestCase):
    def test_wins_ranks_and_lost_to(self):
        events = parse_events([
            _line(["Descript", "Riverside"]),
            _line(["Riverside", "Descript", "Podcastle"]),
            _line(["Descript"]),
        ])
        stats = aggregate(events)

        descript = stats["Descript"]
        self.assertEqual(descript["appearances"], 3)
        self.assertEqual(descript["wins"], 2)
        self.assertEqual(descript["lost_to"], {"Riverside": 1})

        riverside = stats["Riverside"]
        self.assertEqual(riverside["appearances"], 2)
        self.assertEqual(riverside["wins"], 1)
        self.assertEqual(riverside["lost_to"], {"Descript": 1})

        podcastle = stats["Podcastle"]
        self.assertEqual(podcastle["appearances"], 1)
        self.assertEqual(podcastle["wins"], 0)
        # Ranked third: lost to both tools above it.
        self.assertEqual(podcastle["lost_to"], {"Riverside": 1, "Descript": 1})
        self.assertEqual(podcastle["rank_sum"], 3)

    def test_goals_and_queries_counted(self):
        events = parse_events([
            _line(["Descript"], goal="editing podcasts", q="podcast editor"),
            _line(["Descript"], goal="editing podcasts", q="edit my podcast"),
        ])
        stats = aggregate(events)
        self.assertEqual(stats["Descript"]["goals"], {"editing podcasts": 2})
        self.assertEqual(
            stats["Descript"]["queries"],
            {"podcast editor": 1, "edit my podcast": 1},
        )


class RenderTests(unittest.TestCase):
    def _stats(self):
        return aggregate(parse_events([
            _line(["Descript", "Riverside"]),
            _line(["Riverside", "Descript"]),
        ]))

    def test_overview_lists_tools(self):
        text = render_overview(self._stats())
        self.assertIn("Descript", text)
        self.assertIn("Riverside", text)

    def test_tool_view_shows_the_sellable_story(self):
        text = render_tool(self._stats(), "descript")  # case-insensitive
        self.assertIn("Shortlisted: 2 times", text)
        self.assertIn("Won (ranked #1): 1 times (50%)", text)
        self.assertIn("Riverside: 1 times", text)

    def test_tool_view_unknown_tool(self):
        self.assertIn("No SHORTLIST events", render_tool(self._stats(), "Nope"))

    def test_overview_empty(self):
        self.assertIn("No SHORTLIST events", render_overview({}))

    def test_json_is_parseable_and_complete(self):
        data = json.loads(to_json(self._stats()))
        self.assertEqual(data["Descript"]["appearances"], 2)
        self.assertEqual(data["Descript"]["win_rate"], 0.5)
        self.assertEqual(data["Descript"]["lost_to"], {"Riverside": 1})


if __name__ == "__main__":
    unittest.main()
