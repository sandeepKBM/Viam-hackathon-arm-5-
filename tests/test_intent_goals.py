import statistics
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from components.affordances import resolve_goal
from components.constants import WORKSPACE_CORNERS
from components.voice import DEFAULT_MOVES, _finalize_mapped


def _in_ws():
    xs = [p[0] for p in WORKSPACE_CORNERS]
    ys = [p[1] for p in WORKSPACE_CORNERS]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def _block(color, *, x=None, y=None, depth=510, area=1000, z=0.0):
    cx, cy = _in_ws()
    return SimpleNamespace(
        color=color,
        x=cx if x is None else x,
        y=cy if y is None else y,
        z=z,
        depth_mm=depth,
        u=10,
        v=10,
        yaw=0.0,
        shape=SimpleNamespace(area=area, box=(0, 0, 10, 10), aspect_ratio=1.0),
    )


def _llm(text, data):
    return _finalize_mapped(text, data)


class FinalizeIntentTests(unittest.TestCase):
    def test_thirsty_sets_hydrate_without_claiming_object(self):
        out = _llm("I am thirsty", {"task": "sort", "goal": "hydrate", "say": "On it.", "moves": []})
        self.assertEqual(out["goal"], "hydrate")
        self.assertEqual(out["task"], "sort")
        self.assertEqual(out["moves"], [])

    def test_llm_invented_bottle_is_stripped(self):
        out = _llm(
            "I need a drink",
            {
                "task": "sort",
                "goal": "hydrate",
                "say": "Okay",
                "moves": [{"object": "bottle", "place": "handoff", "count": 1}],
            },
        )
        self.assertEqual(out["goal"], "hydrate")
        self.assertEqual(out["moves"], [])

    def test_write_sets_goal(self):
        out = _llm("I want to write", {"task": "sort", "goal": "write", "say": "Okay.", "moves": []})
        self.assertEqual(out["goal"], "write")
        self.assertEqual(out["moves"], [])

    def test_explicit_pen_bin2_wins(self):
        out = _llm(
            "put the pen in bin 2",
            {
                "task": "sort",
                "goal": "write",
                "say": "Okay.",
                "moves": [{"object": "pen", "place": "bin2", "count": 1}],
            },
        )
        self.assertEqual(out["moves"], [{"object": "pen", "place": "bin2", "count": 1}])
        self.assertIsNone(out["goal"])

    def test_hand_me_two_bottles_keeps_count(self):
        out = _llm(
            "hand me two bottles",
            {
                "task": "sort",
                "goal": None,
                "say": "Okay.",
                "moves": [{"object": "bottle", "place": "handoff", "count": 1}],
            },
        )
        self.assertEqual(out["moves"], [{"object": "bottle", "place": "handoff", "count": 2}])
        self.assertIsNone(out["goal"])

    def test_generic_sort_keeps_defaults(self):
        out = _llm("sort the blocks", {"task": "sort", "goal": None, "say": "Sorting.", "moves": []})
        self.assertEqual(out["moves"], [dict(m) for m in DEFAULT_MOVES])

    def test_ambiguous_never_defaults(self):
        out = _llm("do something", {"task": "unknown", "goal": None, "say": "", "moves": []})
        self.assertEqual(out["task"], "unknown")
        self.assertEqual(out["moves"], [])
        self.assertIn("?", out["say"])

    def test_malformed_invalid_fields(self):
        out = _llm("hmm", {"task": "explode", "goal": "pour", "moves": [{"object": "laser"}]})
        self.assertEqual(out["task"], "unknown")
        self.assertEqual(out["moves"], [])
        self.assertIsNone(out["goal"])

    def test_one_intent_model_call(self):
        from components import voice

        calls = {"n": 0}

        class _Client:
            def __init__(self, *a, **k):
                self.chat = SimpleNamespace(
                    completions=SimpleNamespace(create=self._create)
                )

            def _create(self, **kwargs):
                calls["n"] += 1
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content='{"task":"sort","goal":"hydrate","say":"On it.","moves":[]}'
                            )
                        )
                    ]
                )

        with patch("openai.OpenAI", _Client), patch.dict(
            "os.environ", {"OPENAI_API_KEY": "test"}, clear=False
        ):
            out = voice.map_task("I am thirsty")
        self.assertEqual(calls["n"], 1)
        self.assertEqual(out["goal"], "hydrate")
        self.assertEqual(out["moves"], [])


class AffordanceTests(unittest.TestCase):
    def test_thirsty_bottle_to_handoff(self):
        got = resolve_goal("hydrate", [_block("bottle"), _block("cup", area=9000)])
        self.assertTrue(got["ok"])
        self.assertEqual(got["moves"], [{"object": "bottle", "place": "handoff", "count": 1}])

    def test_thirsty_cup_only_no_motion(self):
        got = resolve_goal("hydrate", [_block("cup")])
        self.assertFalse(got["ok"])
        self.assertEqual(got["moves"], [])
        self.assertIn("bottle", got["say"])
        self.assertNotIn("pour", got["say"].lower())

    def test_thirsty_bottle_and_cup_picks_bottle(self):
        got = resolve_goal("hydrate", [_block("cup", area=5000), _block("bottle", area=800)])
        self.assertEqual(got["moves"][0]["object"], "bottle")

    def test_write_pen(self):
        got = resolve_goal("write", [_block("pen")])
        self.assertEqual(got["moves"], [{"object": "pen", "place": "handoff", "count": 1}])

    def test_write_no_pen(self):
        got = resolve_goal("write", [_block("bottle")])
        self.assertFalse(got["ok"])
        self.assertEqual(got["moves"], [])

    def test_no_depth_is_unsafe(self):
        got = resolve_goal("hydrate", [_block("bottle", depth=0)])
        self.assertFalse(got["ok"])

    def test_out_of_workspace_is_unsafe(self):
        got = resolve_goal("hydrate", [_block("bottle", x=-5000, y=-5000)])
        self.assertFalse(got["ok"])

    def test_resolution_p95_under_10ms(self):
        items = [_block("bottle"), _block("pen"), _block("cup")]
        times = []
        for _ in range(24):
            t0 = time.perf_counter()
            resolve_goal("hydrate", items)
            times.append((time.perf_counter() - t0) * 1000.0)
        times.sort()
        p50 = statistics.median(times)
        p95 = times[int(0.95 * (len(times) - 1))]
        print(f"resolve_goal replay n={len(times)} p50={p50:.3f}ms p95={p95:.3f}ms")
        self.assertLess(p95, 10.0)


class SortGoalTests(unittest.TestCase):
    def test_goal_uses_one_vision_pass_and_no_default_bins(self):
        import sort_blocks

        bottle = _block("bottle")
        cup = _block("cup")
        vision = SimpleNamespace(locate_blocks=AsyncMock(return_value=[bottle, cup]))
        sorter = SimpleNamespace(sort_blocks=AsyncMock(return_value={"placed": [{"color": "bottle"}], "skipped": []}))

        async def _run():
            return await sort_blocks.main(
                None,
                None,
                goal="hydrate",
                timings={},
                _vision=vision,
                _sorter=sorter,
                _skip_motion_setup=True,
            )

        import asyncio

        result = asyncio.run(_run())
        vision.locate_blocks.assert_awaited_once()
        self.assertEqual(vision.locate_blocks.await_args.kwargs.get("colors") or vision.locate_blocks.await_args.args[0], ("bottle",))
        sorter.sort_blocks.assert_awaited_once()
        bins, counts = sorter.sort_blocks.await_args.args[1], sorter.sort_blocks.await_args.args[2]
        self.assertEqual(bins, {"bottle": "handoff"})
        self.assertEqual(counts, {"bottle": 1})
        self.assertTrue(result["ok"])
        self.assertTrue(result["moved"])

    def test_goal_missing_object_does_not_sort(self):
        import sort_blocks

        vision = SimpleNamespace(locate_blocks=AsyncMock(return_value=[_block("cup")]))
        sorter = SimpleNamespace(sort_blocks=AsyncMock())

        import asyncio

        result = asyncio.run(
            sort_blocks.main(
                None,
                None,
                goal="hydrate",
                timings={},
                _vision=vision,
                _sorter=sorter,
                _skip_motion_setup=True,
            )
        )
        sorter.sort_blocks.assert_not_called()
        self.assertFalse(result["moved"])
        self.assertIn("bottle", result["say"])

    def test_failed_grasp_is_not_success(self):
        import sort_blocks

        vision = SimpleNamespace(locate_blocks=AsyncMock(return_value=[_block("pen")]))
        sorter = SimpleNamespace(
            sort_blocks=AsyncMock(return_value={"placed": [], "skipped": [{"error": "gripper did not grab"}]})
        )
        import asyncio

        result = asyncio.run(
            sort_blocks.main(
                None,
                None,
                goal="write",
                timings={},
                _vision=vision,
                _sorter=sorter,
                _skip_motion_setup=True,
            )
        )
        self.assertFalse(result["ok"])
        self.assertIn("could not", result["say"].lower())


if __name__ == "__main__":
    unittest.main()
