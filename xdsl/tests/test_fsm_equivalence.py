from pathlib import Path

from xdsl.dialects import arith
from xdsl.dialects.builtin import ModuleOp, i1, i64

from spechls.dialect import SpeculationConfigAttr, SpeculationEntryAttr, SpeculationSlowPathAttr, StructType, TaskOp, GammaOp, CommitOp
from spechls.fsm_equivalence import compare_fsms, export_xdsl_fsm, load_java_fsm_fixture
from spechls.transforms import infer_configured_speculation_fsms


def test_java_fixture_matches_exported_xdsl_machine_on_supported_structure_and_trace():
    task = TaskOp(StructType("result", ["enable"], [i1]), "worker", [])
    enable = arith.ConstantOp.from_int_and_width(1, 1)
    selector = arith.ConstantOp.from_int_and_width(0, 64)
    false = arith.ConstantOp.from_int_and_width(0, 1)
    gamma = GammaOp("g0", selector.result, [enable.result, false.result])
    task.body.block.add_ops([selector, false, gamma, enable, CommitOp([enable])])
    task.attributes["spechls.speculation_config"] = SpeculationConfigAttr([
        SpeculationEntryAttr(1, 3, "g0", 0, [SpeculationSlowPathAttr(2, 1, True, True, [], [], [], 1)])
    ])
    xdsl = export_xdsl_fsm(infer_configured_speculation_fsms(ModuleOp([task]))[0])
    java = load_java_fsm_fixture(Path(__file__).parent / "fixtures" / "java_single_recovery_fsm.json")

    result = compare_fsms(java, xdsl, [[{"mispec_0": 0}, {"mispec_0": 0}]])

    assert result.equivalent


def test_ambiguous_selector_trace_is_an_explicit_gap_not_an_equivalence_claim():
    java = load_java_fsm_fixture(Path(__file__).parent / "fixtures" / "java_single_recovery_fsm.json")

    result = compare_fsms(java, java, [[{"mispec_0": 0}, {"mispec_0": 2}]])

    assert not result.equivalent
    assert any("not deterministic" in gap for gap in result.gaps)
