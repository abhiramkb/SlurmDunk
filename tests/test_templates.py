"""The 'collect' pattern templates: '{name}' / '{name:type}' placeholders in a literal line."""
import math

import pytest

from slurmdunk.collect import compile_template, convert_auto


def match(template, line):
    regex, converters = compile_template(template)
    m = regex.search(line)
    return None if m is None else {name: conv(m.group(name)) for name, conv in converters.items()}


def same(got, want):
    """Equal values of equal type; nan equals nan."""
    if got is None or want is None:
        return got is want
    if got.keys() != want.keys():
        return False
    for key, w in want.items():
        g = got[key]
        if type(g) is not type(w):
            return False
        if not (g == w or (isinstance(w, float) and math.isnan(w) and math.isnan(g))):
            return False
    return True


@pytest.mark.parametrize("template, line, want", [
    # values of the motivating examples, and the VEGAS line from the NLO diffraction logs
    ("Time for integration: {t_int} s", "Time for integration: 10.2 s", {"t_int": 10.2}),
    ("Result of code = [{result},{error}]", "Result of code = [0.52,1e-3]", {"result": 0.52, "error": 0.001}),
    ("Final results: {result} +/- {error}",
     "[INFO] (vegasflow.monte_carlo)  > Final results: -0.520629 +/- 0.000366309",
     {"result": -0.520629, "error": 0.000366309}),
    # int/float follow how the number is written, unless a type says otherwise
    ("count: {n}", "count: 42", {"n": 42}),
    ("count: {n:float}", "count: 42", {"n": 42.0}),
    ("count: {n:int}", "count: 42", {"n": 42}),
    # spacing is flexible; template edges are ignored
    ("Q2 = {q2}", "Q2   =\t  10.5", {"q2": 10.5}),
    ("  Q2 = {q2}  ", "Q2 = 10.5", {"q2": 10.5}),
    # literal braces and regex metacharacters
    ("{{a}} = {a}", "{a} = 3", {"a": 3}),
    ("x(+.*) {v}", "x(+.*) 7", {"v": 7}),
    # exponent styles, nan/inf, and INFO is not inf
    ("v = {v}", "v = 1.2D-05", {"v": 1.2e-05}),
    ("v = {v}", "v = 1.2E+03", {"v": 1200.0}),
    ("v = {v}", "v = nan", {"v": float("nan")}),
    ("v = {v}", "v = -Infinity", {"v": float("-inf")}),
    ("{v}", "[INFO] nothing here", None),
    # numbers are matched whole
    ("{n:int} s", "took 10.5 s", None),
    ("took {n:int}", "took 10.5", None),
    ("{n} items", "x10 items", None),
    ("{a:int}.{b:int}", "10.5", {"a": 10, "b": 5}),
    ("value is {v}", "value is 100.", {"v": 100}),
    ("L{line}", "L12", {"line": 12}),
    # str / text stop at the next literal, or run to the end of the line
    ("Saved JSON results to {path:str}", "Saved JSON results to /a/b,c/result.json", {"path": "/a/b,c/result.json"}),
    ("file={f:str}, size", "file=out.dat, size=3", {"f": "out.dat"}),
    ("status: {msg:text} done", "status: all is well done", {"msg": "all is well"}),
    ("note: {msg:text}", "note: two words here ", {"msg": "two words here"}),
    ("hash {h:str}", "hash 0012ab", {"h": "0012ab"}),
])
def test_template_matches(template, line, want):
    got = match(template, line)
    assert same(got, want), got


@pytest.mark.parametrize("raw, want", [
    ("0.00035", 0.00035), ("400000000.0", 4e8), ("12", 12), ("abc", "abc"),
    ("/p/mvgam,balsd/85.dat", "/p/mvgam,balsd/85.dat"), ("1e-3", 0.001),
])
def test_convert_auto(raw, want):
    got = convert_auto(raw)
    assert got == want and type(got) is type(want)


@pytest.mark.parametrize("template, message", [
    ("no placeholder here", "no placeholder"),
    ("{x:complex}", "unknown type 'complex'"),
    ("unbalanced { here {x}", "unbalanced '{'"),
    ("close } {x}", "unbalanced '}'"),
    ("{x} and {x}", "appears twice"),
    ("{:float}", "needs a name"),
    ("{1bad}", "needs a name"),
    ("{x:}", "unknown type ''"),
])
def test_template_errors(template, message):
    with pytest.raises(ValueError, match=message):
        compile_template(template)
