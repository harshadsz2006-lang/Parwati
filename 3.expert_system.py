"""
Rule-based expert system (propositional logic, Horn clauses) with
forward chaining (data-driven) and backward chaining (goal-driven).
Domain: simple medical diagnosis (illustrative only, not medical advice).
Rule format: IF p1 AND p2 AND ... THEN q
"""
from collections import namedtuple

Rule = namedtuple("Rule", "name premises conclusion")


class KnowledgeBase:
    def __init__(self):
        self.rules = []

    def add_rule(self, name, premises, conclusion):
        self.rules.append(Rule(name, set(premises), conclusion))

    def show(self):
        for r in self.rules:
            print(f"   {r.name}: IF {' AND '.join(sorted(r.premises))} THEN {r.conclusion}")

    def forward_chaining(self, facts, goal=None, verbose=True):
        known, step, changed = set(facts), 0, True
        while changed and goal not in known:
            changed = False
            for r in self.rules:
                if r.conclusion not in known and r.premises <= known:
                    known.add(r.conclusion)
                    changed, step = True, step + 1
                    if verbose:
                        print(f"  Step {step}: fired {r.name} -> inferred '{r.conclusion}'")
                    if r.conclusion == goal:
                        return True, known
        return (goal in known if goal else None), known

    def backward_chaining(self, goal, facts, verbose=True, depth=0, stack=frozenset()):
        pad = "  " * (depth + 1)
        if goal in facts:
            if verbose:
                print(f"{pad}'{goal}' is a known fact  [OK]")
            return True
        if goal in stack:  # avoid circular reasoning
            return False
        for r in self.rules:
            if r.conclusion == goal:
                if verbose:
                    print(f"{pad}Goal '{goal}': trying {r.name} (needs {sorted(r.premises)})")
                if all(self.backward_chaining(p, facts, verbose, depth + 1, stack | {goal})
                       for p in sorted(r.premises)):
                    if verbose:
                        print(f"{pad}'{goal}' PROVED via {r.name}  [OK]")
                    return True
        if verbose:
            print(f"{pad}'{goal}' cannot be proved  [FAIL]")
