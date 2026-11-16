# CELL 1: Import Required Libraries
# No external libraries required


# CELL 2: Define Rule Class
"""
PR4: Knowledge Representation and Expert System (CO1, CO2)
Rule-based expert system using propositional logic with
Forward Chaining (data-driven) and Backward Chaining (goal-driven).

Domain: Simple medical diagnosis (illustrative only, not medical advice).
Rule format (Horn clause): IF p1 AND p2 AND ... THEN q
"""


class Rule:
    def __init__(self, name, premises, conclusion):
        self.name = name
        self.premises = set(premises)   # propositional symbols (conjunction)
        self.conclusion = conclusion    # single propositional symbol

    def __str__(self):
        return f"{self.name}: IF {' AND '.join(sorted(self.premises))} THEN {self.conclusion}"


# CELL 3: Define Knowledge Base Class
class KnowledgeBase:
    def __init__(self):
        self.rules = []

    def add_rule(self, name, premises, conclusion):
        self.rules.append(Rule(name, premises, conclusion))

    def show(self):
        for r in self.rules:
            print("  ", r)

    # ---------------- FORWARD CHAINING (data-driven) ----------------
    def forward_chaining(self, facts, goal=None, verbose=True):
        known = set(facts)
        fired = set()
        step = 0
        changed = True
        while changed:
            changed = False
            for r in self.rules:
                if r.name not in fired and r.premises <= known:
                    fired.add(r.name)
                    if r.conclusion not in known:
                        known.add(r.conclusion)
                        changed = True
                        step += 1
                        if verbose:
                            print(f"  Step {step}: fired {r.name} -> inferred '{r.conclusion}'")
                        if goal and r.conclusion == goal:
                            return True, known
        return (goal in known if goal else None), known

    # ---------------- BACKWARD CHAINING (goal-driven) ----------------
    def backward_chaining(self, goal, facts, verbose=True, _depth=0, _stack=None):
        _stack = _stack or set()
        pad = "  " * (_depth + 1)
        if goal in facts:
            if verbose:
                print(f"{pad}'{goal}' is a known fact  [OK]")
            return True
        if goal in _stack:                      # avoid circular reasoning
            return False
        _stack = _stack | {goal}
        for r in self.rules:
            if r.conclusion == goal:
                if verbose:
                    print(f"{pad}Goal '{goal}': trying {r.name} (needs {sorted(r.premises)})")
                if all(self.backward_chaining(p, facts, verbose, _depth + 1, _stack)
                       for p in sorted(r.premises)):
                    if verbose:
                        print(f"{pad}'{goal}' PROVED via {r.name}  [OK]")
                    return True
        if verbose:
            print(f"{pad}'{goal}' cannot be proved  [FAIL]")
        return False


# CELL 4: Build Knowledge Base
def build_kb():
    kb = KnowledgeBase()
    kb.add_rule("R1", ["fever", "cough"], "respiratory_infection")
    kb.add_rule("R2", ["fever", "body_ache", "fatigue"], "flu_like_illness")
    kb.add_rule("R3", ["respiratory_infection", "flu_like_illness"], "flu")
    kb.add_rule("R4", ["cough", "sneezing", "runny_nose"], "common_cold")
    kb.add_rule("R5", ["flu"], "needs_rest_and_fluids")
    kb.add_rule("R6", ["flu", "breathing_difficulty"], "consult_doctor_urgently")
    kb.add_rule("R7", ["common_cold"], "needs_rest_and_fluids")
    return kb


# CELL 5: Main Function
def main():
    kb = build_kb()
    print("=" * 60)
    print("RULE-BASED EXPERT SYSTEM (Propositional Logic)")
    print("=" * 60)
    print("Knowledge Base:")
    kb.show()

    facts = {"fever", "cough", "body_ache", "fatigue"}
    print("\nInitial facts:", sorted(facts))

    # CELL 6: Forward Chaining
    print("\n--- FORWARD CHAINING (all conclusions) ---")
    _, derived = kb.forward_chaining(facts)
    print("  Derived facts:", sorted(derived - facts))

    # CELL 7: Forward Chaining with Goal
    goal = "flu"
    print(f"\n--- FORWARD CHAINING (goal = '{goal}') ---")
    ok, _ = kb.forward_chaining(facts, goal)
    print(f"  Result: '{goal}' is {'TRUE' if ok else 'NOT derivable'}")

    # CELL 8: Backward Chaining
    print(f"\n--- BACKWARD CHAINING (goal = '{goal}') ---")
    print("  Result:", "PROVED" if kb.backward_chaining(goal, facts) else "NOT PROVED")

    # CELL 9: Backward Chaining with Second Goal
    goal2 = "consult_doctor_urgently"
    print(f"\n--- BACKWARD CHAINING (goal = '{goal2}') ---")
    print("  Result:", "PROVED" if kb.backward_chaining(goal2, facts) else "NOT PROVED")

    # CELL 10: Interactive Consultation
    print("\n--- INTERACTIVE CONSULTATION ---")
    symptoms = ["fever", "cough", "body_ache", "fatigue", "sneezing",
                "runny_nose", "breathing_difficulty"]
    user_facts = set()
    for s in symptoms:
        if input(f"  Do you have {s.replace('_', ' ')}? (y/n): ").strip().lower() == "y":
            user_facts.add(s)
    _, result = kb.forward_chaining(user_facts, verbose=False)
    advice = sorted(result - user_facts)
    print("\n  Diagnosis / advice:", ", ".join(advice) if advice else "No conclusion reached")


# CELL 11: Execute Main Function
if __name__ == "__main__":
    main()
