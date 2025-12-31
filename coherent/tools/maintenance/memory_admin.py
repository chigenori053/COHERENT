import sys
import argparse
from typing import Optional

# Adjust path if run as script
# In a real app these imports would work via installed package or pythonpath
from coherent.core.memory.types import Action
from coherent.core.memory.utility import UtilityTable
from coherent.core.memory.human.utility_shaping import HumanUtilityShaper
from coherent.core.memory.decision import DecisionEngine
from coherent.core.memory.types import StateDistribution, ExpectedUtility

class MemoryAdminTool:
    def __init__(self):
        self.utility_table = UtilityTable()
        self.human_shaper = HumanUtilityShaper()
        self.decision_engine = DecisionEngine(self.utility_table, self.human_shaper)

    def view_utility(self, state: Optional[str] = None):
        print("=== Utility Table ===")
        if state and state in self.utility_table.matrix:
            print(f"State: {state}")
            for act, u in self.utility_table.matrix[state].items():
                mod = self.human_shaper.get_modifier(state, act)
                total = u + mod
                print(f"  {act.name}: {total:.2f} (Base: {u}, Mod: {mod})")
        else:
            for s, actions in self.utility_table.matrix.items():
                print(f"State: {s}")
                for act, u in actions.items():
                    mod = self.human_shaper.get_modifier(s, act)
                    total = u + mod
                    print(f"  {act.name}: {total:.2f} (Base: {u}, Mod: {mod})")

    def set_override(self, state: str, action_name: str, delta: float):
        try:
            action = Action(action_name.lower()) # Enum values are lowercase
        except ValueError:
            print(f"Invalid action: {action_name}. Valid examples: store_new, merge_soft...")
            return

        print(f"Applying override: {state} + {action.name} += {delta}")
        self.human_shaper.set_modifier(state, action, delta)
        print("Done.")

    def dry_run(self, state_dist_str: str):
        """
        Simulate decision.
        Format: "TrulyUnique=0.8,Variant=0.2"
        """
        try:
            parts = state_dist_str.split(',')
            probs = {}
            for p in parts:
                k, v = p.split('=')
                probs[k.strip()] = float(v)
            
            dist = StateDistribution(probs=probs)
            eu = self.decision_engine.compute_expected_utility(dist)
            decision = self.decision_engine.decide(eu)
            
            print(f"\nDist: {probs}")
            print(f"Expected Utilities:")
            for act, val in eu.values.items():
                print(f"  {act.name}: {val:.4f}")
            print(f"DECISION => {decision.name}")

        except Exception as e:
            print(f"Error parse state dist: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MemorySpace Admin Tool")
    subparsers = parser.add_subparsers(dest="command")

    # view
    p_view = subparsers.add_parser("view", help="View utility table")
    p_view.add_argument("--state", type=str, help="Filter by state")

    # set
    p_set = subparsers.add_parser("set", help="Set utility override")
    p_set.add_argument("state", type=str)
    p_set.add_argument("action", type=str)
    p_set.add_argument("delta", type=float)

    # dry-run
    p_run = subparsers.add_parser("dry-run", help="Dry run decision logic")
    p_run.add_argument("dist", type=str, help="e.g. TrulyUnique=0.8,Variant=0.2")

    args = parser.parse_args()
    
    tool = MemoryAdminTool()

    if args.command == "view":
        tool.view_utility(args.state)
    elif args.command == "set":
        tool.set_override(args.state, args.action, args.delta)
    elif args.command == "dry-run":
        tool.dry_run(args.dist)
    else:
        parser.print_help()
