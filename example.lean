import Mathlib.Data.Nat.Factorial.Basic
import Mathlib.Data.Nat.Choose.Basic


#align_import data.nat.choose.basic from "leanprover-community/mathlib"@"2f3994e1b117b1e1da49bcfb67334f33460c3ce4"

open Nat

@[simp]
theorem choose_one (n : ℕ) : choose n 1 = n := by
  induction n <;>
  simp [*, choose]
  simp [add_comm]
