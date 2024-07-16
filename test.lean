import Lean4Repl
import Mathlib

open Real Nat Topology
open scoped BigOperators

set_option maxHeartbeats 999999999999999999999999

-- theorem hello_world (a b c : Nat)
--   : a + b + c = a + c + b := by
--   rw [add_assoc, add_comm b, ←add_assoc]

theorem abc(a b c : Nat) : a + b + c = a + c + b:= by
  rw [Nat.add_assoc, Nat.add_right_comm]
  rw [Nat.add_comm]
  rw [Nat.add_assoc]
  rw [Nat.add_assoc, Nat.add_comm b c]
  sorry


theorem abcd (a b c : Nat) : b + (c + a) = a + b + c := by
  -- rw [Nat.add_assoc, Nat.add_right_comm] at abc
  rw [← Nat.add_assoc]
  rw [← Nat.add_comm]
  rw [← Nat.add_right_comm, ← Nat.add_assoc]
  rw[abc a b c ]

theorem eq_add_of_sub_eq'' {a b c : Nat} (hle : b ≤ a) (h : a - b = c) : a = c + b := by
  rw[h.symm]
  sorry

-- theorem aaa{a b c : Nat} {hle : b ≤ a} (h : a - b = c): a = a - b + b := by
--   have eq_add (a b c : Nat)(hle : b ≤ a) (h : a - b = c): a = c + b := sorry
--   rw [h.symm] at eq_add_of_sub_eq''
--   rw [eq_add]



theorem mathd_numbertheory_185
  (n : ℕ)
  (h₀ : n % 5 = 3) :
  (2 * n) % 5 = 1 := by
  rw [two_mul]
  rw [add_mod]
  simp [h₀]

-- theorem mathd
--   (n : ℕ)
--   (h₀ : n % 5 = 3) :
--   (n % 5 + n % 5) % 5 = 1 := by
--   rw [← add_mod]
--   rw [← two_mul]
--   rw [mathd_numbertheory_185 n h₀]

theorem mathd
  (n : ℕ)
  (h₀ : n % 5 = 3) :
  (n % 5 + n % 5) % 5 = 1 := by
  have mathd_185
  (n : ℕ)
  (h₀ : n % 5 = 3) :
  (2 * n) % 5 = 1 := by sorry
  rw[two_mul] at mathd_185
