open Nat (add_assoc add_comm)

theorem hello_world (a b c : Nat)
  : a + b + c = a + c + b := by
  rw [add_assoc, add_comm b, ←add_assoc]

example(a b c : Nat) : a + b + c = a + c + b:= by
  rw [Nat.add_assoc, Nat.add_right_comm]
  rw [Nat.add_assoc, Nat.add_comm b c]