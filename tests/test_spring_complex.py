"""PROVEN: wing operators define spring-plane complex multiplication."""
import unittest

from aethos_spring_complex import (
    SpringPoint,
    i_act,
    conj_act,
    neg_act,
    reflect_x,
    swap_xy,
    spring_mul,
    verify_critical_line_rotation,
    verify_i_act_axioms,
    spring_complex_field_check,
)


class TestSpringComplex(unittest.TestCase):
    def test_i_act_axioms(self):
        r = verify_i_act_axioms()
        self.assertTrue(all(r.values()), r)

    def test_i_squared_negation(self):
        p = SpringPoint(3, 5)
        self.assertEqual(neg_act(p).x, -p.x)
        self.assertEqual(neg_act(p).y, -p.y)

    def test_i_on_real_unit(self):
        self.assertEqual(i_act(SpringPoint(1, 0)), SpringPoint(0, 1))

    def test_conj(self):
        p = SpringPoint(3, -4)
        self.assertEqual(conj_act(p), SpringPoint(3, 4))

    def test_field_sample(self):
        self.assertTrue(spring_complex_field_check())

    def test_mul_matches_complex(self):
        a = SpringPoint(2, 3)
        b = SpringPoint(-1, 4)
        self.assertAlmostEqual(
            spring_mul(a, b).to_complex(),
            a.to_complex() * b.to_complex(),
        )

    def test_critical_line_rotation(self):
        r = verify_critical_line_rotation()
        self.assertTrue(all(r.values()), r)


if __name__ == "__main__":
    unittest.main()
