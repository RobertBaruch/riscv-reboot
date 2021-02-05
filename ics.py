# Disable pylint's "your name is too short" warning.
# pylint: disable=C0103
"""Modules for ICs."""
from typing import List, Tuple

from nmigen import Signal, Module, Elaboratable, Cat
from nmigen.build import Platform
from nmigen.asserts import Assert, Assume

from util import main


class IC_7416373(Elaboratable):
    """Logic for the 7416373 16-bit transparent latch."""

    def __init__(self):
        pass

    def elaborate(self, _: Platform) -> Module:
        """Implements the logic of the 7416373 chip."""
        m = Module()
        return m


class IC_74181(Elaboratable):
    """Logic for the 74181 4-bit ALU.

    Attributes:
        a: The A input.
        b: The B input.
        f: The output.
        s: The function to perform.
        m: The mode: 0 = arithmetic, 1 = logical.
        n_carryin: Carry in, active low.
        n_carryout: Carry out, active low.
        x: X output (only meaningful in add/subtract mode). This output is NOT
           the same as a carry propagate, but it does work with the 74182 CLU
           when hooked up to its np input -- and then the group_np output of the
           CLU is actually the group X.
        y: Y output (only meaningful in add/subtract mode). This output is NOT
           the same as a carry generate, but it does work with the 74182 CLU
           when hooked up to its ng input -- and then the group_ng output of the
           CLU is actually the group Y.
        a_eq_b: Equality (only meaningful in subtract mode with n_carryin = 1).
    """

    a: Signal
    b: Signal
    s: Signal
    f: Signal
    m: Signal
    n_carryin: Signal
    n_carryout: Signal
    x: Signal
    y: Signal
    a_eq_b: Signal

    def __init__(self):
        self.a = Signal(4)
        self.b = Signal(4)
        self.s = Signal(4)
        self.f = Signal(4)
        self.m = Signal()
        self.n_carryin = Signal()
        self.n_carryout = Signal()
        self.x = Signal()
        self.y = Signal()
        self.a_eq_b = Signal()  # open-collector in the actual chip

    def ports(self):
        return [self.a, self.b, self.s, self.f, self.m, self.n_carryin,
                self.n_carryout, self.x, self.y, self.a_eq_b]

    def elaborate(self, _: Platform) -> Module:
        """Implements the logic of the 74181 chip.

        The logic is from the logic diagram in the data
        sheet, so that we're not fooled by any quirks.
        """
        m = Module()
        a = self.a
        b = self.b
        s = self.s
        n_cin = self.n_carryin
        arith = ~self.m

        # Intermediate signals, nor of ands.
        x = [None] * 8
        for i in range(4):
            ab_a0 = a[i]
            ab_a1 = b[i] & s[0]
            ab_a2 = ~b[i] & s[1]
            x[2*i] = ~(ab_a0 | ab_a1 | ab_a2)

            ab_b0 = a[i] & ~b[i] & s[2]
            ab_b1 = a[i] & b[i] & s[3]
            x[2*i+1] = ~(ab_b0 | ab_b1)

        x0 = x[0]
        x1 = x[1]
        x2 = x[2]
        x3 = x[3]
        x4 = x[4]
        x5 = x[5]
        x6 = x[6]
        x7 = x[7]

        # Next set of intermediate signals.

        y0a = arith & n_cin
        y0 = ~y0a

        y1 = x0 ^ x1

        y2a = arith & x0
        y2b = arith & x1 & n_cin
        y2 = ~(y2a | y2b)

        y3 = x2 ^ x3

        y4a = arith & x2
        y4b = arith & x0 & x3
        y4c = arith & x1 & x3 & n_cin
        y4 = ~(y4a | y4b | y4c)

        y5 = x4 ^ x5

        y6a = arith & x4
        y6b = arith & x2 & x5
        y6c = arith & x0 & x3 & x5
        y6d = arith & x1 & x3 & x5 & n_cin
        y6 = ~(y6a | y6b | y6c | y6d)

        y7 = x6 ^ x7

        y8 = ~(x1 & x3 & x5 & x7)
        y9 = ~(x1 & x3 & x5 & x7 & n_cin)

        y10a = x0 & x3 & x5 & x7
        y10b = x2 & x5 & x7
        y10c = x4 & x7
        y10d = x6
        y10 = ~(y10a | y10b | y10c | y10d)

        m.d.comb += self.f[0].eq(y0 ^ y1)
        m.d.comb += self.f[1].eq(y2 ^ y3)
        m.d.comb += self.f[2].eq(y4 ^ y5)
        m.d.comb += self.f[3].eq(y6 ^ y7)
        # This only works if function is minus with n_cin = 1.
        # F = A - B - 1 = 1111 only if A = B.
        m.d.comb += self.a_eq_b.eq(self.f[0]
                                   & self.f[1] & self.f[2] & self.f[3])
        m.d.comb += self.x.eq(y8)
        m.d.comb += self.y.eq(y10)
        m.d.comb += self.n_carryout.eq(~y9 | ~y10)

        return m

    @classmethod
    def formal_single(cls) -> Tuple[Module, List[Signal]]:
        """Formal verification for a single 74181."""
        m = Module()
        m.submodules.alu = alu = IC_74181()

        cin = Signal()
        m.d.comb += cin.eq(~alu.n_carryin)
        cout = Signal()
        m.d.comb += cout.eq(~alu.n_carryout)

        with m.If(alu.m):
            with m.Switch(alu.s):
                with m.Case(0):
                    m.d.comb += Assert(alu.f == ~alu.a)
                with m.Case(1):
                    m.d.comb += Assert(alu.f == ~(alu.a | alu.b))
                with m.Case(2):
                    m.d.comb += Assert(alu.f == ~alu.a & alu.b)
                with m.Case(3):
                    m.d.comb += Assert(alu.f == 0)
                with m.Case(4):
                    m.d.comb += Assert(alu.f == ~(alu.a & alu.b))
                with m.Case(5):
                    m.d.comb += Assert(alu.f == ~alu.b)
                with m.Case(6):
                    m.d.comb += Assert(alu.f == alu.a ^ alu.b)
                with m.Case(7):
                    m.d.comb += Assert(alu.f == alu.a & ~alu.b)
                with m.Case(8):
                    m.d.comb += Assert(alu.f == ~alu.a | alu.b)
                with m.Case(9):
                    m.d.comb += Assert(alu.f == ~(alu.a ^ alu.b))
                with m.Case(10):
                    m.d.comb += Assert(alu.f == alu.b)
                with m.Case(11):
                    m.d.comb += Assert(alu.f == alu.a & alu.b)
                with m.Case(12):
                    m.d.comb += Assert(alu.f == 0xF)
                with m.Case(13):
                    m.d.comb += Assert(alu.f == alu.a | ~alu.b)
                with m.Case(14):
                    m.d.comb += Assert(alu.f == alu.a | alu.b)
                with m.Case(15):
                    m.d.comb += Assert(alu.f == alu.a)
        with m.Else():
            sm = Signal(5)
            inv = Signal()
            m.d.comb += inv.eq(0)

            with m.Switch(alu.s):
                with m.Case(0):
                    m.d.comb += sm.eq(alu.a + cin)
                with m.Case(1):
                    m.d.comb += sm.eq((alu.a | alu.b) + cin)
                with m.Case(2):
                    m.d.comb += sm.eq((alu.a | ~alu.b) + cin)
                with m.Case(3):
                    m.d.comb += sm.eq(0xF + cin)
                with m.Case(4):
                    m.d.comb += sm.eq(alu.a + (alu.a & ~alu.b) + cin)
                with m.Case(5):
                    m.d.comb += sm.eq((alu.a | alu.b) + (alu.a & ~alu.b) + cin)

                with m.Case(6):
                    m.d.comb += sm.eq(alu.a - alu.b - ~cin)
                    m.d.comb += inv.eq(1)

                    # Check how equality, unsigned gt, and unsigned gte comparisons work.
                    with m.If(~cin):
                        m.d.comb += Assert(alu.a_eq_b == (alu.a == alu.b))
                        m.d.comb += Assert(cout == (alu.a > alu.b))
                    with m.Else():
                        m.d.comb += Assert(cout == (alu.a >= alu.b))

                with m.Case(7):
                    m.d.comb += sm.eq((alu.a & ~alu.b) - ~cin)
                    m.d.comb += inv.eq(1)
                with m.Case(8):
                    m.d.comb += sm.eq(alu.a + (alu.a & alu.b) + cin)

                with m.Case(9):
                    m.d.comb += sm.eq(alu.a + alu.b + cin)

                with m.Case(10):
                    m.d.comb += sm.eq((alu.a | ~alu.b) + (alu.a & alu.b) + cin)
                with m.Case(11):
                    m.d.comb += sm.eq((alu.a & alu.b) - ~cin)
                    m.d.comb += inv.eq(1)
                with m.Case(12):
                    m.d.comb += sm.eq(alu.a + alu.a + cin)
                with m.Case(13):
                    m.d.comb += sm.eq((alu.a | alu.b) + alu.a + cin)
                with m.Case(14):
                    m.d.comb += sm.eq((alu.a | ~alu.b) + alu.a + cin)
                with m.Case(15):
                    m.d.comb += sm.eq(alu.a - ~cin)
                    m.d.comb += inv.eq(1)
            m.d.comb += Assert(alu.f == sm[:4])
            m.d.comb += Assert(cout == inv ^ sm[4])

        return m, alu.ports()

    @classmethod
    def formal_ripple(cls) -> Tuple[Module, List[Signal]]:
        """Formal verification for a bunch of ALUs in ripple-carry mode."""
        m = Module()

        alus = [None] * 8
        m.submodules.alu0 = alus[0] = IC_74181()
        m.submodules.alu1 = alus[1] = IC_74181()
        m.submodules.alu2 = alus[2] = IC_74181()
        m.submodules.alu3 = alus[3] = IC_74181()
        m.submodules.alu4 = alus[4] = IC_74181()
        m.submodules.alu5 = alus[5] = IC_74181()
        m.submodules.alu6 = alus[6] = IC_74181()
        m.submodules.alu7 = alus[7] = IC_74181()

        a = Signal(32)
        b = Signal(32)
        f = Signal(32)
        cin = Signal()
        cout = Signal()
        s = Signal(4)
        mt = Signal()

        for x in range(8):
            m.d.comb += alus[x].a.eq(a[x*4:x*4+4])
            m.d.comb += alus[x].b.eq(b[x*4:x*4+4])
            m.d.comb += f[x*4:x*4+4].eq(alus[x].f)
            m.d.comb += alus[x].m.eq(mt)
            m.d.comb += alus[x].s.eq(s)
        for x in range(7):
            m.d.comb += alus[x+1].n_carryin.eq(alus[x].n_carryout)
        m.d.comb += alus[0].n_carryin.eq(~cin)
        m.d.comb += cout.eq(~alus[7].n_carryout)

        add_mode = (s == 9) & (mt == 0)
        sub_mode = (s == 6) & (mt == 0)
        m.d.comb += Assume(add_mode | sub_mode)

        y = Signal(33)
        with m.If(add_mode):
            m.d.comb += y.eq(a + b + cin)
            m.d.comb += Assert(f == y[:32])
            m.d.comb += Assert(cout == y[32])
        with m.Elif(sub_mode):
            m.d.comb += y.eq(a - b - ~cin)
            m.d.comb += Assert(f == y[:32])
            m.d.comb += Assert(cout == ~y[32])

            # Check how equality, unsigned gt, and unsigned gte comparisons work.
            with m.If(cin == 0):
                all_eq = Cat(*[i.a_eq_b for i in alus]).all()
                m.d.comb += Assert(all_eq == (a == b))
                m.d.comb += Assert(cout == (a > b))
            with m.Else():
                m.d.comb += Assert(cout == (a >= b))

        return m, [a, b, f, cin, cout, s, mt, y]

    @classmethod
    def formal(cls) -> Tuple[Module, List[Signal]]:
        """Formal verification for the 74181 chip."""
        m = Module()

        m1, ports1 = cls.formal_single()
        m2, ports2 = cls.formal_ripple()

        m.submodules += [m1, m2]
        return m, ports1 + ports2

    @classmethod
    def toRTL(cls) -> Tuple[Module, List[Signal]]:
        m = Module()
        m.submodules.chip = chip = cls()

        return m, [chip.a, chip.b, chip.s, chip.f, chip.m, chip.n_carryin, chip.n_carryout, chip.x, chip.y, chip.a_eq_b]


class IC_74182_active_low(Elaboratable):
    """Logic for the active low 74182 4-bit carry look-ahead unit.

    Attributes:
        ng: The four input generate signals, active low.
        np: The four input propagate signals, active low.
        carryin: The carry in input.
        carryout_x: The carry out for the first group.
        carryout_y: The carry out for the second group.
        carryout_z: The carry out for the third group.
        group_ng: The group generate output, active low.
        group_np: The group propagate output, active low.
    """

    ng: Signal
    np: Signal
    carryin: Signal
    carryout_x: Signal
    carryout_y: Signal
    carryout_z: Signal
    group_ng: Signal
    group_np: Signal

    def __init__(self):
        self.ng = Signal(4)
        self.np = Signal(4)
        self.carryin = Signal()
        self.carryout_x = Signal()
        self.carryout_y = Signal()
        self.carryout_z = Signal()
        self.group_ng = Signal()
        self.group_np = Signal()

    def ports(self):
        return [self.ng, self.np, self.carryin,
                self.carryout_x, self.carryout_y, self.carryout_z,
                self.group_ng, self.group_np]

    def elaborate(self, _: Platform) -> Module:
        """Implements the logic of the 74182 chip.

        The logic is from the logic diagram in the data
        sheet, so that we're not fooled by any quirks.
        """
        m = Module()

        np0 = self.np[0]
        np1 = self.np[1]
        np2 = self.np[2]
        np3 = self.np[3]
        ng0 = self.ng[0]
        ng1 = self.ng[1]
        ng2 = self.ng[2]
        ng3 = self.ng[3]
        n_cin = ~self.carryin

        m.d.comb += self.group_np.eq(np0 | np1 | np2 | np3)

        cx0 = np0 & ng0
        cx1 = n_cin & ng0
        m.d.comb += self.carryout_x.eq(~(cx0 | cx1))

        cy0 = np1 & ng1
        cy1 = ng0 & ng1 & np0
        cy2 = n_cin & ng0 & ng1
        m.d.comb += self.carryout_y.eq(~(cy0 | cy1 | cy2))

        cz0 = np2 & ng2
        cz1 = ng1 & ng2 & np1
        cz2 = ng0 & ng1 & ng2 & np0
        cz3 = n_cin & ng0 & ng1 & ng2
        m.d.comb += self.carryout_z.eq(~(cz0 | cz1 | cz2 | cz3))

        gg0 = np3 & ng3
        gg1 = ng2 & ng3 & np2
        gg2 = ng1 & ng2 & ng3 & np1
        gg3 = ng0 & ng1 & ng2 & ng3
        m.d.comb += self.group_ng.eq(gg0 | gg1 | gg2 | gg3)

        return m

    @classmethod
    def formal(cls) -> Tuple[Module, List[Signal]]:
        """Formal verification for the active low 74182 chip."""
        m = Module()
        m.submodules.clu = clu = IC_74182_active_low()

        # Verify the truth tables in the datasheet

        with m.If(clu.np.matches("0000")):
            m.d.comb += Assert(clu.group_np == 0)
        with m.Else():
            m.d.comb += Assert(clu.group_np == 1)

        with m.If(clu.ng.matches("0---") & clu.np.matches("----")):
            m.d.comb += Assert(clu.group_ng == 0)
        with m.Elif(clu.ng.matches("-0--") & clu.np.matches("0---")):
            m.d.comb += Assert(clu.group_ng == 0)
        with m.Elif(clu.ng.matches("--0-") & clu.np.matches("00--")):
            m.d.comb += Assert(clu.group_ng == 0)
        with m.Elif(clu.ng.matches("---0") & clu.np.matches("000-")):
            m.d.comb += Assert(clu.group_ng == 0)
        with m.Else():
            m.d.comb += Assert(clu.group_ng == 1)

        with m.If(clu.ng[0] == 0):
            m.d.comb += Assert(clu.carryout_x == 1)
        with m.Elif((clu.np[0] == 0) & (clu.carryin == 1)):
            m.d.comb += Assert(clu.carryout_x == 1)
        with m.Else():
            m.d.comb += Assert(clu.carryout_x == 0)

        with m.If(clu.ng.matches("--0-") & clu.np.matches("----")):
            m.d.comb += Assert(clu.carryout_y == 1)
        with m.Elif(clu.ng.matches("---0") & clu.np.matches("--0-")):
            m.d.comb += Assert(clu.carryout_y == 1)
        with m.Elif(clu.ng.matches("----") & clu.np.matches("--00") & (clu.carryin == 1)):
            m.d.comb += Assert(clu.carryout_y == 1)
        with m.Else():
            m.d.comb += Assert(clu.carryout_y == 0)

        with m.If(clu.ng.matches("-0--") & clu.np.matches("----")):
            m.d.comb += Assert(clu.carryout_z == 1)
        with m.Elif(clu.ng.matches("--0-") & clu.np.matches("-0--")):
            m.d.comb += Assert(clu.carryout_z == 1)
        with m.Elif(clu.ng.matches("---0") & clu.np.matches("-00-")):
            m.d.comb += Assert(clu.carryout_z == 1)
        with m.Elif(clu.ng.matches("----") & clu.np.matches("-000") & (clu.carryin == 1)):
            m.d.comb += Assert(clu.carryout_z == 1)
        with m.Else():
            m.d.comb += Assert(clu.carryout_z == 0)

        return m, clu.ports()


class IC_74182_active_high(Elaboratable):
    """Logic for the active high 74182 4-bit carry look-ahead unit.

    This is just a renaming of the signals to correspond to the
    active-high set.

    Attributes:
        y: The four input generate signals, active low.
        x: The four input propagate signals, active low.
        n_carryin: The carry in input.
        n_carryout_x: The carry out for the first group.
        n_carryout_y: The carry out for the second group.
        n_carryout_z: The carry out for the third group.
        group_y: The group generate output, active low.
        group_x: The group propagate output, active low.
    """

    y: Signal
    x: Signal
    n_carryin: Signal
    n_carryout_x: Signal
    n_carryout_y: Signal
    n_carryout_z: Signal
    group_y: Signal
    group_x: Signal

    def __init__(self):
        self.y = Signal(4)
        self.x = Signal(4)
        self.n_carryin = Signal()
        self.n_carryout_x = Signal()
        self.n_carryout_y = Signal()
        self.n_carryout_z = Signal()
        self.group_y = Signal()
        self.group_x = Signal()

    def ports(self):
        return [self.y, self.x, self.n_carryin,
                self.n_carryout_x, self.n_carryout_y, self.n_carryout_z,
                self.group_y, self.group_x]

    def elaborate(self, _: Platform) -> Module:
        """Implements the logic of the active high 74182 chip."""
        m = Module()
        m.submodules.low = low = IC_74182_active_low()

        m.d.comb += [
            low.ng.eq(self.y),
            low.np.eq(self.x),
            low.carryin.eq(self.n_carryin),
            self.n_carryout_x.eq(low.carryout_x),
            self.n_carryout_z.eq(low.carryout_z),
            self.n_carryout_z.eq(low.carryout_z),
            self.group_y.eq(low.group_ng),
            self.group_x.eq(low.group_np),
        ]

        return m

    @classmethod
    def formal(cls) -> Tuple[Module, List[Signal]]:
        """Formal verification for the active high 74182 chip.

        Used with an active high 74181.
        """
        m = Module()
        m.submodules.clu = clu = IC_74182_active_high()
        m.submodules.alu = alu = IC_74181()

        add_mode = (alu.s == 9) & (alu.m == 0)
        m.d.comb += Assume(add_mode)

        m.d.comb += [
            clu.x[0].eq(alu.x),
            clu.y[0].eq(alu.y),
            clu.x[1:].eq(0),
            clu.y[1:].eq(0),
            clu.n_carryin.eq(alu.n_carryin),
        ]

        m.d.comb += Assert(clu.n_carryout_x == alu.n_carryout)

        return m, clu.ports() + alu.ports()


class IC_74181_and_or_layer1(Elaboratable):
    """Logic for the first and-or layer of the 74181 4-bit ALU.
    """

    def __init__(self):
        self.a = Signal(4)
        self.b = Signal(4)
        self.s = Signal(4)
        self.x = Signal(8)
        self.m = Signal()
        self.n_carryin = Signal()

    def ports(self):
        return [self.a, self.b, self.s, self.x, self.m, self.n_carryin]

    def elaborate(self, _: Platform) -> Module:
        """Implements the logic of the 74181 chip.

        The logic is from the logic diagram in the data
        sheet, so that we're not fooled by any quirks.
        """
        m = Module()
        a = self.a
        b = self.b
        s = self.s
        x = self.x

        for i in range(4):
            ab_a0 = a[i]
            ab_a1 = b[i] & s[0]
            ab_a2 = ~b[i] & s[1]
            m.d.comb += x[2*i].eq(~(ab_a0 | ab_a1 | ab_a2))

            ab_b0 = a[i] & ~b[i] & s[2]
            ab_b1 = a[i] & b[i] & s[3]
            m.d.comb += x[2*i+1].eq(~(ab_b0 | ab_b1))

        return m

    @classmethod
    def toRTL(cls) -> Tuple[Module, List[Signal]]:
        m = Module()
        m.submodules.chip = chip = cls()

        return m, chip.ports()


class IC_74181_and_or_layer2(Elaboratable):
    """Logic for the second and-or layer of the 74181 4-bit ALU.
    """

    def __init__(self):
        self.x = Signal(8)
        self.m = Signal()
        self.n_carryin = Signal()
        self.n = Signal(7)

    def ports(self):
        return [self.x, self.m, self.n_carryin, self.n]

    def elaborate(self, _: Platform) -> Module:
        """Implements the logic of the 74181 chip.

        The logic is from the logic diagram in the data
        sheet, so that we're not fooled by any quirks.
        """
        m = Module()
        n_cin = self.n_carryin
        arith = ~self.m
        x = self.x
        n = self.n

        x0 = x[0]
        x1 = x[1]
        x2 = x[2]
        x3 = x[3]
        x4 = x[4]
        x5 = x[5]
        x6 = x[6]
        x7 = x[7]

        # Next set of intermediate signals.

        m.d.comb += n[0].eq(~(arith & n_cin))

        y2a = arith & x0
        y2b = arith & x1 & n_cin
        m.d.comb += n[1].eq(~(y2a | y2b))

        y4a = arith & x2
        y4b = arith & x0 & x3
        y4c = arith & x1 & x3 & n_cin
        m.d.comb += n[2].eq(~(y4a | y4b | y4c))

        y6a = arith & x4
        y6b = arith & x2 & x5
        y6c = arith & x0 & x3 & x5
        y6d = arith & x1 & x3 & x5 & n_cin
        m.d.comb += n[3].eq(~(y6a | y6b | y6c | y6d))

        m.d.comb += n[4].eq(~(x1 & x3 & x5 & x7 & n_cin))

        y10a = x0 & x3 & x5 & x7
        y10b = x2 & x5 & x7
        y10c = x4 & x7
        y10d = x6
        m.d.comb += n[5].eq(~(y10a | y10b | y10c | y10d))

        m.d.comb += n[6].eq(~(x1 & x3 & x5 & x7))

        return m

    @classmethod
    def toRTL(cls) -> Tuple[Module, List[Signal]]:
        m = Module()
        m.submodules.chip = chip = cls()

        return m, chip.ports()


class IC_74181_and_or_layer3(Elaboratable):
    """Logic for the third and-or layer of the 74181 4-bit ALU.
    """

    def __init__(self):
        self.f = Signal(4)
        self.n = Signal(7)
        self.n_carryout = Signal()
        self.a_eq_b = Signal()  # open-collector in the actual chip

    def ports(self):
        return [self.f, self.n, self.n_carryout, self.a_eq_b]

    def elaborate(self, _: Platform) -> Module:
        """Implements the logic of the 74181 chip.

        The logic is from the logic diagram in the data
        sheet, so that we're not fooled by any quirks.
        """
        m = Module()
        f = self.f
        n = self.n
        n_cout = self.n_carryout
        a_eq_b = self.a_eq_b

        # This only works if function is minus with n_cin = 1.
        # F = A - B - 1 = 1111 only if A = B.
        m.d.comb += a_eq_b.eq(f[0] & f[1] & f[2] & f[3])
        m.d.comb += n_cout.eq(~n[4] | ~n[5])

        return m

    @classmethod
    def toRTL(cls) -> Tuple[Module, List[Signal]]:
        m = Module()
        m.submodules.chip = chip = cls()

        return m, chip.ports()


class IC_74688_and_or_layer1(Elaboratable):
    def __init__(self):
        self.x = Signal(8)
        self.p_neq_q = Signal()

    def ports(self):
        return [self.x, self.p_neq_q]

    def elaborate(self, _: Platform) -> Module:
        m = Module()
        x = self.x

        m.d.comb += self.p_neq_q.eq(x[0] | x[1] |
                                    x[2] | x[3] | x[4] | x[5] | x[6] | x[7])

        return m

    @ classmethod
    def toRTL(cls) -> Tuple[Module, List[Signal]]:
        m = Module()
        m.submodules.chip = chip = cls()

        return m, chip.ports()


if __name__ == "__main__":
    main(IC_74688_and_or_layer1)
