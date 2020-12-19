# Disable pylint's "your name is too short" warning.
# pylint: disable=C0103
# Disable protected access warnings
# pylint: disable=W0212
from typing import List, Tuple

from nmigen import Signal, Module, Elaboratable, signed, ClockSignal, ClockDomain, Array
from nmigen.build import Platform
from nmigen.asserts import Assert, Assume, Cover, Stable, Past

from util import main


class IC_7416244_sub(Elaboratable):
    """Contains logic for a 7416244 4-bit buffer.
    """

    def __init__(self):
        self.a = Signal(4)
        self.n_oe = Signal()
        self.y = Signal(4)

    def elaborate(self, _: Platform) -> Module:
        """Implements the logic of the buffer."""
        m = Module()

        m.d.comb += self.y.eq(0)
        with m.If(self.n_oe == 0):
            m.d.comb += self.y.eq(self.a)

        return m


class IC_7416244(Elaboratable):
    """Contains logic for a 7416244 16-bit buffer.
    """

    def __init__(self):
        # Inputs
        self.a0 = Signal(4)
        self.a1 = Signal(4)
        self.a2 = Signal(4)
        self.a3 = Signal(4)

        # Output enables
        self.n_oe0 = Signal()
        self.n_oe1 = Signal()
        self.n_oe2 = Signal()
        self.n_oe3 = Signal()

        # Outputs
        self.y0 = Signal(4)
        self.y1 = Signal(4)
        self.y2 = Signal(4)
        self.y3 = Signal(4)

    def elaborate(self, _: Platform) -> Module:
        """Implements the logic of the buffer."""
        m = Module()

        s0 = IC_7416244_sub()
        s1 = IC_7416244_sub()
        s2 = IC_7416244_sub()
        s3 = IC_7416244_sub()

        m.submodules += [s0, s1, s2, s3]

        m.d.comb += s0.a.eq(self.a0)
        m.d.comb += s1.a.eq(self.a1)
        m.d.comb += s2.a.eq(self.a2)
        m.d.comb += s3.a.eq(self.a3)

        m.d.comb += s0.n_oe.eq(self.n_oe0)
        m.d.comb += s1.n_oe.eq(self.n_oe1)
        m.d.comb += s2.n_oe.eq(self.n_oe2)
        m.d.comb += s3.n_oe.eq(self.n_oe3)

        m.d.comb += self.y0.eq(s0.y)
        m.d.comb += self.y1.eq(s1.y)
        m.d.comb += self.y2.eq(s2.y)
        m.d.comb += self.y3.eq(s3.y)

        return m

    @classmethod
    def formal(cls) -> Tuple[Module, List[Signal]]:
        m = Module()
        s = IC_7416244()

        m.submodules += s

        with m.If(s.n_oe0):
            m.d.comb += Assert(s.y0 == 0)
        with m.Else():
            m.d.comb += Assert(s.y0 == s.a0)
        with m.If(s.n_oe1):
            m.d.comb += Assert(s.y1 == 0)
        with m.Else():
            m.d.comb += Assert(s.y1 == s.a1)
        with m.If(s.n_oe2):
            m.d.comb += Assert(s.y2 == 0)
        with m.Else():
            m.d.comb += Assert(s.y2 == s.a2)
        with m.If(s.n_oe3):
            m.d.comb += Assert(s.y3 == 0)
        with m.Else():
            m.d.comb += Assert(s.y3 == s.a3)

        return m, [s.a0, s.a1, s.a2, s.a3, s.n_oe0, s.n_oe1, s.n_oe2, s.n_oe3]


if __name__ == "__main__":
    main(IC_7416244)
