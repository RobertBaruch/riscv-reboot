"""Module containing stuff made out of 7416244 16-bit buffers."""
# Disable pylint's "your name is too short" warning.
# pylint: disable=C0103
# Disable protected access warnings
# pylint: disable=W0212
from typing import List, Tuple

from nmigen import Signal, Module, Elaboratable, Array
from nmigen.build import Platform
from nmigen.asserts import Assert

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


class IC_buff32(Elaboratable):
    """A pair of 7416244s, with OEs tied together."""

    def __init__(self):
        self.a = Signal(32)
        self.n_oe = Signal()
        self.y = Signal(32)

    def elaborate(self, _: Platform) -> Module:
        """Implements the logic of a 32-bit buffer.

        Made out of a pair of 7416244s.
        """
        m = Module()

        buffs = [IC_7416244(), IC_7416244()]
        m.submodules += buffs

        for i in range(2):
            m.d.comb += buffs[i].n_oe0.eq(self.n_oe)
            m.d.comb += buffs[i].n_oe1.eq(self.n_oe)
            m.d.comb += buffs[i].n_oe2.eq(self.n_oe)
            m.d.comb += buffs[i].n_oe3.eq(self.n_oe)

        for i in range(2):
            m.d.comb += buffs[i].a0.eq(self.a[16*i:16*i+4])
            m.d.comb += buffs[i].a1.eq(self.a[16*i+4:16*i+8])
            m.d.comb += buffs[i].a2.eq(self.a[16*i+8:16*i+12])
            m.d.comb += buffs[i].a3.eq(self.a[16*i+12:16*i+16])

        for i in range(2):
            m.d.comb += self.y[16*i:16*i+4].eq(buffs[i].y0)
            m.d.comb += self.y[16*i+4:16*i+8].eq(buffs[i].y1)
            m.d.comb += self.y[16*i+8:16*i+12].eq(buffs[i].y2)
            m.d.comb += self.y[16*i+12:16*i+16].eq(buffs[i].y3)

        return m


class IC_mux32(Elaboratable):
    """An N-input 32-bit multiplexer made of 7416244s.

    Select lines are separate. Activating more than one is a really
    bad idea.
    """

    def __init__(self, N: int):
        self.N = N
        self.a = Array([Signal(32, name=f"mux_in{i}") for i in range(N)])
        self.n_sel = Signal(N)
        self.y = Signal(32)

    def elaborate(self, _: Platform) -> Module:
        """Implements the logic of an N-input 32-bit multiplexer."""
        m = Module()

        buffs = [IC_buff32() for _ in range(self.N)]
        m.submodules += buffs

        for i in range(self.N):
            m.d.comb += buffs[i].a.eq(self.a[i])
            m.d.comb += buffs[i].n_oe.eq(self.n_sel[i])

        combine = 0
        for i in range(0, self.N):
            combine |= buffs[i].y
        m.d.comb += self.y.eq(combine)

        return m


if __name__ == "__main__":
    main(IC_7416244)
