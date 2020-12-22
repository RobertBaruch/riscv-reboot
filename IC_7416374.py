"""Module containing stuff made out of 7416374 16-bit registers."""
# Disable pylint's "your name is too short" warning.
# pylint: disable=C0103
# Disable protected access warnings
# pylint: disable=W0212
from typing import List, Tuple

from nmigen import Signal, Module, Elaboratable, Array, ClockSignal, ResetSignal, ClockDomain
from nmigen.build import Platform
from nmigen.asserts import Assert, Assume, Past, Rose, Fell

from IC_7416244 import IC_mux32
from util import main


class IC_7416374(Elaboratable):
    """Contains logic for a 7416374 16-bit register.
    """

    def __init__(self, clk, ext_init: bool = False):
        attrs = [] if not ext_init else [("uninitialized", "")]

        self.clk = clk

        self.d = Signal(16)
        self.n_oe = Signal()
        self.q = Signal(16)

        self._q = Signal(16, attrs=attrs)

    def elaborate(self, _: Platform) -> Module:
        """Implements the logic of the register."""
        m = Module()

        self.clk += self._q.eq(self.d)
        m.d.comb += self.q.eq(0)
        with m.If(~self.n_oe):
            m.d.comb += self.q.eq(self._q)

        return m

    @classmethod
    def formal(cls) -> Tuple[Module, List[Signal]]:
        m = Module()
        ph = ClockDomain("clk")
        clk = ClockSignal("ph")

        m.domains += ph
        m.d.sync += clk.eq(~clk)

        s = IC_7416374(m.d.ph)

        m.submodules += s

        with m.If(s.n_oe):
            m.d.comb += Assert(s.q == 0)

        with m.If(~s.n_oe & Rose(clk)):
            m.d.comb += Assert(s.q == Past(s.d))

        with m.If(~s.n_oe & Fell(clk) & ~Past(s.n_oe)):
            m.d.comb += Assert(s.q == Past(s.q))

        sync_clk = ClockSignal("sync")
        sync_rst = ResetSignal("sync")

        # Make sure the clock is clocking
        m.d.comb += Assume(sync_clk == ~Past(sync_clk))

        # Don't want to test what happens when we reset.
        m.d.comb += Assume(~sync_rst)
        m.d.comb += Assume(~ResetSignal("ph"))

        return m, [sync_clk, sync_rst, s.n_oe, s.q, s.d]


class IC_reg32(Elaboratable):
    """A 32-bit register from a pair of 16-bit registers."""

    def __init__(self, clk, ext_init: bool = False):
        self.clk = clk
        self.d = Signal(32)
        self.n_oe = Signal()
        self.q = Signal(32)
        self.ext_init = ext_init

    def elaborate(self, _: Platform) -> Module:
        """Implements the logic of the register."""
        m = Module()
        regs = [IC_7416374(self.clk, self.ext_init),
                IC_7416374(self.clk, self.ext_init)]
        m.submodules += regs

        for i in range(2):
            m.d.comb += regs[i].n_oe.eq(self.n_oe)
            m.d.comb += regs[i].d.eq(self.d[i*16:i*16+16])
            m.d.comb += self.q[i*16:i*16+16].eq(regs[i].q)

        return m


class IC_reg32_with_mux(Elaboratable):
    """A 32-bit register that multiplexes any of its inputs, or itself.

    There is no output enable input.
    """

    def __init__(self, clk, N: int, ext_init: bool = False):
        self.N = N
        self.clk = clk
        self.d = Array([Signal(32) for _ in range(N)])
        self.n_sel = Signal(N)
        self.q = Signal(32)
        self.ext_init = ext_init

    def elaborate(self, _: Platform) -> Module:
        """The logic."""
        m = Module()
        r = IC_reg32(self.clk, self.ext_init)
        mux = IC_mux32(self.N + 1)
        m.submodules += [r, mux]

        m.d.comb += r.n_oe.eq(0)
        m.d.comb += r.d.eq(mux.y)
        m.d.comb += self.q.eq(r.q)

        m.d.comb += mux.n_sel.eq(self.n_sel)
        m.d.comb += mux.n_sel[-1].eq(~self.n_sel != 0)

        for i in range(self.N):
            m.d.comb += mux.a[i].eq(self.d[i])
        m.d.comb += mux.a[-1].eq(r.q)

        return m

    @classmethod
    def formal(cls) -> Tuple[Module, List[Signal]]:
        m = Module()
        ph = ClockDomain("clk")
        clk = ClockSignal("ph")

        m.domains += ph
        m.d.sync += clk.eq(~clk)

        s = IC_reg32_with_mux(m.d.ph, 2)

        m.submodules += s

        sync_clk = ClockSignal("sync")
        sync_rst = ResetSignal("sync")

        with m.If(Rose(clk)):
            with m.Switch(~Past(s.n_sel)):
                with m.Case(0b11):
                    m.d.comb += Assert(0)
                with m.Case(0b01):
                    m.d.comb += Assert(s.q == Past(s.d[0]))
                with m.Case(0b10):
                    m.d.comb += Assert(s.q == Past(s.d[1]))
                with m.Default():
                    m.d.comb += Assert(s.q == Past(s.q))

        # Make sure the clock is clocking
        m.d.comb += Assume(sync_clk == ~Past(sync_clk))

        # Don't want to test what happens when we reset.
        m.d.comb += Assume(~sync_rst)
        m.d.comb += Assume(~ResetSignal("ph"))

        m.d.comb += Assume(s.n_sel != 0)

        return m, [sync_clk, sync_rst, s.d[0], s.d[1], s.n_sel, s.q]


class IC_reg32_with_load(Elaboratable):
    """A 32-bit register that loads or retains its value."""

    def __init__(self, clk):
        self.clk = clk
        self.d = Signal(32)
        self.n_oe = Signal()
        self.load = Signal()
        self.q = Signal(32)

    def elaborate(self, _: Platform) -> Module:
        """Implements the logic of the register."""
        m = Module()
        reg = IC_reg32(self.clk)
        mux = IC_mux32(2)

        m.submodules += [reg, mux]

        m.d.comb += reg.n_oe.eq(self.n_oe)

        m.d.comb += mux.a[0].eq(reg.q)
        m.d.comb += mux.a[1].eq(self.d)
        m.d.comb += mux.n_sel[0].eq(self.load)
        m.d.comb += mux.n_sel[1].eq(~self.load)
        m.d.comb += reg.d.eq(mux.y)
        m.d.comb += self.q.eq(reg.q)

        return m

    @classmethod
    def formal(cls) -> Tuple[Module, List[Signal]]:
        m = Module()
        ph = ClockDomain("clk")
        clk = ClockSignal("ph")

        m.domains += ph
        m.d.sync += clk.eq(~clk)

        s = IC_reg32_with_load(m.d.ph)

        m.submodules += s

        sync_clk = ClockSignal("sync")
        sync_rst = ResetSignal("sync")

        with m.If(s.n_oe):
            m.d.comb += Assert(s.q == 0)

        with m.Else():
            with m.If(~Past(s.n_oe) & ~Past(s.load)):
                m.d.comb += Assert(s.q == Past(s.q))

            with m.If(Past(s.load) & Rose(clk)):
                m.d.comb += Assert(s.q == Past(s.d))

            with m.If(~Past(s.n_oe) & Fell(clk)):
                m.d.comb += Assert(s.q == Past(s.q))

        # Make sure the clock is clocking
        m.d.comb += Assume(sync_clk == ~Past(sync_clk))

        # Don't want to test what happens when we reset.
        m.d.comb += Assume(~sync_rst)
        m.d.comb += Assume(~ResetSignal("ph"))

        return m, [sync_clk, sync_rst, s.n_oe, s.load, s.d]


if __name__ == "__main__":
    main(IC_reg32_with_mux)
