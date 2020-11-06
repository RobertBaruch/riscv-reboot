# Disable pylint's "your name is too short" warning.
# pylint: disable=C0103
"""
This module provides an implementation of a transparent latch.
"""
from typing import List, Tuple

from nmigen import Signal, Module, Elaboratable, ClockDomain
from nmigen import Mux
from nmigen.build import Platform
from nmigen.sim import Simulator, Delay
from nmigen.asserts import Assert, Cover, Fell, Past

from util import main


class TransparentLatch(Elaboratable):
    """Logic for a transparent latch like the 74373.

    Attributes:
        data_in: The input data.
        data_out: The output data. Since nMigen doesn't support Z, the
            output is 0 when the output is disabled.
        n_oe: Output enable, active low. When n_oe is 1, the output is 0.
        le: Latch enable. When 1, the output is the input, that is, the
            latch is transparent. Otherwise, the output is the input when
            le was last 1, that is, the latch is latched.
    """
    def __init__(self, size: int):
        """Constructs a transparent latch.

        Args:
            size: The number of bits in the latch.
        """
        assert size > 0
        self.data_in = Signal(size)
        self.data_out = Signal(size)
        self.le = Signal()
        self.n_oe = Signal()

        self.size = size

    def elaborate(self, _: Platform) -> Module:
        """Implements the logic of a transparent latch."""
        m = Module()

        internal_reg = Signal(self.size, reset=0, reset_less=True)

        # Local clock domain so we can clock data into the
        # internal memory on the negative edge of le.
        le_clk = ClockDomain("le_clk", clk_edge="neg", local=True)
        m.domains.le_clk = le_clk
        le_clk.clk = self.le

        m.d.le_clk += internal_reg.eq(self.data_in)
        m.d.comb += self.data_out.eq(Mux(self.n_oe, 0, internal_reg))
        with m.If(~self.le & ~self.n_oe):
            m.d.comb += self.data_out.eq(self.data_in)

        return m

    @classmethod
    def sim(cls):
        """A quick simulation of the transparent latch."""
        m = Module()
        m.submodules.latch = latch = TransparentLatch(32)

        sim = Simulator(m)

        def process():
            yield latch.n_oe.eq(1)
            yield latch.le.eq(1)
            yield Delay(1e-6)
            yield latch.data_in.eq(0xAAAA1111)
            yield Delay(1e-6)
            yield latch.data_in.eq(0x1111AAAA)
            yield Delay(1e-6)
            yield latch.le.eq(0)
            yield Delay(1e-6)
            yield latch.data_in.eq(0xAAAA1111)
            yield Delay(1e-6)
            yield latch.le.eq(1)
            yield Delay(1e-6)

        sim.add_process(process)
        with sim.write_vcd("latch.vcd"):
            sim.run_until(10e-6)

    @classmethod
    def formal(cls) -> Tuple[Module, List[Signal]]:
        """Formal verification for the transparent latch.

        Note that you MUST have multiclock on in the sby file, because there is
        more than one clock in the system -- the default formal clock and the
        local clock inside the transparent latch.
        """
        m = Module()
        m.submodules.latch = latch = TransparentLatch(32)

        m.d.sync += Cover((latch.data_out == 0xAAAAAAAA) & (latch.le == 0)
                          & (Past(latch.data_out, 2) == 0xBBBBBBBB)
                          & (Past(latch.le, 2) == 0))

        with m.If(latch.n_oe == 1):
            m.d.comb += Assert(latch.data_out == 0)

        with m.If((latch.n_oe == 0) & (latch.le == 1)):
            m.d.comb += Assert(latch.data_out == latch.data_in)

        with m.If((latch.n_oe == 0) & Fell(latch.le)):
            m.d.sync += Assert(latch.data_out == Past(latch.data_in))

        return m, [latch.data_in, latch.le, latch.n_oe]


if __name__ == "__main__":
    main(TransparentLatch)
