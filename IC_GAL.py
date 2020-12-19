# Disable pylint's "your name is too short" warning.
# pylint: disable=C0103
# Disable protected access warnings
# pylint: disable=W0212
from typing import List, Tuple

from nmigen import Signal, Module, Elaboratable, signed, ClockSignal, ClockDomain, Array
from nmigen.build import Platform
from nmigen.asserts import Assert, Assume, Cover, Stable, Past

from util import main


class IC_GAL_imm_format_decoder(Elaboratable):
    """Contains logic for a 7416244 16-bit buffer.
    """

    def __init__(self):
        # Inputs
        self.opcode = Signal(7)

        # Outputs
        self.i_n_oe = Signal()
        self.s_n_oe = Signal()
        self.u_n_oe = Signal()
        self.b_n_oe = Signal()
        self.j_n_oe = Signal()
        self.sys_n_oe = Signal()

    def elaborate(self, _: Platform) -> Module:
        """Implements the logic of the buffer."""
        m = Module()

        # OE_I = 0000011 | 0001111 | 0010011
        # OE_S = 0100011
        # OE_U = 0010111 | 0110111
        # OE_B = 1100011
        # OE_J = 1100111 | 1101111
        # OE_SYS = 1110011

        m.d.comb += [
            self.i_n_oe.eq(1),
            self.s_n_oe.eq(1),
            self.u_n_oe.eq(1),
            self.b_n_oe.eq(1),
            self.j_n_oe.eq(1),
            self.sys_n_oe.eq(1),
        ]
        with m.Switch(self.opcode):
            with m.Case(0b0000011, 0b0001111, 0b0010011):  # I
                m.d.comb += self.i_n_oe.eq(0)
            with m.Case(0b0100011):  # S
                m.d.comb += self.s_n_oe.eq(0)
            with m.Case(0b0010111, 0b0110111):  # U
                m.d.comb += self.u_n_oe.eq(0)
            with m.Case(0b1100011):
                m.d.comb += self.b_n_oe.eq(0)
            with m.Case(0b1100111, 0b1101111):
                m.d.comb += self.j_n_oe.eq(0)
            with m.Default():
                m.d.comb += self.sys_n_oe.eq(0)
        return m


if __name__ == "__main__":
    main(IC_GAL_imm_format_decoder)
