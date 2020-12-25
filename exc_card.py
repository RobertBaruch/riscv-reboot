# Disable pylint's "your name is too short" warning.
# pylint: disable=C0103
from typing import List

from nmigen import Signal, Module, Elaboratable
from nmigen.build import Platform

from IC_7416374 import IC_reg32_with_mux
from consts import CSRAddr


class ExcCard(Elaboratable):
    """Logic for the exception card.

    Attributes:
        data_x_in: The X bus, read from.
        data_x_out: The X bus, written to.
        data_y_in: The Y bus, always read from.
        data_z_in: The Z bus, always read from.
        csr_num: The CSR number for access.
        csr_to_x: Read the CSR (output to X).
        z_to_csr: Write the CSR (input from Z).
        save_trap_csrs: Signal that the buses should be saved to the CSRs.

    The card holds the MCAUSE, MEPC, and MTVAL registers. Aside from responding
    to the normal CSR read/write instructions, it can also store X -> MCAUSE,
    Y -> MEPC, and Z -> MTVAL when requested.
    """

    data_x_in: Signal
    data_x_out: Signal
    data_y_in: Signal
    data_z_in: Signal
    csr_num: Signal
    csr_to_x: Signal
    z_to_csr: Signal
    save_trap_csrs: Signal

    def __init__(self, ext_init: bool = False):
        """Constructs an exception card."""
        self.ext_init = ext_init
        attrs = [] if not ext_init else [("uninitialized", "")]

        # Buses
        self.data_x_in = Signal(32)
        self.data_x_out = Signal(32)
        self.data_y_in = Signal(32)
        self.data_z_in = Signal(32)

        # Controls
        self.csr_num = Signal(CSRAddr)
        self.csr_to_x = Signal()
        self.z_to_csr = Signal()
        self.save_trap_csrs = Signal()

        # Internals
        self._mcause = Signal(32)
        self._mepc = Signal(32)
        self._mtval = Signal(32)

    def elaborate(self, _: Platform) -> Module:
        """Implements the logic of the exception card."""
        m = Module()

        self.multiplex_to_reg(m, clk="ph2w", reg=self._mcause,
                              sels=[
                                  self.z_to_csr & (
                                      self.csr_num == CSRAddr.MCAUSE),
                                  self.save_trap_csrs
                              ],
                              sigs=[self.data_z_in, self.data_x_in])
        self.multiplex_to_reg(m, clk="ph2w", reg=self._mepc,
                              sels=[
                                  self.z_to_csr & (
                                      self.csr_num == CSRAddr.MEPC),
                                  self.save_trap_csrs
                              ],
                              sigs=[self.data_z_in, self.data_y_in])
        self.multiplex_to_reg(m, clk="ph2w", reg=self._mtval,
                              sels=[
                                  (self.z_to_csr & (self.csr_num == CSRAddr.MTVAL)) |
                                  self.save_trap_csrs,
                              ],
                              sigs=[self.data_z_in])

        m.d.comb += self.data_x_out.eq(0)
        with m.If(self.csr_to_x & ~self.save_trap_csrs):
            with m.Switch(self.csr_num):
                with m.Case(CSRAddr.MCAUSE):
                    m.d.comb += self.data_x_out.eq(self._mcause)
                with m.Case(CSRAddr.MEPC):
                    m.d.comb += self.data_x_out.eq(self._mepc)
                with m.Case(CSRAddr.MTVAL):
                    m.d.comb += self.data_x_out.eq(self._mtval)

        return m

    def multiplex_to_reg(self, m: Module, clk: str, reg: Signal, sels: List[Signal], sigs: List[Signal]):
        """Sets up a multiplexer with a register.

        clk is the clock domain on which the register is clocked.

        reg is the register signal.

        sels is an array of Signals which select that input for the multiplexer (active high). If
        no select is active, then the register retains its value.

        sigs is an array of Signals which are the inputs to the multiplexer.
        """
        assert len(sels) == len(sigs)

        muxreg = IC_reg32_with_mux(
            clk=clk, N=len(sels), ext_init=self.ext_init, faster=True)
        m.submodules += muxreg
        m.d.comb += reg.eq(muxreg.q)
        for i in range(len(sels)):
            m.d.comb += muxreg.n_sel[i].eq(~sels[i])
            m.d.comb += muxreg.d[i].eq(sigs[i])
