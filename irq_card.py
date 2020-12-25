# Disable pylint's "your name is too short" warning.
# pylint: disable=C0103
from typing import List

from nmigen import Signal, Module, Elaboratable
from nmigen.build import Platform

from IC_7416244 import IC_mux32
from IC_7416374 import IC_reg32_with_mux
from consts import CSRAddr, MStatus, MInterrupt


class IrqCard(Elaboratable):
    """Logic for the interrupt card.

    The card holds the MSTATUS, MIE and MIP registers. Aside from responding
    to the normal CSR read/write instructions, it can also do other stuff.
    """

    def __init__(self, ext_init: bool = False):
        """Constructs an interrupt card."""
        self._ext_init = ext_init

        # Buses
        self.data_x_out = Signal(32)
        self.data_z_in = Signal(32)

        # Controls
        self.csr_num = Signal(CSRAddr)
        self.csr_to_x = Signal()
        self.z_to_csr = Signal()
        self.trap = Signal()
        self.time_irq = Signal()
        self.ext_irq = Signal()
        self.enter_trap = Signal()
        self.exit_trap = Signal()
        self.clear_pend_mti = Signal()
        self.clear_pend_mei = Signal()
        self.mei_pend = Signal()
        self.mti_pend = Signal()

        # Internals
        self._mstatus = Signal(32)
        self._mie = Signal(32)
        self._mip = Signal(32)

        self._pend_mti = Signal()
        self._pend_mei = Signal()
        self._clear_pend_mti = Signal()
        self._clear_pend_mei = Signal()

    def elaborate(self, _: Platform) -> Module:
        """Implements the logic of the interrupt card."""
        m = Module()

        m.d.comb += [
            self._pend_mti.eq(0),
            self._pend_mei.eq(0),
            self._clear_pend_mti.eq(0),
            self._clear_pend_mei.eq(0),
        ]

        with m.If(~self.trap):
            with m.If(self._mstatus[MStatus.MIE]):

                with m.If(self._mie[MInterrupt.MTI]):
                    with m.If(~self._mip[MInterrupt.MTI]):
                        m.d.comb += self._pend_mti.eq(self.time_irq)
                with m.Else():
                    m.d.comb += self._clear_pend_mti.eq(1)

                with m.If(self._mie[MInterrupt.MEI]):
                    with m.If(~self._mip[MInterrupt.MEI]):
                        m.d.comb += self._pend_mei.eq(self.ext_irq)
                with m.Else():
                    m.d.comb += self._clear_pend_mei.eq(1)

            with m.Else():
                m.d.comb += self._clear_pend_mti.eq(1)
                m.d.comb += self._clear_pend_mei.eq(1)

        m.d.comb += self.mei_pend.eq((self.ext_irq & self._mie[MInterrupt.MEI]) |
                                     self._mip[MInterrupt.MEI])
        m.d.comb += self.mti_pend.eq((self.time_irq & self._mie[MInterrupt.MTI]) |
                                     self._mip[MInterrupt.MTI])

        enter_trap_mstatus = self._mstatus
        enter_trap_mstatus &= ~(1 << MStatus.MIE)  # clear MIE
        enter_trap_mstatus &= ~(1 << MStatus.MPIE)  # clear MPIE
        enter_trap_mstatus |= (
            self._mstatus[MStatus.MIE] << MStatus.MPIE)  # set MPIE

        exit_trap_mstatus = self._mstatus
        exit_trap_mstatus |= (1 << MStatus.MPIE)  # set MPIE
        exit_trap_mstatus &= ~(1 << MStatus.MIE)  # clear MIE
        exit_trap_mstatus |= (
            self._mstatus[MStatus.MPIE] << MStatus.MIE)  # set MIE

        self.multiplex_to_reg(m, clk="ph2w", reg=self._mstatus,
                              sels=[
                                  self.z_to_csr & (
                                      self.csr_num == CSRAddr.MSTATUS),
                                  self.enter_trap,
                                  self.exit_trap,
                              ],
                              sigs=[
                                  self.data_z_in,
                                  enter_trap_mstatus,
                                  exit_trap_mstatus,
                              ],
                              ext_init=self._ext_init)

        self.multiplex_to_reg(m, clk="ph2w", reg=self._mie,
                              sels=[
                                  self.z_to_csr & (
                                      self.csr_num == CSRAddr.MIE),
                              ],
                              sigs=[self.data_z_in],
                              ext_init=self._ext_init)

        mtip = Signal()
        meip = Signal()

        m.d.comb += mtip.eq(self._mip[MInterrupt.MTI])
        m.d.comb += meip.eq(self._mip[MInterrupt.MEI])
        with m.If(self._pend_mti):
            m.d.comb += mtip.eq(1)
        with m.Elif(self._clear_pend_mti | self.clear_pend_mti):
            m.d.comb += mtip.eq(0)
        with m.If(self._pend_mei):
            m.d.comb += meip.eq(1)
        with m.Elif(self._clear_pend_mei | self.clear_pend_mei):
            m.d.comb += meip.eq(0)

        # Pending machine interrupts are not writable.
        mip_load = Signal(32)
        m.d.comb += mip_load.eq(self.data_z_in)
        m.d.comb += mip_load[MInterrupt.MTI].eq(mtip)
        m.d.comb += mip_load[MInterrupt.MEI].eq(meip)
        m.d.comb += mip_load[MInterrupt.MSI].eq(self._mip[MInterrupt.MSI])

        load_mip = self.z_to_csr & (self.csr_num == CSRAddr.MIP)

        mip_pend = Signal(32)
        m.d.comb += mip_pend.eq(self._mip)
        m.d.comb += mip_pend[MInterrupt.MTI].eq(mtip)
        m.d.comb += mip_pend[MInterrupt.MEI].eq(meip)

        # Either we load MIP, or set/clear/retain bits in MIP. We never have
        # to leave MIP alone.

        self.multiplex_to_reg(m, clk="ph2w", reg=self._mip,
                              sels=[load_mip, ~load_mip],
                              sigs=[mip_load, mip_pend])

        self.multiplex_to_bus(m, bus=self.data_x_out,
                              sels=[
                                  self.csr_to_x & (
                                      self.csr_num == CSRAddr.MSTATUS),
                                  self.csr_to_x & (
                                      self.csr_num == CSRAddr.MIE),
                                  self.csr_to_x & (
                                      self.csr_num == CSRAddr.MIP),
                              ],
                              sigs=[self._mstatus,
                                    self._mie,
                                    self._mip,
                                    ])

        return m

    def multiplex_to_reg(self, m: Module, clk: str, reg: Signal, sels: List[Signal], sigs: List[Signal],
                         ext_init: bool = False):
        """Sets up a multiplexer with a register.

        clk is the clock domain on which the register is clocked.

        reg is the register signal.

        sels is an array of Signals which select that input for the multiplexer (active high). If
        no select is active, then the register retains its value.

        sigs is an array of Signals which are the inputs to the multiplexer.
        """
        assert len(sels) == len(sigs)

        muxreg = IC_reg32_with_mux(
            clk=clk, N=len(sels), ext_init=ext_init, faster=True)
        m.submodules += muxreg
        m.d.comb += reg.eq(muxreg.q)
        for i in range(len(sels)):
            m.d.comb += muxreg.n_sel[i].eq(~sels[i])
            m.d.comb += muxreg.d[i].eq(sigs[i])

    def multiplex_to_bus(self, m: Module, bus: Signal, sels: List[Signal], sigs: List[Signal]):
        """Sets up a multiplexer to a bus."""
        assert len(sels) == len(sigs)

        mux = IC_mux32(N=len(sels), faster=True)
        m.submodules += mux
        m.d.comb += bus.eq(mux.y)
        for i in range(len(sels)):
            m.d.comb += mux.n_sel[i].eq(~sels[i])
            m.d.comb += mux.a[i].eq(sigs[i])
