# Disable pylint's "your name is too short" warning.
# pylint: disable=C0103
from typing import List, Tuple

from nmigen import Signal, Module, Elaboratable, ClockDomain, ClockSignal
from nmigen import Mux
from nmigen.build import Platform
from nmigen.asserts import Assert, Assume, Cover, Past, Stable, Rose, AnyConst, Initial

from async_memory import AsyncMemory
from transparent_latch import TransparentLatch
from util import main


class RegCard(Elaboratable):
    """Logic for the register card.

    Attributes:
        data_x: The X bus, always written to.
        data_y: The Y bus, always written to.
        data_z: The Z bus, always read from.
        reg_to_x: The control line indicating we want to output a register to X.
        reg_to_y: The control line indicating we want to output a register to Y.
        reg_x: The register to output to X.
        reg_y: The register to output to Y.
        reg_z: The register to write from Z.
        reg_page: The 32-register page to access.

    The card optionally outputs the data in reg_x to the data_x bus,
    and the data in reg_y to the data_y bus. It also writes the data
    on the data_z bus to reg_z. Of course, reg_z should be zero if you
    don't really want to write anything.

    This module uses two system-wide clocks: ph1 and ph2. The phases look
    like this:

           ________          ________          
    ph1  _|   RD   |___WR___|   RD   |___WR___|
         ___     ____     ____     ____     ___
    ph2     |___|    |___|    |___|    |___|

    ph1 controls whether we're reading or writing the memories, while
    ph2 controls the read/write pulse (and memory input buffers) to the memories
    if we are writing and latches if we are reading.

    The phases also don't both change at the same time.

    Strictly speaking that's 6 clocks per machine cycle.
    """

    data_x: Signal
    data_y: Signal
    data_z: Signal
    reg_to_x: Signal
    reg_to_y: Signal
    reg_x: Signal
    reg_y: Signal
    reg_z: Signal
    reg_page: Signal

    def __init__(self):
        """Constructs a register card."""
        # Buses
        self.data_x = Signal(32)
        self.data_y = Signal(32)
        self.data_z = Signal(32)

        # Controls
        self.reg_to_x = Signal()
        self.reg_to_y = Signal()
        self.reg_x = Signal(5)
        self.reg_y = Signal(5)
        self.reg_z = Signal(5)
        self.reg_page = Signal()

        # Submodules
        self._x_bank = AsyncMemory(width=32, addr_lines=6)
        self._y_bank = AsyncMemory(width=32, addr_lines=6)
        self._x_latch = TransparentLatch(size=32)
        self._y_latch = TransparentLatch(size=32)
        self._x_bank_wr_latch = TransparentLatch(size=32)
        self._y_bank_wr_latch = TransparentLatch(size=32)

    def elaborate(self, _: Platform) -> Module:
        """Implements the logic of the register card."""
        m = Module()

        ph1 = ClockSignal("ph1")
        ph2 = ClockSignal("ph2")
        x_bank = self._x_bank
        y_bank = self._y_bank
        x_latch = self._x_latch
        y_latch = self._y_latch
        x_bank_wr_latch = self._x_bank_wr_latch
        y_bank_wr_latch = self._y_bank_wr_latch

        m.submodules.x_bank = x_bank
        m.submodules.y_bank = y_bank
        m.submodules.x_latch = x_latch
        m.submodules.y_latch = y_latch
        m.submodules.x_bank_wr_latch = x_bank_wr_latch
        m.submodules.y_bank_wr_latch = y_bank_wr_latch

        # Some aliases
        read_phase = ph1
        write_phase = ~ph1
        read_pulse = ph1 & ~ph2  # high pulse during read phase
        write_pulse = ph1 | ph2  # low pulse during write phase

        # Checks for register 0
        x_reg_0 = Signal()
        y_reg_0 = Signal()
        m.d.comb += [
            x_reg_0.eq(self.reg_x == 0),
            y_reg_0.eq(self.reg_y == 0),
        ]

        # Write signals for both banks are tied together.
        n_wr = Signal()
        m.d.comb += [
            x_bank.n_wr.eq(n_wr),
            y_bank.n_wr.eq(n_wr),
        ]

        # Pass through the write pulse to the memory banks.
        m.d.comb += n_wr.eq(write_pulse)

        # We also read from the memories on ph1, but only if we're
        # not reading from register 0. Strictly speaking this doesn't
        # matter, since if we are reading from register 0, the memory
        # output is switched out by the output multiplexer.
        x_bank_oe = ~x_reg_0 & read_phase
        y_bank_oe = ~y_reg_0 & read_phase
        m.d.comb += [
            x_bank.n_oe.eq(~x_bank_oe),
            y_bank.n_oe.eq(~y_bank_oe),
        ]

        # The address for each bank is switched between x/y during the
        # read phase, and z during the write phase.
        x_addr = Signal(6)
        y_addr = Signal(6)
        m.d.comb += [
            x_addr[:5].eq(Mux(read_phase, self.reg_x, self.reg_z)),
            y_addr[:5].eq(Mux(read_phase, self.reg_y, self.reg_z)),
            x_addr[5].eq(self.reg_page),
            y_addr[5].eq(self.reg_page),
            x_bank.addr.eq(x_addr),
            y_bank.addr.eq(y_addr),
        ]

        # The memory data is the data coming out of the memory, or the
        # data coming out of the write latches. Since high-Z is simulated
        # by outputting zeros, this is fine.
        x_mem_data = Signal(32)
        y_mem_data = Signal(32)
        m.d.comb += [
            x_mem_data.eq(x_bank.data_out | x_bank_wr_latch.data_out),
            y_mem_data.eq(y_bank.data_out | y_bank_wr_latch.data_out),
        ]

        # The memory data inputs are the memory data.
        m.d.comb += [
            x_bank.data_in.eq(x_mem_data),
            y_bank.data_in.eq(y_mem_data),
        ]

        # The write latches aren't actually used as latches, just
        # tri-state buffers. They are always transparent. But, n_oe
        # is used to control the output.
        wr_latch_oe = ~n_wr
        m.d.comb += [
            x_bank_wr_latch.data_in.eq(self.data_z),
            y_bank_wr_latch.data_in.eq(self.data_z),
            x_bank_wr_latch.le.eq(1),
            y_bank_wr_latch.le.eq(1),
            x_bank_wr_latch.n_oe.eq(~wr_latch_oe),
            y_bank_wr_latch.n_oe.eq(~wr_latch_oe),
        ]

        # The bus output latches take data from the memory data and
        # optionally pass it to the x/y buses, transparently during
        # the read cycle, but the data is latched so it remains during
        # the write cycle. However, the output of the latches to the
        # buses is controlled by the control signals.
        #
        # Also, multiplex between the memory data and zero depending
        # on whether we're reading register 0.
        x_zero_oe = x_reg_0 & read_phase
        y_zero_oe = y_reg_0 & read_phase

        m.d.comb += [
            x_latch.data_in.eq(Mux(x_zero_oe, 0, x_mem_data)),
            y_latch.data_in.eq(Mux(y_zero_oe, 0, y_mem_data)),
            x_latch.le.eq(read_pulse),
            y_latch.le.eq(read_pulse),
            x_latch.n_oe.eq(~self.reg_to_x),
            y_latch.n_oe.eq(~self.reg_to_y),
        ]

        # The x and y buses get the output latch outputs.
        m.d.comb += [
            self.data_x.eq(x_latch.data_out),
            self.data_y.eq(y_latch.data_out),
        ]

        return m

    @classmethod
    def formal(cls) -> Tuple[Module, List[Signal]]:
        """Formal verification for the register card."""
        m = Module()

        ph1 = ClockDomain("ph1")
        ph2 = ClockDomain("ph2")
        regs = RegCard()

        m.domains += [ph1, ph2]
        m.submodules += regs

        # Generate the ph1 and ph2 clocks.
        cycle_count = Signal(8, reset=0, reset_less=True)
        phase_count = Signal(3, reset=0, reset_less=True)

        m.d.sync += phase_count.eq(phase_count + 1)
        with m.Switch(phase_count):
            with m.Case(0, 1, 2):
                m.d.comb += ph1.clk.eq(1)
            with m.Default():
                m.d.comb += ph1.clk.eq(0)
        with m.Switch(phase_count):
            with m.Case(1, 4):
                m.d.comb += ph2.clk.eq(0)
            with m.Default():
                m.d.comb += ph2.clk.eq(1)
        with m.If(phase_count == 5):
            m.d.sync += phase_count.eq(0)
            m.d.sync += cycle_count.eq(cycle_count + 1)

        # This is how we expect to use the card.
        with m.If(phase_count > 0):
            m.d.comb += [
                Assume(Stable(regs.reg_x)),
                Assume(Stable(regs.reg_y)),
                Assume(Stable(regs.reg_z)),
                Assume(Stable(regs.reg_page)),
                Assume(Stable(regs.reg_to_x)),
                Assume(Stable(regs.reg_to_y)),
                Assume(Stable(regs.data_z)),
            ]

        # Figure out how to get to the point where X and Y are nonzero and different.
        m.d.comb += Cover((regs.data_x != 0) & (regs.data_y != 0)
                          & (regs.data_x != regs.data_y))

        # X and Y buses should not change during a cycle, except for the first phase
        with m.Switch(phase_count):
            with m.Case(2, 3, 4, 5):
                with m.If(regs.data_x != 0):
                    m.d.comb += Assert(Stable(regs.data_x))
                with m.If(regs.data_y != 0):
                    m.d.comb += Assert(Stable(regs.data_y))

        # X and Y buses should be zero if there is no data transfer.
        with m.If(regs.reg_to_x == 0):
            m.d.comb += Assert(regs.data_x == 0)
        with m.If(regs.reg_to_y == 0):
            m.d.comb += Assert(regs.data_y == 0)

        with m.If(phase_count > 0):
            # X and Y buses should be zero if we read from register 0.
            with m.If(regs.reg_to_x & (regs.reg_x == 0)):
                m.d.comb += Assert(regs.data_x == 0)
            with m.If(regs.reg_to_y & (regs.reg_y == 0)):
                m.d.comb += Assert(regs.data_y == 0)

        write_pulse = Signal()
        m.d.comb += write_pulse.eq(phase_count != 4)

        # On write, the data should have been written to both banks.
        past_mem_addr = Signal(6)
        m.d.comb += past_mem_addr[:5].eq(Past(regs.reg_z))
        m.d.comb += past_mem_addr[5].eq(Past(regs.reg_page))
        past_z = Past(regs.data_z)
        with m.If(Rose(write_pulse)):
            m.d.comb += Assert(regs._x_bank._mem[past_mem_addr] == past_z)
            m.d.comb += Assert(regs._y_bank._mem[past_mem_addr] == past_z)

        # Pick an register, any register, except 0. We assert that unless
        # it is written, its data will not change.

        check_addr = AnyConst(5)
        check_page = AnyConst(1)
        saved_data = Signal(32)
        stored_x_data = Signal(32)
        stored_y_data = Signal(32)

        write_pulse_domain = ClockDomain("write_pulse_domain", local=True)
        m.domains.write_pulse_domain = write_pulse_domain
        write_pulse_domain.clk = write_pulse

        mem_addr = Signal(6)

        m.d.comb += Assume(check_addr != 0)
        m.d.comb += [
            mem_addr[:5].eq(check_addr),
            mem_addr[5].eq(check_page),
            stored_x_data.eq(regs._x_bank._mem[mem_addr]),
            stored_y_data.eq(regs._y_bank._mem[mem_addr]),
        ]

        with m.If((regs.reg_z == check_addr) & (regs.reg_page == check_page)):
            m.d.write_pulse_domain += saved_data.eq(regs.data_z)

        with m.If(Initial()):
            m.d.comb += Assume(saved_data == stored_x_data)
            m.d.comb += Assume(stored_x_data == stored_y_data)
        with m.Else():
            m.d.comb += Assert(saved_data == stored_x_data)
            m.d.comb += Assert(saved_data == stored_y_data)

        return m, [regs.data_z, regs.reg_to_x, regs.reg_to_y,
                   regs.reg_x, regs.reg_y, regs.reg_z, regs.reg_page, ph1.clk, ph2.clk, saved_data,
                   stored_x_data, stored_y_data]


if __name__ == "__main__":
    main(RegCard)
