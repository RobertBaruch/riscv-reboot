# Disable pylint's "your name is too short" warning.
# pylint: disable=C0103
from typing import List, Tuple

from nmigen import Array, Signal, Module, Elaboratable, ClockDomain
from nmigen.build import Platform
from nmigen.sim import Simulator, Delay
from nmigen.asserts import Assert, Assume, Cover, Past, Stable, Rose, Fell, AnyConst, Initial

from util import main


class AsyncMemory(Elaboratable):
    """Logic for a typical asynchronous memory.

    Attributes:
        addr: The address to read or write.
        data_in: The input data (when writing).
        data_out: The output data (when reading).
        n_oe: Output enable, active low. When n_oe is 1, the output is 0.
        n_wr: Write, active low.
    """

    addr: Signal
    data_in: Signal
    data_out: Signal
    n_oe: Signal
    n_wr: Signal

    def __init__(self, width: int, addr_lines: int, ext_init: bool = False):
        """Constructs an asynchronous memory.

        Args:
            width: The number of bits in each memory cell ("word").
            addr_lines: The number of address lines. We only support memories
                up to 16 address lines (so 64k words).
        """
        assert width > 0
        assert addr_lines > 0
        assert addr_lines <= 16

        attrs = [] if not ext_init else [("uninitialized", "")]

        self.addr = Signal(addr_lines)
        self.n_oe = Signal()
        self.n_wr = Signal()

        # There's no such thing as a tristate or bidirectional port
        # in nMigen, so we do this instead.
        self.data_in = Signal(width)
        self.data_out = Signal(width)

        # The actual memory
        self._mem = Array(
            [Signal(width, reset_less=True, attrs=attrs) for _ in range(2**addr_lines)])

    def elaborate(self, _: Platform) -> Module:
        """Implements the logic of an asynchronous memory.

        Essentially implements the wr-controlled write cycle, where
        the oe signal doesn't matter. We can't really implement the
        delays involved, so be safe with your signals :)
        """
        m = Module()

        # Local clock domain so we can clock data into the memory
        # on the positive edge of n_wr.
        wr_clk = ClockDomain("wr_clk", local=True)
        m.domains.wr_clk = wr_clk
        wr_clk.clk = self.n_wr

        m.d.comb += self.data_out.eq(0)
        with m.If(~self.n_oe & self.n_wr):
            m.d.comb += self.data_out.eq(self._mem[self.addr])
        m.d.wr_clk += self._mem[self.addr].eq(self.data_in)

        return m

    @classmethod
    def sim(cls):
        """A quick simulation of the async memory."""
        m = Module()
        m.submodules.mem = mem = AsyncMemory(width=32, addr_lines=5)

        sim = Simulator(m)

        def process():
            yield mem.n_oe.eq(0)
            yield mem.n_wr.eq(1)
            yield mem.data_in.eq(0xFFFFFFFF)
            yield Delay(1e-6)
            yield mem.addr.eq(1)
            yield mem.n_oe.eq(0)
            yield Delay(1e-6)
            yield mem.n_oe.eq(1)
            yield Delay(1e-6)
            yield mem.data_in.eq(0xAAAA1111)
            yield Delay(1e-6)
            yield mem.n_wr.eq(0)
            yield Delay(0.2e-6)
            yield mem.n_wr.eq(1)
            yield Delay(0.2e-6)
            yield mem.data_in.eq(0xFFFFFFFF)
            yield mem.n_oe.eq(0)
            yield Delay(1e-6)
            yield mem.addr.eq(0)
            yield Delay(1e-6)

        sim.add_process(process)
        with sim.write_vcd("async_memory.vcd"):
            sim.run_until(10e-6)

    @classmethod
    def formal(cls) -> Tuple[Module, List[Signal]]:
        """Formal verification for the async memory.

        Note that you MUST have multiclock on in the sby file, because there is
        more than one clock in the system -- the default formal clock and the
        local clock inside the memory.
        """
        m = Module()
        m.submodules.mem = mem = AsyncMemory(width=32, addr_lines=5)

        # Assume "good practices":
        # * n_oe and n_wr are never simultaneously 0, and any changes
        #   are separated by at least a cycle to allow buffers to turn off.
        # * memory address remains stable throughout a write cycle, and
        #   is also stable just before a write cycle.

        m.d.comb += Assume(mem.n_oe | mem.n_wr)

        # Paren placement is very important! While Python logical operators
        # and, or have lower precedence than ==, the bitwise operators
        # &, |, ^ have *higher* precedence. It can be confusing when it looks
        # like you're writing a boolean expression, but you're actually writing
        # a bitwise expression.
        with m.If(Fell(mem.n_oe)):
            m.d.comb += Assume((mem.n_wr == 1) & (Past(mem.n_wr) == 1))
        with m.If(Fell(mem.n_wr)):
            m.d.comb += Assume((mem.n_oe == 1) & (Past(mem.n_oe) == 1))
        with m.If(Rose(mem.n_wr) | (mem.n_wr == 0)):
            m.d.comb += Assume(Stable(mem.addr))

        m.d.comb += Cover((mem.data_out == 0xAAAAAAAA)
                          & (Past(mem.data_out) == 0xBBBBBBBB))

        # Make sure that when the output is disabled, the output is zero, and
        # when enabled, it's whatever we're pointing at in memory.
        with m.If(mem.n_oe == 1):
            m.d.comb += Assert(mem.data_out == 0)
        with m.Else():
            m.d.comb += Assert(mem.data_out == mem._mem[mem.addr])

        # If we just wrote data, make sure that cell that we pointed at
        # for writing now contains the data we wrote.
        with m.If(Rose(mem.n_wr)):
            m.d.comb += Assert(mem._mem[Past(mem.addr)] == Past(mem.data_in))

        # Pick an address, any address.
        check_addr = AnyConst(5)

        # We assert that unless that address is written, its data will not
        # change. To know when we've written the data, we have to create
        # a clock domain to let us save the data when written.

        saved_data_clk = ClockDomain("saved_data_clk")
        m.domains.saved_data_clk = saved_data_clk
        saved_data_clk.clk = mem.n_wr

        saved_data = Signal(32)
        with m.If(mem.addr == check_addr):
            m.d.saved_data_clk += saved_data.eq(mem.data_in)

        with m.If(Initial()):
            m.d.comb += Assume(saved_data == mem._mem[check_addr])
            m.d.comb += Assume(mem.n_wr == 1)
        with m.Else():
            m.d.comb += Assert(saved_data == mem._mem[check_addr])

        return m, [mem.addr, mem.data_in, mem.n_wr, mem.n_oe]


if __name__ == "__main__":
    main(AsyncMemory)
