# Disable pylint's "your name is too short" warning.
# pylint: disable=C0103
from typing import List, Tuple

from nmigen import Signal, Module, Elaboratable, Mux, Repl
from nmigen.build import Platform
from nmigen.asserts import Assert, Cover

from consts import AluOp
from transparent_latch import TransparentLatch
from util import main


class _ConditionalShiftRight(Elaboratable):
    """A right-shifter that shifts by N or 0, logical or arithmetic.

    For a 16-bit version of this, we might use the SN74CBT16233.

    Attributes:
        data_in: The input data.
        data_out: The output data.
        en: Whether shifting is enabled or disabled.
        arithmetic: Whether shifting is arithmetic or not (i.e. logical).
    """

    en: Signal
    arithmetic: Signal
    data_in: Signal
    data_out: Signal

    def __init__(self, width: int, N: int):
        """Constructs a conditional right-shifter.

        Args:
            width: The number of bits in the shifter.
            N: The number of bits to shift by.
        """
        self._N = N

        self.en = Signal()
        self.arithmetic = Signal()
        self.data_in = Signal(width)
        self.data_out = Signal(width)

    def elaborate(self, _: Platform) -> Module:
        """Implements the logic for the conditional right-shifter."""
        m = Module()
        N = self._N

        with m.If(self.en):
            # The high N bits get either 0 or the most significant
            # bit in data_in, depending on arithmetic.
            msb = self.data_in[-1]
            w = len(self.data_out)

            m.d.comb += self.data_out.bit_select(w-N, N).eq(
                Mux(self.arithmetic, Repl(msb, N), 0))

            # The rest are moved over.
            m.d.comb += self.data_out.bit_select(0, w-N).eq(
                self.data_in.bit_select(N, w-N))

        with m.Else():
            m.d.comb += self.data_out.eq(self.data_in)

        return m


class _ConditionalReverser(Elaboratable):
    """A conditional bit reverser.

    Attributes:
        data_in: The input data.
        data_out: The output data.
        en: Whether reversing is enabled or disabled.
    """

    def __init__(self, width: int):
        """Constructs a conditional reverser.

        Args:
            width: The number of bits in the reverser.
        """
        self.en = Signal()
        self.data_in = Signal(width)
        self.data_out = Signal(width)

    def elaborate(self, _: Platform) -> Module:
        """Implements the logic for the reverser."""
        m = Module()

        m.d.comb += self.data_out.eq(Mux(self.en,
                                         self.data_in[::-1],
                                         self.data_in))

        return m


class ShiftCard(Elaboratable):
    """Logic for the shifter card.

    This implements a logarithmic shifter. If shifting left, the input
    and output are inverted so that only shift right needs to be implemented.
    """

    data_x: Signal
    data_y: Signal
    data_z: Signal
    alu_op: Signal

    def __init__(self):
        # Buses
        self.data_x = Signal(32)
        self.data_y = Signal(32)
        self.data_z = Signal(32)

        # Controls
        self.alu_op = Signal(AluOp)

    def elaborate(self, _: Platform) -> Module:
        """Implements the logic of the shifter card."""
        m = Module()

        input_reverse = _ConditionalReverser(width=32)
        shift1 = _ConditionalShiftRight(width=32, N=1)
        shift2 = _ConditionalShiftRight(width=32, N=2)
        shift4 = _ConditionalShiftRight(width=32, N=4)
        shift8 = _ConditionalShiftRight(width=32, N=8)
        shift16 = _ConditionalShiftRight(width=32, N=16)
        output_reverse = _ConditionalReverser(width=32)
        output_buffer = TransparentLatch(size=32)

        m.submodules += [input_reverse, shift1, shift2,
                         shift4, shift8, shift16, output_reverse, output_buffer]

        # Hook up inputs and outputs

        m.d.comb += [
            input_reverse.data_in.eq(self.data_x),
            shift1.data_in.eq(input_reverse.data_out),
            shift2.data_in.eq(shift1.data_out),
            shift4.data_in.eq(shift2.data_out),
            shift8.data_in.eq(shift4.data_out),
            shift16.data_in.eq(shift8.data_out),
            output_reverse.data_in.eq(shift16.data_out),
            output_buffer.data_in.eq(output_reverse.data_out),
            self.data_z.eq(output_buffer.data_out),
        ]

        # Some flags
        shift_arith = Signal()
        shift_left = Signal()

        m.d.comb += shift_arith.eq(self.alu_op == AluOp.SRA)
        m.d.comb += shift_left.eq(self.alu_op == AluOp.SLL)

        m.d.comb += [
            input_reverse.en.eq(shift_left),
            output_reverse.en.eq(shift_left),
            shift1.arithmetic.eq(shift_arith),
            shift2.arithmetic.eq(shift_arith),
            shift4.arithmetic.eq(shift_arith),
            shift8.arithmetic.eq(shift_arith),
            shift16.arithmetic.eq(shift_arith),
        ]

        # Shift amount
        shamt = self.data_y[:5]

        m.d.comb += [
            shift1.en.eq(shamt[0]),
            shift2.en.eq(shamt[1]),
            shift4.en.eq(shamt[2]),
            shift8.en.eq(shamt[3]),
            shift16.en.eq(shamt[4]),
        ]

        m.d.comb += output_buffer.le.eq(1)
        m.d.comb += output_buffer.n_oe.eq(1)

        with m.Switch(self.alu_op):
            with m.Case(AluOp.SLL, AluOp.SRL, AluOp.SRA):
                m.d.comb += output_buffer.n_oe.eq(0)
            with m.Default():
                m.d.comb += output_buffer.n_oe.eq(1)

        return m

    @classmethod
    def formal(cls) -> Tuple[Module, List[Signal]]:
        """Formal verification for the shifter."""
        m = Module()
        m.submodules.shifter = shifter = ShiftCard()

        shamt = Signal(5)
        m.d.comb += shamt.eq(shifter.data_y[:5])

        with m.If(shamt > 0):
            m.d.comb += Cover(shifter.data_z == 0xFFFFAAA0)

        with m.Switch(shifter.alu_op):
            with m.Case(AluOp.SLL):
                m.d.comb += Assert(shifter.data_z ==
                                   (shifter.data_x << shamt)[:32])

            with m.Case(AluOp.SRL):
                m.d.comb += Assert(shifter.data_z ==
                                   (shifter.data_x >> shamt))

            with m.Case(AluOp.SRA):
                m.d.comb += Assert(shifter.data_z ==
                                   (shifter.data_x.as_signed() >> shamt))

            with m.Default():
                m.d.comb += Assert(shifter.data_z == 0)

        return m, [shifter.alu_op, shifter.data_x, shifter.data_y, shifter.data_z]


if __name__ == "__main__":
    main(ShiftCard)
