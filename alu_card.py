# Disable pylint's "your name is too short" warning.
# pylint: disable=C0103
from typing import List, Tuple

from nmigen import Signal, Module, Elaboratable
from nmigen.build import Platform
from nmigen.asserts import Assert

from consts import AluOp
from transparent_latch import TransparentLatch
from util import main


class AluCard(Elaboratable):
    """Logic for the ALU card.

    Attributes:
        data_x: The X bus, always read from.
        data_y: The Y bus, always read from.
        data_z: The Z bus, written to when ALU is active.
        alu_op: The ALU op to perform.
        alu_eq: Set if the result of a SUB is zero.
        alu_lt: Set if the result of a signed SUB is less than zero.
        alu_ltu: Set if the result of an unsigned SUB is less than zero.
    """

    data_x: Signal
    data_y: Signal
    data_z: Signal
    alu_op: Signal
    alu_eq: Signal
    alu_lt: Signal
    alu_ltu: Signal

    def __init__(self):
        # Buses
        self.data_x = Signal(32)
        self.data_y = Signal(32)
        self.data_z = Signal(32)

        # Controls
        self.alu_op = Signal(AluOp)

        # Outputs for branch compares using SUB:
        self.alu_eq = Signal()
        self.alu_lt = Signal()
        self.alu_ltu = Signal()

    def elaborate(self, _: Platform) -> Module:
        """Implements the logic of the ALU card."""
        m = Module()

        output_buffer = TransparentLatch(size=32)

        m.submodules += output_buffer
        m.d.comb += output_buffer.le.eq(1)
        m.d.comb += output_buffer.data_in.eq(0)
        m.d.comb += self.data_z.eq(output_buffer.data_out)

        m.d.comb += self.alu_eq.eq(0)
        m.d.comb += self.alu_lt.eq(0)
        m.d.comb += self.alu_ltu.eq(0)

        x = self.data_x
        y = self.data_y

        m.d.comb += self.alu_eq.eq(output_buffer.data_in == 0)
        m.d.comb += self.alu_ltu.eq(x < y)
        m.d.comb += self.alu_lt.eq(x.as_signed() < y.as_signed())

        with m.Switch(self.alu_op):
            with m.Case(AluOp.ADD):
                m.d.comb += output_buffer.data_in.eq(x + y)
                m.d.comb += output_buffer.n_oe.eq(0)

            with m.Case(AluOp.SUB):
                m.d.comb += output_buffer.data_in.eq(x - y)
                m.d.comb += output_buffer.n_oe.eq(0)

            with m.Case(AluOp.SLTU):
                m.d.comb += output_buffer.data_in.eq(self.alu_ltu)
                m.d.comb += output_buffer.n_oe.eq(0)

            with m.Case(AluOp.SLT):
                m.d.comb += output_buffer.data_in.eq(self.alu_lt)
                m.d.comb += output_buffer.n_oe.eq(0)

            with m.Case(AluOp.AND):
                m.d.comb += output_buffer.data_in.eq(x & y)
                m.d.comb += output_buffer.n_oe.eq(0)

            with m.Case(AluOp.AND_NOT):
                m.d.comb += output_buffer.data_in.eq(x & ~y)
                m.d.comb += output_buffer.n_oe.eq(0)

            with m.Case(AluOp.OR):
                m.d.comb += output_buffer.data_in.eq(x | y)
                m.d.comb += output_buffer.n_oe.eq(0)

            with m.Case(AluOp.XOR):
                m.d.comb += output_buffer.data_in.eq(x ^ y)
                m.d.comb += output_buffer.n_oe.eq(0)

            with m.Case(AluOp.X):
                m.d.comb += output_buffer.data_in.eq(x)
                m.d.comb += output_buffer.n_oe.eq(0)

            with m.Case(AluOp.Y):
                m.d.comb += output_buffer.data_in.eq(y)
                m.d.comb += output_buffer.n_oe.eq(0)

            with m.Default():
                m.d.comb += output_buffer.n_oe.eq(1)

        return m

    @classmethod
    def formal(cls) -> Tuple[Module, List[Signal]]:
        """Formal verification for the ALU."""
        m = Module()
        m.submodules.alu = alu = AluCard()

        with m.Switch(alu.alu_op):
            with m.Case(AluOp.ADD):
                m.d.comb += Assert(alu.data_z ==
                                   (alu.data_x + alu.data_y)[:32])

            with m.Case(AluOp.SUB):
                m.d.comb += [
                    Assert(alu.data_z ==
                           (alu.data_x - alu.data_y)[:32]),
                    Assert(alu.alu_eq == (alu.data_x == alu.data_y)),
                    Assert(alu.alu_ltu == (alu.data_x < alu.data_y)),
                    Assert(alu.alu_lt == (alu.data_x.as_signed()
                                          < alu.data_y.as_signed())),
                ]

            with m.Case(AluOp.AND):
                m.d.comb += Assert(alu.data_z == (alu.data_x & alu.data_y))

            with m.Case(AluOp.AND_NOT):
                m.d.comb += Assert(alu.data_z ==
                                   (alu.data_x & ~alu.data_y))

            with m.Case(AluOp.OR):
                m.d.comb += Assert(alu.data_z == (alu.data_x | alu.data_y))

            with m.Case(AluOp.XOR):
                m.d.comb += Assert(alu.data_z == (alu.data_x ^ alu.data_y))

            with m.Case(AluOp.SLTU):
                with m.If(alu.data_x < alu.data_y):
                    m.d.comb += Assert(alu.data_z == 1)
                with m.Else():
                    m.d.comb += Assert(alu.data_z == 0)

            with m.Case(AluOp.SLT):
                with m.If(alu.data_x.as_signed() < alu.data_y.as_signed()):
                    m.d.comb += Assert(alu.data_z == 1)
                with m.Else():
                    m.d.comb += Assert(alu.data_z == 0)

            with m.Case(AluOp.X):
                m.d.comb += Assert(alu.data_z == alu.data_x)

            with m.Case(AluOp.Y):
                m.d.comb += Assert(alu.data_z == alu.data_y)

            with m.Default():
                m.d.comb += Assert(alu.data_z == 0)

        return m, [alu.alu_op, alu.data_x, alu.data_y, alu.data_z]


if __name__ == "__main__":
    main(AluCard)
