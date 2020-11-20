# Disable pylint's "your name is too short" warning.
# pylint: disable=C0103

from enum import Enum, unique

from nmigen import Signal, Module, Elaboratable, signed, ClockDomain
from nmigen.build import Platform
from nmigen.asserts import Assert, Assume, Cover
from nmigen.cli import main_parser, main_runner
from nmigen.sim import Simulator, Delay, Settle, Tick


@unique
class OpcodeFormat(Enum):
    """Opcode formats."""
    R = 0
    I = 1
    U = 2
    S = 3
    B = 4
    J = 5


@unique
class AluOp(Enum):
    """ALU operations."""
    ADD = 0b0000
    SUB = 0b1000
    SLL = 0b0001
    SLT = 0b0010
    SLTU = 0b1011
    XOR = 0b0100
    SRL = 0b0101
    SRA = 0b1101
    OR = 0b0110
    AND = 0b0111
