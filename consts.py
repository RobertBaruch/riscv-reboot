# Disable pylint's "your name is too short" warning.
# pylint: disable=C0103

from enum import Enum, unique

from nmigen import Signal, Module, Elaboratable, signed, ClockDomain
from nmigen.build import Platform
from nmigen.asserts import Assert, Assume, Cover
from nmigen.cli import main_parser, main_runner
from nmigen.sim import Simulator, Delay, Settle, Tick


@unique
class Opcode(Enum):
    """Opcodes."""
    LOAD = 0b000_0011
    OP_IMM = 0b001_0011
    STORE = 0b010_0011
    OP = 0b011_0011
    BRANCH = 0b110_0011
    SYSTEM = 0b111_0011
    LUI = 0b011_0111
    JALR = 0b110_0111
    MISC_MEM = 0b000_1111
    AUIPC = 0b001_0111
    JAL = 0b110_1111


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
class BranchCond(Enum):
    """Branch conditions."""
    EQ = 0b000
    NE = 0b001
    LT = 0b100
    GE = 0b101
    LTU = 0b110
    GEU = 0b111


@unique
class MemAccessWidth(Enum):
    """Memory access widths."""
    B = 0b000
    H = 0b001
    W = 0b010
    BU = 0b100
    HU = 0b101


@unique
class AluOp(Enum):
    "ALU card operations."
    NONE = 0b0000
    ADD = 0b0001
    SUB = 0b0010
    SLL = 0b0011
    SLT = 0b0100
    SLTU = 0b0101
    XOR = 0b0110
    SRL = 0b0111
    SRA = 0b1000
    OR = 0b1001
    AND = 0b1010


@unique
class AluFunc(Enum):
    """ALU functions."""
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
