# Disable pylint's "your name is too short" warning.
# pylint: disable=C0103

from enum import IntEnum, unique


@unique
class Opcode(IntEnum):
    """Opcodes."""
    LOAD = 0b000_0011      # 0x03
    OP_IMM = 0b001_0011    # 0x13
    STORE = 0b010_0011     # 0x23
    OP = 0b011_0011        # 0x33
    BRANCH = 0b110_0011    # 0x63
    SYSTEM = 0b111_0011    # 0x73
    LUI = 0b011_0111       # 0x37
    JALR = 0b110_0111      # 0x67
    MISC_MEM = 0b000_1111  # 0x0F
    AUIPC = 0b001_0111     # 0x17
    JAL = 0b110_1111       # 0x6F


@unique
class OpcodeFormat(IntEnum):
    """Opcode formats."""
    R = 0
    I = 1
    U = 2
    S = 3
    B = 4
    J = 5
    SYS = 6


@unique
class BranchCond(IntEnum):
    """Branch conditions."""
    EQ = 0b000
    NE = 0b001
    LT = 0b100
    GE = 0b101
    LTU = 0b110
    GEU = 0b111


@unique
class MemAccessWidth(IntEnum):
    """Memory access widths."""
    B = 0b000
    H = 0b001
    W = 0b010
    BU = 0b100
    HU = 0b101


@unique
class AluOp(IntEnum):
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
    X = 0b1011
    Y = 0b1100
    AND_NOT = 0b1101


@unique
class AluFunc(IntEnum):
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


@unique
class SystemFunc(IntEnum):
    """System opcode functions."""
    PRIV = 0b000
    CSRRW = 0b001
    CSRRS = 0b010
    CSRRC = 0b011
    CSRRWI = 0b101
    CSRRSI = 0b110
    CSRRCI = 0b111


@unique
class PrivFunc(IntEnum):
    """Privileged functions, funct12 value."""
    # Functions for which rd and rs1 must be 0:
    ECALL = 0b000000000000
    EBREAK = 0b000000000001
    URET = 0b000000000010
    SRET = 0b000100000010
    MRET = 0b001100000010
    WFI = 0b000100000101


@unique
class TrapCause(IntEnum):
    """Trap causes."""
    INT_USER_SOFTWARE = 0x80000000
    INT_SUPV_SOFTWARE = 0x80000001
    INT_MACH_SOFTWARE = 0x80000003
    INT_USER_TIMER = 0x80000004
    INT_SUPV_TIMER = 0x80000005
    INT_MACH_TIMER = 0x80000007
    INT_USER_EXTERNAL = 0x80000008
    INT_SUPV_EXTERNAL = 0x80000009
    INT_MACH_EXTERNAL = 0x8000000B

    EXC_INSTR_ADDR_MISALIGN = 0x00000000
    EXC_INSTR_ACCESS_FAULT = 0x00000001
    EXC_ILLEGAL_INSTR = 0x00000002
    EXC_BREAKPOINT = 0x00000003
    EXC_LOAD_ADDR_MISALIGN = 0x00000004
    EXC_LOAD_ACCESS_FAULT = 0x00000005
    EXC_STORE_AMO_ADDR_MISALIGN = 0x00000006
    EXC_STORE_AMO_ACCESS_FAULT = 0x00000007
    EXC_ECALL_FROM_USER_MODE = 0x00000008
    EXC_ECALL_FROM_SUPV_MODE = 0x00000009
    EXC_ECALL_FROM_MACH_MODE = 0x0000000B
    EXC_INSTR_PAGE_FAULT = 0x0000000C
    EXC_LOAD_PAGE_FAULT = 0x0000000D
    EXC_STORE_AMO_PAGE_FAULT = 0x0000000F


@unique
class CSRAddr(IntEnum):
    """CSR addresses."""
    MSTATUS = 0x300
    MIE = 0x304
    MTVEC = 0x305
    MEPC = 0x341
    MCAUSE = 0x342
    MTVAL = 0x343
    MIP = 0x344
    LAST = 0xFFF


@unique
class MStatus(IntEnum):
    """Bits for mstatus."""
    MIE = 3   # Machine interrupts global enable                  (00000008)
    MPIE = 7  # Machine interrupts global enable (previous value) (00000080)


@unique
class MInterrupt(IntEnum):
    """Bits for mie and mip."""
    MSI = 3   # Machine software interrupt enabled/pending (00000008)
    MTI = 7   # Machine timer interrupt enabled/pending    (00000080)
    MEI = 11  # Machine external interrupt enabled/pending (00000800)


@unique
class InstrReg(IntEnum):
    """Which register number to put on *_reg."""
    ZERO = 0
    RS1 = 1
    RS2 = 2
    RD = 3
