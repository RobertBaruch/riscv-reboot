# Disable pylint's "your name is too short" warning.
# pylint: disable=C0103
from typing import List, Tuple

from nmigen import Signal, Module, Elaboratable, signed, ClockSignal, ClockDomain
from nmigen.build import Platform
from nmigen.asserts import Assert, Assume, Cover, Stable, Past

from consts import AluOp, OpcodeFormat
from transparent_latch import TransparentLatch
from util import main

OPCODE_LOAD = 0b000_0011
OPCODE_OP_IMM = 0b001_0011
OPCODE_STORE = 0b010_0011
OPCODE_OP = 0b011_0011
OPCODE_BRANCH = 0b110_0011
OPCODE_SYSTEM = 0b111_0011
OPCODE_LUI = 0b011_0111
OPCODE_JALR = 0b110_0111
OPCODE_MISC_MEM = 0b000_1111
OPCODE_AUIPC = 0b001_0111
OPCODE_JAL = 0b110_1111

BRANCH_EQ = 0b000
BRANCH_NE = 0b001
BRANCH_LT = 0b100
BRANCH_GE = 0b101
BRANCH_LTU = 0b110
BRANCH_GEU = 0b111

LOAD_STORE_B = 0b000
LOAD_STORE_H = 0b001
LOAD_STORE_W = 0b010
LOAD_STORE_BU = 0b100
LOAD_STORE_HU = 0b101


class SequencerCard(Elaboratable):
    """Logic for the sequencer card.

    Control lines indicate to the other cards what to do now,
    if the control lines control combinatorial logic, or
    to do next, if the control lines control sequential logic.

    This module uses two system-wide clocks: ph1 and ph2. The phases look
    like this:

           ________          ________          
    ph1  _|   RD   |___WR___|   RD   |___WR___|
         ___     ____     ____     ____     ___
    ph2     |___|    |___|    |___|    |___|
    """

    def __init__(self):
        # A clock-based signal, high only at the end of a machine
        # cycle (i.e. phase 5, the very end of the write phase).
        self.mcycle_end = Signal()

        # Control signals. 28 by my count, which includes
        # alu_to_z, x_to_reg, and y_to_reg.
        self.alu_op = Signal(AluOp)  # 4 bits
        self.alu_eq = Signal()
        self.alu_lt = Signal()
        self.alu_ltu = Signal()

        self.z_to_reg = Signal()
        self.x_reg = Signal(5)
        self.y_reg = Signal(5)
        self.z_reg = Signal(5)

        # Raised when the instruction is illegal. We also
        # halt the processor when one is encountered.
        self.illegal = Signal(reset=0)
        # Raised on the last phase of an instruction.
        self.instr_complete = Signal(reset=0)
        # How does this work?
        self.ext_interrupt = Signal()

        # Buses, bidirectional
        self.data_x_in = Signal(32)
        self.data_x_out = Signal(32)
        self.data_y_in = Signal(32)
        self.data_y_out = Signal(32)
        self.data_z_in = Signal(32)
        self.data_z_out = Signal(32)

        # Memory
        self.mem_rd = Signal(reset=1)
        self.mem_wr = Signal(reset=0)
        # Bytes in memory word to write
        self.mem_wr_mask = Signal(4)
        self.memaddr = Signal(32, reset=0)

        # Memory bus, bidirectional
        self.memdata_rd = Signal(32)
        self.memdata_wr = Signal(32)

        # Internals

        # This opens the instr transparent latch to memdata. The enable
        # (i.e. load_instr) on the latch is a register, so setting load_instr
        # now opens the transparent latch next.
        self._load_instr = Signal(reset=1)

        self._instr = Signal(32)
        self._instr_latch = TransparentLatch(32)
        self._pc = Signal(32, reset=0)
        self._pc_plus_4 = Signal(32)
        self._instr_phase = Signal(2, reset=0)
        self._next_instr_phase = Signal(len(self._instr_phase))
        self._stored_alu_eq = Signal()
        self._stored_alu_lt = Signal()
        self._stored_alu_ltu = Signal()
        self._is_last_instr_cycle = Signal()

        self._opcode = Signal(7)
        self._rs1 = Signal(5)
        self._rs2 = Signal(5)
        self._rd = Signal(5)
        self._funct3 = Signal(3)
        self._funct7 = Signal(7)
        self._imm_format = Signal(OpcodeFormat)
        self._imm = Signal(32)

        # -> X
        self.reg_to_x = Signal()
        self._pc_to_x = Signal()
        self._memdata_to_x = Signal()

        # -> Y
        self.reg_to_y = Signal()
        self._imm_to_y = Signal()
        self._shamt_to_y = Signal()

        # -> Z
        self.alu_to_z = Signal()
        self._pc_plus_4_to_z = Signal()

        # -> PC
        self._pc_plus_4_to_pc = Signal()
        self._z_to_pc = Signal()
        self._x_to_pc = Signal()
        self._memaddr_to_pc = Signal()

        # -> memaddr
        self._pc_plus_4_to_memaddr = Signal()
        self._z_to_memaddr = Signal()
        self._x_to_memaddr = Signal()

        # -> memdata
        self._z_to_memdata = Signal()

        # memory load shamt
        self._shamt = Signal(5)

    def elaborate(self, _: Platform) -> Module:
        """Implements the logic of the sequencer card."""
        m = Module()

        # Instruction latch
        m.submodules += self._instr_latch
        latch_instr = Signal()

        # Defaults
        m.d.comb += [
            self._next_instr_phase.eq(0),
            self.reg_to_x.eq(0),
            self._pc_to_x.eq(0),
            self._memdata_to_x.eq(0),
            self.reg_to_y.eq(0),
            self._imm_to_y.eq(0),
            self._shamt_to_y.eq(0),
            self.alu_to_z.eq(0),
            self._pc_plus_4_to_z.eq(0),
            self._pc_plus_4_to_pc.eq(0),
            self._x_to_pc.eq(0),
            self._z_to_pc.eq(0),
            self._memaddr_to_pc.eq(0),
            self._pc_plus_4_to_memaddr.eq(0),
            self._x_to_memaddr.eq(0),
            self._z_to_memaddr.eq(0),
            self._z_to_memdata.eq(0),
            self.mem_rd.eq(0),
            self._load_instr.eq(0),
            self.mem_wr.eq(0),
            self.mem_wr_mask.eq(0),
            self._shamt.eq(0),
            self.data_x_out.eq(0),
            self.data_y_out.eq(0),
            self.data_z_out.eq(0),
            self._is_last_instr_cycle.eq(0),
            self.instr_complete.eq(0),
        ]
        m.d.ph2 += self.illegal.eq(0)
        m.d.comb += self._pc_plus_4.eq(self._pc + 4)

        with m.If(self._instr_phase == 0):
            m.d.comb += self._load_instr.eq(1)
            m.d.comb += self.mem_rd.eq(1)

        with m.If(self._is_last_instr_cycle):
            m.d.comb += self.instr_complete.eq(self.mcycle_end)
            with m.If(~self._x_to_pc & ~self._z_to_pc & ~self._memaddr_to_pc):
                m.d.comb += self._pc_plus_4_to_pc.eq(1)
            with m.If(~self._x_to_memaddr & ~self._z_to_memaddr):
                m.d.comb += self._pc_plus_4_to_memaddr.eq(1)

        read_pulse = ClockSignal("ph1") & ~ClockSignal("ph2")
        m.d.comb += [
            latch_instr.eq(read_pulse & self._load_instr & ~self.illegal),
            self._instr.eq(self._instr_latch.data_out),
            self._instr_latch.data_in.eq(self.memdata_rd),
            self._instr_latch.n_oe.eq(0),
            self._instr_latch.le.eq(latch_instr),
        ]

        with m.If(~self.illegal):
            # Updates to registers
            m.d.ph1 += self._instr_phase.eq(self._next_instr_phase)
            m.d.ph1 += self._stored_alu_eq.eq(self.alu_eq)
            m.d.ph1 += self._stored_alu_lt.eq(self.alu_lt)
            m.d.ph1 += self._stored_alu_ltu.eq(self.alu_ltu)

            with m.If(self._pc_plus_4_to_pc):
                m.d.ph1 += self._pc.eq(self._pc_plus_4)
            with m.Elif(self._x_to_pc):
                m.d.ph1 += self._pc.eq(self.data_x_in)
            with m.Elif(self._z_to_pc):
                m.d.ph1 += self._pc.eq(self.data_z_in)
            with m.Elif(self._memaddr_to_pc):
                # This is the result of a JAL or JALR instruction.
                # See the comment on JAL for why this is okay to do.
                m.d.ph1 += self._pc[1:].eq(self.memaddr[1:])
                m.d.ph1 += self._pc[0].eq(0)

            # Because we don't support the C (compressed instructions)
            # extension, the PC must be 32-bit aligned.
            with m.If(self._pc[0:2] != 0):
                m.d.ph2 += self.illegal.eq(1)

            with m.If(self._pc_plus_4_to_memaddr):
                m.d.ph1 += self.memaddr.eq(self._pc_plus_4)
            with m.Elif(self._x_to_memaddr):
                m.d.ph1 += self.memaddr.eq(self.data_x_in)
            with m.Elif(self._z_to_memaddr):
                m.d.ph1 += self.memaddr.eq(self.data_z_in)

            with m.If(self._z_to_memdata):
                m.d.ph1 += self.memdata_wr.eq(self.data_z_in)

        # Updates to multiplexers
        with m.If(self._pc_to_x):
            m.d.comb += self.data_x_out.eq(self._pc)
        with m.Elif(self._memdata_to_x):
            m.d.comb += self.data_x_out.eq(self.memdata_rd)

        with m.If(self._imm_to_y):
            m.d.comb += self.data_y_out.eq(self._imm)
        with m.Elif(self._shamt_to_y):
            m.d.comb += self.data_y_out.eq(self._shamt)

        with m.If(self._pc_plus_4_to_z):
            m.d.comb += self.data_z_out.eq(self._pc_plus_4)

        # Decode instruction
        m.d.comb += [
            self._opcode.eq(self._instr[:7]),
            self._rs1.eq(self._instr[15:20]),
            self._rs2.eq(self._instr[20:25]),
            self._rd.eq(self._instr[7:12]),
            self._funct3.eq(self._instr[12:15]),
            self._funct7.eq(self._instr[25:])
        ]
        self.decode_imm(m)

        with m.If(self._instr[:16] == 0):
            m.d.ph2 += self.illegal.eq(1)
        with m.Elif(self._instr == 0xFFFFFFFF):
            m.d.ph2 += self.illegal.eq(1)
        with m.Else():
            # Output control signals
            with m.Switch(self._opcode):
                with m.Case(OPCODE_LUI):
                    self.handle_lui(m)

                with m.Case(OPCODE_AUIPC):
                    self.handle_auipc(m)

                with m.Case(OPCODE_OP_IMM):
                    self.handle_op_imm(m)

                with m.Case(OPCODE_OP):
                    self.handle_op(m)

                with m.Case(OPCODE_JAL):
                    self.handle_jal(m)

                with m.Case(OPCODE_JALR):
                    self.handle_jalr(m)

                with m.Case(OPCODE_BRANCH):
                    self.handle_branch(m)

                with m.Case(OPCODE_LOAD):
                    self.handle_load(m)

                with m.Case(OPCODE_STORE):
                    self.handle_store(m)

                with m.Default():
                    m.d.comb += self._is_last_instr_cycle.eq(1)

        return m

    def decode_imm(self, m: Module):
        """Decodes the immediate value out of the instruction."""
        with m.Switch(self._imm_format):
            # Format I instructions. Surprisingly, SLTIU (Set if Less Than
            # Immediate Unsigned) actually does sign-extend the immediate
            # value, and then compare as if the sign-extended immediate value
            # were unsigned!
            with m.Case(OpcodeFormat.I):
                tmp = Signal(signed(12))
                m.d.comb += tmp.eq(self._instr[20:])
                m.d.comb += self._imm.eq(tmp)

            # Format S instructions:
            with m.Case(OpcodeFormat.S):
                tmp = Signal(signed(12))
                m.d.comb += tmp[0:5].eq(self._instr[7:12])
                m.d.comb += tmp[5:].eq(self._instr[25:])
                m.d.comb += self._imm.eq(tmp)

            # Format R instructions:
            with m.Case(OpcodeFormat.R):
                m.d.comb += self._imm.eq(0)

            # Format U instructions:
            with m.Case(OpcodeFormat.U):
                m.d.comb += self._imm.eq(0)
                m.d.comb += self._imm[12:].eq(self._instr[12:])

            # Format B instructions:
            with m.Case(OpcodeFormat.B):
                tmp = Signal(signed(13))
                m.d.comb += [
                    tmp[12].eq(self._instr[31]),
                    tmp[11].eq(self._instr[7]),
                    tmp[5:11].eq(self._instr[25:31]),
                    tmp[1:5].eq(self._instr[8:12]),
                    tmp[0].eq(0),
                    self._imm.eq(tmp),
                ]

            # Format J instructions:
            with m.Case(OpcodeFormat.J):
                tmp = Signal(signed(21))
                m.d.comb += [
                    tmp[20].eq(self._instr[31]),
                    tmp[12:20].eq(self._instr[12:20]),
                    tmp[11].eq(self._instr[20]),
                    tmp[1:11].eq(self._instr[21:31]),
                    tmp[0].eq(0),
                    self._imm.eq(tmp),
                ]

    def handle_lui(self, m: Module):
        """Adds the LUI logic to the given module.

        rd <- r0 + imm
        PC <- PC + 4

        r0      -> X
        imm     -> Y
        ALU ADD -> Z
        Z       -> rd
        PC + 4  -> PC
        PC + 4  -> memaddr
        """
        m.d.comb += [
            self._imm_format.eq(OpcodeFormat.U),
            self.reg_to_x.eq(1),
            self.x_reg.eq(0),
            self._imm_to_y.eq(1),
            self.alu_op.eq(AluOp.ADD),
            self.alu_to_z.eq(1),
            self.z_to_reg.eq(1),
            self.z_reg.eq(self._rd),
        ]
        m.d.comb += self._is_last_instr_cycle.eq(1)

    def handle_auipc(self, m: Module):
        """Adds the AUIPC logic to the given module.

        rd <- PC + imm
        PC <- PC + 4

        PC      -> X
        imm     -> Y
        ALU ADD -> Z
        Z       -> rd
        PC + 4  -> PC
        PC + 4  -> memaddr
        """
        m.d.comb += [
            self._imm_format.eq(OpcodeFormat.U),
            self._pc_to_x.eq(1),
            self._imm_to_y.eq(1),
            self.alu_op.eq(AluOp.ADD),
            self.alu_to_z.eq(1),
            self.z_to_reg.eq(1),
            self.z_reg.eq(self._rd),
        ]
        m.d.comb += self._is_last_instr_cycle.eq(1)

    def handle_op_imm(self, m: Module):
        """Adds the OP_IMM logic to the given module.

        rd <- rs1 op imm
        PC <- PC + 4

        rs1     -> X
        imm     -> Y
        ALU op  -> Z
        Z       -> rd
        PC + 4  -> PC
        PC + 4  -> memaddr
        """
        m.d.comb += [
            self._imm_format.eq(OpcodeFormat.I),
            self.reg_to_x.eq(1),
            self.x_reg.eq(self._rs1),
            self._imm_to_y.eq(1),
            self.alu_op[3].eq(self._funct7[5]),
            self.alu_op[0:3].eq(self._funct3),
            self.alu_to_z.eq(1),
            self.z_to_reg.eq(1),
            self.z_reg.eq(self._rd),
        ]
        m.d.comb += self._is_last_instr_cycle.eq(1)

    def handle_op(self, m: Module):
        """Adds the OP logic to the given module.

        rd <- rs1 op rs2
        PC <- PC + 4

        rs1     -> X
        rs2     -> Y
        ALU op  -> Z
        Z       -> rd
        PC + 4  -> PC
        PC + 4  -> memaddr
        """
        m.d.comb += [
            self._imm_format.eq(OpcodeFormat.R),
            self.reg_to_x.eq(1),
            self.x_reg.eq(self._rs1),
            self.reg_to_y.eq(1),
            self.y_reg.eq(self._rs2),
            self.alu_op[3].eq(self._funct7[5]),
            self.alu_op[0:3].eq(self._funct3),
            self.alu_to_z.eq(1),
            self.z_to_reg.eq(1),
            self.z_reg.eq(self._rd),
        ]
        m.d.comb += self._is_last_instr_cycle.eq(1)

    def handle_jal(self, m: Module):
        """Adds the JAL logic to the given module.

        rd <- PC + 4, PC <- PC + imm

        PC      -> X
        imm     -> Y
        ALU ADD -> Z
        Z       -> memaddr
        ---------------------
        PC + 4  -> Z
        Z       -> rd
        memaddr -> PC   # This will zero the least significant bit

        Note that because the immediate value for JAL has its least
        significant bit set to zero by definition, and the PC is also
        assumed to be aligned, there is no loss in generality to clear
        the least significant bit when transferring memaddr to PC.
        """
        m.d.comb += self._imm_format.eq(OpcodeFormat.J)

        with m.If(self._instr_phase == 0):
            m.d.comb += [
                self._pc_to_x.eq(1),
                self._imm_to_y.eq(1),
                self.alu_op.eq(AluOp.ADD),
                self.alu_to_z.eq(1),
                self._z_to_memaddr.eq(1),
                self._next_instr_phase.eq(1),
            ]
        with m.Else():
            m.d.comb += [
                self._pc_plus_4_to_z.eq(1),
                self.z_to_reg.eq(1),
                self.z_reg.eq(self._rd),
                self._memaddr_to_pc.eq(1),
            ]
            m.d.comb += self._is_last_instr_cycle.eq(1)

    def handle_jalr(self, m: Module):
        """Adds the JALR logic to the given module.

        rd <- PC + 4, PC <- (rs1 + imm) & 0xFFFFFFFE

        rs1     -> X
        imm     -> Y
        ALU ADD -> Z
        Z       -> memaddr
        ---------------------
        PC + 4  -> Z
        Z       -> rd
        memaddr -> PC  # This will zero the least significant bit
        """
        m.d.comb += self._imm_format.eq(OpcodeFormat.J)

        with m.If(self._instr_phase == 0):
            m.d.comb += [
                self.reg_to_x.eq(1),
                self.x_reg.eq(self._rs1),
                self._imm_to_y.eq(1),
                self.alu_op.eq(AluOp.ADD),
                self.alu_to_z.eq(1),
                self._z_to_memaddr.eq(1),
                self._next_instr_phase.eq(1),
            ]
        with m.Else():
            m.d.comb += [
                self._pc_plus_4_to_z.eq(1),
                self.z_to_reg.eq(1),
                self.z_reg.eq(self._rd),
                self._memaddr_to_pc.eq(1),
            ]
            m.d.comb += self._is_last_instr_cycle.eq(1)

    def handle_branch(self, m: Module):
        """Adds the BRANCH logic to the given module.

        cond <- rs1 - rs2 < 0, rs1 - rs2 == 0
        if f(cond):
            PC <- PC + imm
        else:
            PC <- PC + 4

        rs1     -> X
        rs2     -> Y
        ALU SUB -> Z, cond
        --------------------- cond == 1
        PC      -> X
        imm     -> Y
        ALU ADD -> Z
        Z       -> PC
        Z       -> memaddr
        --------------------- cond == 0
        PC + 4  -> PC
        PC + 4  -> memaddr
        """
        m.d.comb += self._imm_format.eq(OpcodeFormat.B)

        with m.If(self._instr_phase == 0):
            m.d.comb += [
                self.reg_to_x.eq(1),
                self.x_reg.eq(self._rs1),
                self.reg_to_y.eq(1),
                self.y_reg.eq(self._rs2),
                self.alu_op.eq(AluOp.SUB),
                self.alu_to_z.eq(1),
                self._next_instr_phase.eq(1),
            ]
        with m.Else():
            cond = Signal()
            with m.Switch(self._funct3):
                with m.Case(BRANCH_EQ):
                    m.d.comb += cond.eq(self._stored_alu_eq == 1)
                with m.Case(BRANCH_NE):
                    m.d.comb += cond.eq(self._stored_alu_eq == 0)
                with m.Case(BRANCH_LT):
                    m.d.comb += cond.eq(self._stored_alu_lt == 1)
                with m.Case(BRANCH_GE):
                    m.d.comb += cond.eq(self._stored_alu_lt == 0)
                with m.Case(BRANCH_LTU):
                    m.d.comb += cond.eq(self._stored_alu_ltu == 1)
                with m.Case(BRANCH_GEU):
                    m.d.comb += cond.eq(self._stored_alu_ltu == 0)

            with m.If(cond):
                m.d.comb += [
                    self._pc_to_x.eq(1),
                    self._imm_to_y.eq(1),
                    self.alu_op.eq(AluOp.ADD),
                    self.alu_to_z.eq(1),
                    self._z_to_pc.eq(1),
                    self._z_to_memaddr.eq(1),
                ]
            m.d.comb += self._is_last_instr_cycle.eq(1)

    def handle_load(self, m: Module):
        """Adds the LOAD logic to the given module.

        Note that byte loads are byte-aligned, half-word loads
        are 16-bit aligned, and word loads are 32-bit aligned.
        Attempting to load unaligned will lead to undefined
        behavior.

        Operation is to load 32 bits from a 32-bit aligned
        address, and then perform at most two shifts to get
        the desired behavior: a shift left to get the most
        significant byte into the leftmost position, then a
        shift right to zero or sign extend the value.

        For example, for loading a half-word starting at
        address A where A%4=0, we first load the full 32
        bits at that address, resulting in XYHL, where X and
        Y are unwanted and H and L are the half-word we want
        to load. Then we shift left by 16: HL00. And finally
        we shift right by 16, either signed or unsigned
        depending on whether we are doing an LH or an LHU:
        ssHL / 00HL.

        addr <- rs1 + imm
        rd <- data at addr, possibly sign-extended
        PC <- PC + 4

        If we let N be addr%4, then:

        instr   N   shift1  shift2
        --------------------------
        LB      0   SLL 24  SRA 24
        LB      1   SLL 16  SRA 24
        LB      2   SLL  8  SRA 24
        LB      3   SLL  0  SRA 24
        LBU     0   SLL 24  SRL 24
        LBU     1   SLL 16  SRL 24
        LBU     2   SLL  8  SRL 24
        LBU     3   SLL  0  SRL 24
        LH      0   SLL 16  SRA 16
        LH      2   SLL  0  SRA 16
        LHU     0   SLL 16  SRL 16
        LHU     2   SLL  0  SRL 16
        LW      0   SLL  0  SRA  0
        (all other N are misaligned accesses)

        Where there is an SLL 0, the machine cycle
        could be skipped, but in the interests of
        simpler logic, we will not do that.

        rs1     -> X
        imm     -> Y
        ALU ADD -> Z
        Z       -> memaddr
        ---------------------
        memdata -> X
        shamt1  -> Y
        ALU SLL -> Z
        Z       -> rd
        ---------------------
        rd          -> X
        shamt2      -> Y
        ALU SRA/SRL -> Z
        Z           -> rd
        PC + 4      -> PC
        PC + 4      -> memaddr
        """
        m.d.comb += self._imm_format.eq(OpcodeFormat.I)

        with m.If(self._instr_phase == 0):
            m.d.comb += [
                self.reg_to_x.eq(1),
                self.x_reg.eq(self._rs1),
                self._imm_to_y.eq(1),
                self.alu_op.eq(AluOp.ADD),
                self.alu_to_z.eq(1),
                self._z_to_memaddr.eq(1),
                self._next_instr_phase.eq(1),
            ]

        with m.Elif(self._instr_phase == 1):
            m.d.comb += [
                self.mem_rd.eq(1),
                self._memdata_to_x.eq(1),
                self._shamt_to_y.eq(1),
                self.alu_op.eq(AluOp.SLL),
                self.alu_to_z.eq(1),
                self.z_to_reg.eq(1),
                self.z_reg.eq(self._rd),
                self._next_instr_phase.eq(2),
            ]

            with m.Switch(self._funct3):

                with m.Case(LOAD_STORE_B, LOAD_STORE_BU):
                    with m.Switch(self.memaddr[0:2]):
                        with m.Case(0):
                            m.d.comb += self._shamt.eq(24)
                        with m.Case(1):
                            m.d.comb += self._shamt.eq(16)
                        with m.Case(2):
                            m.d.comb += self._shamt.eq(8)
                        with m.Case(3):
                            m.d.comb += self._shamt.eq(0)

                with m.Case(LOAD_STORE_H, LOAD_STORE_HU):
                    with m.Switch(self.memaddr[0:2]):
                        with m.Case(0):
                            m.d.comb += self._shamt.eq(16)
                        with m.Case(2):
                            m.d.comb += self._shamt.eq(0)
                        with m.Default():
                            m.d.ph2 += self.illegal.eq(1)

                with m.Case(LOAD_STORE_W):
                    with m.Switch(self.memaddr[0:2]):
                        with m.Case(0):
                            m.d.comb += self._shamt.eq(0)
                        with m.Default():
                            m.d.ph2 += self.illegal.eq(1)

        with m.Else():
            m.d.comb += [
                self.reg_to_x.eq(1),
                self.x_reg.eq(self._rd),
                self._shamt_to_y.eq(1),
                self.alu_to_z.eq(1),
                self.z_to_reg.eq(1),
                self.z_reg.eq(self._rd),
            ]

            with m.Switch(self._funct3):
                with m.Case(LOAD_STORE_B):
                    m.d.comb += [
                        self._shamt.eq(24),
                        self.alu_op.eq(AluOp.SRA),
                    ]
                with m.Case(LOAD_STORE_BU):
                    m.d.comb += [
                        self._shamt.eq(24),
                        self.alu_op.eq(AluOp.SRL),
                    ]
                with m.Case(LOAD_STORE_H):
                    m.d.comb += [
                        self._shamt.eq(16),
                        self.alu_op.eq(AluOp.SRA),
                    ]
                with m.Case(LOAD_STORE_HU):
                    m.d.comb += [
                        self._shamt.eq(16),
                        self.alu_op.eq(AluOp.SRL),
                    ]
                with m.Case(LOAD_STORE_W):
                    m.d.comb += [
                        self._shamt.eq(0),
                        self.alu_op.eq(AluOp.SRL),
                    ]

            m.d.comb += self._is_last_instr_cycle.eq(1)

    def handle_store(self, m: Module):
        """Adds the STORE logic to the given module.

        Note that byte stores are byte-aligned, half-word stores
        are 16-bit aligned, and word stores are 32-bit aligned.
        Attempting to stores unaligned will lead to undefined
        behavior.

        addr <- rs1 + imm
        data <- rs2
        PC <- PC + 4

        rs1     -> X
        imm     -> Y
        ALU ADD -> Z
        Z       -> memaddr
        ---------------------
        rs2     -> X
        shamt   -> Y
        ALU SLL -> Z
        Z       -> wrdata
                -> wrmask
        ---------------------
        PC + 4  -> PC
        PC + 4  -> memaddr
        """
        m.d.comb += self._imm_format.eq(OpcodeFormat.S)

        with m.If(self._instr_phase == 0):
            m.d.comb += [
                self.reg_to_x.eq(1),
                self.x_reg.eq(self._rs1),
                self._imm_to_y.eq(1),
                self.alu_op.eq(AluOp.ADD),
                self.alu_to_z.eq(1),
                self._z_to_memaddr.eq(1),
                self._next_instr_phase.eq(1),
            ]

        with m.Elif(self._instr_phase == 1):
            m.d.comb += [
                self.reg_to_x.eq(1),
                self.x_reg.eq(self._rs2),
                self._shamt_to_y.eq(1),
                self.alu_op.eq(AluOp.SLL),
                self._z_to_memdata.eq(1),
                self.mem_wr.eq(1),
                self._next_instr_phase.eq(2),
            ]

            with m.Switch(self._funct3):

                with m.Case(LOAD_STORE_B, LOAD_STORE_BU):
                    with m.Switch(self.memaddr[0:2]):
                        with m.Case(0):
                            m.d.comb += self._shamt.eq(0)
                            m.d.comb += self.mem_wr_mask.eq(0b0001)
                        with m.Case(1):
                            m.d.comb += self._shamt.eq(8)
                            m.d.comb += self.mem_wr_mask.eq(0b0010)
                        with m.Case(2):
                            m.d.comb += self._shamt.eq(16)
                            m.d.comb += self.mem_wr_mask.eq(0b0100)
                        with m.Case(3):
                            m.d.comb += self._shamt.eq(24)
                            m.d.comb += self.mem_wr_mask.eq(0b1000)

                with m.Case(LOAD_STORE_H, LOAD_STORE_HU):
                    with m.Switch(self.memaddr[0:2]):
                        with m.Case(0):
                            m.d.comb += self._shamt.eq(0)
                            m.d.comb += self.mem_wr_mask.eq(0b0011)
                        with m.Case(2):
                            m.d.comb += self._shamt.eq(16)
                            m.d.comb += self.mem_wr_mask.eq(0b1100)
                        with m.Default():
                            m.d.ph2 += self.illegal.eq(1)

                with m.Case(LOAD_STORE_W):
                    with m.Switch(self.memaddr[0:2]):
                        with m.Case(0):
                            m.d.comb += self._shamt.eq(0)
                            m.d.comb += self.mem_wr_mask.eq(0b1111)
                        with m.Default():
                            m.d.ph2 += self.illegal.eq(1)

        with m.Else():
            m.d.comb += self._is_last_instr_cycle.eq(1)

    @classmethod
    def formal(cls) -> Tuple[Module, List[Signal]]:
        """Formal verification for the sequencer."""
        m = Module()

        ph1 = ClockDomain("ph1")
        ph2 = ClockDomain("ph2")
        seq = SequencerCard()

        m.domains += [ph1, ph2]
        m.submodules += seq

        # Generate the ph1 and ph2 clocks.
        cycle_count = Signal(8, reset=0, reset_less=True)
        phase_count = Signal(3, reset=0, reset_less=True)

        m.d.sync += phase_count.eq(phase_count + 1)
        with m.If(phase_count == 5):
            m.d.sync += phase_count.eq(0)
            m.d.sync += cycle_count.eq(cycle_count + 1)

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

        m.d.comb += seq.mcycle_end.eq(phase_count == 5)

        m.d.comb += Cover(seq.instr_complete & (seq._opcode == OPCODE_STORE))
        m.d.comb += Cover((seq._pc > 0x100))
        m.d.comb += Cover((cycle_count > 1) &
                          Past(seq.illegal, 6) & Past(seq.illegal, 12) & seq.illegal)

        with m.If(phase_count > 1):
            m.d.comb += [
                Assume(Stable(seq.memdata_rd)),
                Assume(Stable(seq.data_x_in)),
                Assume(Stable(seq.data_y_in)),
                Assume(Stable(seq.data_z_in)),
                Assume(Stable(seq.alu_eq)),
                Assume(Stable(seq.alu_lt)),
                Assume(Stable(seq.alu_ltu)),
            ]

            m.d.comb += [
                Assert(Stable(seq.x_reg)),
                Assert(Stable(seq.y_reg)),
                Assert(Stable(seq.z_reg)),
                Assert(Stable(seq.reg_to_x)),
                Assert(Stable(seq.reg_to_y)),
                Assert(Stable(seq.z_to_reg)),
                Assert(Stable(seq.alu_op)),
                Assert(Stable(seq.alu_to_z)),

                Assert(Stable(seq.data_x_out)),
                Assert(Stable(seq.data_y_out)),
                Assert(Stable(seq.data_z_out)),

                Assert(Stable(seq.mem_rd)),
                Assert(Stable(seq.mem_wr)),
                Assert(Stable(seq.mem_wr_mask)),
                Assert(Stable(seq.memaddr)),
                Assert(Stable(seq.memdata_wr)),
            ]

        return m, [seq.memdata_rd, seq.data_x_in, seq.data_y_in, seq.data_z_in] + [seq._pc_plus_4_to_memaddr, seq._x_to_memaddr, seq._z_to_memdata, seq._imm, seq._imm_format, seq._is_last_instr_cycle, seq.instr_complete]


if __name__ == "__main__":
    main(SequencerCard)
