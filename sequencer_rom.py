# Disable pylint's "your name is too short" warning.
# pylint: disable=C0103
# Disable protected access warnings
# pylint: disable=W0212
from nmigen import Signal, Module, Elaboratable
from nmigen.build import Platform

from consts import AluOp, AluFunc, BranchCond, MemAccessWidth
from consts import OpcodeFormat, SystemFunc, TrapCauseSelect
from consts import InstrReg, OpcodeSelect
from consts import NextPC


class SequencerROM(Elaboratable):
    """ROM for the sequencer card state machine."""

    def __init__(self):
        # Control line
        self.enable_sequencer_rom = Signal()

        # Inputs: 9 + 11 decoder
        # Since this is implemented by a ROM, the address lines
        # must be stable in order for the outputs to start becoming
        # stable. This means that if any input address depends on
        # any output data combinatorically, there's a danger of
        # going unstable. Therefore, all address lines must be
        # registered, or come combinatorically from registered data.

        self.memaddr_2_lsb = Signal(2)
        self.branch_cond = Signal()
        self._instr_phase = Signal(2)
        # Only used on instruction phase 1 in BRANCH.
        self.data_z_in_2_lsb0 = Signal()
        self.imm0 = Signal()
        self.rd0 = Signal()
        self.rs1_0 = Signal()

        # Instruction decoding
        self.opcode_select = Signal(OpcodeSelect)  # 4 bits
        self._funct3 = Signal(3)
        self._alu_func = Signal(4)

        ##############
        # Outputs (48 bits + 28 other bits = 76 bits total)
        ##############

        # Raised on the last phase of an instruction.
        self.set_instr_complete = Signal()

        # Raised when the exception card should store trap data.
        self.save_trap_csrs = Signal()

        # CSR lines
        self.csr_to_x = Signal()
        self.z_to_csr = Signal()

        # Memory
        self.mem_rd = Signal(reset=1)
        self.mem_wr = Signal()
        # Bytes in memory word to write
        self.mem_wr_mask = Signal(4)

        self._next_instr_phase = Signal(2)

        self._imm_format = Signal(OpcodeFormat)  # 3 bits

        self._x_reg_select = Signal(InstrReg)  # 2 bits
        self._y_reg_select = Signal(InstrReg)  # 2 bits
        self._z_reg_select = Signal(InstrReg)  # 2 bits

        # -> X
        self.reg_to_x = Signal()
        self._pc_to_x = Signal()
        self._memdata_to_x = Signal()
        self._trapcause_to_x = Signal()

        # -> Y
        self.reg_to_y = Signal()
        self._imm_to_y = Signal()
        self._shamt_to_y = Signal()
        self._pc_to_y = Signal()
        self._pc_plus_4_to_y = Signal()
        self._mtvec_30_to_y = Signal()

        # -> Z
        self._pc_plus_4_to_z = Signal()
        self._tmp_to_z = Signal()
        self.alu_op_to_z = Signal(AluOp)  # 4 bits
        self._pc_to_z = Signal()
        self._instr_to_z = Signal()
        self._memaddr_to_z = Signal()
        self._memaddr_lsb_masked_to_z = Signal()

        # -> PC
        self._pc_plus_4_to_pc = Signal()
        self._z_to_pc = Signal()
        self._x_to_pc = Signal()
        self._memaddr_to_pc = Signal()
        self._memdata_to_pc = Signal()
        self._z_30_to_pc = Signal()

        # -> tmp
        self._x_to_tmp = Signal()
        self._z_to_tmp = Signal()

        # -> csr_num
        self._funct12_to_csr_num = Signal()
        self._mepc_num_to_csr_num = Signal()
        self._mcause_to_csr_num = Signal()

        # -> memaddr
        self._pc_plus_4_to_memaddr = Signal()
        self._z_to_memaddr = Signal()
        self._x_to_memaddr = Signal()
        self._memdata_to_memaddr = Signal()
        self._z_30_to_memaddr = Signal()

        # -> memdata
        self._z_to_memdata = Signal()

        # memory load shamt
        # There's not a lot of point in multiplexing this, since
        # there are 5 possible values, requiring 3 bits instead of
        # 5. Not much of a savings.
        self._shamt = Signal(5)

        # -> various CSRs
        self._trapcause_select = Signal(TrapCauseSelect)  # 4 bits
        self.clear_pend_mti = Signal()
        self.clear_pend_mei = Signal()

        self.enter_trap = Signal()
        self.exit_trap = Signal()

        # Signals for next registers
        self.load_trap = Signal()
        self.next_trap = Signal()
        self.load_exception = Signal()
        self.next_exception = Signal()
        self.next_fatal = Signal()

    def elaborate(self, _: Platform) -> Module:
        """Implements the logic of the sequencer card."""
        m = Module()

        # Defaults
        m.d.comb += [
            self._next_instr_phase.eq(0),
            self._trapcause_select.eq(TrapCauseSelect.NONE),
            self.reg_to_x.eq(0),
            self._pc_to_x.eq(0),
            self._memdata_to_x.eq(0),
            self._trapcause_to_x.eq(0),
            self.reg_to_y.eq(0),
            self._imm_to_y.eq(0),
            self._shamt_to_y.eq(0),
            self._pc_to_y.eq(0),
            self._pc_plus_4_to_y.eq(0),
            self._mtvec_30_to_y.eq(0),
            self.alu_op_to_z.eq(AluOp.NONE),
            self._pc_plus_4_to_z.eq(0),
            self._tmp_to_z.eq(0),
            self._pc_to_z.eq(0),
            self._instr_to_z.eq(0),
            self._memaddr_to_z.eq(0),
            self._memaddr_lsb_masked_to_z.eq(0),
            self._pc_plus_4_to_pc.eq(0),
            self._x_to_pc.eq(0),
            self._z_to_pc.eq(0),
            self._z_30_to_pc.eq(0),
            self._memaddr_to_pc.eq(0),
            self._memdata_to_pc.eq(0),
            self._x_to_tmp.eq(0),
            self._z_to_tmp.eq(0),
            self._pc_plus_4_to_memaddr.eq(0),
            self._x_to_memaddr.eq(0),
            self._z_to_memaddr.eq(0),
            self._memdata_to_memaddr.eq(0),
            self._z_30_to_memaddr.eq(0),
            self._z_to_memdata.eq(0),
            self.mem_rd.eq(0),
            self.mem_wr.eq(0),
            self.mem_wr_mask.eq(0),
            self._shamt.eq(0),
            self.csr_to_x.eq(0),
            self.z_to_csr.eq(0),
            self._funct12_to_csr_num.eq(0),
            self._mepc_num_to_csr_num.eq(0),
            self._mcause_to_csr_num.eq(0),
            self._x_reg_select.eq(0),
            self._y_reg_select.eq(0),
            self._z_reg_select.eq(0),
            self.enter_trap.eq(0),
            self.exit_trap.eq(0),
            self.clear_pend_mti.eq(0),
            self.clear_pend_mei.eq(0),
            self.save_trap_csrs.eq(0),
        ]

        m.d.comb += [
            self.load_trap.eq(0),
            self.next_trap.eq(0),
            self.load_exception.eq(0),
            self.next_exception.eq(0),
            self.next_fatal.eq(0),
        ]

        with m.If(self.enable_sequencer_rom):

            # Output control signals
            with m.Switch(self.opcode_select):
                with m.Case(OpcodeSelect.LUI):
                    self.handle_lui(m)

                with m.Case(OpcodeSelect.AUIPC):
                    self.handle_auipc(m)

                with m.Case(OpcodeSelect.OP_IMM):
                    self.handle_op_imm(m)

                with m.Case(OpcodeSelect.OP):
                    self.handle_op(m)

                with m.Case(OpcodeSelect.JAL):
                    self.handle_jal(m)

                with m.Case(OpcodeSelect.JALR):
                    self.handle_jalr(m)

                with m.Case(OpcodeSelect.BRANCH):
                    self.handle_branch(m)

                with m.Case(OpcodeSelect.LOAD):
                    self.handle_load(m)

                with m.Case(OpcodeSelect.STORE):
                    self.handle_store(m)

                with m.Case(OpcodeSelect.CSRS):
                    self.handle_csrs(m)

                with m.Case(OpcodeSelect.MRET):
                    self.handle_MRET(m)

                with m.Case(OpcodeSelect.ECALL):
                    self.handle_ECALL(m)

                with m.Case(OpcodeSelect.EBREAK):
                    self.handle_EBREAK(m)

                with m.Default():
                    self.handle_illegal_instr(m)

        return m

    def next_instr(self, m: Module, next_pc: NextPC = NextPC.PC_PLUS_4):
        """Sets signals to advance to the next instruction.

        next_pc is the signal to load the PC and MEMADDR registers with
        at the end of the instruction cycle.
        """
        m.d.comb += self.set_instr_complete.eq(1)
        if next_pc == NextPC.PC_PLUS_4:
            m.d.comb += self._pc_plus_4_to_pc.eq(1)
            m.d.comb += self._pc_plus_4_to_memaddr.eq(1)
        elif next_pc == NextPC.MEMADDR:
            m.d.comb += self._memaddr_to_pc.eq(1)
        elif next_pc == NextPC.Z:
            m.d.comb += self._z_to_pc.eq(1)
            m.d.comb += self._z_to_memaddr.eq(1)
        elif next_pc == NextPC.X:
            m.d.comb += self._x_to_pc.eq(1)
            m.d.comb += self._x_to_memaddr.eq(1)

    def set_exception(self, m: Module, exc: TrapCauseSelect, mtval: Signal, fatal: bool = True):
        m.d.comb += self.load_exception.eq(1)
        m.d.comb += self.next_exception.eq(1)
        m.d.comb += self.next_fatal.eq(1 if fatal else 0)

        m.d.comb += self._trapcause_select.eq(exc)
        m.d.comb += self._trapcause_to_x.eq(1)

        m.d.comb += mtval.eq(1)  # what goes to z

        if fatal:
            m.d.comb += self._pc_to_y.eq(1)
        else:
            m.d.comb += self._pc_plus_4_to_y.eq(1)

        # X -> MCAUSE, Y -> MEPC, Z -> MTVAL
        m.d.comb += self.save_trap_csrs.eq(1)
        m.d.comb += self.load_trap.eq(1)
        m.d.comb += self.next_trap.eq(1)
        m.d.comb += self._next_instr_phase.eq(0)

    def handle_illegal_instr(self, m: Module):
        self.set_exception(m, TrapCauseSelect.EXC_ILLEGAL_INSTR,
                           mtval=self._instr_to_z)

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
            self._x_reg_select.eq(InstrReg.ZERO),
            self._imm_to_y.eq(1),
            self.alu_op_to_z.eq(AluOp.ADD),
            self._z_reg_select.eq(InstrReg.RD),
        ]
        self.next_instr(m)

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
            self.alu_op_to_z.eq(AluOp.ADD),
            self._z_reg_select.eq(InstrReg.RD),
        ]
        self.next_instr(m)

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
        with m.If(~self._alu_func.matches(AluFunc.ADD, AluFunc.SUB, AluFunc.SLL, AluFunc.SLT,
                                          AluFunc.SLTU, AluFunc.XOR, AluFunc.SRL, AluFunc.SRA, AluFunc.OR, AluFunc.AND)):
            self.handle_illegal_instr(m)
        with m.Else():
            m.d.comb += [
                self._imm_format.eq(OpcodeFormat.I),
                self.reg_to_x.eq(1),
                self._x_reg_select.eq(InstrReg.RS1),
                self._imm_to_y.eq(1),
                self._z_reg_select.eq(InstrReg.RD),
            ]
            with m.Switch(self._alu_func):
                with m.Case(AluFunc.ADD):
                    m.d.comb += self.alu_op_to_z.eq(AluOp.ADD)
                with m.Case(AluFunc.SUB):
                    m.d.comb += self.alu_op_to_z.eq(AluOp.SUB)
                with m.Case(AluFunc.SLL):
                    m.d.comb += self.alu_op_to_z.eq(AluOp.SLL)
                with m.Case(AluFunc.SLT):
                    m.d.comb += self.alu_op_to_z.eq(AluOp.SLT)
                with m.Case(AluFunc.SLTU):
                    m.d.comb += self.alu_op_to_z.eq(AluOp.SLTU)
                with m.Case(AluFunc.XOR):
                    m.d.comb += self.alu_op_to_z.eq(AluOp.XOR)
                with m.Case(AluFunc.SRL):
                    m.d.comb += self.alu_op_to_z.eq(AluOp.SRL)
                with m.Case(AluFunc.SRA):
                    m.d.comb += self.alu_op_to_z.eq(AluOp.SRA)
                with m.Case(AluFunc.OR):
                    m.d.comb += self.alu_op_to_z.eq(AluOp.OR)
                with m.Case(AluFunc.AND):
                    m.d.comb += self.alu_op_to_z.eq(AluOp.AND)
            self.next_instr(m)

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
        with m.If(~self._alu_func.matches(AluFunc.ADD, AluFunc.SUB, AluFunc.SLL, AluFunc.SLT,
                                          AluFunc.SLTU, AluFunc.XOR, AluFunc.SRL, AluFunc.SRA, AluFunc.OR, AluFunc.AND)):
            self.handle_illegal_instr(m)
        with m.Else():
            m.d.comb += [
                self._imm_format.eq(OpcodeFormat.R),
                self.reg_to_x.eq(1),
                self._x_reg_select.eq(InstrReg.RS1),
                self.reg_to_y.eq(1),
                self._y_reg_select.eq(InstrReg.RS2),
                self._z_reg_select.eq(InstrReg.RD),
            ]
            with m.Switch(self._alu_func):
                with m.Case(AluFunc.ADD):
                    m.d.comb += self.alu_op_to_z.eq(AluOp.ADD)
                with m.Case(AluFunc.SUB):
                    m.d.comb += self.alu_op_to_z.eq(AluOp.SUB)
                with m.Case(AluFunc.SLL):
                    m.d.comb += self.alu_op_to_z.eq(AluOp.SLL)
                with m.Case(AluFunc.SLT):
                    m.d.comb += self.alu_op_to_z.eq(AluOp.SLT)
                with m.Case(AluFunc.SLTU):
                    m.d.comb += self.alu_op_to_z.eq(AluOp.SLTU)
                with m.Case(AluFunc.XOR):
                    m.d.comb += self.alu_op_to_z.eq(AluOp.XOR)
                with m.Case(AluFunc.SRL):
                    m.d.comb += self.alu_op_to_z.eq(AluOp.SRL)
                with m.Case(AluFunc.SRA):
                    m.d.comb += self.alu_op_to_z.eq(AluOp.SRA)
                with m.Case(AluFunc.OR):
                    m.d.comb += self.alu_op_to_z.eq(AluOp.OR)
                with m.Case(AluFunc.AND):
                    m.d.comb += self.alu_op_to_z.eq(AluOp.AND)
            self.next_instr(m)

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
                self.alu_op_to_z.eq(AluOp.ADD),
                self._z_to_memaddr.eq(1),
                self._next_instr_phase.eq(1),
            ]
        with m.Else():
            with m.If(self.memaddr_2_lsb[1] != 0):
                self.set_exception(
                    m, TrapCauseSelect.EXC_INSTR_ADDR_MISALIGN, mtval=self._memaddr_to_z)
            with m.Else():
                m.d.comb += [
                    self._pc_plus_4_to_z.eq(1),
                    self._z_reg_select.eq(InstrReg.RD),
                ]
                self.next_instr(m, NextPC.MEMADDR)

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
                self._x_reg_select.eq(InstrReg.RS1),
                self._imm_to_y.eq(1),
                self.alu_op_to_z.eq(AluOp.ADD),
                self._z_to_memaddr.eq(1),
                self._next_instr_phase.eq(1),
            ]
        with m.Else():
            with m.If(self.memaddr_2_lsb[1] != 0):
                self.set_exception(
                    m, TrapCauseSelect.EXC_INSTR_ADDR_MISALIGN, mtval=self._memaddr_lsb_masked_to_z)
            with m.Else():
                m.d.comb += [
                    self._pc_plus_4_to_z.eq(1),
                    self._z_reg_select.eq(InstrReg.RD),
                ]
                self.next_instr(m, NextPC.MEMADDR)

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
        imm/4   -> Y (imm for cond == 1, 4 otherwise)
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
                self._x_reg_select.eq(InstrReg.RS1),
                self.reg_to_y.eq(1),
                self._y_reg_select.eq(InstrReg.RS2),
                self.alu_op_to_z.eq(AluOp.SUB),
                self._next_instr_phase.eq(1),
            ]
        with m.Elif(self._instr_phase == 1):
            with m.If(~self._funct3.matches(BranchCond.EQ, BranchCond.NE,
                                            BranchCond.LT, BranchCond.GE,
                                            BranchCond.LTU, BranchCond.GEU)):
                self.handle_illegal_instr(m)

            with m.Else():
                with m.If(self.branch_cond):
                    m.d.comb += self._imm_to_y.eq(1)
                with m.Else():
                    m.d.comb += self._shamt.eq(4)
                    m.d.comb += self._shamt_to_y.eq(1)

                m.d.comb += [
                    self._pc_to_x.eq(1),
                    self.alu_op_to_z.eq(AluOp.ADD),
                ]

                with m.If(self.data_z_in_2_lsb0):
                    self.next_instr(m, NextPC.Z)

                with m.Else():
                    m.d.comb += self._next_instr_phase.eq(2)
                    m.d.comb += self._z_to_tmp.eq(1)

        with m.Else():
            self.set_exception(
                m, TrapCauseSelect.EXC_INSTR_ADDR_MISALIGN, mtval=self._tmp_to_z)

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
                self._x_reg_select.eq(InstrReg.RS1),
                self._imm_to_y.eq(1),
                self.alu_op_to_z.eq(AluOp.ADD),
                self._z_to_memaddr.eq(1),
                self._next_instr_phase.eq(1),
            ]

        with m.Elif(self._instr_phase == 1):
            # Check for exception conditions first
            with m.If(self._funct3.matches(MemAccessWidth.H, MemAccessWidth.HU) &
                      self.memaddr_2_lsb[0]):
                self.set_exception(
                    m, TrapCauseSelect.EXC_LOAD_ADDR_MISALIGN, mtval=self._memaddr_to_z)

            with m.Elif((self._funct3 == MemAccessWidth.W) &
                        (self.memaddr_2_lsb != 0)):
                self.set_exception(
                    m, TrapCauseSelect.EXC_LOAD_ADDR_MISALIGN, mtval=self._memaddr_to_z)

            with m.Elif(~self._funct3.matches(MemAccessWidth.B, MemAccessWidth.BU,
                                              MemAccessWidth.H, MemAccessWidth.HU,
                                              MemAccessWidth.W)):
                self.handle_illegal_instr(m)

            with m.Else():
                m.d.comb += [
                    self.mem_rd.eq(1),
                    self._memdata_to_x.eq(1),
                    self._shamt_to_y.eq(1),
                    self.alu_op_to_z.eq(AluOp.SLL),
                    self._z_reg_select.eq(InstrReg.RD),
                    self._next_instr_phase.eq(2),
                ]

                with m.Switch(self._funct3):

                    with m.Case(MemAccessWidth.B, MemAccessWidth.BU):
                        with m.Switch(self.memaddr_2_lsb):
                            with m.Case(0):
                                m.d.comb += self._shamt.eq(24)
                            with m.Case(1):
                                m.d.comb += self._shamt.eq(16)
                            with m.Case(2):
                                m.d.comb += self._shamt.eq(8)
                            with m.Case(3):
                                m.d.comb += self._shamt.eq(0)

                    with m.Case(MemAccessWidth.H, MemAccessWidth.HU):
                        with m.Switch(self.memaddr_2_lsb):
                            with m.Case(0):
                                m.d.comb += self._shamt.eq(16)
                            with m.Case(2):
                                m.d.comb += self._shamt.eq(0)

                    with m.Case(MemAccessWidth.W):
                        m.d.comb += self._shamt.eq(0)

        with m.Else():
            m.d.comb += [
                self.reg_to_x.eq(1),
                self._x_reg_select.eq(InstrReg.RD),
                self._shamt_to_y.eq(1),
                self._z_reg_select.eq(InstrReg.RD),
            ]

            with m.Switch(self._funct3):
                with m.Case(MemAccessWidth.B):
                    m.d.comb += [
                        self._shamt.eq(24),
                        self.alu_op_to_z.eq(AluOp.SRA),
                    ]
                with m.Case(MemAccessWidth.BU):
                    m.d.comb += [
                        self._shamt.eq(24),
                        self.alu_op_to_z.eq(AluOp.SRL),
                    ]
                with m.Case(MemAccessWidth.H):
                    m.d.comb += [
                        self._shamt.eq(16),
                        self.alu_op_to_z.eq(AluOp.SRA),
                    ]
                with m.Case(MemAccessWidth.HU):
                    m.d.comb += [
                        self._shamt.eq(16),
                        self.alu_op_to_z.eq(AluOp.SRL),
                    ]
                with m.Case(MemAccessWidth.W):
                    m.d.comb += [
                        self._shamt.eq(0),
                        self.alu_op_to_z.eq(AluOp.SRL),
                    ]

            self.next_instr(m)

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
                self._x_reg_select.eq(InstrReg.RS1),
                self._imm_to_y.eq(1),
                self.alu_op_to_z.eq(AluOp.ADD),
                self._z_to_memaddr.eq(1),
                self._next_instr_phase.eq(1),
            ]

        with m.Elif(self._instr_phase == 1):
            # Check for exception conditions first
            with m.If((self._funct3 == MemAccessWidth.H) & self.memaddr_2_lsb[0]):
                self.set_exception(
                    m, TrapCauseSelect.EXC_STORE_AMO_ADDR_MISALIGN, mtval=self._memaddr_to_z)

            with m.Elif((self._funct3 == MemAccessWidth.W) & (self.memaddr_2_lsb != 0)):
                self.set_exception(
                    m, TrapCauseSelect.EXC_STORE_AMO_ADDR_MISALIGN, mtval=self._memaddr_to_z)

            with m.Elif(~self._funct3.matches(MemAccessWidth.B,
                                              MemAccessWidth.H,
                                              MemAccessWidth.W)):
                self.handle_illegal_instr(m)

            with m.Else():
                m.d.comb += [
                    self.reg_to_x.eq(1),
                    self._x_reg_select.eq(InstrReg.RS2),
                    self._shamt_to_y.eq(1),
                    self.alu_op_to_z.eq(AluOp.SLL),
                    self._z_to_memdata.eq(1),
                    self._next_instr_phase.eq(2),
                ]

                with m.Switch(self._funct3):

                    with m.Case(MemAccessWidth.B):
                        with m.Switch(self.memaddr_2_lsb):
                            with m.Case(0):
                                m.d.comb += self._shamt.eq(0)
                            with m.Case(1):
                                m.d.comb += self._shamt.eq(8)
                            with m.Case(2):
                                m.d.comb += self._shamt.eq(16)
                            with m.Case(3):
                                m.d.comb += self._shamt.eq(24)

                    with m.Case(MemAccessWidth.H):
                        with m.Switch(self.memaddr_2_lsb):
                            with m.Case(0):
                                m.d.comb += self._shamt.eq(0)
                            with m.Case(2):
                                m.d.comb += self._shamt.eq(16)

                    with m.Case(MemAccessWidth.W):
                        m.d.comb += self._shamt.eq(0)

        with m.Else():
            with m.Switch(self._funct3):

                with m.Case(MemAccessWidth.B):
                    with m.Switch(self.memaddr_2_lsb):
                        with m.Case(0):
                            m.d.comb += self.mem_wr_mask.eq(0b0001)
                        with m.Case(1):
                            m.d.comb += self.mem_wr_mask.eq(0b0010)
                        with m.Case(2):
                            m.d.comb += self.mem_wr_mask.eq(0b0100)
                        with m.Case(3):
                            m.d.comb += self.mem_wr_mask.eq(0b1000)

                with m.Case(MemAccessWidth.H):
                    with m.Switch(self.memaddr_2_lsb):
                        with m.Case(0):
                            m.d.comb += self.mem_wr_mask.eq(0b0011)
                        with m.Case(2):
                            m.d.comb += self.mem_wr_mask.eq(0b1100)

                with m.Case(MemAccessWidth.W):
                    m.d.comb += self.mem_wr_mask.eq(0b1111)

            m.d.comb += self.mem_wr.eq(1)
            self.next_instr(m)

    def handle_csrs(self, m: Module):
        """Adds the SYSTEM (CSR opcodes) logic to the given module.

        Some points of interest:

        * Attempts to write a read-only register
          result in an illegal instruction exception.
        * Attempts to access a CSR that doesn't exist
          result in an illegal instruction exception.
        * Attempts to write read-only bits to a read/write CSR
          are ignored.

        Because we're building this in hardware, which is
        expensive, we're not implementing any CSRs that aren't
        strictly necessary. The documentation for the misa, mvendorid,
        marchid, and mimpid registers state that they can return zero if
        unimplemented. This implies that unimplemented CSRs still
        exist.

        The mhartid, because we only have one HART, can just return zero.
        """
        m.d.comb += self._imm_format.eq(OpcodeFormat.SYS)

        with m.Switch(self._funct3):
            with m.Case(SystemFunc.CSRRW):
                self.handle_CSRRW(m)
            with m.Case(SystemFunc.CSRRWI):
                self.handle_CSRRWI(m)
            with m.Case(SystemFunc.CSRRS):
                self.handle_CSRRS(m)
            with m.Case(SystemFunc.CSRRSI):
                self.handle_CSRRSI(m)
            with m.Case(SystemFunc.CSRRC):
                self.handle_CSRRC(m)
            with m.Case(SystemFunc.CSRRCI):
                self.handle_CSRRCI(m)
            with m.Default():
                self.handle_illegal_instr(m)

    def handle_CSRRW(self, m: Module):
        m.d.comb += self._funct12_to_csr_num.eq(1)

        with m.If(self._instr_phase == 0):
            with m.If(self.rd0):
                m.d.comb += [
                    self._x_reg_select.eq(InstrReg.ZERO),
                    self.reg_to_x.eq(1),
                ]
            with m.Else():
                m.d.comb += [
                    self.csr_to_x.eq(1)
                ]
            m.d.comb += [
                self._y_reg_select.eq(InstrReg.RS1),
                self.reg_to_y.eq(1),
                self.alu_op_to_z.eq(AluOp.Y),
                self.z_to_csr.eq(1),
                self._x_to_tmp.eq(1),
                self._next_instr_phase.eq(1),
            ]

        with m.Else():
            m.d.comb += [
                self._tmp_to_z.eq(1),
                self._z_reg_select.eq(InstrReg.RD),
            ]
            self.next_instr(m)

    def handle_CSRRWI(self, m: Module):
        m.d.comb += self._funct12_to_csr_num.eq(1)

        with m.If(self._instr_phase == 0):
            with m.If(self.rd0):
                m.d.comb += [
                    self._x_reg_select.eq(InstrReg.ZERO),
                    self.reg_to_x.eq(1),
                ]
            with m.Else():
                m.d.comb += [
                    self.csr_to_x.eq(1)
                ]
            m.d.comb += [
                self._imm_to_y.eq(1),
                self.alu_op_to_z.eq(AluOp.Y),
                self.z_to_csr.eq(1),
                self._x_to_tmp.eq(1),
                self._next_instr_phase.eq(1),
            ]

        with m.Else():
            m.d.comb += [
                self._tmp_to_z.eq(1),
                self._z_reg_select.eq(InstrReg.RD),
            ]
            self.next_instr(m)

    def handle_CSRRS(self, m: Module):
        m.d.comb += self._funct12_to_csr_num.eq(1)

        with m.If(self._instr_phase == 0):
            m.d.comb += [
                self.csr_to_x.eq(1),
                self._y_reg_select.eq(InstrReg.RS1),
                self.reg_to_y.eq(1),
                self.alu_op_to_z.eq(AluOp.OR),
                self.z_to_csr.eq(~self.rs1_0),
                self._x_to_tmp.eq(1),
                self._next_instr_phase.eq(1),
            ]

        with m.Else():
            m.d.comb += [
                self._tmp_to_z.eq(1),
                self._z_reg_select.eq(InstrReg.RD),
            ]
            self.next_instr(m)

    def handle_CSRRSI(self, m: Module):
        m.d.comb += self._funct12_to_csr_num.eq(1)

        with m.If(self._instr_phase == 0):
            m.d.comb += [
                self.csr_to_x.eq(1),
                self._imm_to_y.eq(1),
                self.alu_op_to_z.eq(AluOp.OR),
                self.z_to_csr.eq(~self.imm0),
                self._x_to_tmp.eq(1),
                self._next_instr_phase.eq(1),
            ]

        with m.Else():
            m.d.comb += [
                self._tmp_to_z.eq(1),
                self._z_reg_select.eq(InstrReg.RD),
            ]
            self.next_instr(m)

    def handle_CSRRC(self, m: Module):
        m.d.comb += self._funct12_to_csr_num.eq(1)

        with m.If(self._instr_phase == 0):
            m.d.comb += [
                self.csr_to_x.eq(1),
                self._y_reg_select.eq(InstrReg.RS1),
                self.reg_to_y.eq(1),
                self.alu_op_to_z.eq(AluOp.AND_NOT),
                self.z_to_csr.eq(~self.rs1_0),
                self._x_to_tmp.eq(1),
                self._next_instr_phase.eq(1),
            ]

        with m.Else():
            m.d.comb += [
                self._tmp_to_z.eq(1),
                self._z_reg_select.eq(InstrReg.RD),
            ]
            self.next_instr(m)

    def handle_CSRRCI(self, m: Module):
        m.d.comb += self._funct12_to_csr_num.eq(1)

        with m.If(self._instr_phase == 0):
            m.d.comb += [
                self.csr_to_x.eq(1),
                self._imm_to_y.eq(1),
                self.alu_op_to_z.eq(AluOp.AND_NOT),
                self.z_to_csr.eq(~self.imm0),
                self._x_to_tmp.eq(1),
                self._next_instr_phase.eq(1),
            ]

        with m.Else():
            m.d.comb += [
                self._tmp_to_z.eq(1),
                self._z_reg_select.eq(InstrReg.RD),
            ]
            self.next_instr(m)

    def handle_MRET(self, m: Module):
        m.d.comb += [
            self._mepc_num_to_csr_num.eq(1),
            self.csr_to_x.eq(1),
            self.exit_trap.eq(1),
        ]
        self.next_instr(m, NextPC.X)

    def handle_ECALL(self, m: Module):
        """Handles the ECALL instruction.

        Note that normally, ECALL is used from a lower privelege mode, which stores
        the PC of the instruction in the appropriate lower EPC CSR (e.g. SEPC or UEPC).
        This allows interrupts to be handled during the call, because we're in a higher
        privelege level. However, in machine mode, there is no higher privelege level,
        so we have no choice but to disable interrupts for an ECALL.
        """
        self.set_exception(
            m, TrapCauseSelect.EXC_ECALL_FROM_MACH_MODE, mtval=self._pc_to_z, fatal=False)

    def handle_EBREAK(self, m: Module):
        """Handles the EBREAK instruction.

        Note that normally, EBREAK is used from a lower privelege mode, which stores
        the PC of the instruction in the appropriate lower EPC CSR (e.g. SEPC or UEPC).
        This allows interrupts to be handled during the call, because we're in a higher
        privelege level. However, in machine mode, there is no higher privelege level,
        so we have no choice but to disable interrupts for an EBREAK.
        """
        self.set_exception(
            m, TrapCauseSelect.EXC_BREAKPOINT, mtval=self._pc_to_z, fatal=False)