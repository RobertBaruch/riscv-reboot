# Disable pylint's "your name is too short" warning.
# pylint: disable=C0103
# Disable protected access warnings
# pylint: disable=W0212
import sys
from typing import List, Tuple

from nmigen import Array, Signal, Module, Elaboratable, ClockDomain, Mux, Repl
from nmigen import ClockSignal, ResetSignal
from nmigen.build import Platform
from nmigen.asserts import Assert, Assume, Cover, Stable, Past, Initial, AnyConst, Rose, Fell

from alu_card import AluCard
from consts import AluFunc, AluOp, BranchCond, CSRAddr, MemAccessWidth, Opcode
from consts import SystemFunc, TrapCause, MStatus, MInterrupt
from exc_card import ExcCard
from irq_card import IrqCard
from reg_card import RegCard
from sequencer_card import SequencerCard, SequencerState
from shift_card import ShiftCard
from util import main

mode = ""
MRET = 0x30200073
ECALL = 0x00000073
EBREAK = 0x00100073


class FormalCPU(Elaboratable):
    """Formal verification for the CPU."""

    def __init__(self):
        # CPU bus
        self.mcycle_end = Signal()
        self.instr_complete = Signal()
        self.x_bus = Signal(32)
        self.y_bus = Signal(32)
        self.z_bus = Signal(32)
        self.alu_op = Signal(AluOp)
        self.alu_to_z = Signal()
        self.x_reg = Signal(5)
        self.y_reg = Signal(5)
        self.z_reg = Signal(5)
        self.reg_to_x = Signal()
        self.reg_to_y = Signal()
        self.alu_eq = Signal()
        self.alu_lt = Signal()
        self.alu_ltu = Signal()
        self.csr_num = Signal(CSRAddr)
        self.csr_to_x = Signal()
        self.z_to_csr = Signal()
        self.trap = Signal()
        self.exception = Signal()
        self.fatal = Signal()
        self.save_trap_csrs = Signal()
        self.time_irq = Signal()
        self.ext_irq = Signal()

        # Memory bus
        self.mem_rd = Signal()
        self.mem_wr = Signal()
        self.mem_wr_mask = Signal(4)
        self.memaddr = Signal(32)
        self.memdata_rd = Signal(32)
        self.memdata_wr = Signal(32)

        # Formal verification fake CSR read value
        self.csr_rd_data = Signal(32)

        self.regs = RegCard(ext_init=True)
        self.alu = AluCard()
        self.shifter = ShiftCard()
        self.exc = ExcCard(ext_init=True)
        self.irq = IrqCard(ext_init=True)
        self.seq = SequencerCard(ext_init=True, chips=True)

    def elaborate(self, _: Platform) -> Module:
        """Implements a CPU."""
        m = Module()

        m.submodules.alu = self.alu
        m.submodules.regs = self.regs
        m.submodules.shifter = self.shifter
        m.submodules.exc = self.exc
        m.submodules.irq = self.irq
        m.submodules.sequencer = self.seq

        # Faking a CSR card
        _csr_data_x_out = Signal(32)
        m.d.comb += _csr_data_x_out.eq(0)
        with m.If(self.csr_to_x):
            with m.Switch(self.csr_num):
                with m.Case(CSRAddr.MCAUSE, CSRAddr.MTVEC, CSRAddr.MEPC,
                            CSRAddr.MTVAL, CSRAddr.MSTATUS, CSRAddr.MIE,
                            CSRAddr.MIP):
                    m.d.comb += _csr_data_x_out.eq(self.seq.data_x_out)
                with m.Default():
                    m.d.comb += _csr_data_x_out.eq(self.csr_rd_data)

        # Hook up the buses
        m.d.comb += [
            self.alu.data_x.eq(self.x_bus),
            self.alu.data_y.eq(self.y_bus),

            self.exc.data_x_in.eq(self.x_bus),
            self.exc.data_y_in.eq(self.y_bus),
            self.exc.data_z_in.eq(self.z_bus),

            self.shifter.data_x.eq(self.x_bus),
            self.shifter.data_y.eq(self.y_bus),

            self.regs.data_z.eq(self.z_bus),

            self.seq.data_x_in.eq(self.x_bus),
            self.seq.data_y_in.eq(self.y_bus),
            self.seq.data_z_in.eq(self.z_bus),

            self.irq.data_z_in.eq(self.z_bus),

            self.x_bus.eq(self.seq.data_x_out | self.exc.data_x_out |
                          self.irq.data_x_out |
                          self.regs.data_x | _csr_data_x_out),
            self.y_bus.eq(self.seq.data_y_out | self.regs.data_y),
            self.z_bus.eq(self.alu.data_z | self.shifter.data_z |
                          self.seq.data_z_out),
        ]

        # Hook up the control lines
        m.d.comb += [
            self.alu.alu_op.eq(self.alu_op),
            self.shifter.alu_op.eq(self.alu_op),
            self.alu_op.eq(self.seq.alu_op_to_z),

            self.alu_eq.eq(self.alu.alu_eq),
            self.alu_lt.eq(self.alu.alu_lt),
            self.alu_ltu.eq(self.alu.alu_ltu),

            self.seq.alu_eq.eq(self.alu_eq),
            self.seq.alu_lt.eq(self.alu_lt),
            self.seq.alu_ltu.eq(self.alu_ltu),
            self.seq.time_irq.eq(self.time_irq),
            self.seq.ext_irq.eq(self.ext_irq),
            self.trap.eq(self.seq.state.trap),

            self.regs.reg_x.eq(self.x_reg),
            self.regs.reg_y.eq(self.y_reg),
            self.regs.reg_z.eq(self.z_reg),
            self.x_reg.eq(self.seq.x_reg),
            self.y_reg.eq(self.seq.y_reg),
            self.z_reg.eq(self.seq.z_reg),

            self.regs.reg_to_x.eq(self.reg_to_x),
            self.regs.reg_to_y.eq(self.reg_to_y),
            self.reg_to_x.eq(self.seq.reg_to_x),
            self.reg_to_y.eq(self.seq.reg_to_y),

            self.csr_num.eq(self.seq.csr_num),
            self.csr_to_x.eq(self.seq.csr_to_x),
            self.z_to_csr.eq(self.seq.z_to_csr),

            self.exc.csr_num.eq(self.csr_num),
            self.exc.csr_to_x.eq(self.csr_to_x),
            self.exc.z_to_csr.eq(self.z_to_csr),
            self.exc.save_trap_csrs.eq(self.save_trap_csrs),

            self.irq.csr_num.eq(self.csr_num),
            self.irq.csr_to_x.eq(self.csr_to_x),
            self.irq.z_to_csr.eq(self.z_to_csr),
            self.irq.trap.eq(self.trap),
            self.irq.time_irq.eq(self.time_irq),
            self.irq.ext_irq.eq(self.ext_irq),
            self.irq.enter_trap.eq(self.seq.enter_trap),
            self.irq.exit_trap.eq(self.seq.exit_trap),
            self.irq.clear_pend_mti.eq(self.seq.clear_pend_mti),
            self.irq.clear_pend_mei.eq(self.seq.clear_pend_mei),
            self.seq.mei_pend.eq(self.irq.mei_pend),
            self.seq.mti_pend.eq(self.irq.mti_pend),
        ]

        # Clock line
        m.d.comb += self.seq.mcycle_end.eq(self.mcycle_end)

        # Memory bus
        m.d.comb += [
            self.mem_rd.eq(self.seq.mem_rd),
            self.mem_wr.eq(self.seq.mem_wr),
            self.mem_wr_mask.eq(self.seq.mem_wr_mask),
            self.memaddr.eq(self.seq.state.memaddr),
            self.seq.memdata_rd.eq(self.memdata_rd),
            self.memdata_wr.eq(self.seq.state.memdata_wr),
        ]

        # Other sequencer signals
        m.d.comb += [
            self.exception.eq(self.seq.state.exception),
            self.fatal.eq(self.seq.state.fatal),
            self.instr_complete.eq(self.seq.instr_complete),
            self.save_trap_csrs.eq(self.seq.save_trap_csrs),
        ]

        return m

    @classmethod
    def decode_imm(cls, m: Module, instr: Signal) -> Signal:
        """Decodes the immediate value out of the instruction."""
        imm = Signal(32)

        opcode = instr[:7]
        with m.Switch(opcode):
            # Taken straight from the RISC-V Unprivileged ISA spec.
            with m.Case(Opcode.LUI, Opcode.AUIPC):
                # Format U
                m.d.comb += [
                    imm[12:].eq(instr[12:]),
                    imm[0:12].eq(0),
                ]

            with m.Case(Opcode.OP_IMM, Opcode.LOAD):
                # Format I
                m.d.comb += [
                    imm[11:].eq(Repl(instr[31], 32)),
                    imm[5:11].eq(instr[25:]),
                    imm[1:5].eq(instr[21:]),
                    imm[0].eq(instr[20]),
                ]

            with m.Case(Opcode.OP):
                # Format R
                m.d.comb += imm.eq(0)

            with m.Case(Opcode.JAL, Opcode.JALR):
                # Format J
                m.d.comb += [
                    imm[20:].eq(Repl(instr[31], 32)),
                    imm[12:20].eq(instr[12:]),
                    imm[11].eq(instr[20]),
                    imm[5:11].eq(instr[25:]),
                    imm[1:5].eq(instr[21:]),
                    imm[0].eq(0),
                ]

            with m.Case(Opcode.BRANCH):
                # Format B
                m.d.comb += [
                    imm[12:].eq(Repl(instr[31], 32)),
                    imm[11].eq(instr[7]),
                    imm[5:11].eq(instr[25:]),
                    imm[1:5].eq(instr[8:]),
                    imm[0].eq(0),
                ]

            with m.Case(Opcode.STORE):
                # Format S
                m.d.comb += [
                    imm[11:].eq(Repl(instr[31], 32)),
                    imm[5:11].eq(instr[25:]),
                    imm[1:5].eq(instr[8:]),
                    imm[0].eq(instr[7]),
                ]

            with m.Case(Opcode.SYSTEM):
                m.d.comb += [
                    imm[5:].eq(0),
                    imm[0:5].eq(instr[15:]),
                ]

            with m.Default():
                m.d.comb += imm.eq(0)

        return imm

    class Collected:
        """Data collected throughout an instruction."""

        def __init__(self, m: Module, cpu: "FormalCPU"):
            self.cpu = cpu

            attrs = [("uninitialized", "")]

            self.regs_before = Array(
                [Signal(32, reset_less=True, name=f"reg_before_{i:02X}", attrs=attrs) for i in range(32)])
            self.regs_after = Array(
                [Signal(32, reset_less=True, name=f"reg_after_{i:02X}") for i in range(32)])
            self.state_before = SequencerState()
            self.state = cpu.seq.state
            self.instr = self.state_before._instr

            self.mcause = Signal(32)
            self.mepc = Signal(32)
            self.mtval = Signal(32)
            self.mstatus = Signal(32)
            self.mie = Signal(32)
            self.mip = Signal(32)
            self.mcause_before = Signal(32)
            self.mepc_before = Signal(32)
            self.mtval_before = Signal(32)
            self.mstatus_before = Signal(32)
            self.mie_before = Signal(32)
            self.mip_before = Signal(32)

            # Instr decode data
            self.opcode = Signal(7)
            self.rs1 = Signal(5)
            self.rs2 = Signal(5)
            self.rd = Signal(5)
            self.funct3 = Signal(3)
            self.funct7 = Signal(7)
            self.funct12 = Signal(12)
            self.csr_num = Signal(12)
            self.alu_func = Signal(AluFunc)
            self.is_unknown_opcode = Signal()
            self.is_zero_instr = Signal()
            self.is_ones_instr = Signal()
            self.is_misaligned_load = Signal()
            self.is_misaligned_store = Signal()
            self.is_instr_addr_misaligned = Signal()
            self.is_illegal_instr = Signal()

            self.load_store_addr = Signal(32)
            self.branch_target = Signal(32)

            # Captured access data
            self.did_mem_rd = Signal()
            self.did_mem_wr = Signal()
            self.memaddr_accessed = Signal(32)
            self.mem_rd_data = Signal(32)
            self.mem_wr_data = Signal(32)
            self.mem_wr_mask = Signal(4)
            self.csr_accessed = Signal(CSRAddr)
            self.did_csr_rd = Signal()
            self.did_csr_wr = Signal()
            self.csr_rd_data = Signal(32)
            self.csr_wr_data = Signal(32)
            self.did_time_irq = Signal()
            self.did_ext_irq = Signal()
            self.int_return_pc = Signal(32)

            m.d.comb += [
                self.opcode.eq(self.instr[:7]),
                self.rs1.eq(self.instr[15:20]),
                self.rs2.eq(self.instr[20:25]),
                self.rd.eq(self.instr[7:12]),
                self.funct3.eq(self.instr[12:15]),
                self.funct7.eq(self.instr[25:]),
                self.alu_func[3].eq(self.funct7[5]),
                self.alu_func[0:3].eq(self.funct3),
                self.funct12.eq(self.instr[20:]),
                self.csr_num.eq(self.funct12),
            ]
            self.imm = FormalCPU.decode_imm(m, self.instr)
            m.d.comb += self.load_store_addr.eq(self.get_before_reg(m,
                                                                    self.rs1) + self.imm)
            m.d.comb += self.branch_target.eq(self.target_for_branch(m))

        def get_before_reg(self, m: Module, rnum: Signal) -> Signal:
            """Gets the given register from before the instruction executed."""
            reg_content = Signal(32)

            m.d.comb += reg_content.eq(Mux(rnum == 0,
                                           0, self.regs_before[rnum]))

            return reg_content

        def verify_regs_same_except(self, m: Module, rnum: Signal):
            for i in range(1, 32):
                with m.If(i != rnum):
                    m.d.comb += Assert(self.regs_after[i]
                                       == self.regs_before[i])

        def verify_add(self, m: Module, arg: Signal):
            result = Signal(32)
            m.d.comb += result.eq(self.regs_before[self.rs1] + arg)
            m.d.comb += Assert(self.regs_after[self.rd] == result)

        def verify_and(self, m: Module, arg: Signal):
            result = Signal(32)
            m.d.comb += result.eq(self.regs_before[self.rs1] & arg)
            m.d.comb += Assert(self.regs_after[self.rd] == result)

        def verify_or(self, m: Module, arg: Signal):
            result = Signal(32)
            m.d.comb += result.eq(self.regs_before[self.rs1] | arg)
            m.d.comb += Assert(self.regs_after[self.rd] == result)

        def verify_xor(self, m: Module, arg: Signal):
            result = Signal(32)
            m.d.comb += result.eq(self.regs_before[self.rs1] ^ arg)
            m.d.comb += Assert(self.regs_after[self.rd] == result)

        def verify_sub(self, m: Module, arg: Signal):
            result = Signal(32)
            m.d.comb += result.eq(self.regs_before[self.rs1] - arg)
            m.d.comb += Assert(self.regs_after[self.rd] == result)

        def verify_sltu(self, m: Module, arg: Signal):
            result = Signal()
            m.d.comb += result.eq(self.regs_before[self.rs1] < arg)
            m.d.comb += Assert(self.regs_after[self.rd] == result)

        def verify_slt(self, m: Module, arg: Signal):
            result = Signal()
            m.d.comb += result.eq(
                self.regs_before[self.rs1].as_signed() < arg.as_signed())
            m.d.comb += Assert(self.regs_after[self.rd] == result)

        def verify_sll(self, m: Module, arg: Signal):
            shamt = Signal(5)
            result = Signal(32)
            m.d.comb += shamt.eq(arg[:5])
            m.d.comb += result.eq(self.regs_before[self.rs1] << shamt)
            m.d.comb += Assert(self.regs_after[self.rd] == result)

        def verify_srl(self, m: Module, arg: Signal):
            shamt = Signal(5)
            result = Signal(32)
            m.d.comb += shamt.eq(arg[:5])
            m.d.comb += result.eq(self.regs_before[self.rs1] >> shamt)
            m.d.comb += Assert(self.regs_after[self.rd] == result)

        def verify_sra(self, m: Module, arg: Signal):
            shamt = Signal(5)
            result = Signal(32)
            m.d.comb += shamt.eq(arg[:5])
            m.d.comb += result.eq(
                self.regs_before[self.rs1].as_signed() >> shamt)
            m.d.comb += Assert(self.regs_after[self.rd] == result)

        def verify_opcode_OP(self, m: Module):
            if mode != "op":
                return
            arg = self.regs_before[self.rs2]
            state = self.state
            before = self.state_before

            with m.Switch(self.alu_func):
                with m.Case(AluFunc.ADD, AluFunc.SUB, AluFunc.AND, AluFunc.OR,
                            AluFunc.XOR, AluFunc.SLTU, AluFunc.SLT, AluFunc.SLL,
                            AluFunc.SRL, AluFunc.SRA):
                    m.d.comb += Assert(~self.did_mem_rd & ~self.did_mem_wr)
                    m.d.comb += Assert(state._pc == (before._pc+4)[:32])
                    self.verify_regs_same_except(m, self.rd)

                    with m.If(self.rd != 0):
                        with m.Switch(self.alu_func):
                            with m.Case(AluFunc.ADD):
                                self.verify_add(m, arg)
                            with m.Case(AluFunc.SUB):
                                self.verify_sub(m, arg)
                            with m.Case(AluFunc.AND):
                                self.verify_and(m, arg)
                            with m.Case(AluFunc.OR):
                                self.verify_or(m, arg)
                            with m.Case(AluFunc.XOR):
                                self.verify_xor(m, arg)
                            with m.Case(AluFunc.SLTU):
                                self.verify_sltu(m, arg)
                            with m.Case(AluFunc.SLT):
                                self.verify_slt(m, arg)
                            with m.Case(AluFunc.SLL):
                                self.verify_sll(m, arg)
                            with m.Case(AluFunc.SRL):
                                self.verify_srl(m, arg)
                            with m.Case(AluFunc.SRA):
                                self.verify_sra(m, arg)

                with m.Default():
                    m.d.comb += Assert(0)

        def verify_opcode_OP_IMM(self, m: Module):
            if mode != "op_imm":
                return
            arg = self.imm
            state = self.state
            before = self.state_before

            with m.Switch(self.alu_func):
                with m.Case(AluFunc.ADD, AluFunc.SUB, AluFunc.AND, AluFunc.OR,
                            AluFunc.XOR, AluFunc.SLTU, AluFunc.SLT, AluFunc.SLL,
                            AluFunc.SRL, AluFunc.SRA):
                    m.d.comb += Assert(~self.did_mem_rd & ~self.did_mem_wr)
                    m.d.comb += Assert(state._pc == (before._pc+4)[:32])
                    self.verify_regs_same_except(m, self.rd)

                    with m.If(self.rd != 0):
                        with m.Switch(self.alu_func):
                            with m.Case(AluFunc.ADD):
                                self.verify_add(m, arg)
                            with m.Case(AluFunc.SUB):
                                self.verify_sub(m, arg)
                            with m.Case(AluFunc.AND):
                                self.verify_and(m, arg)
                            with m.Case(AluFunc.OR):
                                self.verify_or(m, arg)
                            with m.Case(AluFunc.XOR):
                                self.verify_xor(m, arg)
                            with m.Case(AluFunc.SLTU):
                                self.verify_sltu(m, arg)
                            with m.Case(AluFunc.SLT):
                                self.verify_slt(m, arg)
                            with m.Case(AluFunc.SLL):
                                self.verify_sll(m, arg)
                            with m.Case(AluFunc.SRL):
                                self.verify_srl(m, arg)
                            with m.Case(AluFunc.SRA):
                                self.verify_sra(m, arg)

                with m.Default():
                    m.d.comb += Assert(0)

        def verify_opcode_LUI(self, m: Module):
            if mode != "lui":
                return
            m.d.comb += Assert(self.state._pc ==
                               (self.state_before._pc+4)[:32])
            self.verify_regs_same_except(m, self.rd)
            with m.If(self.rd != 0):
                result = Signal(32)
                m.d.comb += result[12:].eq(self.imm[12:])
                m.d.comb += result[0:12].eq(0)
                m.d.comb += Assert(self.regs_after[self.rd] == result)

        def verify_opcode_AUIPC(self, m: Module):
            if mode != "auipc":
                return
            m.d.comb += Assert(self.state._pc ==
                               (self.state_before._pc+4)[:32])
            self.verify_regs_same_except(m, self.rd)
            with m.If(self.rd != 0):
                offset = Signal(32)
                result = Signal(32)
                m.d.comb += offset[12:].eq(self.imm[12:])
                m.d.comb += offset[0:12].eq(0)
                m.d.comb += result.eq(offset + self.state_before._pc)
                m.d.comb += Assert(self.regs_after[self.rd] == result)

        def verify_opcode_JAL(self, m: Module):
            if mode != "jal":
                return
            new_pc = Signal(32)
            m.d.comb += new_pc.eq(self.state_before._pc + self.imm)
            m.d.comb += Assert(self.state._pc == new_pc)
            self.verify_regs_same_except(m, self.rd)
            m.d.comb += Assert(~self.did_mem_rd & ~self.did_mem_wr)
            with m.If(self.rd != 0):
                result = Signal(32)
                m.d.comb += result.eq(self.state_before._pc+4)
                m.d.comb += Assert(self.regs_after[self.rd] == result)

        def verify_opcode_JALR(self, m: Module):
            if mode != "jalr":
                return
            new_pc = Signal(32)
            m.d.comb += new_pc.eq(
                (self.regs_before[self.rs1] + self.imm) & 0xFFFFFFFE)
            m.d.comb += Assert(self.state._pc == new_pc)
            self.verify_regs_same_except(m, self.rd)
            m.d.comb += Assert(~self.did_mem_rd & ~self.did_mem_wr)
            with m.If(self.rd != 0):
                result = Signal(32)
                m.d.comb += result.eq(self.state_before._pc+4)
                m.d.comb += Assert(self.regs_after[self.rd] == result)

        def target_for_branch(self, m: Module) -> Signal:
            target = Signal(32)
            target_rs1 = Signal(32)
            target_rs2 = Signal(32)
            target_if = Signal(32)
            target_else = Signal(32)

            rs1 = self.regs_before[self.rs1]
            rs2 = self.regs_before[self.rs2]
            m.d.comb += target_rs1.eq(rs1)
            m.d.comb += target_rs2.eq(rs2)
            m.d.comb += target_if.eq(self.state_before._pc + self.imm)
            m.d.comb += target_else.eq(self.state_before._pc + 4)

            with m.Switch(self.opcode):
                with m.Case(Opcode.JAL):
                    m.d.comb += target.eq(self.state_before._pc + self.imm)
                with m.Case(Opcode.JALR):
                    m.d.comb += target.eq((rs1 + self.imm) & 0xFFFFFFFE)
                with m.Case(Opcode.BRANCH):
                    with m.Switch(self.funct3):
                        with m.Case(BranchCond.EQ):
                            m.d.comb += target.eq(Mux(rs1 == rs2,
                                                      target_if, target_else))
                        with m.Case(BranchCond.NE):
                            m.d.comb += target.eq(Mux(rs1 != rs2,
                                                      target_if, target_else))
                        with m.Case(BranchCond.LTU):
                            m.d.comb += target.eq(Mux(rs1 < rs2,
                                                      target_if, target_else))
                        with m.Case(BranchCond.LT):
                            m.d.comb += target.eq(Mux(
                                rs1.as_signed() < rs2.as_signed(), target_if, target_else))
                        with m.Case(BranchCond.GEU):
                            m.d.comb += target.eq(Mux(rs1 >= rs2,
                                                      target_if, target_else))
                        with m.Case(BranchCond.GE):
                            m.d.comb += target.eq(Mux(
                                rs1.as_signed() >= rs2.as_signed(), target_if, target_else))

                        with m.Default():
                            m.d.comb += target.eq(target_else)

                with m.Default():
                    m.d.comb += target.eq(0)

            return target

        def verify_opcode_BRANCH(self, m: Module):
            if mode != "branch":
                return
            with m.Switch(self.funct3):
                with m.Case(BranchCond.EQ, BranchCond.NE, BranchCond.LT, BranchCond.GE,
                            BranchCond.LTU, BranchCond.GEU):
                    pass
                with m.Default():
                    m.d.comb += Assert(0)
            target = self.target_for_branch(m)
            self.verify_regs_same_except(m, 0)
            m.d.comb += Assert(~self.did_mem_rd & ~self.did_mem_wr)
            m.d.comb += Assert(self.state._pc == target)

        def signed_byte_to_32(self, m: Module, b: Signal) -> Signal:
            """Returns the given byte, sign-extended to 32 bits.

            For some reason this results in less output and faster
            formal verification than just using as_signed().
            """
            s = Signal(32)
            m.d.comb += s.eq(-b[7])
            m.d.comb += s[0:8].eq(b)
            return s

        def signed_word_to_32(self, m: Module, b: Signal) -> Signal:
            """Returns the given word, sign-extended to 32 bits."""
            s = Signal(32)
            m.d.comb += s.eq(-b[15])
            m.d.comb += s[0:16].eq(b)
            return s

        def verify_LB(self, m: Module):
            if mode != "lb":
                return
            m.d.comb += Assert(self.state._pc ==
                               (self.state_before._pc+4)[:32])
            self.verify_regs_same_except(m, self.rd)
            rs1 = self.regs_before[self.rs1]
            rd = Signal(32)
            m.d.comb += rd.eq(self.regs_after[self.rd])
            target = (rs1 + self.imm)[:32]
            mem = self.mem_rd_data

            m.d.comb += Assert(self.did_mem_rd)
            m.d.comb += Assert(self.memaddr_accessed == target)
            with m.If(self.rd != 0):
                m.d.comb += Assert(rd[8:] == Repl(rd[7], 24))
                with m.Switch(target[:2]):
                    with m.Case(0):
                        m.d.comb += Assert(rd[0:8] == mem[0:8])
                    with m.Case(1):
                        m.d.comb += Assert(rd[0:8] == mem[8:16])
                    with m.Case(2):
                        m.d.comb += Assert(rd[0:8] == mem[16:24])
                    with m.Case(3):
                        m.d.comb += Assert(rd[0:8] == mem[24:])

        def verify_LBU(self, m: Module):
            if mode != "lbu":
                return
            m.d.comb += Assert(self.state._pc ==
                               (self.state_before._pc+4)[:32])
            self.verify_regs_same_except(m, self.rd)
            rs1 = self.regs_before[self.rs1]
            rd = self.regs_after[self.rd]
            target = (rs1 + self.imm)[:32]
            mem = self.mem_rd_data

            m.d.comb += Assert(self.did_mem_rd)
            m.d.comb += Assert(self.memaddr_accessed == target)
            with m.If(self.rd != 0):
                with m.Switch(target[:2]):
                    with m.Case(0):
                        m.d.comb += Assert(rd == mem[:8])
                    with m.Case(1):
                        m.d.comb += Assert(rd == mem[8:16])
                    with m.Case(2):
                        m.d.comb += Assert(rd == mem[16:24])
                    with m.Case(3):
                        m.d.comb += Assert(rd == mem[24:])

        def verify_LH(self, m: Module):
            if mode != "lh":
                return
            m.d.comb += Assert(self.state._pc ==
                               (self.state_before._pc+4)[:32])
            self.verify_regs_same_except(m, self.rd)
            rs1 = self.regs_before[self.rs1]
            rd = Signal(32)
            m.d.comb += rd.eq(self.regs_after[self.rd])
            target = (rs1 + self.imm)[:32]
            mem = self.mem_rd_data

            m.d.comb += Assert(target[0] == 0)
            m.d.comb += Assert(self.did_mem_rd)
            m.d.comb += Assert(self.memaddr_accessed == target)
            with m.If(self.rd != 0):
                with m.Switch(target[1]):
                    m.d.comb += Assert(rd[16:] == Repl(rd[15], 16))
                    with m.Case(0):
                        m.d.comb += Assert(rd[0:16] == mem[:16])
                    with m.Case(1):
                        m.d.comb += Assert(rd[0:16] == mem[16:])

        def verify_LHU(self, m: Module):
            if mode != "lhu":
                return
            m.d.comb += Assert(self.state._pc ==
                               (self.state_before._pc+4)[:32])
            self.verify_regs_same_except(m, self.rd)
            rs1 = self.regs_before[self.rs1]
            rd = self.regs_after[self.rd]
            target = (rs1 + self.imm)[:32]
            mem = self.mem_rd_data

            m.d.comb += Assert(target[0] == 0)
            m.d.comb += Assert(self.did_mem_rd)
            m.d.comb += Assert(self.memaddr_accessed == target)
            with m.If(self.rd != 0):
                with m.If(target[1] == 0):
                    m.d.comb += Assert(rd == mem[:16])
                with m.Else():
                    m.d.comb += Assert(rd == mem[16:])

        def verify_LW(self, m: Module):
            if mode != "lw":
                return
            m.d.comb += Assert(self.state._pc ==
                               (self.state_before._pc+4)[:32])
            self.verify_regs_same_except(m, self.rd)
            rs1 = self.regs_before[self.rs1]
            rd = self.regs_after[self.rd]
            target = (rs1 + self.imm)[:32]
            mem = self.mem_rd_data

            m.d.comb += Assert(target[0:2] == 0)
            m.d.comb += Assert(self.did_mem_rd)
            m.d.comb += Assert(self.memaddr_accessed == target)
            with m.If(self.rd != 0):
                m.d.comb += Assert(rd == mem)

        def verify_opcode_LOAD(self, m: Module):
            with m.Switch(self.funct3):
                with m.Case(MemAccessWidth.BU):
                    self.verify_LBU(m)
                with m.Case(MemAccessWidth.B):
                    self.verify_LB(m)
                with m.Case(MemAccessWidth.HU):
                    self.verify_LHU(m)
                with m.Case(MemAccessWidth.H):
                    self.verify_LH(m)
                with m.Case(MemAccessWidth.W):
                    self.verify_LW(m)
                with m.Default():
                    m.d.comb += Assert(0)

        def verify_SB(self, m: Module):
            if mode != "sb":
                return
            m.d.comb += Assert(self.state._pc ==
                               (self.state_before._pc+4)[:32])
            self.verify_regs_same_except(m, 0)
            rs1 = self.regs_before[self.rs1]
            rs2 = self.get_before_reg(m, self.rs2)
            target = (rs1 + self.imm)[:32]
            mem = self.mem_wr_data

            m.d.comb += Assert(self.did_mem_wr)
            m.d.comb += Assert(self.memaddr_accessed == target)
            with m.Switch(target[:2]):
                with m.Case(0):
                    m.d.comb += Assert(mem[:8] == rs2[:8])
                    m.d.comb += Assert(self.mem_wr_mask == 0b0001)
                with m.Case(1):
                    m.d.comb += Assert(mem[8:16] == rs2[:8])
                    m.d.comb += Assert(self.mem_wr_mask == 0b0010)
                with m.Case(2):
                    m.d.comb += Assert(mem[16:24] == rs2[:8])
                    m.d.comb += Assert(self.mem_wr_mask == 0b0100)
                with m.Case(3):
                    m.d.comb += Assert(mem[24:] == rs2[:8])
                    m.d.comb += Assert(self.mem_wr_mask == 0b1000)

        def verify_SH(self, m: Module):
            if mode != "sh":
                return
            m.d.comb += Assert(self.state._pc ==
                               (self.state_before._pc+4)[:32])
            self.verify_regs_same_except(m, 0)
            rs1 = self.regs_before[self.rs1]
            rs2 = self.get_before_reg(m, self.rs2)
            target = (rs1 + self.imm)[:32]
            mem = self.mem_wr_data

            m.d.comb += Assert(self.did_mem_wr)
            m.d.comb += Assert(self.memaddr_accessed == target)
            m.d.comb += Assert(target[0] == 0)
            with m.If(target[1] == 0):
                m.d.comb += Assert(mem[:16] == rs2[:16])
                m.d.comb += Assert(self.mem_wr_mask == 0b0011)
            with m.Else():
                m.d.comb += Assert(mem[16:] == rs2[:16])
                m.d.comb += Assert(self.mem_wr_mask == 0b1100)

        def verify_SW(self, m: Module):
            if mode != "sw":
                return
            m.d.comb += Assert(self.state._pc ==
                               (self.state_before._pc+4)[:32])
            self.verify_regs_same_except(m, 0)
            rs1 = self.regs_before[self.rs1]
            rs2 = self.get_before_reg(m, self.rs2)
            target = (rs1 + self.imm)[:32]
            mem = self.mem_wr_data

            m.d.comb += Assert(self.did_mem_wr)
            m.d.comb += Assert(self.memaddr_accessed == target)
            m.d.comb += Assert(target[0:2] == 0)
            m.d.comb += Assert(mem == rs2)
            m.d.comb += Assert(self.mem_wr_mask == 0b1111)

        def verify_opcode_STORE(self, m: Module):
            with m.Switch(self.funct3):
                with m.Case(MemAccessWidth.B):
                    self.verify_SB(m)
                with m.Case(MemAccessWidth.H):
                    self.verify_SH(m)
                with m.Case(MemAccessWidth.W):
                    self.verify_SW(m)
                with m.Default():
                    m.d.comb += Assert(0)

        def verify_seq_csr_written(self, m: Module):
            with m.Switch(self.csr_accessed):
                with m.Case(CSRAddr.MCAUSE):
                    m.d.comb += Assert(self.csr_wr_data == self.mcause)
                with m.Case(CSRAddr.MTVEC):
                    m.d.comb += Assert(self.csr_wr_data == self.state._mtvec)
                with m.Case(CSRAddr.MEPC):
                    m.d.comb += Assert(self.csr_wr_data == self.mepc)
                with m.Case(CSRAddr.MTVAL):
                    m.d.comb += Assert(self.csr_wr_data == self.mtval)
                with m.Case(CSRAddr.MSTATUS):
                    m.d.comb += Assert(self.csr_wr_data == self.mstatus)
                with m.Case(CSRAddr.MIE):
                    m.d.comb += Assert(self.csr_wr_data == self.mie)
                with m.Case(CSRAddr.MIP):
                    # Pending machine interrupts are not writable.
                    m.d.comb += Assert(
                        (self.csr_wr_data & 0xFFFFF777) == (self.mip & 0xFFFFF777))
                    m.d.comb += Assert(((self.mip_before ^
                                         self.mip) & 0x00000888) == 0)

        def verify_seq_csr_read(self, m: Module):
            with m.Switch(self.csr_accessed):
                with m.Case(CSRAddr.MCAUSE):
                    m.d.comb += Assert(self.csr_rd_data ==
                                       self.mcause_before)
                with m.Case(CSRAddr.MTVEC):
                    m.d.comb += Assert(self.csr_rd_data ==
                                       self.state_before._mtvec)
                with m.Case(CSRAddr.MEPC):
                    m.d.comb += Assert(self.csr_rd_data ==
                                       self.mepc_before)
                with m.Case(CSRAddr.MTVAL):
                    m.d.comb += Assert(self.csr_rd_data ==
                                       self.mtval_before)
                with m.Case(CSRAddr.MSTATUS):
                    m.d.comb += Assert(self.csr_rd_data ==
                                       self.mstatus_before)
                with m.Case(CSRAddr.MIE):
                    m.d.comb += Assert(self.csr_rd_data ==
                                       self.mie_before)
                with m.Case(CSRAddr.MIP):
                    m.d.comb += Assert(self.csr_rd_data ==
                                       self.mip_before)

        def verify_CSRRW(self, m: Module):
            if mode != "csr":
                return
            m.d.comb += Assert(self.state._pc ==
                               (self.state_before._pc+4)[:32])
            m.d.comb += Assert(~self.did_mem_rd & ~self.did_mem_wr)
            self.verify_regs_same_except(m, self.rd)
            rs1 = self.regs_before[self.rs1]
            rd = self.regs_after[self.rd]

            m.d.comb += Assert(self.did_csr_rd == (self.rd != 0))
            m.d.comb += Assert(self.did_csr_wr)
            m.d.comb += Assert(self.csr_accessed == self.csr_num)
            with m.If(self.did_csr_rd):
                m.d.comb += Assert(rd == self.csr_rd_data)
                self.verify_seq_csr_read(m)
            m.d.comb += Assert(self.csr_wr_data == rs1)
            self.verify_seq_csr_written(m)

        def verify_CSRRWI(self, m: Module):
            if mode != "csr":
                return
            m.d.comb += Assert(self.state._pc ==
                               (self.state_before._pc+4)[:32])
            m.d.comb += Assert(~self.did_mem_rd & ~self.did_mem_wr)
            self.verify_regs_same_except(m, self.rd)
            rd = self.regs_after[self.rd]

            m.d.comb += Assert(self.did_csr_rd == (self.rd != 0))
            m.d.comb += Assert(self.did_csr_wr)
            m.d.comb += Assert(self.csr_accessed == self.csr_num)
            with m.If(self.did_csr_rd):
                m.d.comb += Assert(rd == self.csr_rd_data)
                self.verify_seq_csr_read(m)
            m.d.comb += Assert(self.csr_wr_data == self.imm)
            self.verify_seq_csr_written(m)

        def verify_CSRRS(self, m: Module):
            if mode != "csr":
                return
            m.d.comb += Assert(self.state._pc ==
                               (self.state_before._pc+4)[:32])
            m.d.comb += Assert(~self.did_mem_rd & ~self.did_mem_wr)
            self.verify_regs_same_except(m, self.rd)
            rs1 = self.regs_before[self.rs1]
            rd = self.regs_after[self.rd]

            m.d.comb += Assert(self.did_csr_wr == (self.rs1 != 0))
            m.d.comb += Assert(self.did_csr_rd)
            m.d.comb += Assert(self.csr_accessed == self.csr_num)
            self.verify_seq_csr_read(m)
            with m.If(self.rd != 0):
                m.d.comb += Assert(rd == self.csr_rd_data)
            with m.If(self.rs1 != 0):
                m.d.comb += Assert(self.csr_wr_data ==
                                   (rs1 | self.csr_rd_data))
                self.verify_seq_csr_written(m)

        def verify_CSRRSI(self, m: Module):
            if mode != "csr":
                return
            m.d.comb += Assert(self.state._pc ==
                               (self.state_before._pc+4)[:32])
            m.d.comb += Assert(~self.did_mem_rd & ~self.did_mem_wr)
            self.verify_regs_same_except(m, self.rd)
            rd = self.regs_after[self.rd]

            m.d.comb += Assert(self.did_csr_wr == (self.imm != 0))
            m.d.comb += Assert(self.did_csr_rd)
            m.d.comb += Assert(self.csr_accessed == self.csr_num)
            self.verify_seq_csr_read(m)
            with m.If(self.rd != 0):
                m.d.comb += Assert(rd == self.csr_rd_data)
            with m.If(self.imm != 0):
                m.d.comb += Assert(self.csr_wr_data ==
                                   (self.imm | self.csr_rd_data))
                self.verify_seq_csr_written(m)

        def verify_CSRRC(self, m: Module):
            if mode != "csr":
                return
            m.d.comb += Assert(self.state._pc ==
                               (self.state_before._pc+4)[:32])
            m.d.comb += Assert(~self.did_mem_rd & ~self.did_mem_wr)
            self.verify_regs_same_except(m, self.rd)
            rs1 = self.regs_before[self.rs1]
            rd = self.regs_after[self.rd]

            m.d.comb += Assert(self.did_csr_wr == (self.rs1 != 0))
            m.d.comb += Assert(self.did_csr_rd)
            m.d.comb += Assert(self.csr_accessed == self.csr_num)
            self.verify_seq_csr_read(m)
            with m.If(self.rd != 0):
                m.d.comb += Assert(rd == self.csr_rd_data)
            with m.If(self.rs1 != 0):
                m.d.comb += Assert(self.csr_wr_data ==
                                   (self.csr_rd_data & ~rs1))
                self.verify_seq_csr_written(m)

        def verify_CSRRCI(self, m: Module):
            if mode != "csr":
                return
            m.d.comb += Assert(self.state._pc ==
                               (self.state_before._pc+4)[:32])
            m.d.comb += Assert(~self.did_mem_rd & ~self.did_mem_wr)
            self.verify_regs_same_except(m, self.rd)
            rd = self.regs_after[self.rd]

            m.d.comb += Assert(self.did_csr_wr == (self.imm != 0))
            m.d.comb += Assert(self.did_csr_rd)
            m.d.comb += Assert(self.csr_accessed == self.csr_num)
            self.verify_seq_csr_read(m)
            with m.If(self.rd != 0):
                m.d.comb += Assert(rd == self.csr_rd_data)
            with m.If(self.imm != 0):
                m.d.comb += Assert(self.csr_wr_data ==
                                   (self.csr_rd_data & ~self.imm))
                self.verify_seq_csr_written(m)

        def verify_PRIV(self, m: Module):
            if mode != "csr":
                return
            base = self.state._mtvec & 0xFFFFFFFC

            # Anything other than an MRET will be illegal
            with m.If(self.instr == MRET):
                m.d.comb += [
                    Assert(self.state._pc == self.mepc_before),
                    Assert(self.mstatus[MStatus.MIE] ==
                           self.mstatus_before[MStatus.MPIE]),
                    Assert(self.mstatus[MStatus.MPIE] == 1),
                ]
            with m.Elif(self.instr == ECALL):
                m.d.comb += [
                    Assert(self.mepc == (self.state_before._pc+4)[:32]),
                    Assert(self.mcause ==
                           TrapCause.EXC_ECALL_FROM_MACH_MODE),
                    Assert(self.state._pc == base),
                ]
            with m.Elif(self.instr == EBREAK):
                m.d.comb += [
                    Assert(self.mepc == (self.state_before._pc+4)[:32]),
                    Assert(self.mcause == TrapCause.EXC_BREAKPOINT),
                    Assert(self.state._pc == base),
                ]
            with m.Else():
                m.d.comb += Assert(0)

        def verify_opcode_SYSTEM(self, m: Module):
            with m.Switch(self.funct3):
                with m.Case(SystemFunc.CSRRW):
                    self.verify_CSRRW(m)
                with m.Case(SystemFunc.CSRRWI):
                    self.verify_CSRRWI(m)
                with m.Case(SystemFunc.CSRRS):
                    self.verify_CSRRS(m)
                with m.Case(SystemFunc.CSRRSI):
                    self.verify_CSRRSI(m)
                with m.Case(SystemFunc.CSRRC):
                    self.verify_CSRRC(m)
                with m.Case(SystemFunc.CSRRCI):
                    self.verify_CSRRCI(m)
                with m.Case(SystemFunc.PRIV):
                    self.verify_PRIV(m)
                with m.Default():
                    m.d.comb += Assert(0)

        def verify_instr(self, m: Module):
            with m.Switch(self.opcode):
                with m.Case(Opcode.OP):
                    self.verify_opcode_OP(m)
                with m.Case(Opcode.OP_IMM):
                    self.verify_opcode_OP_IMM(m)
                with m.Case(Opcode.LUI):
                    self.verify_opcode_LUI(m)
                with m.Case(Opcode.AUIPC):
                    self.verify_opcode_AUIPC(m)
                with m.Case(Opcode.JAL):
                    self.verify_opcode_JAL(m)
                with m.Case(Opcode.JALR):
                    self.verify_opcode_JALR(m)
                with m.Case(Opcode.BRANCH):
                    self.verify_opcode_BRANCH(m)
                with m.Case(Opcode.LOAD):
                    self.verify_opcode_LOAD(m)
                with m.Case(Opcode.STORE):
                    self.verify_opcode_STORE(m)
                with m.Case(Opcode.SYSTEM):
                    self.verify_opcode_SYSTEM(m)
                with m.Default():
                    m.d.comb += Assert(0)

        def check_unknown_opcode(self, m: Module):
            with m.Switch(self.opcode):
                with m.Case(Opcode.JAL, Opcode.JALR, Opcode.LUI, Opcode.AUIPC):
                    m.d.comb += self.is_unknown_opcode.eq(0)

                with m.Case(Opcode.LOAD):
                    with m.Switch(self.funct3):
                        with m.Case(MemAccessWidth.B, MemAccessWidth.BU,
                                    MemAccessWidth.H, MemAccessWidth.HU,
                                    MemAccessWidth.W):
                            m.d.comb += self.is_unknown_opcode.eq(0)
                        with m.Default():
                            m.d.comb += self.is_unknown_opcode.eq(1)

                with m.Case(Opcode.STORE):
                    with m.Switch(self.funct3):
                        with m.Case(MemAccessWidth.B, MemAccessWidth.H, MemAccessWidth.W):
                            m.d.comb += self.is_unknown_opcode.eq(0)
                        with m.Default():
                            m.d.comb += self.is_unknown_opcode.eq(1)

                with m.Case(Opcode.BRANCH):
                    with m.Switch(self.funct3):
                        with m.Case(BranchCond.EQ, BranchCond.NE, BranchCond.LT, BranchCond.GE,
                                    BranchCond.LTU, BranchCond.GEU):
                            m.d.comb += self.is_unknown_opcode.eq(0)
                        with m.Default():
                            m.d.comb += self.is_unknown_opcode.eq(1)

                with m.Case(Opcode.OP_IMM, Opcode.OP):
                    with m.Switch(self.alu_func):
                        with m.Case(AluFunc.ADD, AluFunc.AND, AluFunc.SUB, AluFunc.SLL,
                                    AluFunc.SLT, AluFunc.SLTU, AluFunc.XOR, AluFunc.SRL,
                                    AluFunc.SRA, AluFunc.OR):
                            m.d.comb += self.is_unknown_opcode.eq(0)
                        with m.Default():
                            m.d.comb += self.is_unknown_opcode.eq(1)

                with m.Case(Opcode.SYSTEM):
                    with m.Switch(self.funct3):
                        with m.Case(SystemFunc.CSRRW, SystemFunc.CSRRWI, SystemFunc.CSRRS,
                                    SystemFunc.CSRRSI, SystemFunc.CSRRC, SystemFunc.CSRRCI):
                            m.d.comb += self.is_unknown_opcode.eq(0)
                        with m.Default():
                            with m.Switch(self.instr):
                                with m.Case(MRET, ECALL, EBREAK):
                                    m.d.comb += self.is_unknown_opcode.eq(0)
                                with m.Default():
                                    m.d.comb += self.is_unknown_opcode.eq(1)

                with m.Default():
                    m.d.comb += self.is_unknown_opcode.eq(1)

        def check_invalid_instruction(self, m: Module):
            m.d.comb += self.is_zero_instr.eq(self.instr[:16] == 0)
            m.d.comb += self.is_ones_instr.eq(self.instr == 0xFFFFFFFF)

        def check_misaligned_load(self, m: Module):
            addr = self.load_store_addr

            m.d.comb += self.is_misaligned_load.eq(0)
            with m.If(self.opcode == Opcode.LOAD):
                with m.Switch(self.funct3):
                    with m.Case(MemAccessWidth.H, MemAccessWidth.HU):
                        m.d.comb += self.is_misaligned_load.eq(addr[0] != 0)
                    with m.Case(MemAccessWidth.W):
                        m.d.comb += self.is_misaligned_load.eq(addr[0:2] != 0)

        def check_misaligned_store(self, m: Module):
            addr = self.load_store_addr

            m.d.comb += self.is_misaligned_store.eq(0)
            with m.If(self.opcode == Opcode.STORE):
                with m.Switch(self.funct3):
                    with m.Case(MemAccessWidth.H, MemAccessWidth.HU):
                        m.d.comb += self.is_misaligned_store.eq(addr[0] != 0)
                    with m.Case(MemAccessWidth.W):
                        m.d.comb += self.is_misaligned_store.eq(addr[0:2] != 0)

        def check_misaligned_branch(self, m: Module):
            m.d.comb += self.is_instr_addr_misaligned.eq(0)
            with m.If((self.opcode == Opcode.BRANCH) |
                      (self.opcode == Opcode.JAL) |
                      (self.opcode == Opcode.JALR)):
                m.d.comb += self.is_instr_addr_misaligned.eq(
                    self.branch_target[0:2] != 0)

        def check_illegal_instr(self, m: Module):
            m.d.comb += self.is_illegal_instr.eq(0)
            with m.If((self.instr[:16] == 0) | (self.instr == 0xFFFFFFFF)):
                m.d.comb += self.is_illegal_instr.eq(1)
            with m.Else():
                with m.Switch(self.opcode):
                    with m.Case(Opcode.LOAD, Opcode.STORE, Opcode.OP, Opcode.OP_IMM,
                                Opcode.LUI, Opcode.AUIPC, Opcode.JAL, Opcode.JALR,
                                Opcode.BRANCH, Opcode.SYSTEM):
                        m.d.comb += self.is_illegal_instr.eq(0)
                    with m.Default():
                        m.d.comb += self.is_illegal_instr.eq(1)

        def verify_fatal(self, m: Module):
            """Verification for fatal exceptions.

            One of these conditions has to be true:
            * The low 16 bits of an instruction are 0
            * All bits of an instruction are 1
            * A misaligned load was requested
            * A misaligned store was requested
            * The target of a branch or unconditional jump is misaligned
            * An unknown instruction was requested
            """
            cpu = self.cpu

            is_fatal = (self.is_zero_instr | self.is_ones_instr |
                        self.is_misaligned_load | self.is_misaligned_store |
                        self.is_instr_addr_misaligned | self.is_unknown_opcode |
                        self.is_illegal_instr)

            # This signal goes high on the last machine cycle of phase 0 for
            # an fatal-causing instruction.
            mcycle_end_with_fatal = Signal()
            m.d.comb += mcycle_end_with_fatal.eq(is_fatal & cpu.seq.mcycle_end)

            # In all cases we know about the fatal on phase 0. Most instructions
            # trap in phase 1. However, BRANCH traps in phase 2, and invalid
            # instructions trap in phase 0
            with m.If(Rose(mcycle_end_with_fatal, clocks=1) &
                      ((self.opcode == Opcode.INSTR_48A) | (self.instr[:16] == 0) |
                       (self.instr == 0xFFFFFFFF) | (self.opcode == Opcode.SYSTEM))):
                m.d.comb += Assert(cpu.seq.state.trap)
                m.d.comb += Assert(cpu.seq.state.fatal)
            with m.If(Rose(mcycle_end_with_fatal, clocks=7) & (self.opcode != Opcode.BRANCH)):
                m.d.comb += Assert(cpu.seq.state.trap)
                m.d.comb += Assert(cpu.seq.state.fatal)
            with m.Elif(Rose(mcycle_end_with_fatal, clocks=13) & (self.opcode == Opcode.BRANCH)):
                m.d.comb += Assert(cpu.seq.state.trap)
                m.d.comb += Assert(cpu.seq.state.fatal)

            # Trap can also rise on IRQ and non-fatal exceptions.
            with m.If(cpu.fatal & Rose(cpu.seq.state.trap, clocks=6)):
                # Exceptions load mepc with the PC of the instruction that caused the problem.
                m.d.comb += Assert(self.mepc == self.state_before._pc)

                with m.If(self.is_instr_addr_misaligned):
                    m.d.comb += Assert(self.mcause ==
                                       TrapCause.EXC_INSTR_ADDR_MISALIGN)
                    with m.Switch(self.opcode):
                        with m.Case(Opcode.JAL, Opcode.JALR, Opcode.BRANCH):
                            m.d.comb += Assert(self.mtval ==
                                               self.branch_target)
                        with m.Default():
                            m.d.comb += Assert(self.mtval ==
                                               self.state_before._pc)

                with m.Elif(self.is_zero_instr | self.is_ones_instr |
                            self.is_unknown_opcode | self.is_illegal_instr):
                    m.d.comb += Assert(self.mcause ==
                                       TrapCause.EXC_ILLEGAL_INSTR)
                    m.d.comb += Assert(self.mtval == self.instr)

                with m.Elif(self.is_misaligned_load):
                    m.d.comb += Assert(self.mcause ==
                                       TrapCause.EXC_LOAD_ADDR_MISALIGN)
                    m.d.comb += Assert(self.mtval == self.load_store_addr)

                with m.Elif(self.is_misaligned_store):
                    m.d.comb += Assert(self.mcause ==
                                       TrapCause.EXC_STORE_AMO_ADDR_MISALIGN)
                    m.d.comb += Assert(self.mtval == self.load_store_addr)

        def verify_irq(self, m: Module):
            """Verification for interrupts after trap goes low."""
            with m.If(self.did_ext_irq):
                m.d.comb += Assert(self.mcause ==
                                   TrapCause.INT_MACH_EXTERNAL)
            with m.Elif(self.did_time_irq):
                m.d.comb += Assert(self.mcause ==
                                   TrapCause.INT_MACH_TIMER)
            m.d.comb += [
                Assert(self.mepc == self.int_return_pc),
                Assert(self.mstatus[MStatus.MPIE] ==
                       self.mstatus_before[MStatus.MIE]),
                Assert(self.mstatus[MStatus.MIE] == 0),
            ]
            vec_mode = self.state._mtvec[:2]
            base = self.state._mtvec & 0xFFFFFFFC
            with m.If(vec_mode == 0):
                m.d.comb += Assert(self.state._pc == base)
            with m.Elif(vec_mode == 1):
                m.d.comb += Assert(self.state._pc ==
                                   (base + (self.mcause << 2))[:32])

    @ classmethod
    def make_clock(cls, m: Module) -> Tuple[Signal, Signal]:
        """Creates the clock domains and signals."""
        ph1 = ClockDomain("ph1")
        ph2 = ClockDomain("ph2")
        ph2w = ClockDomain("ph2w")
        ph2r = ClockDomain("ph2r")

        m.domains += [ph1, ph2, ph2w, ph2r]

        # Generate the ph1 and ph2 clocks.
        phase_count = Signal(3, reset=0, reset_less=True)
        mcycle_end = Signal()

        m.d.sync += phase_count.eq(phase_count + 1)
        with m.If(phase_count == 5):
            m.d.sync += phase_count.eq(0)

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

        m.d.comb += ph2w.clk.eq(phase_count != 4)
        m.d.comb += ph2r.clk.eq(phase_count != 0)

        m.d.comb += mcycle_end.eq(phase_count == 5)

        return (phase_count, mcycle_end)

    @ classmethod
    def formal(cls) -> Tuple[Module, List[Signal]]:
        """Formal verification for the CPU."""
        m = Module()

        m.submodules.cpu = cpu = FormalCPU()

        phase_count, mcycle_end = FormalCPU.make_clock(m)

        m.d.comb += cpu.mcycle_end.eq(mcycle_end)

        # Collected data throughout an instruction
        data = FormalCPU.Collected(m, cpu)

        mcycle = cpu.seq.state._instr_phase

        # Assume memory and fake CSR data is stable
        with m.If(phase_count > 1):
            m.d.comb += Assume(Stable(cpu.memdata_rd))
            m.d.comb += Assume(Stable(cpu.csr_rd_data))

        bef = data.state_before
        state = cpu.seq.state
        exc = cpu.exc
        irq = cpu.irq
        data.check_unknown_opcode(m)
        data.check_invalid_instruction(m)
        data.check_misaligned_load(m)
        data.check_misaligned_store(m)
        data.check_misaligned_branch(m)
        data.check_illegal_instr(m)

        # Take a snapshot of the state just before latching the instruction.
        with m.If((mcycle == 0) & (phase_count == 1) & ~cpu.fatal):
            m.d.ph2 += bef._pc.eq(state._pc)
            m.d.ph2 += bef._instr_phase.eq(state._instr_phase)
            m.d.ph2 += bef._instr.eq(state._instr)
            m.d.ph2 += bef._stored_alu_eq.eq(state._stored_alu_eq)
            m.d.ph2 += bef._stored_alu_lt.eq(state._stored_alu_lt)
            m.d.ph2 += bef._stored_alu_ltu.eq(state._stored_alu_ltu)
            m.d.ph2 += bef.memaddr.eq(state.memaddr)
            m.d.ph2 += bef.memdata_wr.eq(state.memdata_wr)
            m.d.ph2 += bef._tmp.eq(state._tmp)
            m.d.ph2 += bef.reg_page.eq(state.reg_page)
            m.d.ph2 += bef.trap.eq(state.trap)
            m.d.ph2 += bef.exception.eq(state.exception)
            m.d.ph2 += bef._mtvec.eq(state._mtvec)
            m.d.ph2 += data.mcause_before.eq(exc._mcause)
            m.d.ph2 += data.mepc_before.eq(exc._mepc)
            m.d.ph2 += data.mtval_before.eq(exc._mtval)
            m.d.ph2 += data.mstatus_before.eq(irq._mstatus)
            m.d.ph2 += data.mie_before.eq(irq._mie)
            m.d.ph2 += data.mip_before.eq(irq._mip)
            for i in range(1, 32):
                m.d.ph2 += data.regs_before[i].eq(cpu.regs._x_bank._mem[i])
            m.d.ph2 += data.regs_before[0].eq(0)
            m.d.ph2 += [
                data.did_mem_rd.eq(0),
                data.did_mem_wr.eq(0),
                data.did_csr_rd.eq(0),  # These can get overridden later
                data.did_csr_wr.eq(0),
            ]

        for i in range(1, 32):
            m.d.comb += data.regs_after[i].eq(cpu.regs._x_bank._mem[i])
        m.d.comb += data.regs_after[0].eq(0)

        m.d.comb += data.mcause.eq(exc._mcause)
        m.d.comb += data.mepc.eq(exc._mepc)
        m.d.comb += data.mtval.eq(exc._mtval)
        m.d.comb += data.mstatus.eq(irq._mstatus)
        m.d.comb += data.mie.eq(irq._mie)
        m.d.comb += data.mip.eq(irq._mip)

        # If we read or write memory outside the first machine cycle, save it.
        # But complain if we do more than only one read or one write.
        m.d.comb += Assert(~(data.did_mem_rd & data.did_mem_wr))

        with m.If((mcycle > 0) & cpu.mem_rd & (phase_count == 1) & ~cpu.fatal):
            m.d.comb += Assert(~data.did_mem_rd & ~data.did_mem_wr)
            m.d.ph2 += data.did_mem_rd.eq(1)
            m.d.ph2 += data.memaddr_accessed.eq(cpu.memaddr)
            m.d.ph2 += data.mem_rd_data.eq(cpu.memdata_rd)

        # Yes, phase 1. The fatal signal goes high on phase 2,
        # because we know the memory we're going to write to on
        # phase 1. The memory may not actually get written until
        # phase 4, but the address and data are already set up now.
        with m.Elif((mcycle > 0) & cpu.mem_wr & (phase_count == 1) & ~cpu.fatal):
            m.d.comb += Assert(~data.did_mem_rd & ~data.did_mem_wr)
            m.d.ph2 += data.did_mem_wr.eq(1)
            m.d.ph2 += data.memaddr_accessed.eq(cpu.memaddr)
            m.d.ph2 += data.mem_wr_data.eq(cpu.memdata_wr)
            m.d.ph2 += data.mem_wr_mask.eq(cpu.mem_wr_mask)

        with m.If((mcycle == 0) & (cpu.z_to_csr | cpu.csr_to_x) & (phase_count == 1) & ~cpu.fatal):
            m.d.ph2 += [
                data.did_csr_rd.eq(cpu.csr_to_x),
                data.did_csr_wr.eq(cpu.z_to_csr),
                data.csr_accessed.eq(cpu.csr_num),
                data.csr_rd_data.eq(cpu.x_bus),
                data.csr_wr_data.eq(cpu.z_bus),
            ]

        # Covers
        m.d.comb += Cover(Past(cpu.fatal, clocks=12)
                          & (data.opcode == Opcode.LOAD))
        # m.d.comb += Cover(Past(cpu.time_irq, 18) &
        #                   Past(cpu.seq.instr_complete, 18) &
        #                   ~Past(cpu.seq.state.exception, 18))
        # m.d.comb += Cover(Past(cpu.seq.state._instr, 18) == ECALL)

        # Asserts and Assumptions based on which instructions we're verifying.
        cycles = {
            "op": 1,
            "op_imm": 1,
            "lui": 1,
            "auipc": 1,
            "jal": 2,
            "jalr": 2,
            "branch": 2,
            "csr": 2,
            "lb": 3,
            "lbu": 3,
            "lh": 3,
            "lhu": 3,
            "lw": 3,
            "sb": 3,
            "sh": 3,
            "sw": 3,
        }
        opcodes = {
            "op": Opcode.OP,
            "op_imm": Opcode.OP_IMM,
            "lui": Opcode.LUI,
            "auipc": Opcode.AUIPC,
            "jal": Opcode.JAL,
            "jalr": Opcode.JALR,
            "branch": Opcode.BRANCH,
            "csr": Opcode.SYSTEM,
            "lb": Opcode.LOAD,
            "lbu": Opcode.LOAD,
            "lh": Opcode.LOAD,
            "lhu": Opcode.LOAD,
            "lw": Opcode.LOAD,
            "sb": Opcode.STORE,
            "sh": Opcode.STORE,
            "sw": Opcode.STORE,
        }
        widths = {
            "lb": MemAccessWidth.B,
            "lbu": MemAccessWidth.BU,
            "lh": MemAccessWidth.H,
            "lhu": MemAccessWidth.HU,
            "lw": MemAccessWidth.W,
            "sb": MemAccessWidth.B,
            "sh": MemAccessWidth.H,
            "sw": MemAccessWidth.W,
        }
        if mode in cycles:
            with m.If(phase_count == 5):
                m.d.comb += Assume(cpu.instr_complete ==
                                   (mcycle == cycles[mode]-1))
            m.d.comb += Assert(mcycle < cycles[mode])

        if mode in widths:
            with m.If((mcycle == 0) & (phase_count == 2)):
                m.d.comb += Assume(data.funct3 == widths[mode])

        if mode in opcodes:
            m.d.comb += Assert(~cpu.fatal)
            m.d.comb += Assert(~cpu.seq.state.trap)
            m.d.comb += Assume(~cpu.seq.state.exception)
            m.d.comb += Assume(~cpu.time_irq)
            m.d.comb += Assume(~cpu.ext_irq)
            m.d.comb += Assume(~cpu.seq.mei_pend)
            m.d.comb += Assume(~cpu.seq.mti_pend)
            m.d.comb += Assume(~cpu.irq._mip[MInterrupt.MTI])
            m.d.comb += Assume(~cpu.irq._mip[MInterrupt.MEI])
            with m.If((mcycle == 0) & (phase_count == 2)):
                m.d.comb += Assume(data.opcode == opcodes[mode])
                m.d.comb += Assume((data.instr != ECALL) &
                                   (data.instr != EBREAK))
                # These REALLY speed up the process.
                if mode in ("op", "op_imm", "jalr", "branch", "sh", "sw", "sb", "lw",
                            "lh", "lhu", "lb", "lbu", "csr"):
                    if (opcodes[mode] == "csr"):
                        with m.If((data.funct3 == SystemFunc.CSRRW) |
                                  (data.funct3 == SystemFunc.CSRRS) |
                                  (data.funct3 == SystemFunc.CSRRC)):
                            m.d.comb += Assume(data.rs1 <= 1)
                    else:
                        m.d.comb += Assume(data.rs1 <= 1)
                if mode in ("op", "branch", "sh", "sw", "sb"):
                    m.d.comb += Assume(data.rs2 <= 2)
                if mode in ("jal", "jalr", "lw", "lh", "lhu", "lb", "lbu"):
                    m.d.comb += Assume(data.rd <= 3)
            # Formal verification just after we've completed an instruction.
            with m.If(Past(cpu.instr_complete)):
                # Check everything but the LSb because of JALR.
                m.d.comb += Assert(cpu.seq.state._pc[1:] == cpu.memaddr[1:])
                data.verify_instr(m)

        if mode == "ecall":
            m.d.comb += Assume(~cpu.time_irq)
            m.d.comb += Assume(~cpu.ext_irq)
            m.d.comb += Assume(~cpu.seq.mei_pend)
            m.d.comb += Assume(~cpu.seq.mti_pend)
            m.d.comb += Assume(~cpu.irq._mip[MInterrupt.MTI])
            m.d.comb += Assume(~cpu.irq._mip[MInterrupt.MEI])
            with m.If((mcycle == 0) & (phase_count == 2)):
                m.d.comb += Assume((data.instr == ECALL) |
                                   (data.instr == EBREAK))
            with m.If(Fell(cpu.seq.state.trap)):
                m.d.comb += Assert(cpu.seq.state._pc == cpu.memaddr)
                data.verify_instr(m)

        if mode.startswith("fatal"):
            # This will only verify fatal conditions.
            m.d.comb += Assume(~cpu.time_irq)
            m.d.comb += Assume(~cpu.ext_irq)
            m.d.comb += Assume(~cpu.seq.mei_pend)
            m.d.comb += Assume(~cpu.seq.mti_pend)
            m.d.comb += Assume(~cpu.irq._mip[MInterrupt.MTI])
            m.d.comb += Assume(~cpu.irq._mip[MInterrupt.MEI])

            if mode == "fatal1":
                with m.If((mcycle == 0) & (phase_count == 2)):
                    m.d.comb += Assume(data.opcode == Opcode.LOAD)
                with m.If((mcycle > 0) | (phase_count == 2)):
                    m.d.comb += Assume(data.opcode == Opcode.LOAD)
                with m.If((mcycle == 0) & (Past(mcycle) == 1)):
                    m.d.comb += Assume(Rose(cpu.trap))
                with m.If(cpu.trap):
                    m.d.comb += Assume(data.is_unknown_opcode)

            elif mode == "fatal2":
                with m.If((mcycle == 0) & (phase_count == 2)):
                    m.d.comb += Assume((data.opcode == Opcode.JAL)
                                       | (data.opcode == Opcode.JALR))

            elif mode == "fatal3":
                with m.If((mcycle == 0) & (phase_count == 2)):
                    m.d.comb += Assume((data.opcode == Opcode.OP) |
                                       (data.opcode == Opcode.OP_IMM) |
                                       (data.opcode == Opcode.STORE))

            elif mode == "fatal4":
                m.d.comb += Assume(mcycle == 0)
                with m.If((mcycle == 0) & (phase_count == 2)):
                    m.d.comb += Assume((data.opcode == Opcode.INSTR_48A) |
                                       (data.instr[:16] == 0) |
                                       (data.instr == 0xFFFFFFFF) |
                                       (data.opcode == Opcode.SYSTEM))
                with m.If((mcycle == 0) & Past(cpu.mcycle_end)):
                    m.d.comb += Assume(Rose(cpu.trap))
                    m.d.comb += Assume(cpu.fatal)
                with m.If(cpu.trap):
                    m.d.comb += Assume(data.is_illegal_instr |
                                       data.is_unknown_opcode)

            with m.If((mcycle == 0) & (phase_count == 2)):
                # These REALLY speed up the process.
                with m.Switch(data.opcode):
                    with m.Case(Opcode.OP, Opcode.BRANCH, Opcode.STORE):
                        m.d.comb += Assume(data.rs1 <= 1)
                        m.d.comb += Assume(data.rs2 <= 2)

                    with m.Case(Opcode.OP_IMM, Opcode.JALR):
                        m.d.comb += Assume(data.rs1 <= 1)

                    with m.Case(Opcode.LOAD):
                        m.d.comb += Assume(data.rs1 <= 1)
                        m.d.comb += Assume(data.rd <= 3)

                    with m.Case(Opcode.JAL):
                        m.d.comb += Assume(data.rd <= 3)

                    with m.Case(Opcode.SYSTEM):
                        with m.If((data.funct3 == SystemFunc.CSRRW) |
                                  (data.funct3 == SystemFunc.CSRRS) |
                                  (data.funct3 == SystemFunc.CSRRC)):
                            m.d.comb += Assume(data.rs1 <= 1)
                        with m.If((data.funct3 == SystemFunc.CSRRW) |
                                  (data.funct3 == SystemFunc.CSRRS) |
                                  (data.funct3 == SystemFunc.CSRRC) |
                                  (data.funct3 == SystemFunc.CSRRWI) |
                                  (data.funct3 == SystemFunc.CSRRSI) |
                                  (data.funct3 == SystemFunc.CSRRCI)):
                            m.d.comb += Assume(data.rd <= 3)
                            m.d.comb += Assume((data.funct12 == CSRAddr.MCAUSE)
                                               | (data.funct12 == 0x111))
            data.verify_fatal(m)

        if mode == "irq":
            m.d.comb += Assume(~cpu.fatal)
            m.d.comb += Assume(~cpu.exception)
            mti_ok = Signal()
            mei_ok = Signal()
            m.d.comb += mti_ok.eq(data.mstatus[MStatus.MIE]
                                  & data.mie[MInterrupt.MTI])
            m.d.comb += mei_ok.eq(data.mstatus[MStatus.MIE]
                                  & data.mie[MInterrupt.MEI])

            with m.If((phase_count == 4) & ~cpu.trap):
                m.d.comb += Assume((mti_ok & (cpu.time_irq | cpu.irq.mti_pend)) |
                                   ((mei_ok & (cpu.ext_irq | cpu.irq.mei_pend))))
                m.d.ph2 += data.did_time_irq.eq(data.did_time_irq |
                                                (mti_ok & (cpu.time_irq | cpu.irq.mti_pend)))
                m.d.ph2 += data.did_ext_irq.eq(data.did_ext_irq |
                                               (mei_ok & (cpu.ext_irq | cpu.irq.mei_pend)))
            with m.If(phase_count != 4):
                m.d.comb += Assume(~cpu.time_irq & ~cpu.ext_irq)

            # Make sure that even under induction, we only get interrupts once.
            with m.If(~cpu.trap & (phase_count == 4)):
                m.d.comb += Assume(~data.did_time_irq & ~data.did_ext_irq)

            with m.If(Past(cpu.instr_complete, clocks=2) &
                      ~Past(cpu.trap, 2)):
                m.d.ph2 += data.int_return_pc.eq(data.state._pc)

            with m.If(Past(cpu.instr_complete) & ~Past(cpu.trap)):
                with m.If(data.did_time_irq | data.did_ext_irq):
                    m.d.comb += Assert(data.state.trap)

            with m.If(Fell(cpu.trap)):
                data.verify_irq(m)

        with m.If(Initial()):
            m.d.comb += Assume(~cpu.time_irq)
            m.d.comb += Assume(~cpu.ext_irq)

            init_regs = [0] + [AnyConst(32) for _ in range(1, 32)]
            init_pc = AnyConst(32)
            init_mtvec = AnyConst(32)
            init_mcause = AnyConst(32)
            init_mtval = AnyConst(32)
            init_mepc = AnyConst(32)
            init_mstatus = AnyConst(32)
            init_mie = AnyConst(32)

            m.d.comb += Assume(init_pc[:2] == 0)
            m.d.comb += Assume(init_mepc[:2] == 0)
            m.d.comb += [
                Assume(cpu.seq.state._pc == init_pc),
                Assume(cpu.seq.state._mtvec == init_mtvec),
                Assume(exc._mcause == init_mcause),
                Assume(exc._mtval == init_mtval),
                Assume(exc._mepc == init_mepc),
                Assume(irq._mstatus == init_mstatus),
                Assume(irq._mie == init_mie),
                Assume(irq._mip == 0),
            ]
            for i in range(0, 32):
                m.d.comb += Assume(cpu.regs._x_bank._mem[i] == init_regs[i])
                m.d.comb += Assume(cpu.regs._y_bank._mem[i] == init_regs[i])

            for i in range(0, 32):
                m.d.comb += Assume(data.regs_before[i] == init_regs[i])

        sync_clk = ClockSignal("sync")
        sync_rst = ResetSignal("sync")
        m.d.comb += Assume(sync_clk == ~Past(sync_clk))
        m.d.comb += Assume(~sync_rst)

        #
        # Here start all the assertions so that inductive proofs work.
        #

        m.d.comb += Assert(phase_count <= 5)

        for i in range(1, 32):
            m.d.comb += Assert(cpu.regs._x_bank._mem[i]
                               == cpu.regs._y_bank._mem[i])

        # Assert that various data collection things got cleared and loaded. For the most
        # part this is identical to the part above where things get cleared and
        # loaded on ph2, except they are turned into asserts on the correct phase
        # for inductive proofs.
        # with m.If((mcycle == 0) & (phase_count == 2) & ~Past(cpu.fatal)):
        with m.If((mcycle == 0) & (phase_count == 2) & ~Past(cpu.trap)):
            m.d.comb += Assert(~data.did_mem_rd & ~data.did_mem_wr)
            for i in range(1, 32):
                m.d.comb += Assert(data.regs_before[i]
                                   == Past(cpu.regs._x_bank._mem[i]))
                m.d.comb += Assert(data.regs_before[i]
                                   == Past(cpu.regs._y_bank._mem[i]))
            m.d.comb += Assert(data.regs_before[0] == 0)
            m.d.comb += Assert(bef._pc == Past(state._pc))
            m.d.comb += Assert(bef._instr_phase == Past(state._instr_phase))
            m.d.comb += Assert(bef._instr == Past(state._instr))
            m.d.comb += Assert(bef._stored_alu_eq ==
                               Past(state._stored_alu_eq))
            m.d.comb += Assert(bef._stored_alu_lt ==
                               Past(state._stored_alu_lt))
            m.d.comb += Assert(bef._stored_alu_ltu ==
                               Past(state._stored_alu_ltu))
            m.d.comb += Assert(bef.memaddr == Past(state.memaddr))
            m.d.comb += Assert(bef.memdata_wr == Past(state.memdata_wr))
            m.d.comb += Assert(bef._tmp == Past(state._tmp))
            m.d.comb += Assert(bef.reg_page == Past(state.reg_page))
            m.d.comb += Assert(bef.trap == Past(state.trap))
            m.d.comb += Assert(bef.exception == Past(state.exception))
            m.d.comb += Assert(bef._mtvec == Past(state._mtvec))
            m.d.comb += Assert(data.mcause_before == Past(exc._mcause))
            m.d.comb += Assert(data.mepc_before == Past(exc._mepc))
            m.d.comb += Assert(data.mtval_before == Past(exc._mtval))
            m.d.comb += Assert(data.mstatus_before == Past(irq._mstatus))
            m.d.comb += Assert(data.mie_before == Past(irq._mie))
            m.d.comb += Assert(data.mip_before == Past(irq._mip))

        # If we're in the instruction, or in a fatal condition, nothing
        # collected should change.
        with m.If((mcycle > 0) | (phase_count > 2) | Past(cpu.fatal)):
            m.d.comb += Assert(bef._instr == state._instr)
            m.d.comb += Assert(bef._pc == state._pc)
            for i in range(0, 32):
                m.d.comb += Assert(Stable(data.regs_before[i]))
            m.d.comb += Assert(Stable(bef._instr))
            m.d.comb += Assert(Stable(bef._pc))
            m.d.comb += Assert(Stable(bef._stored_alu_eq))
            m.d.comb += Assert(Stable(bef._stored_alu_lt))
            m.d.comb += Assert(Stable(bef._stored_alu_ltu))
            m.d.comb += Assert(Stable(bef.memaddr))
            m.d.comb += Assert(Stable(bef.memdata_wr))
            m.d.comb += Assert(Stable(bef._tmp))
            m.d.comb += Assert(Stable(bef.reg_page))
            m.d.comb += Assert(Stable(bef.trap))
            m.d.comb += Assert(Stable(bef.exception))
            m.d.comb += Assert(Stable(bef._mtvec))
            m.d.comb += Assert(Stable(data.mcause_before))
            m.d.comb += Assert(Stable(data.mepc_before))
            m.d.comb += Assert(Stable(data.mtval_before))
            m.d.comb += Assert(Stable(data.mstatus_before))
            m.d.comb += Assert(Stable(data.mie_before))
            m.d.comb += Assert(Stable(data.mip_before))

        # But also, if we're in a fatal condition, the PC and instruction
        # shouldn't change.
        with m.If(Past(cpu.fatal)):
            m.d.comb += Assert(Past(cpu.exception))
            m.d.comb += Assert(bef._pc == Past(state._pc))
            m.d.comb += Assert(bef._instr == Past(state._instr))

        with m.If((mcycle == 0) & (phase_count == 2) & ~Past(cpu.trap)):
            m.d.comb += Assert(data.did_csr_wr == Past(cpu.z_to_csr))
            m.d.comb += Assert(data.did_csr_rd == Past(cpu.csr_to_x))

        with m.If((mcycle == 0) &
                  (Past(cpu.z_to_csr) | Past(cpu.csr_to_x)) &
                  (phase_count == 2) &
                  ~Past(cpu.trap)):
            m.d.comb += [
                Assert(data.did_csr_rd == Past(cpu.csr_to_x)),
                Assert(data.did_csr_wr == Past(cpu.z_to_csr)),
                Assert(data.csr_accessed == Past(cpu.csr_num)),
                Assert(data.csr_rd_data == Past(cpu.x_bus)),
                Assert(data.csr_wr_data == Past(cpu.z_bus)),
            ]

        with m.If((mcycle > 0) & Past(cpu.mem_rd) & (phase_count == 2) & ~Past(cpu.trap)):
            m.d.comb += Assert(~Past(data.did_mem_rd) & ~Past(data.did_mem_wr))
            m.d.comb += Assert(data.did_mem_rd == 1)
            m.d.comb += Assert(data.memaddr_accessed == Past(cpu.memaddr))
            m.d.comb += Assert(data.mem_rd_data == Past(cpu.memdata_rd))

        with m.Elif((mcycle > 0) & Past(cpu.mem_wr) & (phase_count == 2) & ~Past(cpu.trap)):
            m.d.comb += Assert(~Past(data.did_mem_rd) & ~Past(data.did_mem_wr))
            m.d.comb += Assert(data.did_mem_wr == 1)
            m.d.comb += Assert(data.memaddr_accessed == Past(cpu.memaddr))
            m.d.comb += Assert(data.mem_wr_data == Past(cpu.memdata_wr))
            m.d.comb += Assert(data.mem_wr_mask == Past(cpu.mem_wr_mask))

        with m.If((mcycle == 0) & (phase_count > 1)):
            m.d.comb += Assert(Stable(cpu.seq.state._instr))
        with m.If(mcycle > 0):
            m.d.comb += Assert(Stable(cpu.seq.state._instr))

        with m.If(cpu.trap):
            with m.If(cpu.fatal):
                m.d.comb += Assert(mcycle == 0)
            with m.Else():
                m.d.comb += Assert(mcycle <= 1)

        m.d.comb += Assert(~(cpu.fatal & cpu.instr_complete))

        with m.If(((mcycle > 0) | (phase_count > 2)) & ~Past(cpu.trap)):
            # Misaligned loads and stores can be detected during phase 0 because they
            # only depend on rs1 and the immediate value. They will never appear on
            # phase 2 because the instruction will have trapped in phase 1.
            with m.If(state._instr == Opcode.LOAD):
                with m.If(data.is_misaligned_load):
                    m.d.comb += Assert(mcycle <= 1)

            with m.If(state._instr == Opcode.STORE):
                with m.If(data.is_misaligned_store):
                    m.d.comb += Assert(mcycle <= 1)

            # Misaligned JAL and JALR are also detected immediately, but they trap in
            # phase 1.
            with m.If((state._instr == Opcode.JAL) | (state._instr == Opcode.JALR)):
                with m.If(data.is_instr_addr_misaligned):
                    m.d.comb += Assert(mcycle <= 1)

            # Misaligned BRANCH is also detected immediately, but traps in phase 2.
            with m.If(state._instr == Opcode.BRANCH):
                with m.If(data.is_instr_addr_misaligned):
                    m.d.comb += Assert(mcycle <= 2)

        # If there's a pending interrupt, then that interrupt must have been enabled.
        with m.If(phase_count == 5):
            with m.If(Past(cpu.irq.mei_pend)):
                m.d.comb += Assert(Past(cpu.irq._mie)[MInterrupt.MEI])
            with m.If(Past(cpu.irq.mti_pend)):
                m.d.comb += Assert(Past(cpu.irq._mie)[MInterrupt.MTI])

        return m, [sync_clk, cpu.memdata_rd, cpu.csr_rd_data, cpu.time_irq, cpu.ext_irq]


if __name__ == "__main__":
    mode = sys.argv[2] if len(sys.argv) > 2 else ""
    filename = f"formal_cpu_{mode}.il" if mode != "" else "toplevel.il"

    main(FormalCPU, filename=filename)
