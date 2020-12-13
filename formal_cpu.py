# Disable pylint's "your name is too short" warning.
# pylint: disable=C0103
import sys
from typing import List, Tuple

from nmigen import Array, Signal, Module, Elaboratable, ClockDomain, Mux, Repl
from nmigen.build import Platform
from nmigen.asserts import Assert, Assume, Cover, Stable, Past, Initial, AnyConst

from alu_card import AluCard
from consts import AluFunc, AluOp, BranchCond, MemAccessWidth, Opcode, SystemFunc
from reg_card import RegCard
from sequencer_card import SequencerCard
from shift_card import ShiftCard
from util import main

mode = ""


class FormalCPU(Elaboratable):
    """Formal verification for the CPU."""

    def __init__(self):
        # CPU bus
        self.mcycle_end = Signal()
        self.illegal = Signal()
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
        self.csr_num = Signal(12)
        self.csr_to_x = Signal()
        self.z_to_csr = Signal()

        # Memory bus
        self.mem_rd = Signal()
        self.mem_wr = Signal()
        self.mem_wr_mask = Signal(4)
        self.memaddr = Signal(32)
        self.memdata_rd = Signal(32)
        self.memdata_wr = Signal(32)

        # Formal verification fake CSR read value
        self.csr_rd_data = Signal(32)

        self.regs = RegCard()
        self.alu = AluCard()
        self.shifter = ShiftCard()
        self.seq = SequencerCard()

    def elaborate(self, _: Platform) -> Module:
        """Implements a CPU."""
        m = Module()

        m.submodules.alu = self.alu
        m.submodules.regs = self.regs
        m.submodules.shifter = self.shifter
        m.submodules.sequencer = self.seq

        # Faking a CSR card
        _csr_data_x_out = Signal(32)
        m.d.comb += _csr_data_x_out.eq(Mux(self.csr_to_x, self.csr_rd_data, 0))

        # Hook up the buses
        m.d.comb += [
            self.alu.data_x.eq(self.x_bus),
            self.alu.data_y.eq(self.y_bus),

            self.shifter.data_x.eq(self.x_bus),
            self.shifter.data_y.eq(self.y_bus),

            self.regs.data_z.eq(self.z_bus),

            self.seq.data_x_in.eq(self.x_bus),
            self.seq.data_y_in.eq(self.y_bus),
            self.seq.data_z_in.eq(self.z_bus),

            self.x_bus.eq(self.seq.data_x_out |
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
        ]

        # Clock line
        m.d.comb += self.seq.mcycle_end.eq(self.mcycle_end)

        # Memory bus
        m.d.comb += [
            self.mem_rd.eq(self.seq.mem_rd),
            self.mem_wr.eq(self.seq.mem_wr),
            self.mem_wr_mask.eq(self.seq.mem_wr_mask),
            self.memaddr.eq(self.seq.memaddr),
            self.seq.memdata_rd.eq(self.memdata_rd),
            self.memdata_wr.eq(self.seq.memdata_wr),
        ]

        # Other sequencer signals
        m.d.comb += [
            self.illegal.eq(self.seq.illegal),
            self.instr_complete.eq(self.seq.instr_complete),
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

        def __init__(self, m: Module):
            self.regs_before = Array(
                [Signal(32, reset_less=True, name=f"reg_before{i}") for i in range(32)])
            self.regs_after = Array(
                [Signal(32, reset_less=True, name=f"reg_after{i}") for i in range(32)])
            self.pc_before = Signal(32, reset_less=True, reset=0)
            self.pc_after = Signal(32)
            self.instr = Signal(32)

            # Instr decode data
            self.opcode = Signal(7)
            self.rs1 = Signal(5)
            self.rs2 = Signal(5)
            self.rd = Signal(5)
            self.funct3 = Signal(3)
            self.funct7 = Signal(7)
            self.alu_func = Signal(AluFunc)
            self.csr_num = Signal(12)

            # Captured access data
            self.did_mem_rd = Signal()
            self.did_mem_wr = Signal()
            self.memaddr_accessed = Signal(32)
            self.mem_rd_data = Signal(32)
            self.mem_wr_data = Signal(32)
            self.mem_wr_mask = Signal(4)
            self.csr_accessed = Signal(12)
            self.did_csr_rd = Signal()
            self.did_csr_wr = Signal()
            self.csr_rd_data = Signal(32)
            self.csr_wr_data = Signal(32)

            m.d.comb += [
                self.opcode.eq(self.instr[:7]),
                self.rs1.eq(self.instr[15:20]),
                self.rs2.eq(self.instr[20:25]),
                self.rd.eq(self.instr[7:12]),
                self.funct3.eq(self.instr[12:15]),
                self.funct7.eq(self.instr[25:]),
                self.alu_func[3].eq(self.funct7[5]),
                self.alu_func[0:3].eq(self.funct3),
                self.csr_num.eq(self.instr[20:]),
            ]
            self.imm = FormalCPU.decode_imm(m, self.instr)

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
            m.d.comb += Assert(self.pc_after == (self.pc_before+4)[:32])
            self.verify_regs_same_except(m, self.rd)
            result = Signal(32)
            with m.If(self.rd != 0):
                m.d.comb += result.eq(self.get_before_reg(m, self.rs1) + arg)
                m.d.comb += Assert(self.regs_after[self.rd] == result)

        def verify_and(self, m: Module, arg: Signal):
            m.d.comb += Assert(self.pc_after == (self.pc_before+4)[:32])
            self.verify_regs_same_except(m, self.rd)
            result = Signal(32)
            with m.If(self.rd != 0):
                m.d.comb += result.eq(self.get_before_reg(m, self.rs1) & arg)
                m.d.comb += Assert(self.regs_after[self.rd] == result)

        def verify_or(self, m: Module, arg: Signal):
            m.d.comb += Assert(self.pc_after == (self.pc_before+4)[:32])
            self.verify_regs_same_except(m, self.rd)
            result = Signal(32)
            with m.If(self.rd != 0):
                m.d.comb += result.eq(self.get_before_reg(m, self.rs1) | arg)
                m.d.comb += Assert(self.regs_after[self.rd] == result)

        def verify_xor(self, m: Module, arg: Signal):
            m.d.comb += Assert(self.pc_after == (self.pc_before+4)[:32])
            self.verify_regs_same_except(m, self.rd)
            result = Signal(32)
            with m.If(self.rd != 0):
                m.d.comb += result.eq(self.get_before_reg(m, self.rs1) ^ arg)
                m.d.comb += Assert(self.regs_after[self.rd] == result)

        def verify_sub(self, m: Module, arg: Signal):
            m.d.comb += Assert(self.pc_after == (self.pc_before+4)[:32])
            self.verify_regs_same_except(m, self.rd)
            result = Signal(32)
            with m.If(self.rd != 0):
                m.d.comb += result.eq(self.get_before_reg(m, self.rs1) - arg)
                m.d.comb += Assert(self.regs_after[self.rd] == result)

        def verify_sltu(self, m: Module, arg: Signal):
            m.d.comb += Assert(self.pc_after == (self.pc_before+4)[:32])
            self.verify_regs_same_except(m, self.rd)
            result = Signal()
            with m.If(self.rd != 0):
                m.d.comb += result.eq(self.get_before_reg(m, self.rs1) < arg)
                m.d.comb += Assert(self.regs_after[self.rd] == result)

        def verify_slt(self, m: Module, arg: Signal):
            m.d.comb += Assert(self.pc_after == (self.pc_before+4)[:32])
            self.verify_regs_same_except(m, self.rd)
            result = Signal()
            with m.If(self.rd != 0):
                m.d.comb += result.eq(self.get_before_reg(m,
                                                          self.rs1).as_signed() < arg.as_signed())
                m.d.comb += Assert(self.regs_after[self.rd] == result)

        def verify_sll(self, m: Module, arg: Signal):
            m.d.comb += Assert(self.pc_after == (self.pc_before+4)[:32])
            self.verify_regs_same_except(m, self.rd)
            shamt = Signal(5)
            result = Signal(32)
            with m.If(self.rd != 0):
                m.d.comb += shamt.eq(arg[:5])
                m.d.comb += result.eq(self.get_before_reg(m,
                                                          self.rs1) << shamt)
                m.d.comb += Assert(self.regs_after[self.rd] == result)

        def verify_srl(self, m: Module, arg: Signal):
            m.d.comb += Assert(self.pc_after == (self.pc_before+4)[:32])
            self.verify_regs_same_except(m, self.rd)
            shamt = Signal(5)
            result = Signal(32)
            with m.If(self.rd != 0):
                m.d.comb += shamt.eq(arg[:5])
                m.d.comb += result.eq(self.get_before_reg(m,
                                                          self.rs1) >> shamt)
                m.d.comb += Assert(self.regs_after[self.rd] == result)

        def verify_sra(self, m: Module, arg: Signal):
            m.d.comb += Assert(self.pc_after == (self.pc_before+4)[:32])
            self.verify_regs_same_except(m, self.rd)
            shamt = Signal(5)
            result = Signal(32)
            with m.If(self.rd != 0):
                m.d.comb += shamt.eq(arg[:5])
                m.d.comb += result.eq(self.get_before_reg(m,
                                                          self.rs1).as_signed() >> shamt)
                m.d.comb += Assert(self.regs_after[self.rd] == result)

        def verify_opcode_OP(self, m: Module):
            if mode != "op":
                return
            arg = self.get_before_reg(m, self.rs2)
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
                # No default case: it is okay to have UB (undefined/unspecified behavior)
                # when executing an instruction that is not implemented, that is,
                # not part of the extension you claim to support.

        def verify_opcode_OP_IMM(self, m: Module):
            if mode != "op_imm":
                return
            arg = self.imm
            with m.Switch(self.alu_func):
                with m.Case(AluFunc.ADD):
                    self.verify_add(m, arg)
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
                # No default case: it is okay to have UB (undefined/unspecified behavior)
                # when executing an instruction that is not implemented, that is,
                # not part of the extension you claim to support.

        def verify_opcode_LUI(self, m: Module):
            if mode != "lui":
                return
            m.d.comb += Assert(self.pc_after == (self.pc_before+4)[:32])
            self.verify_regs_same_except(m, self.rd)
            with m.If(self.rd != 0):
                result = Signal(32)
                m.d.comb += result[12:].eq(self.imm[12:])
                m.d.comb += result[0:12].eq(0)
                m.d.comb += Assert(self.regs_after[self.rd] == result)

        def verify_opcode_AUIPC(self, m: Module):
            if mode != "auipc":
                return
            m.d.comb += Assert(self.pc_after == (self.pc_before+4)[:32])
            self.verify_regs_same_except(m, self.rd)
            with m.If(self.rd != 0):
                offset = Signal(32)
                result = Signal(32)
                m.d.comb += offset[12:].eq(self.imm[12:])
                m.d.comb += offset[0:12].eq(0)
                m.d.comb += result.eq(offset + self.pc_before)
                m.d.comb += Assert(self.regs_after[self.rd] == result)

        def verify_opcode_JAL(self, m: Module):
            if mode != "jal":
                return
            new_pc = Signal(32)
            m.d.comb += new_pc.eq(self.pc_before + self.imm)
            m.d.comb += Assert(self.pc_after == new_pc)
            self.verify_regs_same_except(m, self.rd)
            m.d.comb += Assert(~self.did_mem_rd & ~self.did_mem_wr)
            with m.If(self.rd != 0):
                result = Signal(32)
                m.d.comb += result.eq(self.pc_before+4)
                m.d.comb += Assert(self.regs_after[self.rd] == result)

        def verify_opcode_JALR(self, m: Module):
            if mode != "jalr":
                return
            new_pc = Signal(32)
            m.d.comb += new_pc.eq((self.get_before_reg(m,
                                                       self.rs1) + self.imm) & 0xFFFFFFFE)
            m.d.comb += Assert(self.pc_after == new_pc)
            self.verify_regs_same_except(m, self.rd)
            m.d.comb += Assert(~self.did_mem_rd & ~self.did_mem_wr)
            with m.If(self.rd != 0):
                result = Signal(32)
                m.d.comb += result.eq(self.pc_before+4)
                m.d.comb += Assert(self.regs_after[self.rd] == result)

        def verify_opcode_BRANCH(self, m: Module):
            if mode != "branch":
                return
            rs1 = self.get_before_reg(m, self.rs1)
            rs2 = self.get_before_reg(m, self.rs2)
            target_if = (self.pc_before + self.imm)[:32]
            target_else = (self.pc_before + 4)[:32]

            self.verify_regs_same_except(m, 0)
            m.d.comb += Assert(~self.did_mem_rd & ~self.did_mem_wr)
            with m.Switch(self.funct3):
                with m.Case(BranchCond.EQ):
                    expected_target = Mux(rs1 == rs2, target_if, target_else)
                    m.d.comb += Assert(self.pc_after == expected_target)
                with m.Case(BranchCond.NE):
                    expected_target = Mux(rs1 != rs2, target_if, target_else)
                    m.d.comb += Assert(self.pc_after == expected_target)
                with m.Case(BranchCond.LTU):
                    expected_target = Mux(rs1 < rs2, target_if, target_else)
                    m.d.comb += Assert(self.pc_after == expected_target)
                with m.Case(BranchCond.LT):
                    expected_target = Mux(
                        rs1.as_signed() < rs2.as_signed(), target_if, target_else)
                    m.d.comb += Assert(self.pc_after == expected_target)
                with m.Case(BranchCond.GEU):
                    expected_target = Mux(rs1 >= rs2, target_if, target_else)
                    m.d.comb += Assert(self.pc_after == expected_target)
                with m.Case(BranchCond.GE):
                    expected_target = Mux(
                        rs1.as_signed() >= rs2.as_signed(), target_if, target_else)
                    m.d.comb += Assert(self.pc_after == expected_target)

        def signed_byte_to_32(self, m: Module, b: Signal) -> Signal:
            """Returns the given byte, sign-extended to 32 bits.

            For some reason this results in less output and faster
            formal verification than just using as_signed().
            """
            s = Signal(32)
            m.d.comb += s[0:8].eq(b)
            m.d.comb += s[8:].eq(Repl(b[7], 24))
            return s

        def signed_word_to_32(self, m: Module, b: Signal) -> Signal:
            """Returns the given word, sign-extended to 32 bits."""
            s = Signal(32)
            m.d.comb += s[0:16].eq(b)
            m.d.comb += s[16:].eq(Repl(b[15], 16))
            return s

        def verify_LB(self, m: Module):
            if mode != "lb":
                return
            m.d.comb += Assert(self.pc_after == (self.pc_before+4)[:32])
            self.verify_regs_same_except(m, self.rd)
            rs1 = self.get_before_reg(m, self.rs1)
            rd = self.regs_after[self.rd]
            target = (rs1 + self.imm)[:32]
            mem = self.mem_rd_data

            m.d.comb += Assert(self.did_mem_rd)
            m.d.comb += Assert(self.memaddr_accessed == target)
            with m.If(self.rd != 0):
                with m.Switch(target[:2]):
                    with m.Case(0):
                        m.d.comb += Assert(rd ==
                                           self.signed_byte_to_32(m, mem[0:8]))
                    with m.Case(1):
                        m.d.comb += Assert(rd ==
                                           self.signed_byte_to_32(m, mem[8:16]))
                    with m.Case(2):
                        m.d.comb += Assert(rd ==
                                           self.signed_byte_to_32(m, mem[16:24]))
                    with m.Case(3):
                        m.d.comb += Assert(rd ==
                                           self.signed_byte_to_32(m, mem[24:]))

        def verify_LBU(self, m: Module):
            if mode != "lbu":
                return
            m.d.comb += Assert(self.pc_after == (self.pc_before+4)[:32])
            self.verify_regs_same_except(m, self.rd)
            rs1 = self.get_before_reg(m, self.rs1)
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
            m.d.comb += Assert(self.pc_after == (self.pc_before+4)[:32])
            self.verify_regs_same_except(m, self.rd)
            rs1 = self.get_before_reg(m, self.rs1)
            rd = self.regs_after[self.rd]
            target = (rs1 + self.imm)[:32]
            mem = self.mem_rd_data

            m.d.comb += Assert(self.did_mem_rd)
            m.d.comb += Assert(self.memaddr_accessed == target)
            with m.If(self.rd != 0):
                with m.Switch(target[:2]):
                    with m.Case(0):
                        m.d.comb += Assert(rd ==
                                           self.signed_word_to_32(m, mem[:16]))
                    with m.Case(2):
                        m.d.comb += Assert(rd ==
                                           self.signed_word_to_32(m, mem[16:]))
                    with m.Default():
                        m.d.comb += Assert(0)

        def verify_LHU(self, m: Module):
            if mode != "lhu":
                return
            m.d.comb += Assert(self.pc_after == (self.pc_before+4)[:32])
            self.verify_regs_same_except(m, self.rd)
            rs1 = self.get_before_reg(m, self.rs1)
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
            m.d.comb += Assert(self.pc_after == (self.pc_before+4)[:32])
            self.verify_regs_same_except(m, self.rd)
            rs1 = self.get_before_reg(m, self.rs1)
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

        def verify_SB(self, m: Module):
            if mode != "sb":
                return
            m.d.comb += Assert(self.pc_after == (self.pc_before+4)[:32])
            self.verify_regs_same_except(m, 0)
            rs1 = self.get_before_reg(m, self.rs1)
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
            m.d.comb += Assert(self.pc_after == (self.pc_before+4)[:32])
            self.verify_regs_same_except(m, 0)
            rs1 = self.get_before_reg(m, self.rs1)
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
            m.d.comb += Assert(self.pc_after == (self.pc_before+4)[:32])
            self.verify_regs_same_except(m, 0)
            rs1 = self.get_before_reg(m, self.rs1)
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

        def verify_CSRRW(self, m: Module):
            if mode != "csr":
                return
            m.d.comb += Assert(self.pc_after == (self.pc_before+4)[:32])
            m.d.comb += Assert(~self.did_mem_rd & ~self.did_mem_wr)
            self.verify_regs_same_except(m, self.rd)
            rs1 = self.get_before_reg(m, self.rs1)
            rd = self.regs_after[self.rd]

            m.d.comb += Assert(self.did_csr_rd == (self.rd != 0))
            m.d.comb += Assert(self.did_csr_wr)
            m.d.comb += Assert(self.csr_accessed == self.csr_num)
            with m.If(self.did_csr_rd):
                m.d.comb += Assert(rd == self.csr_rd_data)
            m.d.comb += Assert(self.csr_wr_data == rs1)

        def verify_CSRRWI(self, m: Module):
            if mode != "csr":
                return
            m.d.comb += Assert(self.pc_after == (self.pc_before+4)[:32])
            m.d.comb += Assert(~self.did_mem_rd & ~self.did_mem_wr)
            self.verify_regs_same_except(m, self.rd)
            rd = self.regs_after[self.rd]

            m.d.comb += Assert(self.did_csr_rd == (self.rd != 0))
            m.d.comb += Assert(self.did_csr_wr)
            m.d.comb += Assert(self.csr_accessed == self.csr_num)
            with m.If(self.did_csr_rd):
                m.d.comb += Assert(rd == self.csr_rd_data)
            m.d.comb += Assert(self.csr_wr_data == self.imm)

        def verify_CSRRS(self, m: Module):
            if mode != "csr":
                return
            m.d.comb += Assert(self.pc_after == (self.pc_before+4)[:32])
            m.d.comb += Assert(~self.did_mem_rd & ~self.did_mem_wr)
            self.verify_regs_same_except(m, self.rd)
            rs1 = self.get_before_reg(m, self.rs1)
            rd = self.regs_after[self.rd]

            m.d.comb += Assert(self.did_csr_wr == (self.rs1 != 0))
            m.d.comb += Assert(self.did_csr_rd)
            m.d.comb += Assert(self.csr_accessed == self.csr_num)
            with m.If(self.rd != 0):
                m.d.comb += Assert(rd == self.csr_rd_data)
            with m.If(self.rs1 != 0):
                m.d.comb += Assert(self.csr_wr_data ==
                                   (rs1 | self.csr_rd_data))

        def verify_CSRRSI(self, m: Module):
            if mode != "csr":
                return
            m.d.comb += Assert(self.pc_after == (self.pc_before+4)[:32])
            m.d.comb += Assert(~self.did_mem_rd & ~self.did_mem_wr)
            self.verify_regs_same_except(m, self.rd)
            rd = self.regs_after[self.rd]

            m.d.comb += Assert(self.did_csr_wr == (self.imm != 0))
            m.d.comb += Assert(self.did_csr_rd)
            m.d.comb += Assert(self.csr_accessed == self.csr_num)
            with m.If(self.rd != 0):
                m.d.comb += Assert(rd == self.csr_rd_data)
            with m.If(self.imm != 0):
                m.d.comb += Assert(self.csr_wr_data ==
                                   (self.imm | self.csr_rd_data))

        def verify_CSRRC(self, m: Module):
            if mode != "csr":
                return
            m.d.comb += Assert(self.pc_after == (self.pc_before+4)[:32])
            m.d.comb += Assert(~self.did_mem_rd & ~self.did_mem_wr)
            self.verify_regs_same_except(m, self.rd)
            rs1 = self.get_before_reg(m, self.rs1)
            rd = self.regs_after[self.rd]

            m.d.comb += Assert(self.did_csr_wr == (self.rs1 != 0))
            m.d.comb += Assert(self.did_csr_rd)
            m.d.comb += Assert(self.csr_accessed == self.csr_num)
            with m.If(self.rd != 0):
                m.d.comb += Assert(rd == self.csr_rd_data)
            with m.If(self.rs1 != 0):
                m.d.comb += Assert(self.csr_wr_data ==
                                   (self.csr_rd_data & ~rs1))

        def verify_CSRRCI(self, m: Module):
            if mode != "csr":
                return
            m.d.comb += Assert(self.pc_after == (self.pc_before+4)[:32])
            m.d.comb += Assert(~self.did_mem_rd & ~self.did_mem_wr)
            self.verify_regs_same_except(m, self.rd)
            rd = self.regs_after[self.rd]

            m.d.comb += Assert(self.did_csr_wr == (self.imm != 0))
            m.d.comb += Assert(self.did_csr_rd)
            m.d.comb += Assert(self.csr_accessed == self.csr_num)
            with m.If(self.rd != 0):
                m.d.comb += Assert(rd == self.csr_rd_data)
            with m.If(self.imm != 0):
                m.d.comb += Assert(self.csr_wr_data ==
                                   (self.csr_rd_data & ~self.imm))

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
                with m.Case(Opcode.SYSTEM):
                    self.verify_opcode_SYSTEM(m)
                # No default case: it is okay to have UB (undefined/unspecified behavior)
                # when executing an instruction that is not implemented, that is,
                # not part of the extension you claim to support.

        def verify_illegal(self, m: Module):
            """Verification for when the illegal signal goes high.

            One of these conditions has to be true:
            * The low 16 bits of an instruction are 0
            * All bits of an instruction are 1
            * A misaligned load was requested
            * A misaligned store was requested
            * The PC is not 32-bit aligned
            """
            is_zero_instr = Signal()
            is_ones_instr = Signal()
            is_misaligned_load = Signal()
            is_misaligned_store = Signal()
            is_pc_misaligned = Signal()
            addr = self.memaddr_accessed

            m.d.comb += is_zero_instr.eq(self.instr[:16] == 0)
            m.d.comb += is_ones_instr.eq(self.instr == 0xFFFFFFFF)

            m.d.comb += is_misaligned_load.eq(0)
            with m.If(self.did_mem_rd & (self.opcode == Opcode.LOAD)):
                with m.Switch(self.funct3):
                    with m.Case(MemAccessWidth.H, MemAccessWidth.HU):
                        m.d.comb += is_misaligned_load.eq(addr[0] != 0)
                    with m.Case(MemAccessWidth.W):
                        m.d.comb += is_misaligned_load.eq(addr[0:2] != 0)

            m.d.comb += is_misaligned_store.eq(0)
            with m.If(self.did_mem_wr & (self.opcode == Opcode.STORE)):
                with m.Switch(self.funct3):
                    with m.Case(MemAccessWidth.H, MemAccessWidth.HU):
                        m.d.comb += is_misaligned_store.eq(addr[0] != 0)
                    with m.Case(MemAccessWidth.W):
                        m.d.comb += is_misaligned_store.eq(addr[0:2] != 0)

            m.d.comb += is_pc_misaligned.eq(self.pc_after[0:2] != 0)

            m.d.comb += Assert(is_zero_instr | is_ones_instr |
                               is_misaligned_load | is_misaligned_store |
                               is_pc_misaligned)

    @classmethod
    def make_clock(cls, m: Module) -> Tuple[Signal, Signal]:
        """Creates the clock domains and signals."""
        phase_count = Signal(3, reset=0, reset_less=True)
        mcycle_end = Signal()

        ph1 = ClockDomain("ph1")
        ph2 = ClockDomain("ph2")

        m.domains += [ph1, ph2]

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
        data = FormalCPU.Collected(m)

        # Machine cycle counting
        mcycle = Signal(3, reset_less=True, reset=0)
        with m.If(~cpu.illegal):
            m.d.ph1 += mcycle.eq(mcycle+1)
        with m.If(cpu.instr_complete):
            m.d.ph1 += mcycle.eq(0)

        init_regs = [None] + [AnyConst(32) for _ in range(1, 32)]
        init_pc = AnyConst(32)

        # Initialize registers
        with m.If(Initial()):
            m.d.comb += [
                Assume(cpu.seq._pc == init_pc),
                Assume(init_pc[:2] == 0),
                Assume(data.did_mem_rd == 0),
                Assume(data.did_mem_wr == 0),
            ]
            for i in range(1, 32):
                m.d.comb += [
                    Assume(cpu.regs._x_bank._mem[i] == init_regs[i]),
                    Assume(data.regs_before[i] == init_regs[i]),
                ]

        # Assume memory and CSR data is stable
        with m.If(phase_count > 1):
            m.d.comb += Assume(Stable(cpu.memdata_rd))
            m.d.comb += Assume(Stable(cpu.csr_rd_data))

        # Store the instruction we're working on, and take a snapshot
        # of the registers.
        with m.If((mcycle == 0) & (phase_count == 1) & ~cpu.illegal):
            m.d.ph2 += data.instr.eq(cpu.memdata_rd)
            m.d.ph2 += data.pc_before.eq(cpu.seq._pc)
            for i in range(1, 32):
                m.d.ph2 += data.regs_before[i].eq(cpu.regs._x_bank._mem[i])
            m.d.ph2 += [
                data.did_mem_rd.eq(0),
                data.did_mem_wr.eq(0),
                data.did_csr_rd.eq(0),
                data.did_csr_wr.eq(0),
            ]

        for i in range(1, 32):
            m.d.comb += data.regs_after[i].eq(cpu.regs._x_bank._mem[i])
        m.d.comb += data.pc_after.eq(cpu.seq._pc)

        # If we read or write memory outside the first machine cycle, save it.
        # But complain if we do more than only one read or one write.
        m.d.comb += Assert(~(data.did_mem_rd & data.did_mem_wr))

        with m.If((mcycle > 0) & cpu.mem_rd & (phase_count == 1) & ~cpu.illegal):
            m.d.comb += Assert(~data.did_mem_rd & ~data.did_mem_wr)
            m.d.ph2 += data.did_mem_rd.eq(1)
            m.d.ph2 += data.memaddr_accessed.eq(cpu.memaddr)
            m.d.ph2 += data.mem_rd_data.eq(cpu.memdata_rd)

        # Yes, phase 1. The illegal signal goes high on phase 2,
        # because we know the memory we're going to write to on
        # phase 1. The memory may not actually get written until
        # phase 4, but the address and data are already set up now.
        with m.Elif((mcycle > 0) & cpu.mem_wr & (phase_count == 1) & ~cpu.illegal):
            m.d.comb += Assert(~data.did_mem_rd & ~data.did_mem_wr)
            m.d.ph2 += data.did_mem_wr.eq(1)
            m.d.ph2 += data.memaddr_accessed.eq(cpu.memaddr)
            m.d.ph2 += data.mem_wr_data.eq(cpu.memdata_wr)
            m.d.ph2 += data.mem_wr_mask.eq(cpu.mem_wr_mask)

        with m.If((mcycle == 0) & (cpu.z_to_csr | cpu.csr_to_x) & (phase_count == 1) & ~cpu.illegal):
            m.d.ph2 += [
                data.did_csr_rd.eq(cpu.csr_to_x),
                data.did_csr_wr.eq(cpu.z_to_csr),
                data.csr_accessed.eq(cpu.csr_num),
                data.csr_rd_data.eq(cpu.x_bus),
                data.csr_wr_data.eq(cpu.z_bus),
            ]

        # Past cycle 0, we neither read nor write a CSR
        with m.If((mcycle > 0) & ~cpu.illegal):
            m.d.comb += Assert(~cpu.z_to_csr)
            m.d.comb += Assert(~cpu.csr_to_x)

        # Can't read and write at the same time
        m.d.comb += Assert(~(cpu.mem_rd & cpu.mem_wr))

        # Can't write on the first machine cycle
        with m.If(mcycle == 0):
            m.d.comb += Assert(~cpu.mem_wr)

        # Covers
        m.d.comb += Cover(cpu.instr_complete)

        # These assertions are in place for when I decide to run
        # prove mode. The verifier can set any flipflop to any
        # value, so we have to ensure we tell it what a valid state
        # is.
        for i in range(1, 32):
            m.d.comb += Assert(cpu.regs._y_bank._mem[i]
                               == cpu.regs._x_bank._mem[i])
        m.d.comb += Assert(mcycle == cpu.seq._instr_phase)
        # There are no instructions that take more than 3 cycles.
        m.d.comb += Assert(mcycle < 3)
        with m.If((mcycle == 0) & (phase_count >= 1) & ~cpu.illegal):
            m.d.comb += Assert(cpu.seq._instr == cpu.memdata_rd)
        with m.If(phase_count > 1):
            m.d.comb += Assert(Stable(cpu.seq._instr))
            m.d.comb += Assert(Stable(cpu.seq.alu_eq))
            m.d.comb += Assert(Stable(cpu.seq.alu_lt))
            m.d.comb += Assert(Stable(cpu.seq.alu_ltu))
            m.d.comb += Assert(Stable(cpu.seq.csr_num))
            m.d.comb += Assert(Stable(cpu.seq.csr_to_x))
            m.d.comb += Assert(Stable(cpu.seq.z_to_csr))
        with m.If(mcycle > 0):
            m.d.comb += Assert(Stable(cpu.seq._instr))
        with m.If(phase_count > 0):
            m.d.comb += Assert(Stable(cpu.seq._pc))

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
            with m.If((mcycle > 0) | (phase_count > 1)):
                m.d.comb += Assume(data.funct3 == widths[mode])

        if mode in opcodes:
            m.d.comb += Assume(cpu.illegal == 0)
            with m.If((mcycle > 0) | (phase_count > 1)):
                m.d.comb += Assume(data.opcode == opcodes[mode])
            # Formal verification just after we've completed an instruction.
            with m.If(Past(cpu.instr_complete)):
                data.verify_instr(m)

        if mode == "ill":
            m.d.comb += Assume(cpu.instr_complete == 0)
            with m.If(cpu.illegal & (phase_count == 2)):
                data.verify_illegal(m)

            # Ensure that if we've hit an illegal condition, we don't
            # get out of it, and that the memory access we were doing
            # doesn't change.
            with m.If(Past(cpu.illegal)):
                m.d.comb += Assert(cpu.illegal)
                m.d.comb += [
                    Assert(Stable(cpu.memaddr)),
                    Assert(Stable(cpu.memdata_wr)),
                    Assert(Stable(cpu.mem_wr_mask)),
                    Assert(Stable(cpu.mem_wr)),
                    Assert(Stable(cpu.mem_rd)),
                ]

        return m, [cpu.memdata_rd, cpu.csr_rd_data]


if __name__ == "__main__":
    mode = sys.argv[2] if len(sys.argv) > 2 else ""
    filename = f"formal_cpu_{mode}.il" if mode != "" else "toplevel.il"

    main(FormalCPU, filename=filename)
