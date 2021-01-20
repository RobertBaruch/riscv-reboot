# Disable pylint's "your name is too short" warning.
# pylint: disable=C0103
# Disable protected access warnings
# pylint: disable=W0212
from typing import List, Tuple

from nmigen import Signal, Module, Elaboratable, signed, ClockSignal, ClockDomain, Repl
from nmigen import Mux
from nmigen.build import Platform
from nmigen.asserts import Assert, Assume, Cover, Stable, Past

from consts import AluOp, AluFunc, BranchCond, CSRAddr, MemAccessWidth
from consts import Opcode, OpcodeFormat, SystemFunc, TrapCause, PrivFunc
from consts import InstrReg, OpcodeSelect, TrapCauseSelect
from consts import NextPC, Instr, SeqMuxSelect, ConstSelect
from transparent_latch import TransparentLatch
from util import main, all_true
from IC_7416244 import IC_mux32
from IC_7416374 import IC_reg32_with_mux
from IC_GAL import IC_GAL_imm_format_decoder
from sequencer_rom import SequencerROM
from trap_rom import TrapROM
from irq_load_rom import IrqLoadInstrROM


class SequencerState:
    """Contains only the registers in the sequencer card.

    This is useful to take snapshots of the entire state.
    """

    def __init__(self, ext_init: bool = False):
        attrs = [] if not ext_init else [("uninitialized", "")]

        self._pc = Signal(32, attrs=attrs)
        self._instr_phase = Signal(2)
        # Not quite a register, but the output of a latch
        self._instr = Signal(32)
        self._stored_alu_eq = Signal()
        self._stored_alu_lt = Signal()
        self._stored_alu_ltu = Signal()

        self.memaddr = Signal(32)
        self.memdata_wr = Signal(32)

        self._tmp = Signal(32)
        self.reg_page = Signal()

        # Trap handling
        # Goes high when we are handling a trap condition,
        # but not yet in the trap routine.
        self.trap = Signal()

        # Raised when an exception occurs.
        self.exception = Signal()
        self.fatal = Signal()

        self._mtvec = Signal(32, attrs=attrs)


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

    There's also a clock ph2w which is just ph2 on the WR phase:

         ____________     _____________     ___
    ph2w             |___|             |___|
    """

    def __init__(self, ext_init: bool = False, chips: bool = False):
        self.chips = chips
        self.ext_init = ext_init

        self.state = SequencerState(ext_init)
        self.rom = SequencerROM()
        self.trap_rom = TrapROM()
        self.irq_load_rom = IrqLoadInstrROM()

        # A clock-based signal, high only at the end of a machine
        # cycle (i.e. phase 5, the very end of the write phase).
        self.mcycle_end = Signal()

        # Control signals.
        self.alu_eq = Signal()
        self.alu_lt = Signal()
        self.alu_ltu = Signal()

        self.x_reg = Signal(5)
        self.y_reg = Signal(5)
        self.z_reg = Signal(5)

        # Raised when an interrupt based on an external timer goes off.
        self.time_irq = Signal()
        # Raised when any other external interrupt goes off.
        self.ext_irq = Signal()
        # Raised on the last phase of an instruction.
        self.set_instr_complete = Signal()
        self.instr_complete = Signal()
        # Raised when the exception card should store trap data.
        self.save_trap_csrs = Signal()
        self.is_interrupted = Signal()
        self.instr_misalign = Signal()
        self.bad_instr = Signal()

        # CSR lines
        self.csr_num = Signal(CSRAddr)
        self.csr_num_is_mtvec = Signal()
        self.csr_to_x = Signal()
        self.z_to_csr = Signal()

        # Buses, bidirectional
        self.data_x_in = Signal(32)
        self.data_x_out = Signal(32)
        self.data_y_in = Signal(32)
        self.data_y_out = Signal(32)
        self.data_z_in = Signal(32)
        self.data_z_out = Signal(32)
        self.data_z_in_2_lsb0 = Signal()

        # Memory
        self.mem_rd = Signal(reset=1)
        self.mem_wr = Signal()
        # Bytes in memory word to write
        self.mem_wr_mask = Signal(4)
        self.memaddr_2_lsb = Signal(2)

        # Memory bus, bidirectional
        self.memdata_rd = Signal(32)

        # Internals

        # This opens the instr transparent latch to memdata. The enable
        # (i.e. load_instr) on the latch is a register, so setting load_instr
        # now opens the transparent latch next.
        self._load_instr = Signal(reset=1)

        self._instr_latch = TransparentLatch(32)
        self._pc_plus_4 = Signal(32)
        self._next_instr_phase = Signal(len(self.state._instr_phase))
        self._next_reg_page = Signal()

        # Instruction decoding
        self._opcode = Signal(7)
        self.opcode_select = Signal(OpcodeSelect)
        self._rs1 = Signal(5)
        self._rs2 = Signal(5)
        self._rd = Signal(5)
        self._funct3 = Signal(3)
        self._funct7 = Signal(7)
        self._funct12 = Signal(12)
        self._alu_func = Signal(4)
        self._imm_format = Signal(OpcodeFormat)
        self._imm = Signal(32)
        self.branch_cond = Signal()
        self.imm0 = Signal()
        self.rd0 = Signal()
        self.rs1_0 = Signal()

        self._x_reg_select = Signal(InstrReg)
        self._y_reg_select = Signal(InstrReg)
        self._z_reg_select = Signal(InstrReg)

        self._const = Signal(ConstSelect)

        # -> X
        self.reg_to_x = Signal()
        self.x_mux_select = Signal(SeqMuxSelect)

        # -> Y
        self.reg_to_y = Signal()
        self.y_mux_select = Signal(SeqMuxSelect)

        # -> Z
        self.z_mux_select = Signal(SeqMuxSelect)
        self.alu_op_to_z = Signal(AluOp)  # 4 bits

        # -> PC
        self.pc_mux_select = Signal(SeqMuxSelect)

        # -> tmp
        self.tmp_mux_select = Signal(SeqMuxSelect)

        # -> csr_num
        self._funct12_to_csr_num = Signal()
        self._mepc_num_to_csr_num = Signal()
        self._mcause_to_csr_num = Signal()
        self.mtvec_mux_select = Signal(SeqMuxSelect)

        # -> memaddr
        self.memaddr_mux_select = Signal(SeqMuxSelect)

        # -> memdata
        self.memdata_wr_mux_select = Signal(SeqMuxSelect)

        # memory load shamt
        self._shamt = Signal(5)

        # -> various CSRs
        self._trapcause = Signal(TrapCause)
        self._trapcause_select = Signal(TrapCauseSelect)
        self.clear_pend_mti = Signal()
        self.clear_pend_mei = Signal()
        self.mei_pend = Signal()
        self.mti_pend = Signal()
        self.vec_mode = Signal(2)

        self.enter_trap = Signal()
        self.exit_trap = Signal()

        # Signals for next registers
        self.load_trap = Signal()
        self.next_trap = Signal()
        self.load_exception = Signal()
        self.next_exception = Signal()
        self.next_fatal = Signal()

        self.enable_sequencer_rom = Signal()

    def elaborate(self, _: Platform) -> Module:
        """Implements the logic of the sequencer card."""
        m = Module()

        m.submodules.instr_latch = self._instr_latch
        m.submodules.rom = self.rom
        m.submodules.trap_rom = self.trap_rom
        m.submodules.irq_load_rom = self.irq_load_rom

        m.d.comb += self._pc_plus_4.eq(self.state._pc + 4)
        m.d.comb += self.vec_mode.eq(self.state._mtvec[:2])
        m.d.comb += self.memaddr_2_lsb.eq(self.state.memaddr[0:2])
        m.d.comb += self.imm0.eq(self._imm == 0)
        m.d.comb += self.rd0.eq(self._rd == 0)
        m.d.comb += self.rs1_0.eq(self._rs1 == 0)
        m.d.comb += self.csr_num_is_mtvec.eq(self.csr_num == CSRAddr.MTVEC)
        m.d.comb += self.mtvec_mux_select.eq(Mux(self.z_to_csr & self.csr_num_is_mtvec,
                                                 SeqMuxSelect.Z, SeqMuxSelect.MTVEC))
        # Only used on instruction phase 1 in BRANCH, which is why we can
        # register this on phase 1. Also, because it's an input to a ROM,
        # we have to ensure the signal is registered.
        m.d.ph1 += self.data_z_in_2_lsb0.eq(self.data_z_in[0:2] == 0)

        with m.If(self.set_instr_complete):
            m.d.comb += self.instr_complete.eq(self.mcycle_end)
        with m.Else():
            m.d.comb += self.instr_complete.eq(0)

        # We only check interrupts once we're about to load an instruction but we're not already
        # trapping an interrupt.
        m.d.comb += self.is_interrupted.eq(all_true(self.instr_complete,
                                                    ~self.state.trap,
                                                    (self.mei_pend | self.mti_pend)))

        # Because we don't support the C (compressed instructions)
        # extension, the PC must be 32-bit aligned.
        m.d.comb += self.instr_misalign.eq(
            all_true(self.state._pc[0:2] != 0, ~self.state.trap))

        m.d.comb += self.enable_sequencer_rom.eq(
            ~self.state.trap & ~self.instr_misalign & ~self.bad_instr)

        self.encode_opcode_select(m)
        self.connect_roms(m)
        self.process(m)
        self.updates(m)

        return m

    def connect_roms(self, m: Module):
        m.d.comb += self.rom.enable_sequencer_rom.eq(self.enable_sequencer_rom)
        m.d.comb += self.irq_load_rom.enable_sequencer_rom.eq(
            self.enable_sequencer_rom)

        # Inputs
        m.d.comb += [
            self.trap_rom.is_interrupted.eq(self.is_interrupted),
            self.trap_rom.exception.eq(self.state.exception),
            self.trap_rom.fatal.eq(self.state.fatal),
            self.trap_rom.instr_misalign.eq(self.instr_misalign),
            self.trap_rom.bad_instr.eq(self.bad_instr),
            self.trap_rom.trap.eq(self.state.trap),
            self.trap_rom.mei_pend.eq(self.mei_pend),
            self.trap_rom.mti_pend.eq(self.mti_pend),
            self.trap_rom.vec_mode.eq(self.vec_mode),
            self.trap_rom._instr_phase.eq(self.state._instr_phase),

            self.irq_load_rom.is_interrupted.eq(self.is_interrupted),
            self.irq_load_rom._instr_phase.eq(self.state._instr_phase),

            self.rom._instr_phase.eq(self.state._instr_phase),

            self.rom.memaddr_2_lsb.eq(self.memaddr_2_lsb),
            self.rom.branch_cond.eq(self.branch_cond),
            self.rom.data_z_in_2_lsb0.eq(self.data_z_in_2_lsb0),

            # Instruction decoding
            self.rom.opcode_select.eq(self.opcode_select),
            self.rom._funct3.eq(self._funct3),
            self.rom._alu_func.eq(self._alu_func),

            self.rom.imm0.eq(self.imm0),
            self.rom.rd0.eq(self.rd0),
            self.rom.rs1_0.eq(self.rs1_0),
        ]

        # Outputs
        m.d.comb += [
            # Raised on the last phase of an instruction.
            self.set_instr_complete.eq(Mux(self.enable_sequencer_rom,
                                           self.rom.set_instr_complete, self.trap_rom.set_instr_complete)),

            # Raised when the exception card should store trap data.
            self.save_trap_csrs.eq(Mux(self.enable_sequencer_rom,
                                       self.rom.save_trap_csrs, self.trap_rom.save_trap_csrs)),

            # CSR lines
            self.csr_to_x.eq(Mux(self.enable_sequencer_rom,
                                 self.rom.csr_to_x, self.trap_rom.csr_to_x)),
            self.z_to_csr.eq(self.rom.z_to_csr),

            # Memory
            self.mem_rd.eq(Mux(self.enable_sequencer_rom,
                               self.rom.mem_rd, self.irq_load_rom.mem_rd)),
            self.mem_wr.eq(self.rom.mem_wr),
            # Bytes in memory word to write
            self.mem_wr_mask.eq(self.rom.mem_wr_mask),

            # Internals

            # This opens the instr transparent latch to memdata. The enable
            # (i.e. load_instr) on the latch is a register, so setting load_instr
            # now opens the transparent latch next.
            self._load_instr.eq(self.irq_load_rom._load_instr),
            self._next_instr_phase.eq(Mux(self.enable_sequencer_rom,
                                          self.rom._next_instr_phase, self.trap_rom._next_instr_phase)),

            self._x_reg_select.eq(self.rom._x_reg_select),
            self._y_reg_select.eq(self.rom._y_reg_select),
            self._z_reg_select.eq(self.rom._z_reg_select),

            # -> X
            self.reg_to_x.eq(self.rom.reg_to_x),
            self.x_mux_select.eq(Mux(self.csr_to_x & self.csr_num_is_mtvec, SeqMuxSelect.MTVEC,
                                     Mux(self.enable_sequencer_rom,
                                         self.rom.x_mux_select, self.trap_rom.x_mux_select))),
            self._const.eq(Mux(self.enable_sequencer_rom,
                               self.rom._const, self.trap_rom._const)),

            # -> Y
            self.reg_to_y.eq(self.rom.reg_to_y),
            self.y_mux_select.eq(Mux(self.enable_sequencer_rom,
                                     self.rom.y_mux_select, self.trap_rom.y_mux_select)),

            # -> Z
            self.z_mux_select.eq(Mux(self.enable_sequencer_rom,
                                     self.rom.z_mux_select, self.trap_rom.z_mux_select)),
            self.alu_op_to_z.eq(Mux(self.enable_sequencer_rom, self.rom.alu_op_to_z,
                                    self.trap_rom.alu_op_to_z)),

            # -> PC
            self.pc_mux_select.eq(Mux(self.enable_sequencer_rom,
                                      self.rom.pc_mux_select, self.trap_rom.pc_mux_select)),

            # -> tmp
            self.tmp_mux_select.eq(self.rom.tmp_mux_select),

            # -> csr_num
            self._funct12_to_csr_num.eq(self.rom._funct12_to_csr_num),
            self._mepc_num_to_csr_num.eq(self.rom._mepc_num_to_csr_num),
            self._mcause_to_csr_num.eq(Mux(self.enable_sequencer_rom,
                                           self.rom._mcause_to_csr_num, self.trap_rom._mcause_to_csr_num)),

            # -> memaddr
            self.memaddr_mux_select.eq(Mux(self.enable_sequencer_rom,
                                           self.rom.memaddr_mux_select, self.trap_rom.memaddr_mux_select)),

            # -> memdata
            self.memdata_wr_mux_select.eq(self.rom.memdata_wr_mux_select),

            # -> various CSRs
            self.clear_pend_mti.eq(self.trap_rom.clear_pend_mti),
            self.clear_pend_mei.eq(self.trap_rom.clear_pend_mei),

            self.enter_trap.eq(self.rom.enter_trap | self.trap_rom.enter_trap),
            self.exit_trap.eq(self.rom.exit_trap | self.trap_rom.exit_trap),

            # Signals for next registers
            self.load_trap.eq(
                self.rom.load_trap | self.trap_rom.load_trap | self.irq_load_rom.load_trap),
            self.next_trap.eq(
                self.rom.next_trap | self.trap_rom.next_trap | self.irq_load_rom.next_trap),
            self.load_exception.eq(
                self.rom.load_exception | self.trap_rom.load_exception),
            self.next_exception.eq(
                self.rom.next_exception | self.trap_rom.next_exception),
            self.next_fatal.eq(self.rom.next_fatal | self.trap_rom.next_fatal),
        ]

    def encode_opcode_select(self, m: Module):
        m.d.comb += self.opcode_select.eq(OpcodeSelect.NONE)
        m.d.comb += self._imm_format.eq(OpcodeFormat.R)

        with m.Switch(self._opcode):
            with m.Case(Opcode.LUI):
                m.d.comb += self.opcode_select.eq(OpcodeSelect.LUI)
                m.d.comb += self._imm_format.eq(OpcodeFormat.U)

            with m.Case(Opcode.AUIPC):
                m.d.comb += self.opcode_select.eq(OpcodeSelect.AUIPC)
                m.d.comb += self._imm_format.eq(OpcodeFormat.U)

            with m.Case(Opcode.OP_IMM):
                m.d.comb += self.opcode_select.eq(OpcodeSelect.OP_IMM)
                m.d.comb += self._imm_format.eq(OpcodeFormat.I)

            with m.Case(Opcode.OP):
                m.d.comb += self.opcode_select.eq(OpcodeSelect.OP)
                m.d.comb += self._imm_format.eq(OpcodeFormat.R)

            with m.Case(Opcode.JAL):
                m.d.comb += self.opcode_select.eq(OpcodeSelect.JAL)
                m.d.comb += self._imm_format.eq(OpcodeFormat.J)

            with m.Case(Opcode.JALR):
                m.d.comb += self.opcode_select.eq(OpcodeSelect.JALR)
                m.d.comb += self._imm_format.eq(OpcodeFormat.J)

            with m.Case(Opcode.BRANCH):
                m.d.comb += self.opcode_select.eq(OpcodeSelect.BRANCH)
                m.d.comb += self._imm_format.eq(OpcodeFormat.B)

            with m.Case(Opcode.LOAD):
                m.d.comb += self.opcode_select.eq(OpcodeSelect.LOAD)
                m.d.comb += self._imm_format.eq(OpcodeFormat.I)

            with m.Case(Opcode.STORE):
                m.d.comb += self.opcode_select.eq(OpcodeSelect.STORE)
                m.d.comb += self._imm_format.eq(OpcodeFormat.S)

            with m.Case(Opcode.SYSTEM):
                m.d.comb += self._imm_format.eq(OpcodeFormat.SYS)

                with m.If(self._funct3 == SystemFunc.PRIV):
                    with m.Switch(self.state._instr):
                        with m.Case(Instr.MRET):
                            m.d.comb += self.opcode_select.eq(
                                OpcodeSelect.MRET)
                        with m.Case(Instr.ECALL):
                            m.d.comb += self.opcode_select.eq(
                                OpcodeSelect.ECALL)
                        with m.Case(Instr.EBREAK):
                            m.d.comb += self.opcode_select.eq(
                                OpcodeSelect.EBREAK)
                with m.Else():
                    m.d.comb += self.opcode_select.eq(OpcodeSelect.CSRS)

    def decode_const(self, m: Module):
        const_sig = Signal(32)

        with m.Switch(self._const):
            with m.Case(ConstSelect.EXC_INSTR_ADDR_MISALIGN):
                m.d.comb += const_sig.eq(
                    TrapCause.EXC_INSTR_ADDR_MISALIGN)
            with m.Case(ConstSelect.EXC_ILLEGAL_INSTR):
                m.d.comb += const_sig.eq(TrapCause.EXC_ILLEGAL_INSTR)
            with m.Case(ConstSelect.EXC_BREAKPOINT):
                m.d.comb += const_sig.eq(TrapCause.EXC_BREAKPOINT)
            with m.Case(ConstSelect.EXC_LOAD_ADDR_MISALIGN):
                m.d.comb += const_sig.eq(
                    TrapCause.EXC_LOAD_ADDR_MISALIGN)
            with m.Case(ConstSelect.EXC_STORE_AMO_ADDR_MISALIGN):
                m.d.comb += const_sig.eq(
                    TrapCause.EXC_STORE_AMO_ADDR_MISALIGN)
            with m.Case(ConstSelect.EXC_ECALL_FROM_MACH_MODE):
                m.d.comb += const_sig.eq(
                    TrapCause.EXC_ECALL_FROM_MACH_MODE)
            with m.Case(ConstSelect.INT_MACH_EXTERNAL):
                m.d.comb += const_sig.eq(TrapCause.INT_MACH_EXTERNAL)
            with m.Case(ConstSelect.INT_MACH_TIMER):
                m.d.comb += const_sig.eq(TrapCause.INT_MACH_TIMER)
            with m.Case(ConstSelect.SHAMT_0):
                m.d.comb += const_sig.eq(0)
            with m.Case(ConstSelect.SHAMT_4):
                m.d.comb += const_sig.eq(4)
            with m.Case(ConstSelect.SHAMT_8):
                m.d.comb += const_sig.eq(8)
            with m.Case(ConstSelect.SHAMT_16):
                m.d.comb += const_sig.eq(16)
            with m.Case(ConstSelect.SHAMT_24):
                m.d.comb += const_sig.eq(24)
            with m.Default():
                m.d.comb += const_sig.eq(0)

        return const_sig

    def updates(self, m: Module):
        with m.If(self._load_instr):
            m.d.ph2r += self.state._instr.eq(self.memdata_rd)

        m.d.ph1 += self.state._instr_phase.eq(self._next_instr_phase)
        m.d.ph1 += self.state.reg_page.eq(self._next_reg_page)
        m.d.ph1 += self.state._stored_alu_eq.eq(self.alu_eq)
        m.d.ph1 += self.state._stored_alu_lt.eq(self.alu_lt)
        m.d.ph1 += self.state._stored_alu_ltu.eq(self.alu_ltu)

        with m.If(self.load_trap):
            m.d.ph1 += self.state.trap.eq(self.next_trap)

        with m.If(self.load_exception):
            m.d.ph2 += self.state.exception.eq(self.next_exception)
            m.d.ph2 += self.state.fatal.eq(self.next_fatal)

        if self.chips:
            self.multiplex_to_pc_chips(m)
            self.multiplex_to_memaddr_chips(m)
            self.multiplex_to_memdata_chips(m)
            self.multiplex_to_tmp_chips(m)
            self.multiplex_to_x_chips(m)
            self.multiplex_to_y_chips(m)
            self.multiplex_to_z_chips(m)
            self.multiplex_to_csr_num_chips(m)
            self.multiplex_to_reg_nums_chips(m)
            self.multiplex_to_csrs_chips(m)
        else:
            self.multiplex_to_pc(m)
            self.multiplex_to_memaddr(m)
            self.multiplex_to_memdata(m)
            self.multiplex_to_tmp(m)
            self.multiplex_to_x(m)
            self.multiplex_to_y(m)
            self.multiplex_to_z(m)
            self.multiplex_to_csr_num(m)
            self.multiplex_to_reg_nums(m)
            self.multiplex_to_csrs(m)

    def process(self, m: Module):
        # Decode instruction
        m.d.comb += [
            self._opcode.eq(self.state._instr[:7]),
            self._rs1.eq(self.state._instr[15:20]),
            self._rs2.eq(self.state._instr[20:25]),
            self._rd.eq(self.state._instr[7:12]),
            self._funct3.eq(self.state._instr[12:15]),
            self._funct7.eq(self.state._instr[25:]),
            self._alu_func[:3].eq(self._funct3),
            self._alu_func[3].eq(self._funct7[5]),
            self._funct12.eq(self.state._instr[20:]),
        ]
        if (self.chips):
            self.decode_imm_chips(m)
        else:
            self.decode_imm(m)

        # We don't evaluate the instruction for badness until after it's loaded.
        ph2r_clk = ClockSignal("ph2r")
        m.d.comb += self.bad_instr.eq(ph2r_clk & (
            (self.state._instr[:16] == 0) |
            (self.state._instr == 0xFFFFFFFF)))

        with m.Switch(self._funct3):
            with m.Case(BranchCond.EQ):
                m.d.comb += self.branch_cond.eq(self.state._stored_alu_eq == 1)
            with m.Case(BranchCond.NE):
                m.d.comb += self.branch_cond.eq(self.state._stored_alu_eq == 0)
            with m.Case(BranchCond.LT):
                m.d.comb += self.branch_cond.eq(self.state._stored_alu_lt == 1)
            with m.Case(BranchCond.GE):
                m.d.comb += self.branch_cond.eq(self.state._stored_alu_lt == 0)
            with m.Case(BranchCond.LTU):
                m.d.comb += self.branch_cond.eq(
                    self.state._stored_alu_ltu == 1)
            with m.Case(BranchCond.GEU):
                m.d.comb += self.branch_cond.eq(
                    self.state._stored_alu_ltu == 0)

    def multiplex_to_reg(self, m: Module, clk: str, reg: Signal, sels: List[Signal], sigs: List[Signal]):
        """Sets up a multiplexer with a register.

        clk is the clock domain on which the register is clocked.

        reg is the register signal.

        sels is an array of Signals which select that input for the multiplexer (active high). If
        no select is active, then the register retains its value.

        sigs is an array of Signals which are the inputs to the multiplexer.
        """
        assert len(sels) == len(sigs)

        muxreg = IC_reg32_with_mux(
            clk=clk, N=len(sels), ext_init=self.ext_init, faster=True)
        m.submodules += muxreg
        m.d.comb += reg.eq(muxreg.q)
        for i in range(len(sels)):
            m.d.comb += muxreg.n_sel[i].eq(~sels[i])
            m.d.comb += muxreg.d[i].eq(sigs[i])

    def multiplex_to_bus(self, m: Module, bus: Signal, sels: List[Signal], sigs: List[Signal]):
        """Sets up a multiplexer to a bus."""
        assert len(sels) == len(sigs)

        mux = IC_mux32(N=len(sels), faster=True)
        m.submodules += mux
        m.d.comb += bus.eq(mux.y)
        for i in range(len(sels)):
            m.d.comb += mux.n_sel[i].eq(~sels[i])
            m.d.comb += mux.a[i].eq(sigs[i])

    def multiplex_to_csrs(self, m: Module):
        self.multiplex_to(m, self.state._mtvec,
                          self.mtvec_mux_select, clk="ph2w")
        # with m.If(self.z_to_csr & (self.csr_num == CSRAddr.MTVEC)):
        #     m.d.ph2w += self.state._mtvec.eq(self.data_z_in)

    def multiplex_to_csrs_chips(self, m: Module):
        self.multiplex_to_csrs(m)
        # self.multiplex_to_reg(m, clk="ph2w", reg=self.state._mtvec,
        #                       sels=[
        #                           self.z_to_csr & (
        #                               self.csr_num == CSRAddr.MTVEC)
        #                       ],
        #                       sigs=[self.data_z_in])

    def multiplex_to(self, m: Module, sig: Signal, sel: Signal, clk: str):
        with m.Switch(sel):
            with m.Case(SeqMuxSelect.MEMDATA_WR):
                m.d[clk] += sig.eq(self.state.memdata_wr)
            with m.Case(SeqMuxSelect.MEMDATA_RD):
                m.d[clk] += sig.eq(self.memdata_rd)
            with m.Case(SeqMuxSelect.MEMADDR):
                m.d[clk] += sig.eq(self.state.memaddr)
            with m.Case(SeqMuxSelect.MEMADDR_LSB_MASKED):
                m.d[clk] += sig.eq(self.state.memaddr & 0xFFFFFFFE)
            with m.Case(SeqMuxSelect.PC):
                m.d[clk] += sig.eq(self.state._pc)
            with m.Case(SeqMuxSelect.PC_PLUS_4):
                m.d[clk] += sig.eq(self._pc_plus_4)
            with m.Case(SeqMuxSelect.MTVEC):
                m.d[clk] += sig.eq(self.state._mtvec)
            with m.Case(SeqMuxSelect.MTVEC_LSR2):
                m.d[clk] += sig.eq(self.state._mtvec >> 2)
            with m.Case(SeqMuxSelect.TMP):
                m.d[clk] += sig.eq(self.state._tmp)
            with m.Case(SeqMuxSelect.IMM):
                m.d[clk] += sig.eq(self._imm)
            with m.Case(SeqMuxSelect.INSTR):
                m.d[clk] += sig.eq(self.state._instr)
            with m.Case(SeqMuxSelect.X):
                m.d[clk] += sig.eq(self.data_x_in)
            with m.Case(SeqMuxSelect.Y):
                m.d[clk] += sig.eq(self.data_y_in)
            with m.Case(SeqMuxSelect.Z):
                m.d[clk] += sig.eq(self.data_z_in)
            with m.Case(SeqMuxSelect.Z_LSL2):
                m.d[clk] += sig.eq(self.data_z_in << 2)
            with m.Case(SeqMuxSelect.CONST):
                m.d[clk] += sig.eq(self.decode_const(m))

    def multiplex_to_pc(self, m: Module):
        self.multiplex_to(m, self.state._pc, self.pc_mux_select, clk="ph1")

    def multiplex_to_pc_chips(self, m: Module):
        self.multiplex_to_pc(m)
        # self.multiplex_to_reg(m, clk="ph1", reg=self.state._pc,
        #                       sels=[
        #                           self._pc_plus_4_to_pc,
        #                           self._x_to_pc,
        #                           self._z_to_pc,
        #                           self._memaddr_to_pc,
        #                           self._memdata_to_pc,
        #                           self._z_30_to_pc,
        #                       ],
        #                       sigs=[
        #                           self._pc_plus_4,
        #                           self.data_x_in,
        #                           self.data_z_in,
        #                           self.state.memaddr & 0xFFFFFFFE,
        #                           self.memdata_rd,
        #                           self.data_z_in << 2,
        #                       ])

    def multiplex_to_tmp(self, m: Module):
        self.multiplex_to(m, self.state._tmp, self.tmp_mux_select, clk="ph2w")

    def multiplex_to_tmp_chips(self, m: Module):
        self.multiplex_to_tmp(m)
        # self.multiplex_to_reg(m, clk="ph2w", reg=self.state._tmp,
        #                       sels=[
        #                           self._x_to_tmp,
        #                           self._z_to_tmp,
        #                       ],
        #                       sigs=[
        #                           self.data_x_in,
        #                           self.data_z_in,
        #                       ])

    def multiplex_to_memdata(self, m: Module):
        self.multiplex_to(m, self.state.memdata_wr,
                          self.memdata_wr_mux_select, clk="ph1")

    def multiplex_to_memdata_chips(self, m: Module):
        self.multiplex_to_memdata(m)
        # self.multiplex_to_reg(m, clk="ph1", reg=self.state.memdata_wr,
        #                       sels=[
        #                           self._z_to_memdata,
        #                       ],
        #                       sigs=[
        #                           self.data_z_in,
        #                       ])

    def multiplex_to_memaddr(self, m: Module):
        self.multiplex_to(m, self.state.memaddr,
                          self.memaddr_mux_select, clk="ph1")

    def multiplex_to_memaddr_chips(self, m: Module):
        self.multiplex_to_memaddr(m)
        # self.multiplex_to_reg(m, clk="ph1", reg=self.state.memaddr,
        #                       sels=[
        #                           self._pc_plus_4_to_memaddr,
        #                           self._x_to_memaddr,
        #                           self._z_to_memaddr,
        #                           self._memdata_to_memaddr,
        #                           self._z_30_to_memaddr,
        #                       ],
        #                       sigs=[
        #                           self._pc_plus_4,
        #                           self.data_x_in,
        #                           self.data_z_in,
        #                           self.memdata_rd,
        #                           self.data_z_in << 2,
        #                       ])

    def multiplex_to_x(self, m: Module):
        # with m.If(self.csr_to_x):
        #     with m.Switch(self.csr_num):
        #         with m.Case(CSRAddr.MTVEC):
        #             m.d.comb += self.data_x_out.eq(self.state._mtvec)
        with m.If(self.x_mux_select != SeqMuxSelect.X):
            self.multiplex_to(m, self.data_x_out,
                              self.x_mux_select, clk="comb")

    def multiplex_to_x_chips(self, m: Module):
        self.multiplex_to_x(m)
        # self.multiplex_to_bus(m, bus=self.data_x_out,
        #                       sels=[
        #                           self._pc_to_x,
        #                           self._memdata_to_x,
        #                           self._trapcause_to_x,
        #                           self.csr_to_x & (
        #                               self.csr_num == CSRAddr.MTVEC),
        #                       ],
        #                       sigs=[self.state._pc, self.memdata_rd, self._trapcause,
        #                             self.state._mtvec,
        #                             ])

    def multiplex_to_y(self, m: Module):
        with m.If(self.y_mux_select != SeqMuxSelect.Y):
            self.multiplex_to(m, self.data_y_out,
                              self.y_mux_select, clk="comb")

    def multiplex_to_y_chips(self, m: Module):
        self.multiplex_to_y(m)
        # self.multiplex_to_bus(m, bus=self.data_y_out,
        #                       sels=[
        #                           self._imm_to_y,
        #                           self._shamt_to_y,
        #                           self._pc_to_y,
        #                           self._pc_plus_4_to_y,
        #                           self._mtvec_30_to_y,
        #                       ],
        #                       sigs=[
        #                           self._imm,
        #                           self._shamt,
        #                           self.state._pc,
        #                           self._pc_plus_4,
        #                           self.state._mtvec >> 2,
        #                       ])

    def multiplex_to_z(self, m: Module):
        with m.If(self.z_mux_select != SeqMuxSelect.Z):
            self.multiplex_to(m, self.data_z_out,
                              self.z_mux_select, clk="comb")

    def multiplex_to_z_chips(self, m: Module):
        self.multiplex_to_z(m)
        # self.multiplex_to_bus(m, bus=self.data_z_out,
        #                       sels=[
        #                           self._pc_plus_4_to_z,
        #                           self._tmp_to_z,
        #                           self._pc_to_z,
        #                           self._instr_to_z,
        #                           self._memaddr_to_z,
        #                           self._memaddr_lsb_masked_to_z,
        #                       ],
        #                       sigs=[
        #                           self._pc_plus_4,
        #                           self.state._tmp,
        #                           self.state._pc,
        #                           self.state._instr,
        #                           self.state.memaddr,
        #                           self.state.memaddr & 0xFFFFFFFE,
        #                       ])

    def multiplex_to_csr_num(self, m: Module):
        with m.If(self._funct12_to_csr_num):
            m.d.comb += self.csr_num.eq(self._funct12)
        with m.Elif(self._mepc_num_to_csr_num):
            m.d.comb += self.csr_num.eq(CSRAddr.MEPC)
        with m.Elif(self._mcause_to_csr_num):
            m.d.comb += self.csr_num.eq(CSRAddr.MCAUSE)

    def multiplex_to_csr_num_chips(self, m: Module):
        self.multiplex_to_bus(m, bus=self.csr_num,
                              sels=[
                                  self._funct12_to_csr_num,
                                  self._mepc_num_to_csr_num,
                                  self._mcause_to_csr_num,
                              ],
                              sigs=[
                                  self._funct12,
                                  CSRAddr.MEPC,
                                  CSRAddr.MCAUSE,
                              ])

    def multiplex_to_reg_nums(self, m: Module):
        with m.Switch(self._x_reg_select):
            with m.Case(InstrReg.ZERO):
                m.d.comb += self.x_reg.eq(0)
            with m.Case(InstrReg.RS1):
                m.d.comb += self.x_reg.eq(self._rs1)
            with m.Case(InstrReg.RS2):
                m.d.comb += self.x_reg.eq(self._rs2)
            with m.Case(InstrReg.RD):
                m.d.comb += self.x_reg.eq(self._rd)
        with m.Switch(self._y_reg_select):
            with m.Case(InstrReg.ZERO):
                m.d.comb += self.y_reg.eq(0)
            with m.Case(InstrReg.RS1):
                m.d.comb += self.y_reg.eq(self._rs1)
            with m.Case(InstrReg.RS2):
                m.d.comb += self.y_reg.eq(self._rs2)
            with m.Case(InstrReg.RD):
                m.d.comb += self.y_reg.eq(self._rd)
        with m.Switch(self._z_reg_select):
            with m.Case(InstrReg.ZERO):
                m.d.comb += self.z_reg.eq(0)
            with m.Case(InstrReg.RS1):
                m.d.comb += self.z_reg.eq(self._rs1)
            with m.Case(InstrReg.RS2):
                m.d.comb += self.z_reg.eq(self._rs2)
            with m.Case(InstrReg.RD):
                m.d.comb += self.z_reg.eq(self._rd)

    def multiplex_to_reg_nums_chips(self, m: Module):
        self.multiplex_to_bus(m, bus=self.x_reg,
                              sels=[
                                  self._x_reg_select == InstrReg.ZERO,
                                  self._x_reg_select == InstrReg.RS1,
                                  self._x_reg_select == InstrReg.RS2,
                                  self._x_reg_select == InstrReg.RD,
                              ],
                              sigs=[
                                  0,
                                  self._rs1,
                                  self._rs2,
                                  self._rd,
                              ])
        self.multiplex_to_bus(m, bus=self.y_reg,
                              sels=[
                                  self._y_reg_select == InstrReg.ZERO,
                                  self._y_reg_select == InstrReg.RS1,
                                  self._y_reg_select == InstrReg.RS2,
                                  self._y_reg_select == InstrReg.RD,
                              ],
                              sigs=[
                                  0,
                                  self._rs1,
                                  self._rs2,
                                  self._rd,
                              ])
        self.multiplex_to_bus(m, bus=self.z_reg,
                              sels=[
                                  self._z_reg_select == InstrReg.ZERO,
                                  self._z_reg_select == InstrReg.RS1,
                                  self._z_reg_select == InstrReg.RS2,
                                  self._z_reg_select == InstrReg.RD,
                              ],
                              sigs=[
                                  0,
                                  self._rs1,
                                  self._rs2,
                                  self._rd,
                              ])

    def decode_imm(self, m: Module):
        """Decodes the immediate value out of the instruction."""
        with m.Switch(self._imm_format):
            # Format I instructions. Surprisingly, SLTIU (Set if Less Than
            # Immediate Unsigned) actually does sign-extend the immediate
            # value, and then compare as if the sign-extended immediate value
            # were unsigned!
            with m.Case(OpcodeFormat.I):
                tmp = Signal(signed(12))
                m.d.comb += tmp.eq(self.state._instr[20:])
                m.d.comb += self._imm.eq(tmp)

            # Format S instructions:
            with m.Case(OpcodeFormat.S):
                tmp = Signal(signed(12))
                m.d.comb += tmp[0:5].eq(self.state._instr[7:12])
                m.d.comb += tmp[5:].eq(self.state._instr[25:])
                m.d.comb += self._imm.eq(tmp)

            # Format R instructions:
            with m.Case(OpcodeFormat.R):
                m.d.comb += self._imm.eq(0)

            # Format U instructions:
            with m.Case(OpcodeFormat.U):
                m.d.comb += self._imm.eq(0)
                m.d.comb += self._imm[12:].eq(self.state._instr[12:])

            # Format B instructions:
            with m.Case(OpcodeFormat.B):
                tmp = Signal(signed(13))
                m.d.comb += [
                    tmp[12].eq(self.state._instr[31]),
                    tmp[11].eq(self.state._instr[7]),
                    tmp[5:11].eq(self.state._instr[25:31]),
                    tmp[1:5].eq(self.state._instr[8:12]),
                    tmp[0].eq(0),
                    self._imm.eq(tmp),
                ]

            # Format J instructions:
            with m.Case(OpcodeFormat.J):
                tmp = Signal(signed(21))
                m.d.comb += [
                    tmp[20].eq(self.state._instr[31]),
                    tmp[12:20].eq(self.state._instr[12:20]),
                    tmp[11].eq(self.state._instr[20]),
                    tmp[1:11].eq(self.state._instr[21:31]),
                    tmp[0].eq(0),
                    self._imm.eq(tmp),
                ]

            with m.Case(OpcodeFormat.SYS):
                m.d.comb += [
                    self._imm[0:5].eq(self.state._instr[15:]),
                    self._imm[5:].eq(0),
                ]

    def decode_imm_chips(self, m: Module):
        mux = IC_mux32(N=6, faster=True)
        gal = IC_GAL_imm_format_decoder()

        m.submodules += mux
        m.submodules += gal

        m.d.comb += gal.opcode.eq(self._opcode)
        m.d.comb += mux.n_sel[0].eq(gal.i_n_oe)
        m.d.comb += mux.n_sel[1].eq(gal.s_n_oe)
        m.d.comb += mux.n_sel[2].eq(gal.u_n_oe)
        m.d.comb += mux.n_sel[3].eq(gal.b_n_oe)
        m.d.comb += mux.n_sel[4].eq(gal.j_n_oe)
        m.d.comb += mux.n_sel[5].eq(gal.sys_n_oe)

        instr = self.state._instr

        # Format I
        m.d.comb += [
            mux.a[0][0:12].eq(instr[20:]),
            mux.a[0][12:].eq(Repl(instr[31], 32)),  # sext
        ]

        # Format S
        m.d.comb += [
            mux.a[1][0:5].eq(instr[7:]),
            mux.a[1][5:11].eq(instr[25:]),
            mux.a[1][11:].eq(Repl(instr[31], 32)),  # sext
        ]

        # Format U
        m.d.comb += [
            mux.a[2][0:12].eq(0),
            mux.a[2][12:].eq(instr[12:]),
        ]

        # Format B
        m.d.comb += [
            mux.a[3][0].eq(0),
            mux.a[3][1:5].eq(instr[8:]),
            mux.a[3][5:11].eq(instr[25:]),
            mux.a[3][11].eq(instr[7]),
            mux.a[3][12:].eq(Repl(instr[31], 32)),  # sext
        ]

        # Format J
        m.d.comb += [
            mux.a[4][0].eq(0),
            mux.a[4][1:11].eq(instr[21:]),
            mux.a[4][11].eq(instr[20]),
            mux.a[4][12:20].eq(instr[12:]),
            mux.a[4][20:].eq(Repl(instr[31], 32)),  # sext
        ]

        # Format SYS
        m.d.comb += [
            mux.a[5][0:5].eq(instr[15:]),
            mux.a[5][5:].eq(0),
        ]

        m.d.comb += self._imm.eq(mux.y)
